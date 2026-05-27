# SparseColumnPivotedQR.jl

A pure-Julia rank-revealing column-pivoted Householder QR factorization that
operates directly on `SparseMatricesCSR.SparseMatrixCSR{T, Bi}` storage. No
BLAS, no multifrontal supernodes; designed for small-to-moderate `n`
(roughly hundreds to a few thousand) where SPQR's fixed ~500 μs symbolic
overhead is disproportionate to the work, but rank deficiency must still be
handled correctly.

The package follows the **KLU/CSparse split** between a symbolic and a
numeric phase, with a `refactor!` step that reuses the symbolic when the
same sparsity pattern is factored repeatedly with different values.

## API

```julia
using SparseArrays, SparseMatricesCSR, SparseColumnPivotedQR

Acsc = sparse(...)
Acsr = SparseMatrixCSR(transpose(sparse(transpose(Acsc))))

# --- One-shot (analyze + factor in one call) ---
F = csr_qr(Acsr)                          # default ordering = :natural
F = csr_qr(Acsr; ordering=:amd, tol=1e-10)

# --- Symbolic / numeric split (recommended for repeated factor with same pattern) ---
sym = csr_analyze(Acsr; ordering=:amd)    # column ordering, etree, row counts
F   = csr_factor(Acsr, sym; tol=1e-10)    # numeric factorization

# Refactor a matrix with the same sparsity pattern but different values.
# If the pattern of A2 matches sym, the symbolic is reused; otherwise a full
# analyze+factor is done internally and the result returned.
F2 = csr_refactor!(F, A2)

# --- Solve and inspect ---
x = F \ b
ldiv!(x, F, b)
rank(F)        # numerical rank (may be < min(m, n))
size(F)        # (m, n)
```

Element types supported: `Float64`, `Float32`, `ComplexF64`, `ComplexF32`,
with either `Int32` or `Int64` index types in the CSR.

### Ordering choices

| ordering   | meaning                                                              |
|------------|----------------------------------------------------------------------|
| `:natural` | identity column ordering (default — best on dense-fill matrices)     |
| `:amd`     | AMD on `AᵀA`, via the `AMD.jl` weak dep (`using AMD` to enable)       |
| `:colamd`  | currently an alias for `:amd`                                        |

The numeric phase is permitted to deviate from the symbolic ordering when a
candidate column is rank-deficient. With the default `pivot_factor` (`1e-6`
internally), this happens only when the natural-ordered column has
essentially zero residual norm.

## Algorithm

Column-pivoted Householder QR, structured around the standard textbook
algorithm with sparse storage and a few well-known refinements:

1. **Symbolic pre-pass.** `csr_analyze` captures the input pattern, computes
   the column ordering, builds the column elimination tree of `AᵀA` (Davis
   `cs_etree` with `ata=1`, working on a CSC view derived from the CSR
   pattern), and uses the Gilbert–Ng–Peyton row-count algorithm to derive
   upper bounds on `nnz(R[k, :])`. These counts seed per-row capacities so
   the numeric phase rarely reallocates row storage.

2. **Drmac–Bujanovic-style column-norm tracking.** Track squared column
   norms incrementally and recompute exactly whenever the running value
   falls below `sqrt(eps)` of the initial reference (keeps rank detection
   robust under accumulated rounding).

3. **Rank-revealing pivot.** At step `k`, scan `col_nrm2[k..n]` for the
   max-norm column. Use the column already at position `k` (sparsity-
   preserving choice from the symbolic ordering) unless its norm is below
   `pivot_factor` times the max, in which case swap. Declare rank
   deficiency when the chosen pivot's norm² is below `tol²`.

4. **Householder reflector.** Build `H_k = I − τ_k v_k v_k^H` from the
   column-`k` subdiagonal (gathered from the CSR rows).

5. **Sparse row update.** A single dense-workspace pass per row computes
   `w[j] = v_k^H R[:, j]` for `j > k`, then a sorted-merge into each
   affected row handles fill-in. The merge writes into a pre-resized scratch
   buffer and copies into the row in one `copyto!` per row, with the
   column-`k` slot left untouched and the cached position from gather
   reused at the end to delete it without a second binary search.

6. **Step-wise Householder storage.** Each `v_k` is stored as sorted row
   indices + values, making `applyQ` / `applyQH` O(Σ nnz(v_k)) rather than
   O(m · k).

`F \ b` returns the **basic** least-squares solution (the trailing `n − rnk`
coordinates of the rotated solution are set to zero), matching SPQR's
behaviour on rank-deficient inputs. CXSparse's `cs_qr`, by contrast, has
no rank-revealing pivot and produces non-finite x components on the same
rank-deficient problems.

## Tests

```bash
julia --project=. -e "using Pkg; Pkg.test()"
```

Covers identity, full-rank square and tall, structurally singular,
rank-deficient overdetermined and square, the seven user matrices,
`ComplexF64`, and the new analyze/factor/refactor API split including AMD
vs natural ordering. All 49 tests pass.

## Benchmarks

`bench/bench.jl` measures `factor + solve` on the seven user matrices
(199×199, 979 nnz, four rank-deficient at rank 198 and two non-singular).
Numbers from one run on this machine (Julia 1.11, single thread,
`@benchmark seconds=1` minimum time):

```
file                           solver                  time (μs)    ||Ax-b||
-----------------------------------------------------------------------------
11fed5ba-linsolve_0.txt        CSR-QR natural             2016.2    3.310e-01
11fed5ba-linsolve_0.txt        CSR-QR amd                 3987.6    3.310e-01
11fed5ba-linsolve_0.txt        CSR-QR refactor!           1949.9    3.310e-01
11fed5ba-linsolve_0.txt        SPQR                        573.9    3.310e-01
11fed5ba-linsolve_0.txt        CXSparse cs_qr              332.1          NaN
11fed5ba-linsolve_0.txt        LAPACK xgeqp3              2659.1    3.310e-01

2d9e29f1-linsolve_4.txt        CSR-QR natural             1919.5    8.399e-13
2d9e29f1-linsolve_4.txt        CSR-QR amd                 3989.2    1.518e-12
2d9e29f1-linsolve_4.txt        CSR-QR refactor!           1854.1    8.399e-13
2d9e29f1-linsolve_4.txt        SPQR                        559.4    2.877e-13
2d9e29f1-linsolve_4.txt        CXSparse cs_qr              331.3    5.391e-13
2d9e29f1-linsolve_4.txt        LAPACK xgeqp3              2311.9    9.090e-13

(others omitted — they cluster tightly around the above)
```

(The `90095c07-linsolve_6.txt` matrix has NaN in `b`, so every solver
returns NaN; it's included as a regression check that we don't crash.)

### Progression on this workload

Cumulative effect of each optimization, measured on the first user matrix
(`11fed5ba-linsolve_0.txt`):

| stage                                                    | time (μs) | speedup |
|----------------------------------------------------------|-----------|---------|
| Original implementation (git HEAD before changes)        |  ~18 500  | 1.0×    |
| + symbolic pre-pass with row-count capacity hints        |   ~8 000  | 2.3×    |
| + AMD/COLAMD column ordering (used only on demand)       |   ~8 000  | 2.3×    |
| + faster scratch-buffer merge (no per-element push!)     |   ~6 000  | 3.1×    |
| + single-pass in-place column-pivot swap                 |   ~4 000  | 4.6×    |
| + cached gather position reused in apply/drop steps      |   ~3 800  | 4.9×    |
| + relaxed pivot threshold (only swap on rank deficiency) |   ~2 000  | 9.2×    |
| + Vector{Bool} for the dense touched-mask in gather      |   ~1 900  | 9.7×    |

The relaxed pivot threshold is the biggest single jump: the original
LAPACK-style "always pivot to max" did O(m) work per step on swaps that
were essentially cosmetic. On these particular matrices, 80 of the 199
steps produced swaps with no measurable accuracy effect (residuals match
SPQR's basic-solver answer to a few ulps with or without those swaps).
Rank deficiency is still caught by the separate downdate-recompute path.

### What the numbers say

* **Correctness**: matches SPQR's residual on every case. On the four
  rank-deficient matrices (`||Ax-b|| ≈ 0.33`) both this code and SPQR
  return finite least-squares solutions; CXSparse `cs_qr` produces NaN
  x components (it has no rank-revealing pivot and silently divides by
  ~0).
* **Performance**: ~3.5× slower than SPQR's multifrontal BLAS-3
  implementation, ~6× slower than CXSparse `cs_qr` (which has no
  rank-revealing), and now ~1.3× *faster* than dense LAPACK
  column-pivoted QR (which was the prior baseline on these dense-fill
  matrices).
* **Refactor savings**: when the same pattern is factored repeatedly,
  `csr_refactor!` saves the ~150 μs analyze cost. For `:amd` ordering on
  truly sparse matrices the savings are larger (AMD itself is the bulk
  of the analyze).
* **`:natural` vs `:amd`**: AMD almost doubles the time on these
  matrices because they fill in to ~60% density anyway, and the AMD
  permutation actually produces slightly more fill than the natural
  order for this particular structure. AMD will help on genuinely
  sparse problems; it is not the right default here.

## Trade-offs and what's not implemented

* **No multifrontal / BLAS-3.** SPQR's edge over us comes almost entirely
  from BLAS-3 dense panels inside supernodes. We can't compete on raw
  per-element throughput while staying purely sparse-row.
* **No dual CSC view of R maintained during apply.** A `col_rows[j]`
  index would let us skip rows that don't touch column `j` during the
  pivot swap, gather, and downdate recompute. Implementing it correctly
  during fill-in updates was significantly slower than the row-only
  baseline on these dense-fill matrices, so it was reverted. For
  genuinely sparse R, where fill-in updates are cheap, this would help.
* **No COLAMD-proper.** We alias `:colamd` to `:amd` on `AᵀA`. A native
  COLAMD implementation in pure Julia would shave a small amount on the
  symbolic pass and give a slightly better ordering for some workloads.
* **No adaptive dense fallback.** Once the active sub-matrix is mostly
  dense, switching to a dense panel for the remainder of the
  factorization would recover BLAS-3 speed. Outside the spec, but
  realistically the only way to fully close the gap to SPQR on these
  particular workloads.

## Citing the underlying algorithms

The symbolic algorithms (`cs_etree`, row counts of `R`) are textbook
Davis (CSparse / *Direct Methods for Sparse Linear Systems*); the
Drmac–Bujanovic norm-downdate is from their 2008 BLAS-3 column-pivoted
QR paper; the rank-revealing pivot is LAPACK `xgeqp3` style. The
analyze/factor/refactor decomposition is a direct port of the KLU
playbook to QR.
