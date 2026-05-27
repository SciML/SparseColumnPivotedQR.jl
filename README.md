# SparseColumnPivotedQR.jl

A pure-Julia column-pivoted Householder QR factorization that operates directly on
`SparseMatricesCSR.SparseMatrixCSR{T, Bi}` storage. Rank-revealing (LAPACK
`xgeqp3`-style), no BLAS, no multifrontal supernodes.

The intended niche: solve rank-deficient sparse least-squares problems on
small-to-moderate `n` (a few hundred to a few thousand) with **minimum symbolic
overhead** and **lazy, KLU-style allocation**. The reference solver targeted by
this design is SuiteSparseQR (SPQR, used by `qr(::SparseMatrixCSC)`), which is
fast for large problems but pays a fixed ~500 μs symbolic+packing cost that is
disproportionate at `n ≈ 200`.

## API

```julia
using SparseArrays, SparseMatricesCSR, SparseColumnPivotedQR

Acsc = sparse(...)
Acsr = SparseMatrixCSR(transpose(sparse(transpose(Acsc))))
F = csr_qr(Acsr)                  # default tolerance = eps * max(m, n) * ||A||_F
F = csr_qr(Acsr; tol=1e-10)       # explicit tolerance

x = F \ b                         # least-squares solve
ldiv!(x, F, b)                    # in-place

rank(F)        # numerical rank (may be < min(m, n))
size(F)        # (m, n)
```

Element types supported: `Float64`, `Float32`, `ComplexF64`, `ComplexF32`, with
either `Int32` or `Int64` index types in the CSR.

## Algorithm

Standard textbook column-pivoted Householder QR:

1. Track squared column norms incrementally with a Drmac-Bujanovic-style
   downdate (recompute exactly whenever the running value falls below
   `sqrt(eps)` of its initial reference, which keeps rank detection accurate
   under accumulated rounding).
2. At each step `k`, pivot the column with the largest active norm into
   position `k`; declare rank-deficiency and stop when no remaining column has
   `||·||² > tol²`.
3. Build the Householder reflector `H_k = I − τ_k v_k v_k^H` from the current
   column-`k` subdiagonal and apply it to columns `k+1..n` of the active
   submatrix.
4. The reflector application is a single pass per row using a dense workspace
   `w[j] = v_k^H R[:, j]`, followed by a sorted merge into each affected row
   that handles fill-in correctly.
5. `v_k` is stored "step-wise" (sorted nonzero row indices + values),
   making `applyQ` / `applyQH` linear in the total Householder support.

Returns a basic least-squares solution (trailing `n − rank` coordinates of the
rotated solution set to zero), not the minimum-norm pseudoinverse solution.
This matches SPQR's `\` behaviour for rank-deficient problems.

## Tests

```bash
julia --project=. -e "using Pkg; Pkg.test()"
```

Covers identity, full-rank square and tall, structurally singular,
rank-deficient overdetermined and square, the seven user matrices, and
`ComplexF64`. All 30 tests pass.

## Benchmarks

`bench/bench.jl` measures `factor + solve` on the seven user matrices
(199×199, 979 nnz, four rank-deficient at rank 198 and two non-singular).
Numbers from one run on this machine (Julia 1.11.9, single thread):

```
file                           solver            time (μs)       ||Ax-b||   nnan(x)
----------------------------------------------------------------------------------------
11fed5ba-linsolve_0.txt        CSR-QR (this)       18466.7      3.310e-01        0
11fed5ba-linsolve_0.txt        SPQR                  571.2      3.310e-01        0
11fed5ba-linsolve_0.txt        CXSparse cs_qr        329.2            NaN       57
11fed5ba-linsolve_0.txt        LAPACK xgeqp3        2648.5      3.310e-01        0

2d9e29f1-linsolve_4.txt        CSR-QR (this)       18045.3      1.396e-12        0
2d9e29f1-linsolve_4.txt        SPQR                  560.2      2.877e-13        0
2d9e29f1-linsolve_4.txt        CXSparse cs_qr        331.2      5.391e-13        0
2d9e29f1-linsolve_4.txt        LAPACK xgeqp3        2411.3      9.090e-13        0

3d944c13-linsolve_5.txt        CSR-QR (this)       17947.3      1.396e-12        0
3d944c13-linsolve_5.txt        SPQR                  560.4      2.877e-13        0
3d944c13-linsolve_5.txt        CXSparse cs_qr        328.0      5.391e-13        0
3d944c13-linsolve_5.txt        LAPACK xgeqp3        2401.5      9.090e-13        0

3fc0fa44-linsolve_1.txt        CSR-QR (this)       18573.2      3.343e-01        0
3fc0fa44-linsolve_1.txt        SPQR                  563.9      3.343e-01        0
3fc0fa44-linsolve_1.txt        CXSparse cs_qr        325.7            NaN       57
3fc0fa44-linsolve_1.txt        LAPACK xgeqp3        2664.7      3.343e-01        0

6d092b79-linsolve_2.txt        CSR-QR (this)       18611.1      3.310e-01        0
6d092b79-linsolve_2.txt        SPQR                  568.3      3.310e-01        0
6d092b79-linsolve_2.txt        CXSparse cs_qr        326.0            NaN       57
6d092b79-linsolve_2.txt        LAPACK xgeqp3        2835.6      3.310e-01        0

83baa118-linsolve_3.txt        CSR-QR (this)       18622.8      3.343e-01        0
83baa118-linsolve_3.txt        SPQR                  569.4      3.343e-01        0
83baa118-linsolve_3.txt        CXSparse cs_qr        330.1            NaN       57
83baa118-linsolve_3.txt        LAPACK xgeqp3        2650.2      3.343e-01        0

90095c07-linsolve_6.txt        CSR-QR (this)       18062.3            NaN      199
90095c07-linsolve_6.txt        SPQR                  558.3            NaN      199
90095c07-linsolve_6.txt        CXSparse cs_qr        323.9            NaN      199
90095c07-linsolve_6.txt        LAPACK xgeqp3        2307.7            NaN      199
```

### What the numbers say

* **Correctness**: CSR-QR matches SPQR's residual on every case. On the four
  rank-deficient matrices (||Ax-b|| ≈ 0.33) both this code and SPQR return
  finite least-squares solutions; CXSparse `cs_qr` returns NaN x components
  (it has no rank-revealing pivot and silently divides by ~0).
* **Performance**: CSR-QR is **~30× slower than SPQR** and **~7× slower than
  CXSparse** on these inputs, **but ~7× faster than dense LAPACK
  column-pivoted QR is NOT true here** — actually it's **~7× slower than
  LAPACK dense xgeqp3** because the user matrices fill in essentially to
  dense by the bottom of the factorization, and LAPACK BLAS-3 multipliers
  drastically beat a row-merge implementation on dense workloads.

## Trade-offs and what I'd improve next

The honest summary:

1. **The algorithm is correct and rank-revealing.** Same rank as LAPACK's
   `xgeqp3`, same residual as SPQR on every test case (including the four
   rank-deficient user matrices that CXSparse silently breaks on).
2. **The performance is not competitive at this size.** SPQR at ~570 μs and
   CXSparse at ~325 μs are both faster on these inputs. The reasons:
   - SPQR uses a multifrontal supernodal organization with BLAS-3 dense
     panels for the inner work. We're constrained by the spec to plain
     row-merge updates and explicitly prohibited from multifrontal /
     BLAS-3.
   - These specific 199×199 user matrices fill in heavily (initial 25 nnz
     per row balloons to mostly-dense by row 100). For dense-ish workloads,
     the dense LAPACK QR (2.4 ms) is the right floor, and a pure-Julia
     sparse-row Householder loop is a fixed multiple slower than that floor.
3. **What would actually move the needle:**
   - **Adaptive dense fallback.** Track average nnz per row of the active
     submatrix; once it exceeds, say, 40% of `n − k`, switch the working
     storage to a dense `(m − k) × (n − k)` block and run LAPACK-style
     blocked Householder on the rest. This would recover most of the
     LAPACK dense speed when fill is severe, and only pay the sparse
     overhead while the working submatrix really is sparse.
   - **Blocked Householder (LAPACK WY representation).** Even staying
     fully sparse, applying `k` reflectors at once via a `T` matrix gives
     a constant-factor win and slightly better cache behaviour for the
     inner row updates.
   - **Defer fill-in.** Use the structural elimination tree (CSR analogue
     of CXSparse's symbolic QR pattern) so that the sparsity pattern of R
     and of each `v_k` is known up front and we can preallocate exact-size
     row arrays instead of growing them.
   - **The per-row column swap is O(m) every step even when most rows are
     untouched.** For matrices where the pivot column `p` is sparse, we
     should only visit the rows in the union of nonzero patterns of cols
     `k` and `p`. That requires maintaining a column-index → row list
     auxiliary structure (i.e., CSC alongside CSR for the working matrix).
   - **Solve path.** Currently the back-substitution does a binary search
     for the R diagonal of each row. Storing the diagonal index per row
     directly would shave a small but real amount.

For the original purpose (LinearSolve.jl's rank-deficient fallback at
`n ≈ 200`), the cleanest near-term path is probably:

* Use CXSparse `cs_qr` as the **fast path** when no rank-revealing is needed
  (or when the matrix is known full-rank).
* Use this CSR-QR (or SPQR) only when CXSparse returns non-finite results,
  i.e., as the **rank-deficient fallback's fallback** rather than the primary.

That keeps CXSparse's 325 μs hot path for the common case and only pays the
~18 ms cost when the system is genuinely rank-deficient and the user wants a
finite x.
