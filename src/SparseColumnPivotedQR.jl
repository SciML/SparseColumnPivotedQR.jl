module SparseColumnPivotedQR

using LinearAlgebra
using SparseArrays
using SparseMatricesCSR

import LinearAlgebra: ldiv!, rank
import Base: \, size, eltype

export csr_qr, csr_analyze, csr_factor, csr_refactor!,
       has_amd_extension,
       CSRQRSymbolic, CSRQRFactorization

# CSR offset: rowptr stores 1-based if Bi == 1, 0-based if Bi == 0.
@inline function getoffset(::SparseMatrixCSR{Bi}) where {Bi}
    return Bi == 1 ? 0 : 1
end

# ---------------------------------------------------------------------------
# CSC-internal layout for V and R during numeric phase.
# ---------------------------------------------------------------------------
#
# Both V (Householder vectors) and R (upper-triangular factor) are kept in
# compressed-column form: colptr (length n+1), rowval and nzval (length =
# total nnz). Column k of V is rowval[colptr[k]:colptr[k+1]-1] and same for
# nzval. Same for R.
#
# Pre-sized exactly from the symbolic phase (subject to per-step growth if
# the conservative bound was undershot — handled by `_grow_csc!`).

mutable struct _CSCBuf{T}
    m::Int
    n::Int
    colptr::Vector{Int}
    rowval::Vector{Int}
    nzval::Vector{T}
end

@inline function _alloc_csc(::Type{T}, m::Int, n::Int, nzmax::Int) where {T}
    # colptr left uninitialized; the caller sets colptr[1] = 1 and writes
    # colptr[k+1] at the end of each step.
    return _CSCBuf{T}(m, n, Vector{Int}(undef, n + 1),
                      Vector{Int}(undef, max(nzmax, 1)),
                      Vector{T}(undef, max(nzmax, 1)))
end

@inline function _grow_csc!(B::_CSCBuf{T}, needed::Int) where {T}
    L = length(B.rowval)
    new_L = max(2 * L, needed)
    resize!(B.rowval, new_L)
    resize!(B.nzval, new_L)
    return nothing
end

# ---------------------------------------------------------------------------
# Symbolic
# ---------------------------------------------------------------------------
#
# The `Symbolic` captures:
#   - `q`         : column permutation (1-based, length n)
#   - `pinv`      : row permutation (1-based, length m2; "S_row = pinv[A_row]")
#   - `parent`    : column etree of S = (P A Q)
#   - `leftmost`  : leftmost[i] = first column k where S[i, :] is nonzero
#                   (indexed by permuted row 1..m2; 0 = empty row)
#   - `vnz`/`rnz` : upper bounds on nnz(V), nnz(R) used to pre-size buffers
#   - `m2`        : padded row count
#   - `ordering`  : symbolic ordering kind (:natural / :amd / :colamd)
#   - `pattern_*` : captured CSR pattern of the input (for refactor! pattern
#                   matching). Stored 1-based, internal layout.

struct CSRQRSymbolic
    m::Int
    n::Int
    m2::Int
    q::Vector{Int}
    pinv::Vector{Int}
    parent::Vector{Int}
    leftmost::Vector{Int}
    vnz::Int
    rnz::Int
    ordering::Symbol
    pattern_rowptr::Vector{Int}
    pattern_colval::Vector{Int}
end

Base.size(S::CSRQRSymbolic) = (S.m, S.n)

# ---------------------------------------------------------------------------
# Factorization
# ---------------------------------------------------------------------------
#
# CSC storage of V (Householders) and R, plus beta (Householder coefficients),
# plus the symbolic. Permutations come from `sym`.

struct CSRQRFactorization{T, RT}
    m::Int
    n::Int
    V_colptr::Vector{Int}
    V_rowval::Vector{Int}
    V_nzval::Vector{T}
    R_colptr::Vector{Int}
    R_rowval::Vector{Int}
    R_nzval::Vector{T}
    beta::Vector{RT}
    rnk::Int
    tol::RT
    sym::CSRQRSymbolic
end

LinearAlgebra.rank(F::CSRQRFactorization) = F.rnk
Base.size(F::CSRQRFactorization) = (F.m, F.n)
Base.size(F::CSRQRFactorization, d::Integer) = d == 1 ? F.m : (d == 2 ? F.n : 1)
Base.eltype(::CSRQRFactorization{T}) where {T} = T

# ---------------------------------------------------------------------------
# CSR <-> CSC conversion (pattern + values)
# ---------------------------------------------------------------------------

function _csr_to_csc(A::SparseMatrixCSR{Bi, T}) where {Bi, T}
    m, n = size(A)
    off = getoffset(A)
    rowptr = A.rowptr
    colval = A.colval
    nzval_in = A.nzval
    nnz_total = length(colval)

    colcounts = zeros(Int, n)
    @inbounds for p in 1:nnz_total
        colcounts[Int(colval[p]) + off] += 1
    end
    colptr = Vector{Int}(undef, n + 1)
    colptr[1] = 1
    @inbounds for j in 1:n
        colptr[j + 1] = colptr[j] + colcounts[j]
    end
    rowval = Vector{Int}(undef, nnz_total)
    nzval = Vector{T}(undef, nnz_total)
    work = copy(colptr)
    @inbounds for i in 1:m
        r1 = rowptr[i] + off
        r2 = rowptr[i + 1] + off - 1
        for p in r1:r2
            j = Int(colval[p]) + off
            slot = work[j]
            rowval[slot] = i
            nzval[slot] = nzval_in[p]
            work[j] = slot + 1
        end
    end
    return colptr, rowval, nzval, m, n
end

function _csr_pattern_to_csc(rowptr::Vector{Int}, colval::Vector{Int},
                             m::Int, n::Int)
    nnz_total = length(colval)
    colcounts = zeros(Int, n)
    @inbounds for p in 1:nnz_total
        colcounts[colval[p]] += 1
    end
    colptr = Vector{Int}(undef, n + 1)
    colptr[1] = 1
    @inbounds for j in 1:n
        colptr[j + 1] = colptr[j] + colcounts[j]
    end
    rowval = Vector{Int}(undef, nnz_total)
    work = copy(colptr)
    @inbounds for i in 1:m
        r1 = rowptr[i]; r2 = rowptr[i + 1] - 1
        for p in r1:r2
            j = colval[p]
            rowval[work[j]] = i
            work[j] += 1
        end
    end
    return colptr, rowval
end

function _capture_pattern(A::SparseMatrixCSR{Bi}) where {Bi}
    off = getoffset(A)
    rowptr = Vector{Int}(undef, length(A.rowptr))
    @inbounds for i in eachindex(A.rowptr)
        rowptr[i] = Int(A.rowptr[i]) + off
    end
    colval = Vector{Int}(undef, length(A.colval))
    @inbounds for i in eachindex(A.colval)
        colval[i] = Int(A.colval[i]) + off
    end
    return rowptr, colval
end

function _pattern_matches(S::CSRQRSymbolic, A::SparseMatrixCSR{Bi}) where {Bi}
    m, n = size(A)
    (m == S.m && n == S.n) || return false
    length(A.rowptr) == length(S.pattern_rowptr) || return false
    length(A.colval) == length(S.pattern_colval) || return false
    off = getoffset(A)
    @inbounds for i in eachindex(A.rowptr)
        Int(A.rowptr[i]) + off == S.pattern_rowptr[i] || return false
    end
    @inbounds for i in eachindex(A.colval)
        Int(A.colval[i]) + off == S.pattern_colval[i] || return false
    end
    return true
end

# ---------------------------------------------------------------------------
# Column elimination tree of S = A(:, q) using Davis cs_etree with ata=1.
# ---------------------------------------------------------------------------

function _coletree_ata(colptr::Vector{Int}, rowval::Vector{Int}, m::Int, n::Int)
    parent = zeros(Int, n)
    ancestor = zeros(Int, n)
    prev = zeros(Int, m)
    @inbounds for k in 1:n
        c1 = colptr[k]; c2 = colptr[k + 1] - 1
        for p in c1:c2
            i = rowval[p]
            ii = prev[i]
            while ii != 0 && ii < k
                inext = ancestor[ii]
                ancestor[ii] = k
                if inext == 0
                    parent[ii] = k
                    break
                end
                ii = inext
            end
            prev[i] = k
        end
    end
    return parent
end

# Permute CSC pattern: output column k = input column q[k].
function _permute_cols(colptr::Vector{Int}, rowval::Vector{Int},
                       q::Vector{Int}, m::Int, n::Int)
    nnz_total = length(rowval)
    colptr_q = Vector{Int}(undef, n + 1)
    colptr_q[1] = 1
    @inbounds for k in 1:n
        j = q[k]
        colptr_q[k + 1] = colptr_q[k] + (colptr[j + 1] - colptr[j])
    end
    rowval_q = Vector{Int}(undef, nnz_total)
    @inbounds for k in 1:n
        j = q[k]
        src = colptr[j]; n_in_col = colptr[j + 1] - colptr[j]
        dst = colptr_q[k]
        for t in 0:(n_in_col - 1)
            rowval_q[dst + t] = rowval[src + t]
        end
    end
    return colptr_q, rowval_q
end

# Apply both perms (P A Q): output col k = (PA)[:, q[k]]. Row i becomes pinv[i].
# Also permutes nzval if provided.
function _permute_pq(colptr::Vector{Int}, rowval::Vector{Int},
                     nzval::Union{Nothing, Vector{T}},
                     pinv::Vector{Int}, q::Vector{Int},
                     m::Int, n::Int) where {T}
    nnz_total = length(rowval)
    colptr_pq = Vector{Int}(undef, n + 1)
    colptr_pq[1] = 1
    @inbounds for k in 1:n
        j = q[k]
        colptr_pq[k + 1] = colptr_pq[k] + (colptr[j + 1] - colptr[j])
    end
    rowval_pq = Vector{Int}(undef, nnz_total)
    if nzval === nothing
        @inbounds for k in 1:n
            j = q[k]
            src = colptr[j]; n_in_col = colptr[j + 1] - colptr[j]
            dst = colptr_pq[k]
            for t in 0:(n_in_col - 1)
                rowval_pq[dst + t] = pinv[rowval[src + t]]
            end
        end
        return colptr_pq, rowval_pq, nothing
    else
        nzval_pq = Vector{T}(undef, nnz_total)
        @inbounds for k in 1:n
            j = q[k]
            src = colptr[j]; n_in_col = colptr[j + 1] - colptr[j]
            dst = colptr_pq[k]
            for t in 0:(n_in_col - 1)
                rowval_pq[dst + t] = pinv[rowval[src + t]]
                nzval_pq[dst + t] = nzval[src + t]
            end
        end
        return colptr_pq, rowval_pq, nzval_pq
    end
end

# ---------------------------------------------------------------------------
# Symbolic analysis: build q, pinv, etree, leftmost, vnz, rnz.
# Implements Davis cs_sqr for QR with order = (passed in).
# ---------------------------------------------------------------------------

# Default no-op AMD hook; overridden by the AMD.jl extension.
_amd_colperm(rowptr, colval, m, n) = collect(1:n)

# Set to `true` by the AMD.jl extension on `__init__`. Lets `csr_analyze` /
# `csr_qr` resolve the default ordering (`:default`) to `:amd` only when the
# extension is actually loaded, falling back to `:natural` otherwise. Using a
# Ref so it can be flipped from the extension at load time.
const _AMD_EXT_LOADED = Ref(false)

"""
    has_amd_extension() -> Bool

Returns `true` iff the `AMD.jl` extension has been loaded into the current
session (i.e. `using AMD` has been executed). The default ordering
`:default` resolves to `:amd` only when this is `true`, falling back to
`:natural` otherwise.
"""
has_amd_extension() = _AMD_EXT_LOADED[]

# Resolve `:default` to `:amd` when the AMD extension is loaded, `:natural`
# otherwise. All other symbols pass through.
@inline function _resolve_ordering(ordering::Symbol)
    if ordering === :default
        return _AMD_EXT_LOADED[] ? :amd : :natural
    end
    return ordering
end

# Cheap fill estimator: total depth of the column elimination tree (sum
# over k of distance from k to root). Mirrors what the apply step pays —
# deeper chain → more Householders touched per column. O(n) thanks to
# the bottom-up recursion using a `depth` cache.
function _etree_total_depth(parent::Vector{Int})
    n = length(parent)
    depth = zeros(Int, n)
    total = 0
    @inbounds for k in 1:n
        # parent[k] is always > k (etree of column k has children with
        # smaller indices). Walk up; cache depths so each node is touched
        # once. Equivalent to the recursive depth function with memoisation.
        if depth[k] == 0
            # depth from k to root, computed via iteration with stack-free
            # caching: walk to a node whose depth is known, then assign
            # depths back along the path.
            j = k
            len = 0
            while j != 0 && depth[j] == 0
                len += 1
                j = parent[j]
            end
            base = j == 0 ? 0 : depth[j]
            j = k
            d = base + len
            while j != 0 && depth[j] == 0
                depth[j] = d
                d -= 1
                j = parent[j]
            end
        end
        total += depth[k]
    end
    return total
end

function _build_symbolic(rowptr::Vector{Int}, colval::Vector{Int},
                          m::Int, n::Int, ordering::Symbol)
    # 1) Column permutation `q`.
    q = if ordering === :natural
        collect(1:n)
    elseif ordering === :amd || ordering === :colamd
        _amd_colperm(rowptr, colval, m, n)
    else
        throw(ArgumentError("Unknown ordering :$ordering"))
    end

    # 2) Build CSC pattern of A.
    colptr_A, rowval_A = _csr_pattern_to_csc(rowptr, colval, m, n)

    # 3) Build CSC pattern of A(:, q).
    colptr_q, rowval_q = _permute_cols(colptr_A, rowval_A, q, m, n)

    # 4) Etree of A(:, q)^T A(:, q).
    parent = _coletree_ata(colptr_q, rowval_q, m, n)

    # 5) Compute leftmost[i] for each row i of A(:, q).
    # leftmost (indexed by original row 1..m) = min col k where (A_q)[i,k] != 0.
    leftmost_orig = zeros(Int, m)
    @inbounds for k in 1:n
        c1 = colptr_q[k]; c2 = colptr_q[k + 1] - 1
        for p in c1:c2
            i = rowval_q[p]
            if leftmost_orig[i] == 0
                leftmost_orig[i] = k
            end
        end
    end

    # 6) Build pinv (row permutation) so that rows are grouped by leftmost
    # ascending. Each step k gets one row as the "diagonal"; if there are
    # extra rows with the same leftmost they get trailing slots; if there
    # are no rows with leftmost == k, we add a fictitious row at slot k.
    head = fill(0, n + 1)
    nxt = zeros(Int, m)
    @inbounds for i in m:-1:1
        if leftmost_orig[i] != 0
            k = leftmost_orig[i]
            nxt[i] = head[k]
            head[k] = i
        else
            nxt[i] = head[n + 1]
            head[n + 1] = i
        end
    end

    # Two-pass slot assignment. First count m2.
    # m2 = (# real rows) + (# slot positions k with empty bucket).
    pinv = zeros(Int, m)
    m2 = 0
    @inbounds for k in 1:n
        if head[k] == 0
            m2 += 1            # fictitious row at slot k
        else
            # count rows in bucket
            j = head[k]
            while j != 0
                m2 += 1
                j = nxt[j]
            end
        end
    end
    # Plus empty rows (leftmost == 0) get trailing slots.
    nempty = 0
    @inbounds begin
        ii = head[n + 1]
        while ii != 0
            nempty += 1
            ii = nxt[ii]
        end
    end
    m2 += nempty

    # Second pass: assign pinv. Primary row in bucket k → slot k; extras go
    # to slots n+1, n+2, .... Empty rows go after all of those.
    nextslot = n + 1
    @inbounds for k in 1:n
        i = head[k]
        if i != 0
            pinv[i] = k
            j = nxt[i]
            while j != 0
                pinv[j] = nextslot
                nextslot += 1
                j = nxt[j]
            end
        end
    end
    # Empty rows trailing.
    @inbounds begin
        ii = head[n + 1]
        while ii != 0
            pinv[ii] = nextslot
            nextslot += 1
            ii = nxt[ii]
        end
    end

    # 7) Build leftmost in *permuted* row space (indexed by 1..m2).
    # leftmost_perm[pinv[i]] = leftmost_orig[i] for real rows; fictitious
    # rows get leftmost = their slot k (i.e. they "become live" at exactly k).
    leftmost_perm = zeros(Int, m2)
    @inbounds for i in 1:m
        leftmost_perm[pinv[i]] = leftmost_orig[i]
    end
    # Fictitious rows: their slot k has no real row mapped to it, so
    # leftmost_perm[k] is still 0. But for the numeric phase to behave
    # correctly we should set leftmost_perm[slot] = slot itself (so the
    # row "becomes live" exactly at its own step k). For slots that fall
    # outside 1..n (trailing slots from empty rows), leftmost = slot itself
    # works fine too (they'll never be in any column's pattern since they
    # have no nonzeros).
    @inbounds for slot in 1:m2
        if leftmost_perm[slot] == 0
            leftmost_perm[slot] = slot <= n ? slot : 0
        end
    end

    # 8) Compute exact / upper-bound vnz, rnz.
    vnz, rnz = _vnz_rnz_estimate(colptr_q, rowval_q, parent,
                                  leftmost_orig, m, n)

    # Return original-row leftmost too? No; we only need the permuted one
    # during the numeric phase. We return the permuted one.
    return q, pinv, parent, leftmost_perm, m2, vnz, rnz
end

# Upper-bound estimate of nnz(V) and nnz(R), used to pre-size CSC buffers.
# Tightens iteratively if exceeded by `_grow_csc!`.
function _vnz_rnz_estimate(colptr::Vector{Int}, rowval::Vector{Int},
                            parent::Vector{Int},
                            leftmost_orig::Vector{Int},
                            m::Int, n::Int)
    # rnz: for each column k of S, run ereach to count R[:,k] pattern entries
    # (excluding diagonal). We use a *cheap* upper-bound: for each column k,
    # the number of R entries is at most k itself (full upper triangle), but
    # a tighter and cheaper bound uses the etree depth from the row reach.
    # Compute a conservative bound via per-row contribution: each row i in
    # the original matrix contributes (n - leftmost[i] + 1) entries summed
    # across all R columns it touches. This overcounts but is O(m) instead
    # of O(n * mean_pattern_size).
    # vnz: bounded by sum over real rows of (n - leftmost[i] + 1).
    vnz = 0
    rnz_bound = 0
    @inbounds for i in 1:m
        lm = leftmost_orig[i]
        if lm != 0
            extent = n - lm + 1
            vnz += extent
            rnz_bound += extent
        end
    end
    vnz = vnz + max(16, vnz >> 4)
    rnz = min(rnz_bound, n * (n + 1) ÷ 2) + max(16, rnz_bound >> 4)
    return vnz, rnz
end

# ---------------------------------------------------------------------------
# Public API: csr_analyze / csr_factor / csr_refactor! / csr_qr
# ---------------------------------------------------------------------------

"""
    csr_analyze(A::SparseMatrixCSR; ordering=:default) -> CSRQRSymbolic

Symbolic analysis phase for the sparse column-pivoted Householder QR
factorization. Computes the column permutation `q`, row permutation `pinv`,
column elimination tree, per-row `leftmost`, and the upper-bound sizes of
the V (Householder) and R buffers.

`ordering` selects the column ordering:
  * `:default`  — `:amd` if the AMD.jl extension is loaded (`using AMD`),
                  `:natural` otherwise. **This is the default.**
  * `:natural`  — identity ordering (opt-in; usually slower than `:amd` on
                  dense-fill matrices because the column etree degenerates
                  to a chain).
  * `:amd`      — AMD on AᵀA (requires `using AMD` to enable the extension;
                  throws an `ArgumentError` otherwise).
  * `:colamd`   — alias for `:amd`.
  * `:adaptive` — compute both `:amd` and `:natural` symbolics, pick the
                  one with the smaller column-etree total depth (which
                  bounds total apply work). Requires AMD; ~140 µs extra
                  symbolic overhead vs `:amd` alone.

Returns a `CSRQRSymbolic` that can be passed to `csr_factor` and reused via
`csr_refactor!` for matrices with identical sparsity patterns.
"""
function csr_analyze(A::SparseMatrixCSR{Bi}; ordering::Symbol=:default) where {Bi}
    m, n = size(A)
    rowptr, colval = _capture_pattern(A)
    ordering_use = _resolve_ordering(ordering)
    if (ordering_use === :amd || ordering_use === :colamd ||
        ordering_use === :adaptive) && !_AMD_EXT_LOADED[]
        throw(ArgumentError(
            "ordering=:$ordering_use requires the AMD.jl extension; load it via `using AMD`"
        ))
    end

    if ordering_use === :adaptive
        # Build both candidate symbolics, compare predicted apply work via
        # the column-etree total depth, keep the cheaper one. AMD's etree
        # is branched and typically shallower; on already-well-ordered
        # matrices natural can win and we fall back to it.
        q_a, pinv_a, parent_a, leftmost_a, m2_a, vnz_a, rnz_a =
            _build_symbolic(rowptr, colval, m, n, :amd)
        q_n, pinv_n, parent_n, leftmost_n, m2_n, vnz_n, rnz_n =
            _build_symbolic(rowptr, colval, m, n, :natural)
        d_a = _etree_total_depth(parent_a)
        d_n = _etree_total_depth(parent_n)
        # Tiebreaker prefers :natural: cheaper symbolic, and on shallow
        # etrees the apply-step difference is in the noise.
        if d_a < d_n
            return CSRQRSymbolic(m, n, m2_a, q_a, pinv_a, parent_a,
                                  leftmost_a, vnz_a, rnz_a, :amd,
                                  rowptr, colval)
        else
            return CSRQRSymbolic(m, n, m2_n, q_n, pinv_n, parent_n,
                                  leftmost_n, vnz_n, rnz_n, :natural,
                                  rowptr, colval)
        end
    end

    q, pinv, parent, leftmost_perm, m2, vnz, rnz =
        _build_symbolic(rowptr, colval, m, n, ordering_use)
    return CSRQRSymbolic(m, n, m2, q, pinv, parent, leftmost_perm,
                          vnz, rnz, ordering_use, rowptr, colval)
end

"""
    csr_factor(A::SparseMatrixCSR, sym::CSRQRSymbolic; tol=nothing, drop_tol=0) -> CSRQRFactorization

Numeric factorization given a `CSRQRSymbolic`. Implements the Davis
`cs_qr` algorithm (scatter–apply–emit on a dense workspace) with the V/R
buffers pre-sized from the symbolic phase.

`tol === nothing` selects the default `eps(real(T)) * max(m, n) * ‖A‖_F`.
Columns whose post-Householder diagonal magnitude falls below `tol` are
flagged as rank-deficient: `V[:,k] = 0`, `β_k = 0`, `R[k,k] = 0`. The
numerical rank is the count of columns whose diagonal magnitude is above
threshold.

`drop_tol=0` (default) keeps every nonzero in each Householder vector
`V[:, k]`. Setting `drop_tol > 0` discards entries with
`|v_i| <= drop_tol * ‖v‖` and recomputes `β_k` for the truncated vector;
the resulting factorization is an *approximate* QR (residual `‖A x - b‖`
grows with `drop_tol`) but subsequent `apply_QH` / `apply_Q` over fewer
nonzeros becomes cheaper. Typical safe values are `1e-12` to `1e-8` —
larger values trade accuracy for fill.
"""
function csr_factor(A::SparseMatrixCSR{Bi, T}, sym::CSRQRSymbolic;
                    tol::Union{Nothing, Real}=nothing,
                    drop_tol::Real=0) where {Bi, T}
    return _factor_kernel(A, sym, tol, real(T)(drop_tol))
end

"""
    csr_qr(A::SparseMatrixCSR; tol=nothing, ordering=:default) -> CSRQRFactorization

One-shot convenience: equivalent to `csr_factor(A, csr_analyze(A; ordering); tol)`.

When `ordering=:default` (the default), the column ordering is `:amd` if the
AMD.jl extension is loaded (`using AMD`) and `:natural` otherwise. On the
typical dense-fill matrices that arise from nonlinear solver linsolves,
`:amd` roughly halves the factor time. Pass `ordering=:natural` to opt out
for matrices whose columns are already well-ordered.
"""
function csr_qr(A::SparseMatrixCSR{Bi, T};
                tol::Union{Nothing, Real}=nothing,
                ordering::Symbol=:default,
                drop_tol::Real=0) where {Bi, T}
    sym = csr_analyze(A; ordering=ordering)
    return csr_factor(A, sym; tol=tol, drop_tol=drop_tol)
end

"""
    csr_refactor!(F::CSRQRFactorization, A::SparseMatrixCSR; tol=nothing, drop_tol=0) -> CSRQRFactorization

Numeric refactorization. If the sparsity pattern of `A` matches the one
captured in `F.sym`, the symbolic is reused (skipping the etree / `pinv` /
`leftmost` work). Otherwise a fresh analyze+factor is performed.

The `drop_tol` keyword has the same meaning as in [`csr_factor`](@ref).

Returns a fresh `CSRQRFactorization` (the original is unchanged).
"""
function csr_refactor!(F::CSRQRFactorization{T},
                       A::SparseMatrixCSR{Bi};
                       tol::Union{Nothing, Real}=nothing,
                       drop_tol::Real=0) where {T, Bi}
    dt = real(T)(drop_tol)
    if _pattern_matches(F.sym, A)
        return _factor_kernel(A, F.sym, tol, dt)
    else
        sym = csr_analyze(A; ordering=F.sym.ordering)
        return _factor_kernel(A, sym, tol, dt)
    end
end

# ---------------------------------------------------------------------------
# Numeric kernel — Davis cs_qr on the row+column permuted matrix S = P A Q.
# ---------------------------------------------------------------------------

function _factor_kernel(A::SparseMatrixCSR{Bi, T}, sym::CSRQRSymbolic,
                         tol::Union{Nothing, Real},
                         drop_tol::Real=zero(real(T))) where {Bi, T}
    m, n = size(A)
    (m == sym.m && n == sym.n) ||
        throw(DimensionMismatch("A is $m x $n but symbolic is $(sym.m) x $(sym.n)"))

    RT = real(T)

    # Single-pass CSR -> CSC(A) conversion + Frobenius norm + column-norm
    # cache (the column norms feed the zero-column check below). This fuses
    # what used to be _csr_to_csc, a separate norm-loop, and the col-norm
    # computation inside _maybe_repivot_zero_cols.
    colptr_A, rowval_A, nzval_A, col_nrm2, fro2 = _csr_to_csc_with_norms(A)
    fro = sqrt(fro2)
    tol_use = tol === nothing ? RT(eps(RT) * max(m, n)) * fro : RT(max(tol, 0))
    tol2 = tol_use * tol_use

    # Value-aware refinement of column ordering: push numerically-zero columns
    # to the trailing positions. This makes natural-order back-substitution
    # produce the correct basic LS solution even when A has linearly dependent
    # columns. Without this, a zero column in the middle of R causes back-sub
    # to fail (the row constraint for that index cannot be satisfied by any
    # later z).
    sym_use = _maybe_repivot_zero_cols_from_norms(col_nrm2, sym, fro)

    # Apply row+column permutation: S = (P A Q). Fused: walks colptr_A once,
    # writes the permuted CSC in pinv- / q-permuted order.
    colptr_S, rowval_S, nzval_S =
        _permute_pq(colptr_A, rowval_A, nzval_A, sym_use.pinv, sym_use.q, m, n)

    return _csc_qr_numeric(colptr_S, rowval_S, nzval_S, sym_use,
                           tol_use, tol2, RT(drop_tol))
end

# CSR -> CSC of A, plus per-column squared norms and total ||A||_F^2. The
# norms come for free as we already iterate every nonzero during the
# conversion. Saves a redundant nzval scan in _factor_kernel.
function _csr_to_csc_with_norms(A::SparseMatrixCSR{Bi, T}) where {Bi, T}
    m, n = size(A)
    off = getoffset(A)
    rowptr = A.rowptr
    colval = A.colval
    nzval_in = A.nzval
    nnz_total = length(colval)
    RT = real(T)

    colcounts = zeros(Int, n)
    @inbounds for p in 1:nnz_total
        colcounts[Int(colval[p]) + off] += 1
    end
    colptr = Vector{Int}(undef, n + 1)
    colptr[1] = 1
    @inbounds for j in 1:n
        colptr[j + 1] = colptr[j] + colcounts[j]
    end
    rowval = Vector{Int}(undef, nnz_total)
    nzval = Vector{T}(undef, nnz_total)
    col_nrm2 = zeros(RT, n)
    work = copy(colptr)
    fro2 = zero(RT)
    @inbounds for i in 1:m
        r1 = rowptr[i] + off
        r2 = rowptr[i + 1] + off - 1
        for p in r1:r2
            j = Int(colval[p]) + off
            slot = work[j]
            v = nzval_in[p]
            rowval[slot] = i
            nzval[slot] = v
            work[j] = slot + 1
            v2 = abs2(v)
            col_nrm2[j] += v2
            fro2 += v2
        end
    end
    return colptr, rowval, nzval, col_nrm2, fro2
end

# Inspect column norms of A. If any are below `fro * eps(RT) * n` (i.e.,
# numerically zero), move those columns to the end of `sym.q` and rebuild
# the value-independent symbolic pieces (parent, leftmost, m2, pinv, vnz, rnz)
# for the new ordering. Returns either the original `sym` (no zero columns)
# or a freshly-built one.
function _maybe_repivot_zero_cols_from_norms(col_norms::Vector{RT},
                                              sym::CSRQRSymbolic,
                                              fro_A::Real) where {RT}
    n = sym.n
    eps_zero = RT(fro_A) * RT(eps(RT)) * RT(max(sym.m, n))
    thr2 = eps_zero * eps_zero
    has_zero = false
    @inbounds for j in 1:n
        if col_norms[j] <= thr2
            has_zero = true
            break
        end
    end
    if !has_zero
        return sym
    end
    # Build refined q': non-zero columns in the existing q-order first, then
    # zero columns at the end.
    q_new = Vector{Int}(undef, n)
    pos = 1
    @inbounds for k in 1:n
        j = sym.q[k]
        if col_norms[j] > thr2
            q_new[pos] = j
            pos += 1
        end
    end
    @inbounds for k in 1:n
        j = sym.q[k]
        if col_norms[j] <= thr2
            q_new[pos] = j
            pos += 1
        end
    end
    if q_new == sym.q
        return sym
    end

    # Rebuild symbolic with the new q'.
    q2, pinv2, parent2, leftmost2, m2_2, vnz2, rnz2 =
        _rebuild_symbolic_for_q(sym.pattern_rowptr, sym.pattern_colval,
                                 sym.m, sym.n, q_new)
    return CSRQRSymbolic(sym.m, sym.n, m2_2, q2, pinv2, parent2, leftmost2,
                         vnz2, rnz2, sym.ordering, sym.pattern_rowptr,
                         sym.pattern_colval)
end

# Rebuild symbolic data for a given q (a column permutation).
function _rebuild_symbolic_for_q(rowptr::Vector{Int}, colval::Vector{Int},
                                  m::Int, n::Int, q::Vector{Int})
    # Reuse _build_symbolic but force the q we've chosen.
    colptr_A, rowval_A = _csr_pattern_to_csc(rowptr, colval, m, n)
    colptr_q, rowval_q = _permute_cols(colptr_A, rowval_A, q, m, n)
    parent = _coletree_ata(colptr_q, rowval_q, m, n)

    leftmost_orig = zeros(Int, m)
    @inbounds for k in 1:n
        c1 = colptr_q[k]; c2 = colptr_q[k + 1] - 1
        for p in c1:c2
            i = rowval_q[p]
            if leftmost_orig[i] == 0
                leftmost_orig[i] = k
            end
        end
    end

    head = fill(0, n + 1)
    nxt = zeros(Int, m)
    @inbounds for i in m:-1:1
        if leftmost_orig[i] != 0
            k = leftmost_orig[i]
            nxt[i] = head[k]
            head[k] = i
        else
            nxt[i] = head[n + 1]
            head[n + 1] = i
        end
    end

    pinv = zeros(Int, m)
    m2 = 0
    @inbounds for k in 1:n
        if head[k] == 0
            m2 += 1
        else
            j = head[k]
            while j != 0
                m2 += 1
                j = nxt[j]
            end
        end
    end
    nempty = 0
    @inbounds begin
        ii = head[n + 1]
        while ii != 0
            nempty += 1
            ii = nxt[ii]
        end
    end
    m2 += nempty

    nextslot = n + 1
    @inbounds for k in 1:n
        i = head[k]
        if i != 0
            pinv[i] = k
            j = nxt[i]
            while j != 0
                pinv[j] = nextslot
                nextslot += 1
                j = nxt[j]
            end
        end
    end
    @inbounds begin
        ii = head[n + 1]
        while ii != 0
            pinv[ii] = nextslot
            nextslot += 1
            ii = nxt[ii]
        end
    end

    leftmost_perm = zeros(Int, m2)
    @inbounds for i in 1:m
        leftmost_perm[pinv[i]] = leftmost_orig[i]
    end
    @inbounds for slot in 1:m2
        if leftmost_perm[slot] == 0
            leftmost_perm[slot] = slot <= n ? slot : 0
        end
    end

    vnz, rnz = _vnz_rnz_estimate(colptr_q, rowval_q, parent, leftmost_orig,
                                  m, n)
    return q, pinv, parent, leftmost_perm, m2, vnz, rnz
end

# Numeric loop. Returns a CSRQRFactorization.
function _csc_qr_numeric(colptr::Vector{Int}, rowval::Vector{Int},
                         nzval::Vector{T}, sym::CSRQRSymbolic,
                         tol_use::RT, tol2::RT,
                         drop_tol::RT=zero(RT)) where {T, RT}
    drop_active = drop_tol > zero(RT)
    drop_tol2 = drop_tol * drop_tol
    m, n, m2 = sym.m, sym.n, sym.m2
    parent = sym.parent
    leftmost = sym.leftmost

    V = _alloc_csc(T, m2, n, sym.vnz)
    R = _alloc_csc(T, n, n, sym.rnz)
    beta = Vector{RT}(undef, n)

    # Workspaces. `x` and `w` must be zero-initialised (`x` is the dense
    # scatter slot, `w` is the ereach marker compared against k >= 1).
    x = zeros(T, m2)
    s = Vector{Int}(undef, n)
    w = zeros(Int, n)
    vrows = Vector{Int}(undef, m2)

    # Hoist field accesses for the inner loops.
    Vp = V.colptr; Vi = V.rowval; Vx = V.nzval
    Rp = R.colptr; Ri = R.rowval; Rx = R.nzval

    Vp[1] = 1
    Rp[1] = 1
    rnz_total = 0
    vnz_total = 0
    rnk = 0

    @inbounds for k in 1:n
        # --- 1) ereach pattern of R[:,k] + scatter S[:,k] into x ---------
        top = n + 1
        c1 = colptr[k]; c2 = colptr[k + 1] - 1
        for p in c1:c2
            i = rowval[p]
            x[i] = nzval[p]
            lm = leftmost[i]
            if lm > 0 && lm <= k
                len = 0
                jj = lm
                while jj != 0 && w[jj] != k && jj < k
                    s[len + 1] = jj
                    len += 1
                    w[jj] = k
                    jj = parent[jj]
                end
                # Davis cs_ereach: transfer s[1..len] to s[top-len..top-1] in
                # increasing-column-index order. Iteration is `--top, --len;
                # s[top] = s[len]`, which moves s[len] (largest col) to
                # s[top-1], and s[1] (smallest col = lm) ends up at s[top-len].
                while len > 0
                    top -= 1
                    s[top] = s[len]
                    len -= 1
                end
            end
        end

        # --- 2) Apply previous Householders ------------------------------
        # Pattern in s[top..n] is in increasing column index. Iterate that
        # order: apply H_1, H_2, ..., H_{k-1} (the original order of the
        # outer QR loop).
        #
        # We deliberately do NOT track V[:,k]'s row pattern here. Instead we
        # scan x[k..m2] for nonzeros at the end of the apply phase. This
        # keeps the apply inner loop branch-free (the tau dot-product and
        # AXPY both vectorize well).
        for spos in top:n
            p_idx = s[spos]
            if p_idx == k
                continue
            end
            vc1 = Vp[p_idx]; vc2 = Vp[p_idx + 1] - 1
            bk = beta[p_idx]
            tau = zero(T)
            @simd for vp in vc1:vc2
                tau += conj(Vx[vp]) * x[Vi[vp]]
            end
            if bk != 0 && tau != 0
                tau_b = T(bk) * tau
                @simd for vp in vc1:vc2
                    x[Vi[vp]] -= tau_b * Vx[vp]
                end
            end
            # Emit R[p_idx, k] = x[p_idx]; clear x[p_idx].
            if rnz_total + 1 > length(Ri)
                _grow_csc!(R, rnz_total + 1)
                Ri = R.rowval; Rx = R.nzval
            end
            rnz_total += 1
            Ri[rnz_total] = p_idx
            Rx[rnz_total] = x[p_idx]
            x[p_idx] = zero(T)
        end

        # --- 2b) Build V[:,k]'s row list by scanning x[k..m2] for nonzeros.
        # The row indices are written into vrows[1..vlen], with vrows[1] = k
        # (the diagonal row) and the remaining entries being rows > k with
        # x[i] != 0.
        vrows[1] = k
        vlen = 1
        for i in (k + 1):m2
            if x[i] != zero(T)
                vlen += 1
                vrows[vlen] = i
            end
        end

        # --- 3) Build Householder for x[vrows[1..vlen]] -----------------
        # Compute alpha, beta_k. v[1] = x[vrows[1]] - alpha; v[j>=2] unchanged.
        nrm2 = zero(RT)
        for q in 1:vlen
            nrm2 += abs2(x[vrows[q]])
        end

        # --- 4) Emit R[k,k] and V[:,k], or mark rank-deficient -----------
        # Allocate row for R[k,k] first.
        if rnz_total + 1 > length(Ri)
            _grow_csc!(R, rnz_total + 1)
            Ri = R.rowval; Rx = R.nzval
        end

        if nrm2 <= tol2 || vlen == 0
            # Rank-deficient column. R[k,k] = 0 (but emit, so R has a diagonal).
            rnz_total += 1
            Ri[rnz_total] = k
            Rx[rnz_total] = zero(T)
            Rp[k + 1] = rnz_total + 1
            Vp[k + 1] = vnz_total + 1
            beta[k] = zero(RT)
            # Clear x at v-pattern rows (vrows[1..vlen]).
            for q in 1:vlen
                x[vrows[q]] = zero(T)
            end
            continue
        end

        x1 = x[vrows[1]]
        sgn = if T <: Real
            x1 >= 0 ? one(T) : -one(T)
        else
            x1 == 0 ? one(T) : x1 / abs(x1)
        end
        nrm = sqrt(nrm2)
        alpha = -sgn * nrm
        v1 = x1 - alpha
        # vnorm2 = |v1|^2 + sum_{j>=2} |x[j]|^2 = |v1|^2 + (nrm2 - |x1|^2)
        vnorm2 = abs2(v1) + (nrm2 - abs2(x1))

        if vnorm2 <= zero(RT)
            # Pathological: H = I. Treat as rank-deficient pivot.
            rnz_total += 1
            Ri[rnz_total] = k
            Rx[rnz_total] = T(alpha)
            Rp[k + 1] = rnz_total + 1
            Vp[k + 1] = vnz_total + 1
            beta[k] = zero(RT)
            for q in 1:vlen
                x[vrows[q]] = zero(T)
            end
            if abs(alpha) > tol_use
                rnk += 1
            end
            continue
        end

        beta_k = RT(2) / vnorm2

        # Emit R[k, k] = alpha.
        rnz_total += 1
        Ri[rnz_total] = k
        Rx[rnz_total] = T(alpha)
        Rp[k + 1] = rnz_total + 1

        # Emit V[:,k]: first slot is row k with value v1; remaining slots are
        # vrows[2..vlen] with NONZERO values x[vrows[q]] only.
        # Skipping exact zeros keeps V columns numerically sparse: rows marked
        # by the apply-step row-tracking but cancelled to 0 by Householder
        # arithmetic don't pollute future apply walks.
        #
        # If `drop_tol > 0` is in effect we additionally drop entries
        # j >= 2 whose magnitude satisfies |x[vrows[q]]|^2 <= drop_tol^2 *
        # vnorm2, and then recompute β_k from the surviving |v|^2 so that
        # H̃ = I - β̃ ṽ ṽ^T remains a proper Householder of the truncated ṽ.
        # The diagonal v1 is never dropped.
        if vnz_total + vlen > length(Vi)
            _grow_csc!(V, vnz_total + vlen)
            Vi = V.rowval; Vx = V.nzval
        end
        vnz_total += 1
        Vi[vnz_total] = k
        Vx[vnz_total] = v1
        if drop_active
            thr_drop2 = drop_tol2 * vnorm2
            new_vnorm2 = abs2(v1)
            for q in 2:vlen
                xv = x[vrows[q]]
                if xv != zero(T) && abs2(xv) > thr_drop2
                    vnz_total += 1
                    Vi[vnz_total] = vrows[q]
                    Vx[vnz_total] = xv
                    new_vnorm2 += abs2(xv)
                end
            end
            beta_k = RT(2) / new_vnorm2
        else
            for q in 2:vlen
                xv = x[vrows[q]]
                if xv != zero(T)
                    vnz_total += 1
                    Vi[vnz_total] = vrows[q]
                    Vx[vnz_total] = xv
                end
            end
        end
        Vp[k + 1] = vnz_total + 1

        beta[k] = beta_k
        rnk += 1

        # Clear x at v-pattern rows.
        for q in 1:vlen
            x[vrows[q]] = zero(T)
        end
    end

    # Trim V/R to actual sizes.
    resize!(Vi, vnz_total); resize!(Vx, vnz_total)
    resize!(Ri, rnz_total); resize!(Rx, rnz_total)

    return CSRQRFactorization{T, RT}(sym.m, sym.n,
        Vp, Vi, Vx,
        Rp, Ri, Rx,
        beta, rnk, tol_use, sym)
end

# ---------------------------------------------------------------------------
# Solve path.
# ---------------------------------------------------------------------------
#
# Given F representing P A Q = Q_H R (where Q_H is the product of Householders
# stored in V), to solve A x = b:
#   1) work = P b (i.e. work[pinv[i]] = b[i]; pad slots > m with zeros).
#   2) Apply Q_H^T = H_n ... H_1 to work: work := Q_H^T work.
#   3) Solve R x' = work[1:n] in place (upper-triangular, columnwise).
#   4) x = Q_perm x': x[q[k]] = x'[k].

function _apply_QH!(F::CSRQRFactorization{T}, work::Vector{T}) where {T}
    Vp = F.V_colptr; Vi = F.V_rowval; Vx = F.V_nzval; beta = F.beta
    n = F.n
    @inbounds for k in 1:n
        bk = beta[k]
        bk == 0 && continue
        vc1 = Vp[k]; vc2 = Vp[k + 1] - 1
        vc2 < vc1 && continue
        tau = zero(T)
        @simd for vp in vc1:vc2
            tau += conj(Vx[vp]) * work[Vi[vp]]
        end
        if tau == 0
            continue
        end
        tau_b = T(bk) * tau
        @simd for vp in vc1:vc2
            work[Vi[vp]] -= tau_b * Vx[vp]
        end
    end
    return work
end

function _apply_Q!(F::CSRQRFactorization{T}, work::Vector{T}) where {T}
    Vp = F.V_colptr; Vi = F.V_rowval; Vx = F.V_nzval; beta = F.beta
    n = F.n
    @inbounds for k in n:-1:1
        bk = beta[k]
        bk == 0 && continue
        vc1 = Vp[k]; vc2 = Vp[k + 1] - 1
        vc2 < vc1 && continue
        tau = zero(T)
        @simd for vp in vc1:vc2
            tau += conj(Vx[vp]) * work[Vi[vp]]
        end
        if tau == 0
            continue
        end
        tau_b = T(bk) * tau
        @simd for vp in vc1:vc2
            work[Vi[vp]] -= tau_b * Vx[vp]
        end
    end
    return work
end

# Solve R z = c (R is upper triangular in CSC). Rank-revealing: rows
# (k+1..n) of z are zeroed if R[k,k] is below threshold.
function _usolve!(z::AbstractVector{T}, F::CSRQRFactorization{T},
                   c::AbstractVector{T}) where {T}
    n = F.n
    Rp = F.R_colptr; Ri = F.R_rowval; Rx = F.R_nzval
    tol_use = F.tol
    @inbounds for i in 1:n
        z[i] = c[i]
    end
    # Davis-style usolve on a CSC upper triangular: walk k from n down to 1.
    # For column k of R: the entries are R[i,k] for i <= k. Diagonal is at
    # one of these slots; identify it. Iterate column k from bottom to top.
    @inbounds for k in n:-1:1
        rc1 = Rp[k]; rc2 = Rp[k + 1] - 1
        # Find the diagonal (the entry with row index k). For an in-order
        # CSC R (which our emit guarantees: pattern emitted in ereach
        # topological order, then diag last), the diagonal is at the LAST
        # slot Rp[k+1]-1. Verify and fall back to search if not.
        diag_p = 0
        if rc2 >= rc1 && Ri[rc2] == k
            diag_p = rc2
        else
            for p in rc1:rc2
                if Ri[p] == k
                    diag_p = p; break
                end
            end
        end
        if diag_p == 0
            # Missing diagonal: column is structurally absent. Treat as
            # rank-deficient: set z[k] = 0.
            z[k] = zero(T)
            continue
        end
        d = Rx[diag_p]
        if abs(d) <= tol_use || d == 0
            # Rank-deficient row; set z[k] = 0 (basic LS solution).
            z[k] = zero(T)
            # Still need to subtract z[k] contributions from above rows in
            # higher steps? No: we go k from n down, so above rows handle
            # themselves. We just don't propagate the zero down.
            continue
        end
        z[k] /= d
        zk = z[k]
        # Subtract z[k] * R[i, k] from z[i] for i < k.
        for p in rc1:rc2
            p == diag_p && continue
            i = Ri[p]
            z[i] -= Rx[p] * zk
        end
    end
    return z
end

function LinearAlgebra.ldiv!(x::AbstractVector{T}, F::CSRQRFactorization{T},
                              b::AbstractVector{T}) where {T}
    length(b) == F.m || throw(DimensionMismatch("b length $(length(b)) != m=$(F.m)"))
    length(x) == F.n || throw(DimensionMismatch("x length $(length(x)) != n=$(F.n)"))
    m, n, m2 = F.m, F.n, F.sym.m2
    pinv = F.sym.pinv
    q = F.sym.q

    # Workspace sized to m2 (handles fictitious rows).
    work = zeros(T, m2)
    @inbounds for i in 1:m
        work[pinv[i]] = b[i]
    end
    # Slots m+1..m2 stay zero (fictitious rows).

    _apply_QH!(F, work)

    # Solve R x' = work[1:n].
    xprime = Vector{T}(undef, n)
    @inbounds for k in 1:n
        xprime[k] = work[k]
    end
    _usolve!(xprime, F, xprime)

    # x[q[k]] = x'[k].
    @inbounds for k in 1:n
        x[q[k]] = xprime[k]
    end
    return x
end

function Base.:\(F::CSRQRFactorization{T}, b::AbstractVector{T}) where {T}
    x = zeros(T, F.n)
    ldiv!(x, F, b)
    return x
end

function Base.:\(F::CSRQRFactorization{T}, b::AbstractVector) where {T}
    bb = convert(Vector{T}, b)
    x = zeros(T, F.n)
    ldiv!(x, F, bb)
    return x
end

end # module
