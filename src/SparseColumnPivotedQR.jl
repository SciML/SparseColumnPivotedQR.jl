module SparseColumnPivotedQR

using LinearAlgebra
using SparseArrays
using SparseMatricesCSR
using PrecompileTools

import LinearAlgebra: ldiv!, rank
import Base: \, size, eltype

export csr_qr, csr_analyze, csr_factor, csr_refactor!,
    has_amd_extension,
    CSRQRSymbolic, CSRQRFactorization

# CSR offset: rowptr stores 1-based if Bi == 1, 0-based if Bi == 0.
@inline function getoffset(::SparseMatrixCSR{Bi}) where {Bi}
    return Bi == 1 ? 0 : 1
end

# The adaptive-dense fallback finishes the trailing dense block with LAPACK
# `geqp3!` / `ormqr!`, which are only defined for the four BLAS float types
# (Float32/Float64/ComplexF32/ComplexF64). For any other element type (e.g.
# `BigFloat` or `ForwardDiff.Dual`) the dense path is unavailable, so we
# transparently ignore `adaptive_dense` and run the pure-Julia sparse kernel.
@inline _is_blas_eltype(::Type{T}) where {T} = T <: LinearAlgebra.BlasFloat

# Grow a (rowval, nzval) pair so that both have at least `needed` capacity.
# Used by the numeric kernel to expand V/R output buffers if the symbolic
# bound was undershot.
@inline function _grow_pair!(
        rowval::Vector{Int}, nzval::Vector{T},
        needed::Int
    ) where {T}
    L = length(rowval)
    new_L = max(2 * L, needed)
    resize!(rowval, new_L)
    resize!(nzval, new_L)
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

# ---------------------------------------------------------------------------
# Workspace pool: large per-call buffers shared across `csr_factor` /
# `csr_refactor!` calls.
# ---------------------------------------------------------------------------
#
# A `_CSRQRWorkspace{T, RT}` holds the value-typed scratch buffers used by
# the numeric kernel: intermediate CSC representations of A and S = (P A Q),
# the dense workspace `x`, the etree stack `s`, the marker array `w`, and
# the row-pattern buffer `vrows`. It also owns the column-norm cache used
# by the value-aware repivot.
#
# Lifetime: attached lazily to the `CSRQRSymbolic` (whose type T is only
# known at the first numeric call). Subsequent calls reuse it.

mutable struct _CSRQRWorkspace{T, RT}
    # Intermediate CSC of A.
    colptr_A::Vector{Int}
    rowval_A::Vector{Int}
    nzval_A::Vector{T}
    col_nrm2::Vector{RT}
    work_perm::Vector{Int}
    # Intermediate CSC of S = P A Q.
    colptr_S::Vector{Int}
    rowval_S::Vector{Int}
    nzval_S::Vector{T}
    # Numeric kernel dense workspaces.
    x::Vector{T}
    s::Vector{Int}
    w::Vector{Int}
    vrows::Vector{Int}
    # Positions (in the current q-order) of columns the numeric kernel flags
    # rank-deficient. Reused across calls via `empty!` so the full-rank path
    # does no heap work; consumed by `_factor_kernel`'s deferral re-pass.
    def_pos::Vector{Int}
    # Per-original-column membership marker for the deferral fixed-point loop:
    # `is_indep[j] == 1` iff column j is confirmed leading-independent (a nonzero
    # pivot in the leading block). Allocated empty and only sized (once) on the
    # first rank-deficient call, so the full-rank path does no extra heap work.
    # Cannot alias `w` because the numeric kernel clobbers `w` each pass while
    # this must persist across passes.
    is_indep::Vector{Int}
    # Generation counter used as a "stamp base" for the `w` marker array so
    # that the numeric kernel doesn't have to zero `w` between calls. The
    # stamp value used in the kernel is `gen * (n + 1) + k`; an `n`-sized
    # `w` is then compared against this monotonic generation-encoded stamp.
    wgen::Int
    # Pooled scratch for the solve path so `ldiv!` is allocation-free in steady
    # state (mirrors the zero-alloc factor path). `solve_work` is sized m2,
    # `solve_xprime` is sized n.
    solve_work::Vector{T}
    solve_xprime::Vector{T}
end

function _alloc_workspace(
        ::Type{T}, m::Int, n::Int, m2::Int,
        nnz_A::Int
    ) where {T}
    RT = real(T)
    return _CSRQRWorkspace{T, RT}(
        Vector{Int}(undef, n + 1),
        Vector{Int}(undef, nnz_A),
        Vector{T}(undef, nnz_A),
        Vector{RT}(undef, n),
        Vector{Int}(undef, n + 1),
        Vector{Int}(undef, n + 1),
        Vector{Int}(undef, nnz_A),
        Vector{T}(undef, nnz_A),
        zeros(T, m2),
        Vector{Int}(undef, n),
        zeros(Int, n),
        Vector{Int}(undef, m2),
        Int[],
        Int[],
        0,
        zeros(T, m2),
        Vector{T}(undef, n),
    )
end

mutable struct CSRQRSymbolic
    m::Int
    n::Int
    m2::Int
    q::Vector{Int}
    pinv::Vector{Int}
    parent::Vector{Int}
    leftmost::Vector{Int}
    vnz::Int
    rnz::Int
    rcount::Vector{Int}     # exact nnz per column of R (length n)
    ordering::Symbol
    pattern_rowptr::Vector{Int}
    pattern_colval::Vector{Int}
    # Lazily-attached numeric workspace. Type-erased here because Symbolic
    # is built without knowing the value-element type T.
    workspace::Union{Nothing, _CSRQRWorkspace}
end

Base.size(S::CSRQRSymbolic) = (S.m, S.n)

# ---------------------------------------------------------------------------
# Factorization
# ---------------------------------------------------------------------------
#
# CSC storage of V (Householders) and R, plus beta (Householder coefficients),
# plus the symbolic. Permutations come from `sym`.

mutable struct CSRQRFactorization{T, RT}
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
    # Adaptive dense fallback. When the active submatrix becomes dense enough
    # mid-factorization we switch to LAPACK geqp3 on the trailing block. The
    # fields below describe that block; they are zero-length / `k_dense == 0`
    # when no transition occurred.
    #
    # `k_dense`     : sparse Householders are V[:, 1..k_dense]; dense tail
    #                 covers columns k_dense+1..n.
    # `D`           : (m2 - k_dense) x (n - k_dense) compact LAPACK form from
    #                 geqp3 — strict lower triangle = Householder vectors v,
    #                 upper triangle = R (also redundantly emitted into CSC R).
    # `dtau`        : Householder coefficients τ for the dense tail.
    # `q_eff`       : composed column permutation (length n). Equals sym.q
    #                 when k_dense == 0; otherwise equals
    #                 [sym.q[1..k_dense]; sym.q[k_dense .+ jpvt_dense]].
    k_dense::Int
    D::Matrix{T}
    dtau::Vector{T}
    q_eff::Vector{Int}
end

LinearAlgebra.rank(F::CSRQRFactorization) = F.rnk
Base.size(F::CSRQRFactorization) = (F.m, F.n)
Base.size(F::CSRQRFactorization, d::Integer) = d == 1 ? F.m : (d == 2 ? F.n : 1)
Base.eltype(::CSRQRFactorization{T}) where {T} = T

# ---------------------------------------------------------------------------
# CSR <-> CSC conversion (pattern + values)
# ---------------------------------------------------------------------------

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
# Exact R column counts via an ereach-based symbolic pass.
# ---------------------------------------------------------------------------

# cs_counts: exact column counts of the upper-triangular factor R for the
# QR factorization of A. Inputs are the column-permuted matrix S = A(:,q)
# pattern in CSC form, plus its column etree and a postorder.
#
# Returns Vector{Int} of length n with the exact nnz of each column of R
# (including the diagonal entry R[k,k]). Equivalently this is the number
# of structural nonzeros in R[:, k] (the upper-triangular column k of R)
# under the no-cancellation assumption.
#
# Implementation: this is a symbolic-only pass that mirrors the ereach
# walk used in the numeric kernel. For each column k:
#
#   1. Initialize the column-k row set R[:,k] to the rows of S[:,k] that
#      lie in {1..k}. Mark each such row as "in pattern".
#   2. For each row i in S[:,k] with leftmost[i] = lm <= k, traverse up the
#      etree from lm following `parent[]`, adding each visited node < k to
#      the pattern (deduplicated via a stamp array).
#   3. The diagonal entry R[k,k] is always present (added explicitly).
#
# Time complexity: O(nnz(R)) per call, dominated by the etree walks. This
# is the same complexity as the numeric phase's ereach work, run once.
#
# Note: this is an "ereach-based" exact count (used by Davis cs_qr to
# pre-size the numeric buffers). It is exact under the no-cancellation
# assumption. We keep the name cs_counts for familiarity with CSparse, but
# the algorithm here is the ereach formulation rather than the Gilbert-Ng-
# Peyton union-find one (which is O(nnz·α) but more intricate for QR).
# For the matrix sizes typical in this package (n ~ 200, nnz ~ 1000), the
# ereach formulation is both simpler and competitive.
function _cs_counts(
        colptr::Vector{Int}, rowval::Vector{Int},
        parent::Vector{Int}, leftmost::Vector{Int},
        m::Int, n::Int
    )
    colcount = zeros(Int, n)
    stamp = zeros(Int, n)         # stamp[j] == k means j is in column-k pattern
    @inbounds for k in 1:n
        cnt = 0
        c1 = colptr[k]; c2 = colptr[k + 1] - 1
        for p in c1:c2
            i = rowval[p]
            lm = leftmost[i]
            if lm == 0 || lm > k
                continue
            end
            # Walk up etree from lm until we hit a node >= k or already-stamped.
            jj = lm
            while jj != 0 && jj < k && stamp[jj] != k
                stamp[jj] = k
                cnt += 1
                jj = parent[jj]
            end
        end
        # Diagonal R[k,k].
        colcount[k] = cnt + 1
    end
    return colcount
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

# Fused (CSR pattern → CSC pattern of A(:, q)). Equivalent to
# `_csr_pattern_to_csc` followed by `_permute_cols`, but skips materializing
# the intermediate un-permuted CSC pattern. Returns (colptr_q, rowval_q).
function _csr_pattern_to_csc_permuted(
        rowptr::Vector{Int}, colval::Vector{Int},
        q::Vector{Int}, m::Int, n::Int
    )
    nnz_total = length(colval)
    # qinv: position k in q-order = the column j such that q[k] = j.
    qinv = Vector{Int}(undef, n)
    @inbounds for k in 1:n
        qinv[q[k]] = k
    end
    # Per-column counts (in q-order).
    colcounts = zeros(Int, n)
    @inbounds for p in 1:nnz_total
        colcounts[qinv[colval[p]]] += 1
    end
    colptr_q = Vector{Int}(undef, n + 1)
    colptr_q[1] = 1
    @inbounds for k in 1:n
        colptr_q[k + 1] = colptr_q[k] + colcounts[k]
    end
    rowval_q = Vector{Int}(undef, nnz_total)
    # Scatter row by row using the running counts in colcounts (reset to
    # the column starts).
    @inbounds for k in 1:n
        colcounts[k] = colptr_q[k]
    end
    @inbounds for i in 1:m
        r1 = rowptr[i]; r2 = rowptr[i + 1] - 1
        for p in r1:r2
            kc = qinv[colval[p]]
            rowval_q[colcounts[kc]] = i
            colcounts[kc] += 1
        end
    end
    return colptr_q, rowval_q
end

function _build_symbolic(
        rowptr::Vector{Int}, colval::Vector{Int},
        m::Int, n::Int, ordering::Symbol
    )
    # 1) Column permutation `q`.
    q = if ordering === :natural
        collect(1:n)
    elseif ordering === :amd || ordering === :colamd
        _amd_colperm(rowptr, colval, m, n)
    else
        throw(ArgumentError("Unknown ordering :$ordering"))
    end

    # 2+3) Build CSC pattern of A(:, q) in one pass: scatter row-by-row
    # using the column-bucket counts of A(:, q).
    colptr_q, rowval_q = _csr_pattern_to_csc_permuted(rowptr, colval, q, m, n)

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

    # 8) Compute exact R column counts via the ereach-based cs_counts. The
    # exact rnz is sum(rcount). V's exact column count is bounded by the
    # same row-extent bound used previously; computing the V exact count
    # would require a second pass and gives only marginal savings here.
    rcount = _cs_counts(colptr_q, rowval_q, parent, leftmost_orig, m, n)
    rnz_exact = 0
    @inbounds for k in 1:n
        rnz_exact += rcount[k]
    end
    vnz_bound = _vnz_estimate(leftmost_orig, m, n)
    return q, pinv, parent, leftmost_perm, m2, vnz_bound, rnz_exact, rcount
end

# Cheap O(m) upper bound for nnz(V): each real row contributes
# (n - leftmost[i] + 1) to the V workspace size, with a small headroom.
function _vnz_estimate(leftmost_orig::Vector{Int}, m::Int, n::Int)
    vnz = 0
    @inbounds for i in 1:m
        lm = leftmost_orig[i]
        if lm != 0
            vnz += n - lm + 1
        end
    end
    return vnz + max(16, vnz >> 4)
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
function csr_analyze(A::SparseMatrixCSR{Bi}; ordering::Symbol = :default) where {Bi}
    m, n = size(A)
    rowptr, colval = _capture_pattern(A)
    ordering_use = _resolve_ordering(ordering)
    if (
            ordering_use === :amd || ordering_use === :colamd ||
                ordering_use === :adaptive
        ) && !_AMD_EXT_LOADED[]
        throw(
            ArgumentError(
                "ordering=:$ordering_use requires the AMD.jl extension; load it via `using AMD`"
            )
        )
    end

    if ordering_use === :adaptive
        # Build both candidate symbolics, compare predicted apply work via
        # the column-etree total depth, keep the cheaper one. AMD's etree
        # is branched and typically shallower; on already-well-ordered
        # matrices natural can win and we fall back to it.
        q_a, pinv_a, parent_a, leftmost_a, m2_a, vnz_a, rnz_a, rcount_a =
            _build_symbolic(rowptr, colval, m, n, :amd)
        q_n, pinv_n, parent_n, leftmost_n, m2_n, vnz_n, rnz_n, rcount_n =
            _build_symbolic(rowptr, colval, m, n, :natural)
        d_a = _etree_total_depth(parent_a)
        d_n = _etree_total_depth(parent_n)
        # Tiebreaker prefers :natural: cheaper symbolic, and on shallow
        # etrees the apply-step difference is in the noise.
        if d_a < d_n
            return CSRQRSymbolic(
                m, n, m2_a, q_a, pinv_a, parent_a,
                leftmost_a, vnz_a, rnz_a, rcount_a, :amd,
                rowptr, colval, nothing
            )
        else
            return CSRQRSymbolic(
                m, n, m2_n, q_n, pinv_n, parent_n,
                leftmost_n, vnz_n, rnz_n, rcount_n, :natural,
                rowptr, colval, nothing
            )
        end
    end

    q, pinv, parent, leftmost_perm, m2, vnz, rnz, rcount =
        _build_symbolic(rowptr, colval, m, n, ordering_use)
    return CSRQRSymbolic(
        m, n, m2, q, pinv, parent, leftmost_perm,
        vnz, rnz, rcount, ordering_use, rowptr, colval, nothing
    )
end

"""
    csr_factor(A::SparseMatrixCSR, sym::CSRQRSymbolic; tol=nothing, drop_tol=0,
               adaptive_dense=false, dense_threshold=0.4) -> CSRQRFactorization

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

If `adaptive_dense=true`, the numeric kernel monitors the density of the
just-emitted Householder columns. Once the active submatrix exceeds
`dense_threshold * (m2 - k + 1)` density (default 40%) over several
consecutive columns, it materializes the trailing block as a dense matrix
and finishes with LAPACK `geqp3!`. The composed column permutation is
stored in `F.q_eff`.
"""
function csr_factor(
        A::SparseMatrixCSR{Bi, T}, sym::CSRQRSymbolic;
        tol::Union{Nothing, Real} = nothing,
        drop_tol::Real = 0,
        adaptive_dense::Bool = false,
        dense_threshold::Real = 0.4
    ) where {Bi, T}
    return _factor_kernel(
        A, sym, tol, nothing, real(T)(drop_tol),
        adaptive_dense, real(T)(dense_threshold)
    )
end

"""
    csr_qr(A::SparseMatrixCSR; tol=nothing, ordering=:default, drop_tol=0,
           adaptive_dense=false, dense_threshold=0.4) -> CSRQRFactorization

One-shot convenience: equivalent to `csr_factor(A, csr_analyze(A; ordering); tol)`.

When `ordering=:default` (the default), the column ordering is `:amd` if the
AMD.jl extension is loaded (`using AMD`) and `:natural` otherwise. On the
typical dense-fill matrices that arise from nonlinear solver linsolves,
`:amd` roughly halves the factor time. Pass `ordering=:natural` to opt out
for matrices whose columns are already well-ordered.
"""
function csr_qr(
        A::SparseMatrixCSR{Bi, T};
        tol::Union{Nothing, Real} = nothing,
        ordering::Symbol = :default,
        drop_tol::Real = 0,
        adaptive_dense::Bool = false,
        dense_threshold::Real = 0.4
    ) where {Bi, T}
    sym = csr_analyze(A; ordering = ordering)
    return csr_factor(
        A, sym; tol = tol, drop_tol = drop_tol,
        adaptive_dense = adaptive_dense, dense_threshold = dense_threshold
    )
end

"""
    csr_refactor!(F::CSRQRFactorization, A::SparseMatrixCSR; tol=nothing, drop_tol=0,
                  adaptive_dense=false, dense_threshold=0.4) -> CSRQRFactorization

Numeric refactorization, mutating `F` in place. If the sparsity pattern of
`A` matches the one captured in `F.sym`, the symbolic is reused (skipping
the etree / `pinv` / `leftmost` work) and the pre-allocated numeric
workspace on the symbolic is reused — steady-state calls allocate nothing
(unless `adaptive_dense=true` triggers, which allocates the dense block).
Otherwise a fresh analyze is performed (and a new workspace lazily built)
before refactoring.

The `drop_tol`, `adaptive_dense`, and `dense_threshold` keywords have the
same meaning as in [`csr_factor`](@ref).

`F`'s `V_*`, `R_*`, `beta` buffers are overwritten with the new values
(growing only if the previous bounds were undersized). The return value is
`F` itself.
"""
function csr_refactor!(
        F::CSRQRFactorization{T},
        A::SparseMatrixCSR{Bi};
        tol::Union{Nothing, Real} = nothing,
        drop_tol::Real = 0,
        adaptive_dense::Bool = false,
        dense_threshold::Real = 0.4
    ) where {T, Bi}
    dt = real(T)(drop_tol)
    dth = real(T)(dense_threshold)
    if _pattern_matches(F.sym, A)
        return _factor_kernel(A, F.sym, tol, F, dt, adaptive_dense, dth)
    else
        sym = csr_analyze(A; ordering = F.sym.ordering)
        return _factor_kernel(A, sym, tol, F, dt, adaptive_dense, dth)
    end
end

# ---------------------------------------------------------------------------
# Numeric kernel — Davis cs_qr on the row+column permuted matrix S = P A Q.
# ---------------------------------------------------------------------------

# Get (or lazily create) a numeric workspace of the right type attached to
# the symbolic. Reuses on subsequent calls; falls back to a fresh allocation
# if the cached workspace has a mismatched element type.
@inline function _get_workspace(
        ::Type{T}, sym::CSRQRSymbolic,
        nnz_A::Int
    ) where {T}
    RT = real(T)
    ws = sym.workspace
    if ws isa _CSRQRWorkspace{T, RT}
        # Reuse — but ensure capacities are sufficient. n and m2 are sym-fixed.
        if length(ws.rowval_A) < nnz_A
            resize!(ws.rowval_A, nnz_A)
            resize!(ws.nzval_A, nnz_A)
            resize!(ws.rowval_S, nnz_A)
            resize!(ws.nzval_S, nnz_A)
        end
        return ws
    end
    new_ws = _alloc_workspace(T, sym.m, sym.n, sym.m2, nnz_A)
    sym.workspace = new_ws
    return new_ws
end

function _factor_kernel(
        A::SparseMatrixCSR{Bi, T}, sym::CSRQRSymbolic,
        tol::Union{Nothing, Real},
        F::Union{Nothing, CSRQRFactorization},
        drop_tol::Real = zero(real(T)),
        adaptive_dense::Bool = false,
        dense_threshold::Real = real(T)(0.4)
    ) where {Bi, T}
    m, n = size(A)
    (m == sym.m && n == sym.n) ||
        throw(DimensionMismatch("A is $m x $n but symbolic is $(sym.m) x $(sym.n)"))

    RT = real(T)
    nnz_A = length(A.colval)
    ws = _get_workspace(T, sym, nnz_A)::_CSRQRWorkspace{T, RT}

    # Single-pass CSR -> CSC(A) conversion + Frobenius norm + column-norm
    # cache (the column norms feed the zero-column check below). All buffers
    # come from the workspace.
    fro2 = _csr_to_csc_with_norms!(ws, A)
    fro = sqrt(fro2)
    tol_use = tol === nothing ? RT(eps(RT) * max(m, n)) * fro : RT(max(tol, 0))
    tol2 = tol_use * tol_use

    # Value-aware repivot for numerically-zero columns. The returned sym
    # always carries the same workspace (we pass sym.workspace through in
    # the rebuild path), so `ws` stays valid.
    sym_use = _maybe_repivot_zero_cols_from_norms(ws.col_nrm2, sym, fro)

    # Apply row+column permutation S = (P A Q) into workspace buffers.
    _permute_pq!(ws, sym_use.pinv, sym_use.q, m, n)

    # Pass 1: factor and collect the positions (in sym_use.q order) of any
    # columns the kernel flags rank-deficient. `def_pos` is workspace-owned and
    # reset here, so the common full-rank path does no extra heap work.
    def_pos = ws.def_pos
    empty!(def_pos)
    F1 = _csc_qr_numeric!(
        ws, sym_use, tol_use, tol2, F, RT(drop_tol),
        adaptive_dense, RT(dense_threshold), def_pos
    )

    # Fast path: full-rank (no deficiency) or the adaptive-dense fallback fired
    # (geqp3 already produces a minimum-residual factorization for the dense
    # tail). Either way pass 1's result is correct and final — single pass. This
    # preserves the byte-identical, zero-allocation full-rank behavior.
    if isempty(def_pos) || F1.k_dense != 0
        return F1
    end

    # Rank-deficient: iterate the deferral to a FIXED POINT. The numeric kernel's
    # rank test is order-dependent: a column flagged deficient (residual <= tol)
    # under one elimination order can become an INDEPENDENT pivot under another
    # once the columns spanning it are themselves deferred. So a single re-pass
    # (move pass-1's deficient set to the end) is not a fixed point under
    # chained/overlapping mutual dependence: it can leave an interleaved zero
    # pivot — the exact #23 pathology — and the basic back-substitution in
    # `_usolve!` is then non-minimum.
    #
    # We instead grow a confirmed LEADING-INDEPENDENT set `is_indep` (by original
    # column index) and order q = [is_indep cols, in confirmation order; the rest,
    # in their current order]. Key monotonicity: a column that pivots nonzero with
    # a given set of predecessors still pivots nonzero with any SUBSET of them
    # (removing projections cannot shrink its residual). Placing `is_indep` first,
    # in the stable order they were confirmed, makes each confirmed column's
    # predecessors a subset of its predecessors at confirmation time, so it stays
    # nonzero. Hence `is_indep` only GROWS (bounded by n) — the loop terminates in
    # <= (n - rnk) iterations. At convergence no column outside `is_indep` pivots
    # nonzero, so every zero pivot trails (positions > rnk) and `_usolve!`'s basic
    # back-substitution is the true minimum-residual solve.
    is_indep = ws.is_indep  # length-n membership marker; persists across passes
    if length(is_indep) < n
        resize!(is_indep, n)
    end
    @inbounds for j in 1:n
        is_indep[j] = 0
    end
    # `indep_order` lists confirmed-independent columns in confirmation order
    # (stable predecessors); `def_pos` from pass 1 are the zero-pivot positions,
    # so every OTHER position is an independent pivot to seed the set.
    indep_order = Vector{Int}(undef, n)
    n_indep = 0
    @inbounds for k in def_pos
        is_indep[sym_use.q[k]] = -1  # temporary mark: this position is deficient
    end
    @inbounds for k in 1:n
        j = sym_use.q[k]
        if is_indep[j] == 0
            is_indep[j] = 1
            n_indep += 1
            indep_order[n_indep] = j
        end
    end
    @inbounds for j in 1:n
        is_indep[j] == -1 && (is_indep[j] = 0)
    end

    sym_cur = sym_use
    F_cur = F1
    # Hard safety bound: `is_indep` grows by >= 1 each non-converged iteration and
    # is bounded by n, so n iterations is a strict upper bound. Exceeding it
    # signals a logic error (non-monotone `is_indep`), not a numerical edge case.
    local F_out::CSRQRFactorization
    converged = false
    iters = 0
    while iters < n
        iters += 1

        # q_next = [confirmed-independent cols in confirmation order; remaining
        # cols in current q-order]. The remaining (not-yet-confirmed) columns are
        # the only ones that can still pivot zero. A fresh array is required:
        # `_rebuild_symbolic_for_q` keeps the passed q by reference as `sym.q`, so
        # reusing one buffer across iterations would alias and clobber `sym_cur.q`.
        q_next = Vector{Int}(undef, n)
        @inbounds for t in 1:n_indep
            q_next[t] = indep_order[t]
        end
        pos = n_indep + 1
        @inbounds for k in 1:n
            j = sym_cur.q[k]
            if is_indep[j] == 0
                q_next[pos] = j
                pos += 1
            end
        end

        # If the order already matches and pass `iters-1`'s factorization had all
        # zero pivots trailing, no re-factor is needed. This is true exactly when
        # the current factorization's nonzero pivots are precisely the leading
        # `n_indep` positions, which holds iff q_next == sym_cur.q here.
        if iters == 1 && q_next == sym_cur.q
            # Pass 1 already has every zero pivot trailing (e.g. only the
            # structural zeros pre-trailed by the norm repivot were deficient).
            F_out = F_cur
            converged = true
            break
        end

        # Rebuild the value-independent symbolic for the forced order q_next and
        # re-permute S = (P A Q_next).
        qn, pinvn, parentn, leftmostn, m2n, vnzn, rnzn, rcountn =
            _rebuild_symbolic_for_q(
            sym_cur.pattern_rowptr, sym_cur.pattern_colval,
            sym_cur.m, sym_cur.n, q_next
        )
        sym_next = CSRQRSymbolic(
            sym_cur.m, sym_cur.n, m2n, qn, pinvn, parentn, leftmostn,
            vnzn, rnzn, rcountn, sym_cur.ordering, sym_cur.pattern_rowptr,
            sym_cur.pattern_colval, sym_cur.workspace
        )
        ws_next = _get_workspace(T, sym_next, nnz_A)::_CSRQRWorkspace{T, RT}
        _permute_pq!(ws_next, sym_next.pinv, sym_next.q, m, n)

        # Re-factor in place over F_cur, collecting the zero-pivot positions.
        empty!(def_pos)
        F_cur = _csc_qr_numeric!(
            ws_next, sym_next, tol_use, tol2, F_cur, RT(drop_tol),
            adaptive_dense, RT(dense_threshold), def_pos
        )

        # Promote every NONZERO-pivot column (position not in def_pos) not yet
        # confirmed — a previously-deferred column that pivoted nonzero this pass
        # and must move into the leading block. First mark this pass's zero-pivot
        # DEFERRED columns as -1 so the promotion scan skips them. The `== 0`
        # guard never demotes a confirmed column: those keep nonzero pivots by
        # the subset-of-predecessors argument, so they should not appear in
        # def_pos, and the guard preserves monotonicity if a threshold edge ever
        # flags one.
        @inbounds for k in def_pos
            j = sym_next.q[k]
            is_indep[j] == 0 && (is_indep[j] = -1)
        end
        new_indep = 0
        @inbounds for k in 1:n
            j = sym_next.q[k]
            if is_indep[j] == 0
                is_indep[j] = 1
                n_indep += 1
                indep_order[n_indep] = j
                new_indep += 1
            end
        end
        @inbounds for k in def_pos
            j = sym_next.q[k]
            is_indep[j] == -1 && (is_indep[j] = 0)
        end

        sym_cur = sym_next
        if new_indep == 0
            # Converged: no deferred column pivots nonzero, so the confirmed set
            # is exactly the leading block and all zero pivots trail.
            F_out = F_cur
            converged = true
            @debug "rank-deficient deferral converged" iters n rnk = F_out.rnk
            break
        end
    end

    converged ||
        error(
        "rank-deficient deferral did not converge in $n iterations; " *
            "the confirmed-independent set should grow monotonically and be " *
            "bounded by n (this indicates a logic error, not a numerical edge case)"
    )
    return F_out
end

# In-place CSR -> CSC of A, plus per-column squared norms and ||A||_F^2.
# Writes into ws.colptr_A, ws.rowval_A, ws.nzval_A, ws.col_nrm2. Returns
# the Frobenius-norm-squared. ws.work_perm doubles as a temporary copy of
# colptr used during scatter.
function _csr_to_csc_with_norms!(
        ws::_CSRQRWorkspace{T, RT},
        A::SparseMatrixCSR{Bi, T}
    ) where {Bi, T, RT}
    m, n = size(A)
    off = getoffset(A)
    rowptr = A.rowptr
    colval = A.colval
    nzval_in = A.nzval
    nnz_total = length(colval)

    colptr = ws.colptr_A
    rowval = ws.rowval_A
    nzval = ws.nzval_A
    col_nrm2 = ws.col_nrm2
    work = ws.work_perm

    # Reset accumulators.
    @inbounds for j in 1:n
        col_nrm2[j] = zero(RT)
    end
    # Use colptr as a count buffer first (we'll convert in place).
    @inbounds for j in 1:(n + 1)
        colptr[j] = 0
    end
    @inbounds for p in 1:nnz_total
        colptr[Int(colval[p]) + off + 1] += 1
    end
    # Cumsum: colptr[j+1] = colptr[j] + count[j].
    colptr[1] = 1
    @inbounds for j in 1:n
        colptr[j + 1] += colptr[j]
    end
    @inbounds for j in 1:(n + 1)
        work[j] = colptr[j]
    end
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
    return fro2
end

# In-place P A Q: writes the row- and column-permuted CSC into ws.colptr_S /
# ws.rowval_S / ws.nzval_S. Uses ws.colptr_A / ws.rowval_A / ws.nzval_A as
# input (filled by `_csr_to_csc_with_norms!`).
function _permute_pq!(
        ws::_CSRQRWorkspace{T, RT}, pinv::Vector{Int},
        q::Vector{Int}, m::Int, n::Int
    ) where {T, RT}
    colptr_A = ws.colptr_A; rowval_A = ws.rowval_A; nzval_A = ws.nzval_A
    colptr_S = ws.colptr_S; rowval_S = ws.rowval_S; nzval_S = ws.nzval_S
    colptr_S[1] = 1
    @inbounds for k in 1:n
        j = q[k]
        colptr_S[k + 1] = colptr_S[k] + (colptr_A[j + 1] - colptr_A[j])
    end
    @inbounds for k in 1:n
        j = q[k]
        src = colptr_A[j]; nincol = colptr_A[j + 1] - colptr_A[j]
        dst = colptr_S[k]
        for t in 0:(nincol - 1)
            rowval_S[dst + t] = pinv[rowval_A[src + t]]
            nzval_S[dst + t] = nzval_A[src + t]
        end
    end
    return nothing
end

# Inspect column norms of A. If any are below `fro * eps(RT) * n` (i.e.,
# numerically zero), move those columns to the end of `sym.q` and rebuild
# the value-independent symbolic pieces (parent, leftmost, m2, pinv, vnz, rnz)
# for the new ordering. Returns either the original `sym` (no zero columns)
# or a freshly-built one.
function _maybe_repivot_zero_cols_from_norms(
        col_norms::Vector{RT},
        sym::CSRQRSymbolic,
        fro_A::Real
    ) where {RT}
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
    # Cheap fast-path: if sym.q's trailing positions are already exactly the
    # zero columns (in some order) and the prefix is the non-zero columns
    # (in some order), no rebuild is needed. This catches the common
    # `csr_refactor!` case where F.sym was already repivoted on the first
    # call.
    nzero_count = 0
    @inbounds for j in 1:n
        if col_norms[j] <= thr2
            nzero_count += 1
        end
    end
    already_at_end = nzero_count > 0
    @inbounds for k in (n - nzero_count + 1):n
        j = sym.q[k]
        if col_norms[j] > thr2
            already_at_end = false
            break
        end
    end
    if already_at_end
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
    q2, pinv2, parent2, leftmost2, m2_2, vnz2, rnz2, rcount2 =
        _rebuild_symbolic_for_q(
        sym.pattern_rowptr, sym.pattern_colval,
        sym.m, sym.n, q_new
    )
    return CSRQRSymbolic(
        sym.m, sym.n, m2_2, q2, pinv2, parent2, leftmost2,
        vnz2, rnz2, rcount2, sym.ordering, sym.pattern_rowptr,
        sym.pattern_colval, sym.workspace
    )
end

# Rebuild symbolic data for a given q (a column permutation).
function _rebuild_symbolic_for_q(
        rowptr::Vector{Int}, colval::Vector{Int},
        m::Int, n::Int, q::Vector{Int}
    )
    # Reuse _build_symbolic but force the q we've chosen.
    colptr_q, rowval_q = _csr_pattern_to_csc_permuted(rowptr, colval, q, m, n)
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

    rcount = _cs_counts(colptr_q, rowval_q, parent, leftmost_orig, m, n)
    rnz = 0
    @inbounds for k in 1:n
        rnz += rcount[k]
    end
    vnz = _vnz_estimate(leftmost_orig, m, n)
    return q, pinv, parent, leftmost_perm, m2, vnz, rnz, rcount
end

# Numeric loop. Mutates the workspace and (if F is non-nothing) the output
# factorization's V/R/beta arrays in place. Returns a CSRQRFactorization
# (the mutated F if provided, else a freshly-allocated one).
#
# `drop_tol > 0` enables approximate-QR fill control: entries j >= 2 of a
# Householder vector with `|x[vrows[q]]|^2 <= drop_tol^2 * vnorm2` are
# dropped and `β_k` is recomputed from the surviving `|v|^2`. The diagonal
# v1 is never dropped.
function _csc_qr_numeric!(
        ws::_CSRQRWorkspace{T, RT}, sym::CSRQRSymbolic,
        tol_use::RT, tol2::RT,
        F::Union{Nothing, CSRQRFactorization},
        drop_tol::RT = zero(RT),
        adaptive_dense::Bool = false,
        dense_threshold::RT = RT(0.4),
        def_pos::Union{Nothing, Vector{Int}} = nothing
    ) where {T, RT}
    drop_active = drop_tol > zero(RT)
    drop_tol2 = drop_tol * drop_tol
    # The dense fallback relies on LAPACK geqp3!/ormqr!, which exist only for
    # BLAS float types. For generic T (e.g. BigFloat, ForwardDiff.Dual) just run
    # the pure-Julia sparse kernel to completion.
    if adaptive_dense && !_is_blas_eltype(T)
        adaptive_dense = false
    end
    m, n, m2 = sym.m, sym.n, sym.m2
    parent = sym.parent
    leftmost = sym.leftmost

    # CSC of S = P A Q lives in workspace.
    colptr = ws.colptr_S
    rowval = ws.rowval_S
    nzval = ws.nzval_S

    # Allocate or reuse V/R/beta output buffers.
    if F === nothing
        Vp = Vector{Int}(undef, n + 1)
        Vi = Vector{Int}(undef, max(sym.vnz, 1))
        Vx = Vector{T}(undef, max(sym.vnz, 1))
        Rp = Vector{Int}(undef, n + 1)
        Ri = Vector{Int}(undef, max(sym.rnz, 1))
        Rx = Vector{T}(undef, max(sym.rnz, 1))
        beta = Vector{RT}(undef, n)
    else
        # Reuse F's buffers. They're already sized from the previous call so
        # in steady state no allocation happens. Resize defensively if smaller.
        Vp = F.V_colptr; Vi = F.V_rowval; Vx = F.V_nzval
        Rp = F.R_colptr; Ri = F.R_rowval; Rx = F.R_nzval
        beta = F.beta
        if length(Vp) < n + 1
            resize!(Vp, n + 1); resize!(Rp, n + 1)
        end
        if length(Vi) < sym.vnz
            resize!(Vi, sym.vnz); resize!(Vx, sym.vnz)
        end
        if length(Ri) < sym.rnz
            resize!(Ri, sym.rnz); resize!(Rx, sym.rnz)
        end
        if length(beta) < n
            resize!(beta, n)
        end
    end

    # Dense workspaces from pool. x must be zero (or cleared at end-of-step,
    # which the loop does); w must be zero so the != k stamp check works.
    # Both are zero-initialized at workspace creation. On subsequent calls
    # they remain "clean" (algorithm restores them), so no reset needed —
    # except defensively for the very first call we use zeros(...).
    x = ws.x
    s = ws.s
    w = ws.w
    vrows = ws.vrows
    # Make sure dense buffers can fit if sym.m2 changed (shouldn't, but defend).
    if length(x) < m2
        resize!(x, m2); fill!(x, zero(T))
    end
    if length(vrows) < m2
        resize!(vrows, m2)
    end
    # If `w` capacity grew (because we ran on a smaller sym previously then
    # a larger one), we'd need to zero the tail. In practice n is fixed.
    if length(w) < n
        Lold = length(w)
        resize!(w, n)
        @inbounds for j in (Lold + 1):n
            w[j] = 0
        end
    end
    if length(s) < n
        resize!(s, n)
    end

    Vp[1] = 1
    Rp[1] = 1
    rnz_total = 0
    vnz_total = 0
    rnk = 0

    # Adaptive-dense state. `k_sparse_done == 0` means we never transitioned.
    # When we decide to switch, we break out of the sparse loop with
    # `k_sparse_done > 0` and run the dense fallback below.
    k_sparse_done = 0    # number of sparse columns successfully completed
    consec_dense = 0     # consecutive columns whose V has crossed threshold
    # Minimum number of remaining columns to justify the dense overhead.
    # Below this we just finish sparse — at small block sizes geqp3 setup
    # cost dwarfs any saving.
    dense_min_remaining = 16
    # Require at least this many consecutive dense V columns before we
    # actually switch. This guards against isolated density spikes in an
    # otherwise-sparse matrix.
    dense_consec_required = 4

    # Generation-shifted stamp for the `w` marker. Each numeric call starts
    # a fresh generation (wbase + 1 .. wbase + n). After the call, all stamps
    # in w[] lie in [wbase + 1, wbase + n]; the *next* call uses a strictly
    # larger generation so stale stamps never collide.
    ws.wgen += 1
    wbase = ws.wgen * (n + 1)

    @inbounds for k in 1:n
        kstamp = wbase + k
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
                while jj != 0 && w[jj] != kstamp && jj < k
                    s[len + 1] = jj
                    len += 1
                    w[jj] = kstamp
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
                _grow_pair!(Ri, Rx, rnz_total + 1)
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
            _grow_pair!(Ri, Rx, rnz_total + 1)
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
            def_pos === nothing || push!(def_pos, k)
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
            elseif def_pos !== nothing
                push!(def_pos, k)
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
            _grow_pair!(Vi, Vx, vnz_total + vlen)
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

        # --- Adaptive dense fallback trigger -----------------------------
        # If the just-emitted V[:,k] is dense relative to the (m2 - k + 1)
        # active rows AND we've had `dense_consec_required` consecutive
        # dense columns AND there are enough remaining columns to justify
        # the dense overhead, break and finish in dense.
        #
        # The consecutive-dense guard stops the fallback from firing on
        # AMD-ordered inputs where the etree keeps individual V columns
        # sparse on average even when one isolated column is dense.
        if adaptive_dense && k < n
            active_rows = m2 - k
            if active_rows > 0
                density = vlen / RT(m2 - k + 1)
                if density > dense_threshold
                    consec_dense += 1
                else
                    consec_dense = 0
                end
                if (n - k) >= dense_min_remaining &&
                        consec_dense >= dense_consec_required
                    k_sparse_done = k
                    break
                end
            end
        end
    end

    if k_sparse_done == 0
        # No dense transition. Trim V/R and return pure-sparse.
        resize!(Vi, vnz_total); resize!(Vx, vnz_total)
        resize!(Ri, rnz_total); resize!(Rx, rnz_total)
        if F === nothing
            return CSRQRFactorization{T, RT}(
                sym.m, sym.n,
                Vp, Vi, Vx,
                Rp, Ri, Rx,
                beta, rnk, tol_use, sym,
                0, Matrix{T}(undef, 0, 0), T[], sym.q
            )
        else
            # Mutate F in place. In the steady-state refactor! case (no dense
            # transition, F previously also had no dense tail) we deliberately
            # avoid allocating fresh D/dtau/q_eff: reuse the existing empty
            # arrays and re-point q_eff at sym.q (no copy). This keeps the
            # @allocated refactor! exactly zero. We only reset D/dtau if the
            # previous call HAD transitioned to dense.
            F.m = sym.m; F.n = sym.n
            F.V_colptr = Vp; F.V_rowval = Vi; F.V_nzval = Vx
            F.R_colptr = Rp; F.R_rowval = Ri; F.R_nzval = Rx
            F.beta = beta
            F.rnk = rnk
            F.tol = tol_use
            F.sym = sym
            if F.k_dense != 0
                F.k_dense = 0
                F.D = Matrix{T}(undef, 0, 0)
                F.dtau = T[]
            end
            F.q_eff = sym.q
            return F
        end
    end

    # ----------------------------------------------------------------------
    # Dense fallback. We have V[:,1..k_sparse_done] and R[1..k_sparse_done, :]
    # for cols 1..k_sparse_done. For each remaining column j in
    # (k_sparse_done+1)..n we:
    #   * scatter S[:, j] into x
    #   * apply H_1..H_{k_sparse_done} (sparse) — also picks up R top rows
    #   * emit R[1..k_sparse_done, j] into a per-column staging buffer
    #     (we don't yet know the dense jpvt permutation, so we cannot write
    #     into the final CSC R until after geqp3)
    #   * place the active part x[k_sparse_done+1..m2] into a column of D
    # ----------------------------------------------------------------------
    ks = k_sparse_done
    n_active = n - ks
    m_active = m2 - ks
    # Dense block: column j corresponds to original column ks + j of S.
    D = Matrix{T}(undef, m_active, n_active)
    # Top R staging: top_R[i, j] = R[i, ks + j]. Stored densely; most of it
    # is zero, but at the densities that trigger this fallback the top rows
    # are nearly fully populated anyway.
    top_R = zeros(T, ks, n_active)

    @inbounds for jj in 1:n_active
        kcol = ks + jj
        kstamp = wbase + kcol
        top = n + 1
        c1 = colptr[kcol]; c2 = colptr[kcol + 1] - 1
        # 1) scatter S[:, kcol] into x and compute the ereach for prior H's.
        for p in c1:c2
            i = rowval[p]
            x[i] = nzval[p]
            lm = leftmost[i]
            if lm > 0 && lm <= ks
                len = 0
                jjj = lm
                while jjj != 0 && w[jjj] != kstamp && jjj <= ks
                    s[len + 1] = jjj
                    len += 1
                    w[jjj] = kstamp
                    jjj = parent[jjj]
                end
                while len > 0
                    top -= 1
                    s[top] = s[len]
                    len -= 1
                end
            end
        end

        # 2) Apply sparse H_1..H_{ks}: walk s[top..n] in increasing column
        # order; for each prior column p_idx <= ks we apply that H to x and
        # then emit x[p_idx] into top_R.
        for spos in top:n
            p_idx = s[spos]
            p_idx > ks && continue
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
            # Emit into top_R[p_idx, jj]; clear x[p_idx].
            top_R[p_idx, jj] = x[p_idx]
            x[p_idx] = zero(T)
        end

        # 3) Place active rows x[ks+1..m2] into D[:, jj]. The active vector
        # may have unmarked rows that are nonzero only if leftmost[i] <= ks
        # for that i. We must copy ALL rows ks+1..m2 from x — we have no
        # cheap pattern tracking here. Also clear x as we go.
        for i in 1:m_active
            xi = x[ks + i]
            D[i, jj] = xi
            x[ks + i] = zero(T)
        end
    end

    # 4) Run LAPACK column-pivoted QR on D. Result:
    #    - Upper triangle of D[1:n_active, 1:n_active] = R_dense
    #    - Strict lower triangle = Householder v's
    #    - dtau = Householder coefficients
    #    - jpvt = column permutation of the dense block (1-based)
    jpvt = zeros(LinearAlgebra.BlasInt, n_active)
    dtau = Vector{T}(undef, min(m_active, n_active))
    LinearAlgebra.LAPACK.geqp3!(D, jpvt, dtau)

    # 5) Compose q_eff = [sym.q[1..ks]; sym.q[ks .+ jpvt]].
    q_eff = Vector{Int}(undef, n)
    @inbounds for k in 1:ks
        q_eff[k] = sym.q[k]
    end
    @inbounds for j in 1:n_active
        q_eff[ks + j] = sym.q[ks + Int(jpvt[j])]
    end

    # 6) Emit R columns for the dense tail into CSC R.
    # For final column ks + j (post-jpvt), top rows come from
    # top_R[:, jpvt[j]] and bottom rows from upper triangle of D.
    @inbounds for j in 1:n_active
        kcol = ks + j
        src_col = Int(jpvt[j])
        # Top rows (1..ks): copy from top_R[:, src_col], skip exact zeros.
        for i in 1:ks
            v = top_R[i, src_col]
            if v != zero(T)
                rnz_total += 1
                if rnz_total > length(Ri)
                    _grow_pair!(Ri, Rx, rnz_total)
                end
                Ri[rnz_total] = i
                Rx[rnz_total] = v
            end
        end
        # Bottom (rows ks+1..ks+j) from upper triangle of D.
        for i in 1:j
            v = D[i, j]
            if v != zero(T) || i == j  # always emit diagonal for shape
                rnz_total += 1
                if rnz_total > length(Ri)
                    _grow_pair!(Ri, Rx, rnz_total)
                end
                Ri[rnz_total] = ks + i
                Rx[rnz_total] = v
            end
        end
        Rp[kcol + 1] = rnz_total + 1
        # No sparse V column for the dense tail.
        Vp[kcol + 1] = vnz_total + 1
        beta[kcol] = zero(RT)
    end

    # 7) Determine dense block's rank-revealing rank using consistent tol.
    @inbounds for j in 1:n_active
        d = abs(D[j, j])
        if d > tol_use
            rnk += 1
        end
    end

    # Trim V/R to actual sizes.
    resize!(Vi, vnz_total); resize!(Vx, vnz_total)
    resize!(Ri, rnz_total); resize!(Rx, rnz_total)

    return _finish_factorization!(
        F, sym, Vp, Vi, Vx, Rp, Ri, Rx, beta, rnk, tol_use,
        ks, D, dtau, q_eff
    )
end

# Construct a fresh `CSRQRFactorization` or refresh the fields of the
# provided `F` in place. Shared between the pure-sparse and dense-fallback
# return paths so the dense-tail fields stay in sync with the buffers.
function _finish_factorization!(
        F::Union{Nothing, CSRQRFactorization}, sym::CSRQRSymbolic,
        Vp, Vi::Vector{Int}, Vx::Vector{T},
        Rp, Ri::Vector{Int}, Rx::Vector{T},
        beta::Vector{RT}, rnk::Int, tol_use::RT,
        k_dense::Int, D::Matrix{T}, dtau::Vector{T}, q_eff::Vector{Int}
    ) where {T, RT}
    if F === nothing
        return CSRQRFactorization{T, RT}(
            sym.m, sym.n,
            Vp, Vi, Vx,
            Rp, Ri, Rx,
            beta, rnk, tol_use, sym,
            k_dense, D, dtau, q_eff
        )
    else
        # Mutate F in place, refresh fields.
        F.m = sym.m; F.n = sym.n
        F.V_colptr = Vp; F.V_rowval = Vi; F.V_nzval = Vx
        F.R_colptr = Rp; F.R_rowval = Ri; F.R_nzval = Rx
        F.beta = beta
        F.rnk = rnk
        F.tol = tol_use
        F.sym = sym
        F.k_dense = k_dense
        F.D = D
        F.dtau = dtau
        F.q_eff = q_eff
        return F
    end
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
    # When an adaptive-dense fallback transitioned at column k_dense, only the
    # first k_dense Householders live in V. The remaining ones live in F.D.
    ks = F.k_dense
    n_sparse = ks == 0 ? n : ks
    @inbounds for k in 1:n_sparse
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
    if ks > 0
        # Apply dense Householders to work[ks+1 .. ks+m_active] via LAPACK.
        # The dense block is stored in F.D (m_active x n_active), tau in F.dtau.
        # work has length m2; the dense block was built on rows ks+1..m2 of x.
        # Apply Qᴴ of the dense tail manually: allocation-free (LAPACK ormqr!
        # allocates an internal work buffer per call) and generic over eltype.
        # Q = H₁…H_r, H_j = I − τ_j v_j v_jᴴ, v_j in strict-lower D (v_j[j]=1).
        # Qᴴ applies H_jᴴ for j=1..r: y −= conj(τ_j)(v_jᴴ y) v_j.
        m_active = size(F.D, 1)
        D = F.D; dtau = F.dtau; r = length(dtau)
        if m_active > 0 && r > 0
            @inbounds for j in 1:r
                tj = dtau[j]
                tj == zero(T) && continue
                g = work[ks + j]
                for i in (j + 1):m_active
                    g += conj(D[i, j]) * work[ks + i]
                end
                c = conj(tj) * g
                work[ks + j] -= c
                for i in (j + 1):m_active
                    work[ks + i] -= c * D[i, j]
                end
            end
        end
    end
    return work
end

function _apply_Q!(F::CSRQRFactorization{T}, work::Vector{T}) where {T}
    Vp = F.V_colptr; Vi = F.V_rowval; Vx = F.V_nzval; beta = F.beta
    n = F.n
    ks = F.k_dense
    # If we transitioned to dense, first apply dense H's (in reverse via 'N').
    if ks > 0
        # Apply Q of the dense tail manually (allocation-free): H_j for j=r..1.
        m_active = size(F.D, 1)
        D = F.D; dtau = F.dtau; r = length(dtau)
        if m_active > 0 && r > 0
            @inbounds for j in r:-1:1
                tj = dtau[j]
                tj == zero(T) && continue
                g = work[ks + j]
                for i in (j + 1):m_active
                    g += conj(D[i, j]) * work[ks + i]
                end
                c = tj * g
                work[ks + j] -= c
                for i in (j + 1):m_active
                    work[ks + i] -= c * D[i, j]
                end
            end
        end
    end
    n_sparse = ks == 0 ? n : ks
    @inbounds for k in n_sparse:-1:1
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
function _usolve!(
        z::AbstractVector{T}, F::CSRQRFactorization{T},
        c::AbstractVector{T}
    ) where {T}
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

function LinearAlgebra.ldiv!(
        x::AbstractVector{T}, F::CSRQRFactorization{T},
        b::AbstractVector{T}
    ) where {T}
    length(b) == F.m || throw(DimensionMismatch("b length $(length(b)) != m=$(F.m)"))
    length(x) == F.n || throw(DimensionMismatch("x length $(length(x)) != n=$(F.n)"))
    m, n, m2 = F.m, F.n, F.sym.m2
    pinv = F.sym.pinv
    # Use the composed permutation: equals sym.q when no dense transition,
    # otherwise carries the dense block's geqp3 column pivoting.
    q = F.q_eff

    # Reuse pooled solve scratch on the symbolic's workspace when present
    # (allocation-free steady state); fall back to fresh buffers otherwise.
    ws = F.sym.workspace
    local work::Vector{T}
    local xprime::Vector{T}
    if ws isa _CSRQRWorkspace{T, real(T)} &&
            length(ws.solve_work) >= m2 && length(ws.solve_xprime) >= n
        work = ws.solve_work
        xprime = ws.solve_xprime
    else
        work = Vector{T}(undef, m2)
        xprime = Vector{T}(undef, n)
    end
    @inbounds for i in 1:m2
        work[i] = zero(T)
    end
    @inbounds for i in 1:m
        work[pinv[i]] = b[i]
    end
    # Slots m+1..m2 stay zero (fictitious rows).

    _apply_QH!(F, work)

    # Solve R x' = work[1:n].
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

# ---------------------------------------------------------------------------
# Precompile workload
# ---------------------------------------------------------------------------
#
# Exercise the hot paths (analyze / factor / refactor / solve / rank / size)
# for the four standard BLAS element types and both index types, on tiny
# well-conditioned and rank-deficient inputs, so the specialized methods land
# in the package image. Only the `:natural` ordering is exercised: `:amd`
# lives in a weak-dep extension and cannot be loaded from here.

@setup_workload begin
    @compile_workload begin
        for T in (Float64, Float32, ComplexF64, ComplexF32)
            for Ti in (Int32, Int64)
                # Well-conditioned full-rank 6x6: identity + a few off-diagonals.
                rows = Ti[1, 2, 3, 4, 5, 6, 1, 2, 3, 4]
                cols = Ti[1, 2, 3, 4, 5, 6, 2, 3, 4, 5]
                vals = T[4, 4, 4, 4, 4, 4, 1, 1, 1, 1]
                A = sparsecsr(rows, cols, vals, 6, 6)
                b = ones(T, 6)

                F = csr_qr(A; ordering = :natural)
                F \ b
                rank(F)
                size(F)
                size(F, 1)

                # analyze / factor / refactor! round-trip with the same pattern.
                sym = csr_analyze(A; ordering = :natural)
                G = csr_factor(A, sym)
                csr_refactor!(G, A)
                G \ b

                # Rank-deficient 6x6: column 6 is a copy of column 1 (drop the
                # identity entry there so the matrix is genuinely rank 5).
                drows = Ti[1, 2, 3, 4, 5, 1, 2, 3, 4, 6]
                dcols = Ti[1, 2, 3, 4, 5, 2, 3, 4, 5, 1]
                dvals = T[4, 4, 4, 4, 4, 1, 1, 1, 1, 4]
                Ad = sparsecsr(drows, dcols, dvals, 6, 6)
                Fd = csr_qr(Ad; ordering = :natural)
                Fd \ b
                rank(Fd)
            end
        end
    end
end

end # module
