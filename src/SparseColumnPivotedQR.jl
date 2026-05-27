module SparseColumnPivotedQR

using LinearAlgebra
using SparseArrays
using SparseMatricesCSR

import LinearAlgebra: ldiv!, rank
import Base: \, size, eltype

export csr_qr, csr_analyze, csr_factor, csr_refactor!,
       CSRQRSymbolic, CSRQRFactorization

# SparseMatricesCSR offset: rowptr stores 1-based if Bi == 1, 0-based if Bi == 0.
@inline function getoffset(::SparseMatrixCSR{Bi}) where {Bi}
    return Bi == 1 ? 0 : 1
end

# ---------------------------------------------------------------------------
# Symbolic
# ---------------------------------------------------------------------------
#
# The Symbolic carries the pieces of work that depend only on the *sparsity
# pattern* of A: the column ordering used as the starting pivot order, an
# elimination tree of A^T A, row counts of R, and per-row capacity hints.
#
# Column ordering choices:
#   :natural — keep columns 1..n in input order (cheapest analyze).
#   :amd     — AMD on A^T A (sparsity-preserving column order).
#   :colamd  — currently same as :amd (AMD on A^T A); see the README.
#
# The numeric phase may deviate from this ordering when a candidate column
# turns out to be rank-deficient (column-pivoting kicks in). In that case the
# symbolic capacity hints become loose upper bounds rather than exact counts.

struct CSRQRSymbolic
    m::Int
    n::Int
    colperm::Vector{Int}        # initial column ordering (length n)
    rowcap::Vector{Int}         # per-row capacity hint for R (length m)
    ordering::Symbol            # :natural, :amd, :colamd
    pattern_rowptr::Vector{Int} # captured pattern of input A (for refactor!)
    pattern_colval::Vector{Int}
end

Base.size(S::CSRQRSymbolic) = (S.m, S.n)

# ---------------------------------------------------------------------------
# Numeric factorization
# ---------------------------------------------------------------------------
#
# Storage layout:
# - R is stored as a list of row-sparse vectors (R_cols[i], R_vals[i]), each kept
#   sorted by column index.
# - The k-th Householder vector v_k is stored "step-wise" as (Vstep_idx[k],
#   Vstep_val[k]): the row indices where v_k is nonzero (sorted), and the
#   corresponding values. This makes `applyQ` / `applyQH` O(sum nnz(v_k)).
# - sym is the Symbolic that was used to produce this factorization.

struct CSRQRFactorization{T, RT}
    m::Int
    n::Int
    R_cols::Vector{Vector{Int}}
    R_vals::Vector{Vector{T}}
    Vstep_idx::Vector{Vector{Int}}
    Vstep_val::Vector{Vector{T}}
    tau::Vector{T}
    perm::Vector{Int}
    rnk::Int
    tol::RT
    sym::CSRQRSymbolic
end

LinearAlgebra.rank(F::CSRQRFactorization) = F.rnk
Base.size(F::CSRQRFactorization) = (F.m, F.n)
Base.size(F::CSRQRFactorization, d::Integer) = d == 1 ? F.m : (d == 2 ? F.n : 1)
Base.eltype(::CSRQRFactorization{T}) where {T} = T

# ---------------------------------------------------------------------------
# Small helpers on sorted-(cols,vals) sparse rows
# ---------------------------------------------------------------------------

@inline function row_get(cols::Vector{Int}, vals::Vector{T}, c::Int) where {T}
    idx = searchsortedfirst(cols, c)
    if idx <= length(cols) && cols[idx] == c
        return vals[idx]
    end
    return zero(T)
end

@inline function row_remove!(cols::Vector{Int}, vals::Vector{T}, c::Int) where {T}
    idx = searchsortedfirst(cols, c)
    if idx <= length(cols) && cols[idx] == c
        deleteat!(cols, idx)
        deleteat!(vals, idx)
        return true
    end
    return false
end

@inline function row_set!(cols::Vector{Int}, vals::Vector{T}, c::Int, v::T) where {T}
    idx = searchsortedfirst(cols, c)
    if idx <= length(cols) && cols[idx] == c
        vals[idx] = v
    else
        insert!(cols, idx, c)
        insert!(vals, idx, v)
    end
    return nothing
end

# Convert a SparseMatrixCSR to a list of (cols, vals) row arrays (sorted by col).
# Applies an optional column permutation: if colperm_inv[j] = new_col, then
# input column j ends up as column colperm_inv[j] in the output rows.
#
# When `rowcap` is provided, allocate each row at the upper-bound capacity
# directly (Vector{...}(undef, cap)) and shrink to the actual nnz via resize!.
# This avoids a separate sizehint!() reallocation per row.
function _csr_to_rows(A::SparseMatrixCSR{Bi, T},
                     colperm_inv::Union{Nothing, Vector{Int}}=nothing,
                     rowcap::Union{Nothing, Vector{Int}}=nothing) where {Bi, T}
    m, n = size(A)
    cols = Vector{Vector{Int}}(undef, m)
    vals = Vector{Vector{T}}(undef, m)
    rowptr = A.rowptr
    colval = A.colval
    nzval = A.nzval
    off = getoffset(A)
    @inbounds for i in 1:m
        r1 = rowptr[i] + off
        r2 = rowptr[i + 1] + off - 1
        nz = r2 - r1 + 1
        cap = (rowcap === nothing) ? nz : max(rowcap[i], nz)
        ci = Vector{Int}(undef, cap)
        vi = Vector{T}(undef, cap)
        k = 1
        if colperm_inv === nothing
            for p in r1:r2
                ci[k] = Int(colval[p]) + off
                vi[k] = nzval[p]
                k += 1
            end
        else
            for p in r1:r2
                ci[k] = colperm_inv[Int(colval[p]) + off]
                vi[k] = nzval[p]
                k += 1
            end
        end
        # Sort if needed. We sort only the first nz entries; the tail
        # storage is unused (length stays at nz after the resize! below).
        unsorted = false
        for p in 2:nz
            if ci[p - 1] > ci[p]
                unsorted = true
                break
            end
        end
        if unsorted
            # Sort the first nz entries (the tail beyond nz is undefined memory).
            view_c = view(ci, 1:nz)
            view_v = view(vi, 1:nz)
            pp = sortperm(view_c)
            ci_sorted = view_c[pp]
            vi_sorted = view_v[pp]
            copyto!(view_c, ci_sorted)
            copyto!(view_v, vi_sorted)
        end
        # Logical length = nz (the rest of `cap` is reserved capacity).
        resize!(ci, nz)
        resize!(vi, nz)
        cols[i] = ci
        vals[i] = vi
    end
    return cols, vals
end

# ---------------------------------------------------------------------------
# Pattern capture (for refactor!)
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
# csr_analyze: build a Symbolic from A
# ---------------------------------------------------------------------------

"""
    csr_analyze(A::SparseMatrixCSR; ordering=:natural) -> CSRQRSymbolic

Symbolic analysis phase. Captures the sparsity pattern of `A`, computes an
initial column ordering, and (where applicable) per-row capacity hints for
`R`. The returned `CSRQRSymbolic` can be reused for repeated factorizations
of matrices with identical sparsity patterns via `csr_refactor!`.

`ordering` controls the initial column permutation handed to the numeric
phase. Supported values:

  * `:natural` — identity ordering, columns kept in input order.
  * `:amd`     — AMD ordering of `AᵀA` (requires `AMD.jl` to be loaded; the
                  package extension wires this up automatically). Falls back
                  to `:natural` if AMD is unavailable.
  * `:colamd`  — alias for `:amd` for now (true COLAMD is not yet
                  implemented; AMD on `AᵀA` is a reasonable substitute).

The numeric phase (`csr_factor`) is permitted to deviate from this initial
ordering when a candidate column is rank-deficient (rank-revealing pivot).
"""
function csr_analyze(A::SparseMatrixCSR{Bi}; ordering::Symbol=:natural) where {Bi}
    m, n = size(A)
    rowptr, colval = _capture_pattern(A)

    colperm, rowcap = _analyze_pattern(rowptr, colval, m, n, ordering)

    return CSRQRSymbolic(m, n, colperm, rowcap, ordering, rowptr, colval)
end

# This is the pure-symbolic kernel; it does not depend on T or on A's
# Bi parameter. Extensions (AMD.jl) hook in by overriding _amd_colperm.
function _analyze_pattern(rowptr::Vector{Int}, colval::Vector{Int},
                          m::Int, n::Int, ordering::Symbol)
    colperm = if ordering === :natural
        collect(1:n)
    elseif ordering === :amd || ordering === :colamd
        _amd_colperm(rowptr, colval, m, n)
    else
        throw(ArgumentError("Unknown ordering :$ordering (expected :natural, :amd, or :colamd)"))
    end

    # Row capacity hints (upper bounds on nnz per row of R). For now we
    # compute via the etree + row-count algorithm of Gilbert-Ng-Peyton,
    # operating on A^T A implicitly. If the etree code is unavailable or the
    # ordering deviates substantially during numeric pivoting, these are
    # treated as hints (sizehint!) rather than fixed capacities.
    rowcap = _row_capacity_hint(rowptr, colval, m, n, colperm)

    return colperm, rowcap
end

# Default fallback: natural ordering. Overridden by the AMD.jl extension.
_amd_colperm(rowptr, colval, m, n) = collect(1:n)

# ---------------------------------------------------------------------------
# Etree + row counts of R = qr(A) symbolic. Algorithm from Davis (CSparse),
# adapted to operate on A given in CSR (rows of A). The column elimination
# tree of A is the etree of A^T A; we never form A^T A explicitly.
# ---------------------------------------------------------------------------

# Column elimination tree of A (= etree of A^T A) using CSparse cs_etree
# with ata=1. We iterate over columns of A, so we first bucket-sort the CSR
# pattern into a CSC pattern. Returns parent[1:n] (0 = root) and the CSC
# pattern, which we reuse later for row counts.
#
# CSparse semantics (1-based with 0 = "no parent"):
#   parent[k]   = parent of k in the etree, or 0
#   ancestor[k] = path-compression pointer used during construction
#   prev[i]     = previous column of A that touched row i (for ata=1)
#
# For each column k of A and each nonzero (i, k), walk from prev[i] up
# through ancestors; set ancestor on the way to k, and when we hit a node
# with no ancestor yet, set parent = k.
function _coletree_via_csc(rowptr::Vector{Int}, colval::Vector{Int}, m::Int, n::Int)
    colptrc, rowidxc = _csr_pattern_to_csc(rowptr, colval, m, n)
    parent = zeros(Int, n)
    ancestor = zeros(Int, n)
    prev = zeros(Int, m)
    @inbounds for k in 1:n
        c1 = colptrc[k]
        c2 = colptrc[k + 1] - 1
        for p in c1:c2
            i = rowidxc[p]
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
    return parent, colptrc, rowidxc
end

# Convert CSR pattern -> CSC pattern (Bucketed). Returns colptr (length n+1), rowidx.
function _csr_pattern_to_csc(rowptr::Vector{Int}, colval::Vector{Int}, m::Int, n::Int)
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
    rowidx = Vector{Int}(undef, nnz_total)
    work = copy(colptr)
    @inbounds for i in 1:m
        r1 = rowptr[i]; r2 = rowptr[i + 1] - 1
        for p in r1:r2
            j = colval[p]
            rowidx[work[j]] = i
            work[j] += 1
        end
    end
    return colptr, rowidx
end

# Row counts of R = qr(A) via the standard algorithm. Given the column etree
# `parent` (etree of A^T A) and the CSC pattern, return rowcounts[1:n] such
# that nnz(R[k, :]) <= rowcounts[k].
#
# For QR, a *symbolic* upper bound on row k of R is:
#   |Rk| = | union over (k = anc of every i where A[i, :] has any column j with col-etree ancestor=k) |
# A practical, slightly loose upper bound that is still sharper than "n - k + 1"
# is to compute the column counts of L of A^T A (= row counts of R^T = R^T's columns)
# which equals the row counts of R. The CSparse cs_counts gives this directly.
#
# We use a simpler upper bound: for each column k, the count is 1 (the diagonal)
# + the number of columns j > k that descend from k in the etree along with the
# nonzero pattern of A in column k. To keep this robust we just compute the
# "reach" of each row of A and add to the row counts of all ancestors along
# the etree path. This is O(nnz(R)) and matches the standard cs_counts result
# upper bound used by CXSparse for symbolic QR.
function _row_counts(rowptr::Vector{Int}, colval::Vector{Int},
                     parent::Vector{Int}, colptrc::Vector{Int}, rowidxc::Vector{Int},
                     m::Int, n::Int)
    # We use a simple safe upper bound: for each row i of A, walk the columns
    # j ∈ A[i, :]; the row count of R[j, :] gets +1 for each unique row i in
    # the column j's reach. Concretely:
    #   * Process rows i of A in order.
    #   * For each i, find the columns reached (sorted). Each col j contributes
    #     1 to rowcount[j].
    # This double-counts when the same (j) is reached from multiple ancestors;
    # to avoid that we use a "marker" array and ascend etree from each j until
    # we hit a marked node.
    rowcounts = zeros(Int, n)
    mark = fill(0, n)
    @inbounds for i in 1:m
        r1 = rowptr[i]; r2 = rowptr[i + 1] - 1
        # ascend from each j = colval[p..r2] up to common marked ancestor.
        for p in r1:r2
            j = colval[p]
            jj = j
            while jj != 0 && mark[jj] != i
                rowcounts[jj] += 1
                mark[jj] = i
                jj = parent[jj]
            end
        end
    end
    # rowcounts[k] now upper-bounds nnz(R[k, :]). Add a small slack so that
    # numeric column-pivot deviation doesn't blow past this hint.
    return rowcounts
end

# Compute per-row capacity hint. Mapped under the column permutation.
function _row_capacity_hint(rowptr::Vector{Int}, colval::Vector{Int},
                            m::Int, n::Int, colperm::Vector{Int})
    # Permute pattern columns first.
    if colperm == 1:n || (length(colperm) == n && all(colperm[i] == i for i in 1:n))
        colval_p = colval
        rowptr_p = rowptr
    else
        invperm = zeros(Int, n)
        @inbounds for k in 1:n
            invperm[colperm[k]] = k
        end
        colval_p = Vector{Int}(undef, length(colval))
        @inbounds for p in eachindex(colval)
            colval_p[p] = invperm[colval[p]]
        end
        rowptr_p = rowptr
    end
    parent, colptrc, rowidxc = _coletree_via_csc(rowptr_p, colval_p, m, n)
    rowcounts = _row_counts(rowptr_p, colval_p, parent, colptrc, rowidxc, m, n)
    # Map rowcounts (per column k = per row k of R) to per-row capacity for our
    # row-storage. Our R uses row-storage indexed by row i = 1..m. Row counts
    # of R are per column k (= per row of R). Row i of R is empty for i > n.
    # So rowcap[i] = rowcounts[i] for i ≤ n, else 0.
    rowcap = zeros(Int, m)
    @inbounds for i in 1:min(m, n)
        # Add small slack for pivot deviation (5% or +4, whichever larger), capped at n.
        slack = max(4, rowcounts[i] >> 4)
        rowcap[i] = min(rowcounts[i] + slack, n)
    end
    return rowcap
end

# ---------------------------------------------------------------------------
# csr_factor: numeric factorization given a Symbolic
# ---------------------------------------------------------------------------

"""
    csr_factor(A::SparseMatrixCSR, sym::CSRQRSymbolic; tol=nothing) -> CSRQRFactorization

Numeric factorization of `A` using the symbolic info in `sym`. The initial
column ordering in `sym.colperm` is used as the starting pivot order; the
numeric phase may deviate when a candidate column is rank-deficient (the
column with largest residual norm is swapped in).

If `tol === nothing`, the default tolerance `eps(real(T)) * max(m, n) * ||A||_F`
is used (LAPACK `xgeqp3`-style).
"""
function csr_factor(A::SparseMatrixCSR{Bi, T}, sym::CSRQRSymbolic;
                    tol::Union{Nothing, Real}=nothing) where {Bi, T}
    return _factor_kernel(A, sym, tol)
end

"""
    csr_qr(A::SparseMatrixCSR; tol=nothing, ordering=:natural) -> CSRQRFactorization

One-shot convenience: runs `csr_analyze` and `csr_factor` together. Equivalent to
`csr_factor(A, csr_analyze(A; ordering); tol)`.
"""
function csr_qr(A::SparseMatrixCSR{Bi, T}; tol::Union{Nothing, Real}=nothing,
                ordering::Symbol=:natural) where {Bi, T}
    sym = csr_analyze(A; ordering=ordering)
    return csr_factor(A, sym; tol=tol)
end

"""
    csr_refactor!(F::CSRQRFactorization, A::SparseMatrixCSR; tol=nothing) -> CSRQRFactorization

Numeric refactorization of `A` reusing the symbolic info from `F`. `A` must
have the same sparsity pattern as the matrix originally factored. If the
pattern differs, a fresh analyze+factor is performed and a new factorization
is returned.

Note: this currently re-runs the full numeric factorization with the same
`Symbolic`; the gain over `csr_qr` is the avoided `csr_analyze` cost (etree,
AMD, capacity computation). Returns the resulting factorization (which may or
may not alias `F` depending on whether allocation was reused).
"""
function csr_refactor!(F::CSRQRFactorization{T},
                       A::SparseMatrixCSR{Bi};
                       tol::Union{Nothing, Real}=nothing) where {T, Bi}
    if _pattern_matches(F.sym, A)
        return _factor_kernel(A, F.sym, tol)
    else
        sym = csr_analyze(A; ordering=F.sym.ordering)
        return _factor_kernel(A, sym, tol)
    end
end

# ---------------------------------------------------------------------------
# Numeric kernel (the original csr_qr loop, parameterized by Symbolic)
# ---------------------------------------------------------------------------

function _factor_kernel(A::SparseMatrixCSR{Bi, T}, sym::CSRQRSymbolic,
                        tol::Union{Nothing, Real}) where {Bi, T}
    m, n = size(A)
    (m == sym.m && n == sym.n) ||
        throw(DimensionMismatch("A is $m x $n but symbolic is $(sym.m) x $(sym.n)"))

    # Build inverse permutation so input column j -> position invperm[j] in R columns.
    invperm = zeros(Int, n)
    @inbounds for k in 1:n
        invperm[sym.colperm[k]] = k
    end

    R_cols, R_vals = _csr_to_rows(A, invperm, sym.rowcap)

    Vstep_idx = Vector{Vector{Int}}()
    Vstep_val = Vector{Vector{T}}()
    tau       = T[]

    # `perm[k]` is the original column index of the column currently in position k.
    perm = copy(sym.colperm)

    RT = real(T)
    col_nrm2 = zeros(RT, n)
    @inbounds for i in 1:m
        ci = R_cols[i]; vi = R_vals[i]
        for q in eachindex(ci)
            col_nrm2[ci[q]] += abs2(vi[q])
        end
    end
    col_nrm2_init = copy(col_nrm2)

    fro = sqrt(sum(col_nrm2))
    tol_use = tol === nothing ? RT(eps(RT) * max(m, n)) * fro : RT(max(tol, 0))
    tol2 = tol_use * tol_use

    kmax = min(m, n)
    rnk = kmax

    # Reusable workspaces
    w = zeros(T, n)
    w_touched = falses(n)
    mark_cols = Int[]
    sizehint!(mark_cols, n)
    new_cols_buf = Int[]
    new_vals_buf = T[]
    sizehint!(new_cols_buf, 2n)
    sizehint!(new_vals_buf, 2n)
    x_idx = Int[]
    x_val = T[]
    v_vals = T[]
    # x_pos[q] = position in R_cols[x_idx[q]] of the column-k entry (cached during gather,
    # reused during the Householder apply step to avoid a second searchsortedfirst).
    x_pos = Int[]

    # Threshold: prefer the AMD-ordered column at position k (sparsity-preserving)
    # unless it's substantially rank-deficient relative to the best remaining.
    pivot_factor = RT(0.1)  # only deviate if amd-column norm < pivot_factor * max-norm

    for k in 1:kmax
        # --- Pivot column selection ---
        # First, find the best (max-norm) candidate in k..n.
        p_best = k
        best = col_nrm2[k]
        @inbounds for j in (k + 1):n
            if col_nrm2[j] > best
                best = col_nrm2[j]
                p_best = j
            end
        end

        # Default choice = position k (which under sym.colperm is the AMD-preferred
        # column). Deviate only when the AMD candidate looks rank-deficient relative
        # to the max-norm column.
        p = k
        cand = col_nrm2[k]
        if cand < pivot_factor * best
            p = p_best
            cand = best
        end

        # Rank-deficiency stop. Before declaring full rank, do a final recompute of
        # the candidate pivot column's norm if it's suspiciously small relative to
        # initial (avoid keeping a column whose downdated norm is just rounding noise).
        if col_nrm2_init[p] > 0 && cand <= sqrt(eps(RT)) * col_nrm2_init[p]
            s = zero(RT)
            @inbounds for ii in k:m
                ci2 = R_cols[ii]; vi2 = R_vals[ii]
                idx2 = searchsortedfirst(ci2, p)
                if idx2 <= length(ci2) && ci2[idx2] == p
                    s += abs2(vi2[idx2])
                end
            end
            cand = s
            col_nrm2[p] = s
            col_nrm2_init[p] = s
            # If recompute reveals the AMD candidate is truly rank-deficient,
            # re-evaluate against the max-norm candidate.
            if p == k && cand < pivot_factor * best
                # Also recompute best to make a fair comparison.
                s2 = zero(RT)
                @inbounds for ii in k:m
                    ci2 = R_cols[ii]; vi2 = R_vals[ii]
                    idx2 = searchsortedfirst(ci2, p_best)
                    if idx2 <= length(ci2) && ci2[idx2] == p_best
                        s2 += abs2(vi2[idx2])
                    end
                end
                col_nrm2[p_best] = s2
                col_nrm2_init[p_best] = s2
                if s2 > cand
                    p = p_best
                    cand = s2
                end
            end
        end
        if cand <= tol2
            # Final defensive recompute: the best candidate might still be live
            # but the AMD candidate's recomputed value is below tol.
            if p == k && best > cand
                # Recompute best column and try again.
                s2 = zero(RT)
                @inbounds for ii in k:m
                    ci2 = R_cols[ii]; vi2 = R_vals[ii]
                    idx2 = searchsortedfirst(ci2, p_best)
                    if idx2 <= length(ci2) && ci2[idx2] == p_best
                        s2 += abs2(vi2[idx2])
                    end
                end
                col_nrm2[p_best] = s2
                col_nrm2_init[p_best] = s2
                if s2 > tol2
                    p = p_best
                    cand = s2
                end
            end
            if cand <= tol2
                rnk = k - 1
                break
            end
        end

        # --- Swap columns k and p in R, perm, and norms ---
        # Single-pass per row: locate k and p simultaneously and apply the
        # smallest structural change.
        if p != k
            # `p` may be > or < k in general, but the rank-revealing pivot keeps p >= k
            # by construction (we pivot from positions k..n). Ensure k_lo < p_hi.
            k_lo, p_hi = k < p ? (k, p) : (p, k)
            col_nrm2[k], col_nrm2[p] = col_nrm2[p], col_nrm2[k]
            col_nrm2_init[k], col_nrm2_init[p] = col_nrm2_init[p], col_nrm2_init[k]
            perm[k], perm[p] = perm[p], perm[k]
            @inbounds for i in 1:m
                ci = R_cols[i]; vi = R_vals[i]
                L = length(ci)
                L == 0 && continue
                # Find idx_lo = position of k_lo (or insertion position).
                idx_lo = searchsortedfirst(ci, k_lo)
                has_lo = idx_lo <= L && ci[idx_lo] == k_lo
                # Find idx_hi >= idx_lo (binary search restricted).
                idx_hi = searchsortedfirst(view(ci, idx_lo:L), p_hi) + idx_lo - 1
                has_hi = idx_hi <= L && ci[idx_hi] == p_hi
                if !has_lo && !has_hi
                    continue
                end
                if has_lo && has_hi
                    # Both present: swap values in place; no structural change.
                    vi[idx_lo], vi[idx_hi] = vi[idx_hi], vi[idx_lo]
                elseif has_lo
                    # Move k_lo -> p_hi position.
                    # Shift entries [idx_lo+1 : idx_hi-1] left by one, write p_hi at idx_hi-1.
                    v_save = vi[idx_lo]
                    for t in idx_lo:(idx_hi - 2)
                        ci[t] = ci[t + 1]
                        vi[t] = vi[t + 1]
                    end
                    ci[idx_hi - 1] = p_hi
                    vi[idx_hi - 1] = v_save
                else  # has_hi only
                    # Move p_hi -> k_lo position.
                    # Shift entries [idx_lo : idx_hi-1] right by one, write k_lo at idx_lo.
                    v_save = vi[idx_hi]
                    for t in idx_hi:-1:(idx_lo + 1)
                        ci[t] = ci[t - 1]
                        vi[t] = vi[t - 1]
                    end
                    ci[idx_lo] = k_lo
                    vi[idx_lo] = v_save
                end
            end
        end

        # --- Gather column-k entries from rows k..m to build Householder x ---
        empty!(x_idx); empty!(x_val); empty!(x_pos)
        @inbounds for i in k:m
            ci = R_cols[i]; vi = R_vals[i]
            idx = searchsortedfirst(ci, k)
            if idx <= length(ci) && ci[idx] == k
                v = vi[idx]
                if v != 0
                    push!(x_idx, i)
                    push!(x_val, v)
                    push!(x_pos, idx)
                end
            end
        end

        if isempty(x_idx)
            rnk = k - 1
            break
        end

        # Ensure row k is first (so v[1] corresponds to row k -> diagonal slot)
        if x_idx[1] != k
            pushfirst!(x_idx, k)
            pushfirst!(x_val, zero(T))
            # Row k didn't have column k yet; will be inserted later. Sentinel value 0
            # indicates "no cached position" — the apply step will recompute via search.
            pushfirst!(x_pos, 0)
        end

        # --- Compute Householder reflector: alpha, v, tau ---
        normx2 = zero(RT)
        @inbounds for q in eachindex(x_val)
            normx2 += abs2(x_val[q])
        end
        normx = sqrt(normx2)

        x1 = x_val[1]
        if T <: Real
            sgn = x1 >= 0 ? one(T) : -one(T)
        else
            sgn = x1 == 0 ? one(T) : x1 / abs(x1)
        end
        alpha = -sgn * normx

        resize!(v_vals, length(x_val))
        copyto!(v_vals, x_val)
        v_vals[1] = x1 - alpha

        vnorm2 = zero(RT)
        @inbounds for q in eachindex(v_vals)
            vnorm2 += abs2(v_vals[q])
        end

        if vnorm2 == 0
            rnk = k - 1
            break
        end

        tau_k = T(2) / T(vnorm2)

        # --- Apply H = I - tau v v^H to columns j > k of rows in x_idx ---
        empty!(mark_cols)
        nrows_v = length(x_idx)
        @inbounds for q in 1:nrows_v
            i = x_idx[q]
            vi_q = v_vals[q]
            if vi_q == 0
                continue
            end
            cvi_q = conj(vi_q)
            ci = R_cols[i]; vi = R_vals[i]
            # Tail starts immediately after the cached column-k position (if any).
            pos = x_pos[q]
            start = (pos == 0) ? searchsortedfirst(ci, k + 1) : pos + 1
            for p2 in start:length(ci)
                j = ci[p2]
                if !w_touched[j]
                    w_touched[j] = true
                    push!(mark_cols, j)
                end
                w[j] += cvi_q * vi[p2]
            end
        end

        sort!(mark_cols)

        @inbounds for q in 1:nrows_v
            i = x_idx[q]
            vi_q = v_vals[q]
            if vi_q == 0
                continue
            end
            factor = tau_k * vi_q
            ci = R_cols[i]; vi = R_vals[i]
            pos = x_pos[q]
            start = (pos == 0) ? searchsortedfirst(ci, k + 1) : pos + 1
            la = length(ci) - start + 1
            lb = length(mark_cols)

            # Fast path: if there's no overlap between the existing tail and the
            # marked columns AND no fill-in shrinks (all new entries are fill-in),
            # do an in-place merge by appending. We detect this by scanning once.
            # General path: merge into the scratch buffer, then copy back.
            # Resize the scratch buffer up front so push!() doesn't re-grow.
            new_max = la + lb
            resize!(new_cols_buf, new_max)
            resize!(new_vals_buf, new_max)
            nwrite = 0

            a = 1; b = 1
            while a <= la && b <= lb
                ca = ci[start + a - 1]
                cb = mark_cols[b]
                if ca == cb
                    nv = vi[start + a - 1] - factor * w[cb]
                    if nv != 0
                        nwrite += 1
                        new_cols_buf[nwrite] = ca
                        new_vals_buf[nwrite] = nv
                    end
                    a += 1; b += 1
                elseif ca < cb
                    nwrite += 1
                    new_cols_buf[nwrite] = ca
                    new_vals_buf[nwrite] = vi[start + a - 1]
                    a += 1
                else
                    nv = -factor * w[cb]
                    if nv != 0
                        nwrite += 1
                        new_cols_buf[nwrite] = cb
                        new_vals_buf[nwrite] = nv
                    end
                    b += 1
                end
            end
            while a <= la
                nwrite += 1
                new_cols_buf[nwrite] = ci[start + a - 1]
                new_vals_buf[nwrite] = vi[start + a - 1]
                a += 1
            end
            while b <= lb
                cb = mark_cols[b]
                nv = -factor * w[cb]
                if nv != 0
                    nwrite += 1
                    new_cols_buf[nwrite] = cb
                    new_vals_buf[nwrite] = nv
                end
                b += 1
            end

            new_total_len = start - 1 + nwrite
            old_total_len = length(ci)
            if new_total_len != old_total_len
                resize!(ci, new_total_len)
                resize!(vi, new_total_len)
            end
            # Copy nwrite elements from scratch buffer into ci/vi at offset start-1.
            if nwrite > 0
                copyto!(ci, start, new_cols_buf, 1, nwrite)
                copyto!(vi, start, new_vals_buf, 1, nwrite)
            end
        end

        @inbounds for j in mark_cols
            w[j] = zero(T)
            w_touched[j] = false
        end

        # --- Set R[k, k] = alpha; drop R[i, k] for i > k (they go to v storage) ---
        # The apply step preserved x_pos[q] (the column-k entry position in row i),
        # because it only overwrote positions start = x_pos[q] + 1 onward. Reuse it
        # to skip a redundant binary search.
        @inbounds begin
            ck = R_cols[k]; vk = R_vals[k]
            pos1 = x_pos[1]
            if pos1 != 0
                # Row k already had column k.
                vk[pos1] = T(alpha)
            else
                # Insert column k at the right sorted position in row k.
                idx = searchsortedfirst(ck, k)
                insert!(ck, idx, k)
                insert!(vk, idx, T(alpha))
            end
        end
        @inbounds for q in 1:nrows_v
            i = x_idx[q]
            i == k && continue
            pos = x_pos[q]
            ci_i = R_cols[i]; vi_i = R_vals[i]
            if pos != 0 && pos <= length(ci_i) && ci_i[pos] == k
                deleteat!(ci_i, pos)
                deleteat!(vi_i, pos)
            else
                row_remove!(ci_i, vi_i, k)
            end
        end

        # --- Store Householder vector step-wise ---
        push!(tau, tau_k)
        vidx = Int[];   sizehint!(vidx, nrows_v)
        vval = T[];     sizehint!(vval, nrows_v)
        @inbounds for q in 1:nrows_v
            vv = v_vals[q]
            vv == 0 && continue
            push!(vidx, x_idx[q])
            push!(vval, vv)
        end
        push!(Vstep_idx, vidx)
        push!(Vstep_val, vval)

        # --- Downdate column norms for j > k ---
        @inbounds begin
            ck = R_cols[k]; vk_ = R_vals[k]
            startc = searchsortedfirst(ck, k + 1)
            recompute_thresh = RT(sqrt(eps(RT)))
            for p2 in startc:length(ck)
                j = ck[p2]
                old = col_nrm2[j]
                col_nrm2[j] = old - abs2(vk_[p2])
                if col_nrm2[j] < 0
                    col_nrm2[j] = zero(RT)
                end
                if col_nrm2[j] <= recompute_thresh * col_nrm2_init[j]
                    s = zero(RT)
                    for ii in (k + 1):m
                        ci2 = R_cols[ii]; vi2 = R_vals[ii]
                        idx2 = searchsortedfirst(ci2, j)
                        if idx2 <= length(ci2) && ci2[idx2] == j
                            s += abs2(vi2[idx2])
                        end
                    end
                    col_nrm2[j] = s
                    col_nrm2_init[j] = s
                end
            end
        end
        col_nrm2[k] = zero(RT)
    end

    return CSRQRFactorization{T, RT}(m, n, R_cols, R_vals, Vstep_idx, Vstep_val,
                                     tau, perm, rnk, tol_use, sym)
end

# ---------------------------------------------------------------------------
# Solve path
# ---------------------------------------------------------------------------

function applyQH!(y::AbstractVector{T}, F::CSRQRFactorization{T}) where {T}
    nstep = length(F.tau)
    @inbounds for k in 1:nstep
        vidx = F.Vstep_idx[k]; vval = F.Vstep_val[k]
        nz = length(vidx)
        nz == 0 && continue
        beta = zero(T)
        for q in 1:nz
            beta += conj(vval[q]) * y[vidx[q]]
        end
        if beta == 0
            continue
        end
        tk_conj = conj(F.tau[k])
        for q in 1:nz
            y[vidx[q]] -= tk_conj * vval[q] * beta
        end
    end
    return y
end

function applyQ!(y::AbstractVector{T}, F::CSRQRFactorization{T}) where {T}
    nstep = length(F.tau)
    @inbounds for k in nstep:-1:1
        vidx = F.Vstep_idx[k]; vval = F.Vstep_val[k]
        nz = length(vidx)
        nz == 0 && continue
        beta = zero(T)
        for q in 1:nz
            beta += conj(vval[q]) * y[vidx[q]]
        end
        if beta == 0
            continue
        end
        tk = F.tau[k]
        for q in 1:nz
            y[vidx[q]] -= tk * vval[q] * beta
        end
    end
    return y
end

function backsub_R!(z::AbstractVector{T}, F::CSRQRFactorization{T}, c::AbstractVector{T}) where {T}
    rnk = F.rnk
    @inbounds for i in 1:rnk
        z[i] = c[i]
    end
    @inbounds for i in rnk:-1:1
        ci = F.R_cols[i]; vi = F.R_vals[i]
        diag_idx = searchsortedfirst(ci, i)
        if diag_idx > length(ci) || ci[diag_idx] != i
            error("Missing R diagonal at row $i (factorization is corrupt or rank stopped early)")
        end
        s = z[i]
        for p in (diag_idx + 1):length(ci)
            j = ci[p]
            j > rnk && break
            s -= vi[p] * z[j]
        end
        z[i] = s / vi[diag_idx]
    end
    @inbounds for i in (rnk + 1):length(z)
        z[i] = zero(T)
    end
    return z
end

function LinearAlgebra.ldiv!(x::AbstractVector{T}, F::CSRQRFactorization{T}, b::AbstractVector{T}) where {T}
    length(b) == F.m || throw(DimensionMismatch("b length $(length(b)) != m=$(F.m)"))
    length(x) == F.n || throw(DimensionMismatch("x length $(length(x)) != n=$(F.n)"))
    y = copy(b)
    applyQH!(y, F)
    z = Vector{T}(undef, F.n)
    backsub_R!(z, F, y)
    @inbounds for k in 1:F.n
        x[F.perm[k]] = z[k]
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
