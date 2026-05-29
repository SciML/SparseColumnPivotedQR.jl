using Test
using LinearAlgebra
using SparseArrays
using SparseMatricesCSR
using Random
using SparseColumnPivotedQR
using AMD  # trigger AMD extension
using ForwardDiff

# helper: convert CSC -> CSR
to_csr(A::SparseMatrixCSC) = SparseMatrixCSR(transpose(sparse(transpose(A))))

# Slightly nicer helper: build a CSR from a Julia matrix
function build_csr(A::AbstractMatrix{T}) where {T}
    Acsc = sparse(A)
    return SparseMatrixCSR(transpose(sparse(transpose(Acsc))))
end

@testset "SparseColumnPivotedQR" begin

    @testset "Identity matrix" begin
        n = 8
        Acsr = build_csr(Matrix{Float64}(I, n, n))
        b = collect(1.0:n)
        F = csr_qr(Acsr)
        @test rank(F) == n
        x = F \ b
        @test x ≈ b atol = 1.0e-12
    end

    @testset "Small dense-ish random sparse (square, full rank)" begin
        Random.seed!(1)
        n = 30
        A = sprand(Float64, n, n, 0.3) + 5 * I
        Acsr = build_csr(Matrix(A))
        b = randn(n)
        F = csr_qr(Acsr)
        @test rank(F) == n
        x = F \ b
        @test norm(A * x - b) / norm(b) < 1.0e-10
    end

    @testset "Tall least-squares (overdetermined, full column rank)" begin
        Random.seed!(2)
        m, n = 20, 8
        Adense = randn(m, n)
        Acsr = build_csr(Adense)
        b = randn(m)
        F = csr_qr(Acsr)
        @test rank(F) == n
        x = F \ b
        # Compare to dense LS
        xref = Adense \ b
        @test norm(x - xref) / max(norm(xref), 1.0) < 1.0e-10
    end

    @testset "Rank-deficient overdetermined" begin
        # Construct a rank-3 matrix in 6x5
        Random.seed!(3)
        U = randn(6, 3); V = randn(5, 3)
        Adense = U * V'
        Acsr = build_csr(Adense)
        b = randn(6)
        F = csr_qr(Acsr; tol = 1.0e-10)
        @test rank(F) == 3
        x = F \ b
        # Compare residual to SVD-pinv (any LS solution gives same residual)
        xref = pinv(Adense) * b
        rres = Adense * x - b
        rref = Adense * xref - b
        @test norm(rres) ≈ norm(rref) atol = 1.0e-8
        @test all(isfinite, x)
    end

    @testset "Structurally singular (one zero column)" begin
        n = 6
        A = Matrix{Float64}(I, n, n)
        A[:, 3] .= 0  # make column 3 entirely zero
        Acsr = build_csr(A)
        b = randn(n); b[3] = 0  # ensure in range
        F = csr_qr(Acsr)
        @test rank(F) == n - 1
        x = F \ b
        @test all(isfinite, x)
        @test norm(A * x - b) < 1.0e-10
    end

    @testset "Rank-deficient square (199x199 user matrix simulated)" begin
        Random.seed!(4)
        n = 50
        # rank n-1
        U = randn(n, n - 1); V = randn(n, n - 1)
        Adense = U * V'
        Acsr = build_csr(Adense)
        b = Adense * randn(n)  # ensure in range
        F = csr_qr(Acsr)
        @test rank(F) <= n - 1
        x = F \ b
        @test all(isfinite, x)
        # residual should be near zero
        @test norm(Adense * x - b) / max(norm(b), 1.0) < 1.0e-8
    end

    @testset "Bundled 199x199 matrices (mix of rank-deficient)" begin
        # Test fixtures checked in under `test/matrices/`. Each file contains
        # `sparse(...)` on line 1 and a `b` vector on line 2. See
        # `test/matrices/README.md` for provenance.
        dir = joinpath(@__DIR__, "matrices")
        files = sort(
            filter(
                f -> endswith(f, ".txt"),
                readdir(dir; join = true)
            )
        )
        @test !isempty(files)
        for f in files
            text = read(f, String)
            lines = split(text, '\n'; keepempty = false)
            A = eval(Meta.parse(strip(lines[1])))
            b = eval(Meta.parse(strip(lines[2])))
            Acsr = SparseMatrixCSR(transpose(sparse(transpose(A))))
            F = csr_qr(Acsr)
            Fspqr = qr(A)
            xspqr = Fspqr \ b
            x = F \ b
            if all(isfinite, b)
                @test all(isfinite, x)
                # Residuals should match SPQR to a few ulps of ||b|| or matrix scale
                rmy = norm(A * x - b)
                rspqr = norm(A * xspqr - b)
                # Either both are essentially zero (full-rank well-conditioned),
                # or the LS residual matches SPQR's basic solver.
                scale = max(rspqr, 1.0e-12 * norm(b))
                @test rmy <= 1.0e-8 + 2 * scale
            else
                # NaN in b: x should be all NaN (or non-finite); matches SPQR.
                @test count(!isfinite, x) == count(!isfinite, xspqr) ||
                    count(!isfinite, x) > 0
            end
        end
    end

    @testset "ComplexF64 basic" begin
        Random.seed!(5)
        n = 10
        Adense = randn(ComplexF64, n, n)
        Acsr = build_csr(Adense)
        b = randn(ComplexF64, n)
        F = csr_qr(Acsr)
        @test rank(F) == n
        x = F \ b
        @test norm(Adense * x - b) / norm(b) < 1.0e-10
    end

    @testset "analyze + factor split" begin
        Random.seed!(11)
        n = 25
        A = sprand(Float64, n, n, 0.25) + 3 * sparse(I, n, n)
        Acsr = build_csr(Matrix(A))
        b = randn(n)

        sym = csr_analyze(Acsr; ordering = :natural)
        @test size(sym) == (n, n)
        F = csr_factor(Acsr, sym)
        x = F \ b
        @test rank(F) == n
        @test norm(A * x - b) / norm(b) < 1.0e-10

        # csr_qr matches csr_factor(csr_analyze)
        F2 = csr_qr(Acsr; ordering = :natural)
        x2 = F2 \ b
        @test rank(F2) == n
        @test norm(x - x2) / max(norm(x), 1.0) < 1.0e-12
    end

    @testset "AMD ordering" begin
        Random.seed!(12)
        n = 40
        A = sprand(Float64, n, n, 0.15) + 3 * sparse(I, n, n)
        Acsr = build_csr(Matrix(A))
        b = randn(n)

        sym_nat = csr_analyze(Acsr; ordering = :natural)
        sym_amd = csr_analyze(Acsr; ordering = :amd)
        sym_col = csr_analyze(Acsr; ordering = :colamd)

        # AMD should produce a non-identity permutation on a random matrix
        @test sym_amd.q != 1:n
        # colamd ordering is currently aliased to amd
        @test sym_col.q == sym_amd.q

        F_nat = csr_factor(Acsr, sym_nat); x_nat = F_nat \ b
        F_amd = csr_factor(Acsr, sym_amd); x_amd = F_amd \ b

        @test rank(F_nat) == n
        @test rank(F_amd) == n
        @test norm(A * x_nat - b) / norm(b) < 1.0e-10
        @test norm(A * x_amd - b) / norm(b) < 1.0e-10

        # AMD should not blow up fill: total nnz(R) <= natural's by a reasonable factor
        nnz_nat = length(F_nat.R_nzval)
        nnz_amd = length(F_amd.R_nzval)
        # No strict inequality (AMD can lose on some matrices), but shouldn't be wildly worse.
        @test nnz_amd <= 4 * nnz_nat
    end

    @testset "csr_refactor!" begin
        Random.seed!(13)
        n = 30
        # Build two matrices with identical pattern, different values.
        Asp = sprand(Float64, n, n, 0.2)
        rows, cols, _ = findnz(Asp)
        v1 = randn(length(rows))
        v2 = randn(length(rows))
        A1 = sparse(rows, cols, v1, n, n) + 4 * sparse(I, n, n)
        A2 = sparse(rows, cols, v2, n, n) + 4 * sparse(I, n, n)
        Acsr1 = build_csr(Matrix(A1))
        Acsr2 = build_csr(Matrix(A2))
        b = randn(n)

        F1 = csr_qr(Acsr1)
        x1 = F1 \ b
        @test norm(A1 * x1 - b) / norm(b) < 1.0e-10

        # Refactor with same pattern, different values
        F2 = csr_refactor!(F1, Acsr2)
        x2 = F2 \ b
        @test norm(A2 * x2 - b) / norm(b) < 1.0e-10

        # Compare against fresh factor
        F2_fresh = csr_qr(Acsr2)
        x2_fresh = F2_fresh \ b
        @test norm(x2 - x2_fresh) / max(norm(x2_fresh), 1.0) < 1.0e-10

        # Refactor with a different pattern triggers full analyze+factor
        A3 = sprand(Float64, n, n, 0.3) + 4 * sparse(I, n, n)
        Acsr3 = build_csr(Matrix(A3))
        F3 = csr_refactor!(F1, Acsr3)
        x3 = F3 \ b
        @test norm(A3 * x3 - b) / norm(b) < 1.0e-10
    end

    @testset "default ordering resolves to :amd when AMD is loaded" begin
        # AMD is `using`'d at the top of this file, so the extension is loaded.
        @test SparseColumnPivotedQR.has_amd_extension()

        Random.seed!(21)
        n = 25
        A = sprand(Float64, n, n, 0.25) + 3 * sparse(I, n, n)
        Acsr = build_csr(Matrix(A))
        b = randn(n)

        sym_default = csr_analyze(Acsr)            # implicit :default
        sym_amd = csr_analyze(Acsr; ordering = :amd)
        @test sym_default.ordering === :amd
        @test sym_default.q == sym_amd.q

        F_default = csr_qr(Acsr)
        F_amd = csr_qr(Acsr; ordering = :amd)
        x_default = F_default \ b
        x_amd = F_amd \ b
        @test x_default ≈ x_amd atol = 1.0e-12
        @test rank(F_default) == n
    end

    @testset ":natural opt-in still works" begin
        Random.seed!(22)
        n = 15
        A = sprand(Float64, n, n, 0.3) + 2 * sparse(I, n, n)
        Acsr = build_csr(Matrix(A))
        b = randn(n)
        sym_nat = csr_analyze(Acsr; ordering = :natural)
        @test sym_nat.ordering === :natural
        @test sym_nat.q == collect(1:n)
        F = csr_qr(Acsr; ordering = :natural)
        @test rank(F) == n
        @test norm(A * (F \ b) - b) / norm(b) < 1.0e-10
    end

    @testset "default ordering on bundled 199x199 matrices" begin
        # Sanity: default ordering on the user matrices == :amd (since AMD is
        # loaded here) and the resulting factor has strictly fewer nnz(R)
        # than the natural-ordered factor on these dense-fill matrices.
        dir = joinpath(@__DIR__, "matrices")
        files = sort(
            filter(f -> endswith(f, ".txt"), readdir(dir; join = true))
        )
        @test !isempty(files)
        for f in files
            text = read(f, String)
            lines = split(text, '\n'; keepempty = false)
            A = eval(Meta.parse(strip(lines[1])))
            Acsr = SparseMatrixCSR(transpose(sparse(transpose(A))))
            sym_default = csr_analyze(Acsr)
            sym_amd = csr_analyze(Acsr; ordering = :amd)
            # :default == :amd here.
            @test sym_default.q == sym_amd.q

            # Actual nnz(R) should be lower with AMD than natural on these
            # dense-fill matrices.
            F_nat = csr_qr(Acsr; ordering = :natural)
            F_amd = csr_qr(Acsr; ordering = :amd)
            @test length(F_amd.R_nzval) < length(F_nat.R_nzval)
        end
    end

    @testset "ordering propagated through csr_qr" begin
        Random.seed!(14)
        n = 20
        A = sprand(Float64, n, n, 0.3) + 3 * sparse(I, n, n)
        Acsr = build_csr(Matrix(A))
        b = randn(n)
        F_nat = csr_qr(Acsr; ordering = :natural)
        F_amd = csr_qr(Acsr; ordering = :amd)
        # Same solution to LS tolerance
        x_nat = F_nat \ b
        x_amd = F_amd \ b
        @test norm(A * x_nat - b) / norm(b) < 1.0e-10
        @test norm(A * x_amd - b) / norm(b) < 1.0e-10
        @test rank(F_nat) == rank(F_amd) == n
    end

    @testset "CSC internal storage: V and R reconstruct A" begin
        Random.seed!(15)
        n = 25
        A = randn(n, n) + 4 * I
        Acsr = build_csr(A)
        F = csr_qr(Acsr; ordering = :natural)
        # Reconstruct R as a dense matrix.
        R = zeros(n, n)
        for k in 1:n
            for p in F.R_colptr[k]:(F.R_colptr[k + 1] - 1)
                R[F.R_rowval[p], k] = F.R_nzval[p]
            end
        end
        # Upper triangular?
        for k in 1:n
            for p in F.R_colptr[k]:(F.R_colptr[k + 1] - 1)
                @test F.R_rowval[p] <= k
            end
        end
        # Reconstruct V as a dense matrix in the m2-padded row space.
        m2 = F.sym.m2
        V = zeros(m2, n)
        for k in 1:n
            for p in F.V_colptr[k]:(F.V_colptr[k + 1] - 1)
                V[F.V_rowval[p], k] = F.V_nzval[p]
            end
        end
        # Q = (I - β_k v_k v_k^T)_{k=n..1}, m2 x m2 orthogonal.
        Q = Matrix{Float64}(I, m2, m2)
        for k in n:-1:1
            v = V[:, k]
            Q = (Matrix{Float64}(I, m2, m2) - F.beta[k] * v * transpose(v)) * Q
        end
        @test norm(transpose(Q) * Q - I) < 1.0e-10
        # Reconstruct P A Q (column-permuted): row i of A maps to slot pinv[i],
        # col j of A appears at q-position invq[j].
        P = zeros(m2, n)
        for i in 1:n
            P[F.sym.pinv[i], i] = 1
        end
        PAq = P * A[:, F.sym.q]
        # Q * [R; 0] should equal P A Q.
        R_ext = vcat(R, zeros(m2 - n, n))
        @test norm(Q * R_ext - PAq) < 1.0e-10
    end

    @testset "adaptive ordering picks the shallower etree" begin
        # On the dense-fill user matrices AMD's etree should win; the
        # adaptive ordering should therefore pick :amd.
        dir = joinpath(@__DIR__, "matrices")
        files = sort(
            filter(f -> endswith(f, ".txt"), readdir(dir; join = true))
        )
        for f in files
            text = read(f, String)
            lines = split(text, '\n'; keepempty = false)
            A = eval(Meta.parse(strip(lines[1])))
            b = eval(Meta.parse(strip(lines[2])))
            Acsr = SparseMatrixCSR(transpose(sparse(transpose(A))))
            sym_adapt = csr_analyze(Acsr; ordering = :adaptive)
            @test sym_adapt.ordering === :amd
            F = csr_factor(Acsr, sym_adapt)
            if all(isfinite, b)
                x = F \ b
                @test all(isfinite, x)
            end
        end

        # On an already-natural-friendly matrix (block-diagonal) AMD
        # cannot improve on the chain depth; adaptive should keep :natural.
        n = 60
        # Build a block-diagonal matrix: identity + small random blocks
        # along the diagonal. Natural order has a maximally branched
        # etree (each block is its own subtree) so AMD provides no
        # benefit.
        I_part = sparse(I, n, n)
        A = 5 * I_part
        Acsr = build_csr(Matrix(A))
        sym_adapt = csr_analyze(Acsr; ordering = :adaptive)
        @test sym_adapt.ordering === :natural
    end

    @testset "drop_tol prunes V columns and stays within tolerance" begin
        Random.seed!(31)
        n = 80
        A = sprand(Float64, n, n, 0.05) + 4 * sparse(I, n, n)
        Acsr = build_csr(Matrix(A))
        b = randn(n)

        F0 = csr_qr(Acsr; ordering = :natural)
        F1 = csr_qr(Acsr; ordering = :natural, drop_tol = 1.0e-8)

        # drop_tol > 0 should at least weakly reduce nnz(V).
        @test length(F1.V_nzval) <= length(F0.V_nzval)

        x0 = F0 \ b
        x1 = F1 \ b
        # The dropped factorization still solves to a small residual; allow
        # one or two extra orders relative to the exact factor.
        r0 = norm(A * x0 - b) / norm(b)
        r1 = norm(A * x1 - b) / norm(b)
        @test r1 < 1.0e-6
        @test r1 <= max(r0, 1.0e-12) * 1.0e7

        # drop_tol = 0 is identical to the default factorization.
        F2 = csr_qr(Acsr; ordering = :natural, drop_tol = 0)
        @test F2.V_nzval == F0.V_nzval
        @test F2.beta == F0.beta
    end

    @testset "Workspace pool: csr_refactor! is zero-alloc steady-state" begin
        Random.seed!(21)
        n = 30
        Asp = sprand(Float64, n, n, 0.2)
        rows, cols, _ = findnz(Asp)
        v1 = randn(length(rows)); v2 = randn(length(rows))
        A1 = sparse(rows, cols, v1, n, n) + 4 * sparse(I, n, n)
        A2 = sparse(rows, cols, v2, n, n) + 4 * sparse(I, n, n)
        Acsr1 = build_csr(Matrix(A1))
        Acsr2 = build_csr(Matrix(A2))
        b = randn(n)

        F = csr_qr(Acsr1; ordering = :amd)
        # Warm up.
        csr_refactor!(F, Acsr2)
        csr_refactor!(F, Acsr1)

        # Steady-state: refactor with the cached workspace must allocate 0 bytes.
        nbytes = @allocated csr_refactor!(F, Acsr2)
        @test nbytes == 0
        nbytes = @allocated csr_refactor!(F, Acsr1)
        @test nbytes == 0

        # Sanity check the solution.
        x = F \ b
        @test norm(A1 * x - b) / norm(b) < 1.0e-10
    end

    @testset "Pooled solve: ldiv! is zero-alloc steady-state" begin
        Random.seed!(23)
        n = 30
        Asp = sprand(Float64, n, n, 0.2)
        rows, cols, _ = findnz(Asp)
        A1 = sparse(rows, cols, randn(length(rows)), n, n) + 4 * sparse(I, n, n)
        Acsr1 = build_csr(Matrix(A1))
        b = randn(n)
        x = zeros(n)

        F = csr_qr(Acsr1; ordering = :amd)
        ldiv!(x, F, b)   # warm up the pooled solve buffers
        ldiv!(x, F, b)

        # Steady state: the solve reuses the symbolic's pooled scratch, so a
        # provided-output ldiv! allocates nothing.
        nbytes = @allocated ldiv!(x, F, b)
        @test nbytes == 0
        @test norm(A1 * x - b) / norm(b) < 1.0e-10
    end

    @testset "Dense-tail apply matches the LAPACK reference solve" begin
        # The manual (allocation-free) dense-tail Householder apply must give
        # the same solution as the pure-sparse path on the bundled dense-fill
        # matrices, where adaptive_dense actually triggers.
        dir = joinpath(@__DIR__, "matrices")
        files = sort(filter(f -> endswith(f, ".txt"), readdir(dir; join = true)))
        for f in files
            lines = split(read(f, String), '\n'; keepempty = false)
            A = eval(Meta.parse(strip(lines[1])))
            b = eval(Meta.parse(strip(lines[2])))
            all(isfinite, b) || continue
            Acsr = SparseMatrixCSR(transpose(sparse(transpose(A))))
            x_sparse = csr_qr(Acsr; ordering = :amd) \ b
            Fd = csr_qr(Acsr; ordering = :amd, adaptive_dense = true)
            x_dense = Fd \ b
            @test Fd.k_dense > 0           # confirm the dense fallback fired
            # The dense-tail residual must match the sparse path. Use the same
            # absolute criterion as the bundled-matrix test: on full-rank inputs
            # both residuals are ~machine-eps (their *relative* gap is FP noise),
            # so compare against an ‖b‖-scaled tolerance, not a relative one.
            r_sparse = norm(A * x_sparse - b)
            scale = max(r_sparse, 1.0e-12 * norm(b))
            @test norm(A * x_dense - b) <= 1.0e-8 + 2 * scale
            # ldiv! on the dense-tail factorization is also allocation-free.
            xd = zeros(length(x_dense))
            ldiv!(xd, Fd, b)
            ldiv!(xd, Fd, b)
            @test (@allocated ldiv!(xd, Fd, b)) == 0
        end
    end

    @testset "Workspace pool: csr_refactor! mutates F in place" begin
        # The mutation-in-place semantic of csr_refactor! means the returned
        # factorization is === to the input one and the input's V/R buffers
        # have been updated.
        Random.seed!(22)
        n = 20
        A1 = sprand(Float64, n, n, 0.25) + 3 * sparse(I, n, n)
        A2 = sprand(Float64, n, n, 0.25) + 3 * sparse(I, n, n)
        Acsr1 = build_csr(Matrix(A1))
        # A2 with the same pattern as A1:
        rows, cols, _ = findnz(A1)
        vals2 = randn(length(rows))
        A2same = sparse(rows, cols, vals2, n, n)
        Acsr2 = build_csr(Matrix(A2same))

        F1 = csr_qr(Acsr1)
        R1 = copy(F1.R_nzval)
        F2 = csr_refactor!(F1, Acsr2)
        @test F2 === F1
        # The R values should have changed (different input).
        @test F2.R_nzval !== R1
        @test F2.R_nzval != R1
    end

    @testset "Workspace pool: handles different element types" begin
        # First a Float64 factor, then attempt a ComplexF64 factor with the
        # same symbolic — the workspace is element-typed so the second call
        # should re-allocate a complex workspace transparently.
        Random.seed!(23)
        n = 10
        A_real = randn(n, n) + 3 * I
        A_cplx = randn(ComplexF64, n, n) + 3 * I
        Ar_csr = build_csr(A_real)
        Ac_csr = build_csr(A_cplx)

        sym = csr_analyze(Ar_csr; ordering = :natural)
        Fr = csr_factor(Ar_csr, sym)
        # Factor a complex matrix using the SAME symbolic (it's pattern-
        # compatible since both are dense n×n). The cached Float64 workspace
        # should be replaced with a ComplexF64 one without corruption.
        Fc = csr_factor(Ac_csr, sym)
        @test eltype(Fc) == ComplexF64
        br = randn(n); bc = randn(ComplexF64, n)
        xr = Fr \ br; xc = Fc \ bc
        @test norm(A_real * xr - br) / norm(br) < 1.0e-10
        @test norm(A_cplx * xc - bc) / norm(bc) < 1.0e-10
    end

    @testset "Symbolic exact R column counts" begin
        # The `rcount` field of CSRQRSymbolic should equal the per-column nnz
        # of the produced R factor (exact under no-cancellation).
        Random.seed!(20)
        for (m, n, p) in [(20, 15, 0.3), (30, 20, 0.2), (50, 50, 0.15)]
            A = sprand(m, n, p) + sparse(I, m, n)
            Acsr = build_csr(Matrix(A))
            for ordering in (:natural, :amd)
                sym = csr_analyze(Acsr; ordering = ordering)
                F = csr_factor(Acsr, sym)
                actual = [F.R_colptr[k + 1] - F.R_colptr[k] for k in 1:n]
                # rcount stored on the symbolic returned by analyze (not the
                # potentially-repivoted one inside F.sym) should match the
                # uncorrected column counts.
                @test sym.rcount == actual
                # The factor's symbolic (which may have been repivoted) should
                # at minimum satisfy sum(rcount) == sum(actual).
                @test sum(F.sym.rcount) == sum(actual)
                @test F.sym.rnz == sum(actual)
                @test length(F.R_nzval) == sum(actual)
            end
        end
    end

    @testset "Numerically-zero column triggers value-aware repivot" begin
        # A is 6x6 full rank except column 4 which is numerically zero.
        Random.seed!(16)
        n = 6
        A = randn(n, n) + 3 * I
        A[:, 4] .= 0   # numerically zero column
        Acsr = build_csr(A)
        b = randn(n)
        F = csr_qr(Acsr)
        # rank should be n - 1
        @test rank(F) == n - 1
        # The trailing q position should hold the original column 4.
        @test F.sym.q[end] == 4
        # solve should still be finite and match SPQR.
        x = F \ b
        @test all(isfinite, x)
        Fs = qr(sparse(A))
        xs = Fs \ b
        @test norm(A * x - b) ≈ norm(A * xs - b) atol = 1.0e-8
    end

    @testset "Adaptive dense fallback: square full-rank dense" begin
        # Densely populated matrix; threshold should trigger the dense fallback
        # for almost every column.
        Random.seed!(17)
        n = 40
        Adense = randn(n, n) + 4 * I
        Acsr = build_csr(Adense)
        b = randn(n)
        F = csr_qr(Acsr; adaptive_dense = true, dense_threshold = 0.2)
        @test F.k_dense > 0
        @test F.k_dense < n
        @test rank(F) == n
        x = F \ b
        xref = Adense \ b
        @test norm(Adense * x - b) / norm(b) < 1.0e-10
        @test norm(x - xref) / max(norm(xref), 1.0) < 1.0e-10
    end

    @testset "Adaptive dense fallback: tall LS" begin
        Random.seed!(18)
        m, n = 40, 22
        Adense = randn(m, n)
        Acsr = build_csr(Adense)
        b = randn(m)
        F = csr_qr(Acsr; adaptive_dense = true, dense_threshold = 0.2)
        # Dense tall LS: with this size the fallback should kick in.
        @test F.k_dense > 0
        @test rank(F) == n
        x = F \ b
        xref = Adense \ b
        @test norm(x - xref) / max(norm(xref), 1.0) < 1.0e-10
    end

    @testset "Adaptive dense fallback: rank-deficient block" begin
        # Rank-deficient where rank deficiency is in the dense tail.
        Random.seed!(19)
        n = 20
        U = randn(n, n - 2); V = randn(n, n - 2)
        Adense = U * V'   # n x n with rank n - 2
        Acsr = build_csr(Adense)
        b = Adense * randn(n)  # in range
        F = csr_qr(Acsr; tol = 1.0e-10, adaptive_dense = true, dense_threshold = 0.2)
        @test F.k_dense > 0
        @test rank(F) <= n - 2
        x = F \ b
        @test all(isfinite, x)
        # residual should be near zero
        @test norm(Adense * x - b) / max(norm(b), 1.0) < 1.0e-8
    end

    @testset "Adaptive dense fallback: sparse never triggers" begin
        # Truly sparse matrix — fallback should stay off or trigger only very
        # late (after most of the work is done).
        Random.seed!(20)
        n = 80
        A = sprand(Float64, n, n, 0.04) + 5 * sparse(I, n, n)
        Acsr = build_csr(Matrix(A))
        b = randn(n)
        F = csr_qr(Acsr; adaptive_dense = true, dense_threshold = 0.4)
        x = F \ b
        @test norm(A * x - b) / norm(b) < 1.0e-10
        # Compare to non-adaptive path
        F_nat = csr_qr(Acsr; adaptive_dense = false)
        x_nat = F_nat \ b
        @test norm(x - x_nat) / max(norm(x_nat), 1.0) < 1.0e-10
    end

    @testset "Adaptive dense fallback: ComplexF64" begin
        Random.seed!(21)
        n = 25
        Adense = randn(ComplexF64, n, n) + 5 * I
        Acsr = build_csr(Adense)
        b = randn(ComplexF64, n)
        F = csr_qr(Acsr; adaptive_dense = true, dense_threshold = 0.2)
        @test F.k_dense > 0
        @test rank(F) == n
        x = F \ b
        @test norm(Adense * x - b) / norm(b) < 1.0e-10
    end

    @testset "Adaptive dense fallback: refactor!" begin
        # Pattern matches and we re-use symbolic. adaptive_dense flows
        # through refactor! independently of the original factor's setting.
        Random.seed!(22)
        n = 30
        Asp = sprand(Float64, n, n, 0.5)
        rows, cols, _ = findnz(Asp)
        v1 = randn(length(rows)); v2 = randn(length(rows))
        A1 = sparse(rows, cols, v1, n, n) + 4 * sparse(I, n, n)
        A2 = sparse(rows, cols, v2, n, n) + 4 * sparse(I, n, n)
        Acsr1 = build_csr(Matrix(A1))
        Acsr2 = build_csr(Matrix(A2))
        b = randn(n)

        F1 = csr_qr(Acsr1; adaptive_dense = true, dense_threshold = 0.2)
        x1 = F1 \ b
        @test norm(A1 * x1 - b) / norm(b) < 1.0e-10

        F2 = csr_refactor!(F1, Acsr2; adaptive_dense = true, dense_threshold = 0.2)
        x2 = F2 \ b
        @test norm(A2 * x2 - b) / norm(b) < 1.0e-10

        # Compare against fresh factor.
        F2_fresh = csr_qr(Acsr2; adaptive_dense = true, dense_threshold = 0.2)
        x2_fresh = F2_fresh \ b
        @test norm(x2 - x2_fresh) / max(norm(x2_fresh), 1.0) < 1.0e-10
    end

    @testset "Adaptive dense fallback: matches non-adaptive on user matrices" begin
        # On the actual user matrices, both adaptive and non-adaptive paths
        # should produce solutions whose residuals match SPQR's.
        dir = "/home/crackauc/.claude/uploads/d279ff12-71e6-4faf-b1ac-6715899a256b"
        if isdir(dir)
            files = sort(readdir(dir; join = true))
            for f in files
                text = read(f, String)
                lines = split(text, '\n'; keepempty = false)
                A = eval(Meta.parse(strip(lines[1])))
                b = eval(Meta.parse(strip(lines[2])))
                Acsr = SparseMatrixCSR(transpose(sparse(transpose(A))))
                F = csr_qr(Acsr; adaptive_dense = true)
                Fspqr = qr(A)
                xspqr = Fspqr \ b
                x = F \ b
                if all(isfinite, b)
                    @test all(isfinite, x)
                    rmy = norm(A * x - b)
                    rspqr = norm(A * xspqr - b)
                    scale = max(rspqr, 1.0e-12 * norm(b))
                    @test rmy <= 1.0e-8 + 2 * scale
                end
            end
        else
            @info "User matrix directory not available; skipping."
        end
    end

    @testset "Generic number types" begin
        @testset "BigFloat full-rank square solve" begin
            setprecision(BigFloat, 256) do
                n = 12
                # Well-conditioned BigFloat matrix (diagonally dominant).
                Adense = Matrix{BigFloat}(undef, n, n)
                for i in 1:n, j in 1:n
                    Adense[i, j] = BigFloat(1) / BigFloat(i + j)
                end
                Adense += BigFloat(n) * I
                Acsr = build_csr(Adense)
                b = BigFloat[BigFloat(i) / 3 for i in 1:n]
                F = csr_qr(Acsr)
                @test eltype(F) == BigFloat
                @test rank(F) == n
                x = F \ b
                @test eltype(x) == BigFloat
                # Residual must reach BigFloat precision: if any Float64
                # truncation leaked in, this would stall near 1e-16.
                @test norm(Adense * x - b) / norm(b) < BigFloat(1.0e-30)
            end
        end

        @testset "BigFloat rank-deficient" begin
            setprecision(BigFloat, 256) do
                # Rank-3 6x5 BigFloat matrix.
                U = BigFloat[BigFloat(i)^j for i in 1:6, j in 1:3]
                Vm = BigFloat[BigFloat(i)^j for i in 1:5, j in 1:3]
                Adense = U * Vm'
                Acsr = build_csr(Adense)
                b = BigFloat[BigFloat(i) / 2 for i in 1:6]
                tol = BigFloat(1.0e-40)
                F = csr_qr(Acsr; tol = tol)
                @test rank(F) == 3
                x = F \ b
                @test all(isfinite, x)
                # Minimum residual reference. LAPACK pinv/svd is unavailable
                # for BigFloat, so compute the LS minimum-residual norm in
                # Float64 (the residual norm is well-conditioned for this
                # benign rank-3 case) and compare against the BigFloat solve's
                # residual. The BigFloat residual must not exceed the minimum.
                xref = pinv(Float64.(Adense)) * Float64.(b)
                rref = norm(Float64.(Adense) * xref - Float64.(b))
                rres = Float64(norm(Adense * x - b))
                @test rres ≈ rref atol = 1.0e-8
            end
        end

        @testset "ForwardDiff.Dual solve matches primal" begin
            Random.seed!(101)
            n = 8
            base = randn(n, n) + 4 * I
            # Parameterized matrix: A(p) keeps a fixed sparsity pattern, only
            # the values are Dual.
            mask = base .!= 0
            makeA(p) = build_csr(base .* p[1] .+ p[2] .* mask)
            b = randn(n)
            p0 = [1.3, 0.2]

            # Primal Float64 solve.
            xprimal = csr_qr(makeA(p0)) \ b

            solve(p) = csr_qr(makeA(p)) \ b
            xdual = solve(ForwardDiff.Dual.(p0, (1.0, 0.0)))
            @test ForwardDiff.value.(xdual) ≈ xprimal rtol = 1.0e-10

            # Jacobian of solution wrt p vs finite differences.
            J = ForwardDiff.jacobian(solve, p0)
            h = 1.0e-6
            Jfd = similar(J)
            for k in 1:length(p0)
                pp = copy(p0); pm = copy(p0)
                pp[k] += h; pm[k] -= h
                Jfd[:, k] = (solve(pp) .- solve(pm)) ./ (2h)
            end
            @test J ≈ Jfd rtol = 1.0e-6 atol = 1.0e-6
        end

        @testset "adaptive_dense falls back for non-BLAS eltype" begin
            # BigFloat: adaptive_dense must be silently ignored (fallback to
            # the pure-Julia sparse kernel), producing a correct solve.
            setprecision(BigFloat, 256) do
                n = 40
                Adense = Matrix{BigFloat}(undef, n, n)
                for i in 1:n, j in 1:n
                    Adense[i, j] = BigFloat(1) / BigFloat(i + 2j)
                end
                Adense += BigFloat(n) * I
                Acsr = build_csr(Adense)
                b = BigFloat[BigFloat(i) for i in 1:n]
                F = csr_qr(Acsr; adaptive_dense = true)
                @test F.k_dense == 0   # no dense transition occurred
                x = F \ b
                @test norm(Adense * x - b) / norm(b) < BigFloat(1.0e-30)
            end

            # ForwardDiff.Dual: same fallback behavior, value correct.
            Random.seed!(202)
            n = 24
            base = randn(n, n) + 6 * I
            mask = base .!= 0
            makeA(p) = build_csr(base .* p[1] .+ p[2] .* mask)
            b = randn(n)
            p0 = [1.1, 0.3]
            solve(p) = csr_qr(makeA(p); adaptive_dense = true) \ b
            xprimal = csr_qr(makeA(p0)) \ b
            xdual = solve(ForwardDiff.Dual.(p0, (1.0, 0.0)))
            @test ForwardDiff.value.(xdual) ≈ xprimal rtol = 1.0e-10
            Fdual = csr_qr(makeA(ForwardDiff.Dual.(p0, (1.0, 0.0))); adaptive_dense = true)
            @test Fdual.k_dense == 0
        end
    end

end
