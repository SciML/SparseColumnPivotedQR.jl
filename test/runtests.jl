using Test
using LinearAlgebra
using SparseArrays
using SparseMatricesCSR
using Random
using SparseColumnPivotedQR
using AMD  # trigger AMD extension

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
        @test sym_amd.colperm != 1:n
        # colamd ordering is currently aliased to amd
        @test sym_col.colperm == sym_amd.colperm

        F_nat = csr_factor(Acsr, sym_nat); x_nat = F_nat \ b
        F_amd = csr_factor(Acsr, sym_amd); x_amd = F_amd \ b

        @test rank(F_nat) == n
        @test rank(F_amd) == n
        @test norm(A * x_nat - b) / norm(b) < 1.0e-10
        @test norm(A * x_amd - b) / norm(b) < 1.0e-10

        # AMD should not blow up fill: total nnz(R) <= natural's by a reasonable factor
        nnz_nat = sum(length, F_nat.R_cols)
        nnz_amd = sum(length, F_amd.R_cols)
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

end
