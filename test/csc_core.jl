# CSC-native core tests. Run in a SEPARATE process that does NOT load
# `SparseMatricesCSR`, so this proves the `SparseMatrixCSC` API works as the
# native path with the CSR extension absent. (Driven from `runtests.jl`.)
using Test
using LinearAlgebra
using SparseArrays
using Random
using SparseColumnPivotedQR
using AMD  # AMD extension is independent of the CSR extension

@assert Base.get_extension(
    SparseColumnPivotedQR, :SparseColumnPivotedQRSparseMatricesCSRExt
) === nothing "SparseMatricesCSR extension must NOT be loaded in the CSC-core test process"

@testset "CSC-native core (no SparseMatricesCSR loaded)" begin
    for T in (Float64, ComplexF64)
        cv(M) = T <: Complex ? (T.(M) .+ T(0.3im) .* (M .!= 0)) : T.(M)

        @testset "square full-rank ($T)" begin
            Random.seed!(1)
            n = 40
            base = sprand(Float64, n, n, 0.2) + 5I
            A = convert(SparseMatrixCSC{T, Int}, cv(Matrix(base)))
            b = ones(T, n)
            F = csr_qr(A)
            x = F \ b
            @test norm(A * x - b) / norm(b) < 1.0e-9
            @test rank(F) == n
        end

        @testset "tall overdetermined least-squares ($T)" begin
            Random.seed!(2)
            m, n = 60, 25
            base = sprand(Float64, m, n, 0.3)
            base = base + sparse(1:n, 1:n, ones(n), m, n)
            A = convert(SparseMatrixCSC{T, Int}, cv(Matrix(base)))
            b = randn(T, m)
            F = csr_qr(A)
            x = F \ b
            # Full-column-rank tall LS: the dense `\` is the unique LS solution.
            @test x ≈ Matrix(A) \ b rtol = 1.0e-8
            @test rank(F) == n
        end

        @testset "wide underdetermined ($T)" begin
            Random.seed!(3)
            m, n = 20, 45
            base = sprand(Float64, m, n, 0.3)
            base = base + sparse(1:m, 1:m, ones(m), m, n)
            A = convert(SparseMatrixCSC{T, Int}, cv(Matrix(base)))
            b = randn(T, m)
            F = csr_qr(A)
            x = F \ b
            @test norm(A * x - b) / norm(b) < 1.0e-8   # consistent system
            @test rank(F) == m
        end

        @testset "rank-deficient ($T)" begin
            Random.seed!(4)
            m, n = 40, 30
            base = sprand(Float64, m, n, 0.25)
            base = base + sparse(1:n, 1:n, ones(n), m, n)
            M = Matrix(base)
            M[:, n] = M[:, 1]          # duplicate column -> rank n-1
            A = convert(SparseMatrixCSC{T, Int}, cv(M))
            b = randn(T, m)
            F = csr_qr(A)
            @test rank(F) == n - 1
            x = F \ b
            # Minimum-residual solve: the residual must match the true
            # least-squares residual (computed via the dense pseudoinverse,
            # which is well-defined for a rank-deficient overdetermined system).
            r_csc = norm(A * x - b)
            r_min = norm(A * (pinv(Matrix(A)) * b) - b)
            @test r_csc ≈ r_min rtol = 1.0e-6
        end

        @testset "csr_refactor! reuse path ($T)" begin
            Random.seed!(5)
            n = 30
            sp = sprand(Float64, n, n, 0.2)
            rows, cols, _ = findnz(sp)
            mk(v) = convert(
                SparseMatrixCSC{T, Int},
                cv(Matrix(sparse(rows, cols, v, n, n) + 4I)),
            )
            A1 = mk(randn(length(rows)))
            A2 = mk(randn(length(rows)))
            b = randn(T, n)
            F = csr_qr(A1; ordering = :amd)
            csr_refactor!(F, A2)
            x2 = F \ b
            @test norm(A2 * x2 - b) / norm(b) < 1.0e-9
            csr_refactor!(F, A1)
            x1 = F \ b
            @test norm(A1 * x1 - b) / norm(b) < 1.0e-9
        end

        @testset "csr_refactor! zero-alloc steady state ($T)" begin
            Random.seed!(6)
            n = 30
            sp = sprand(Float64, n, n, 0.2)
            rows, cols, _ = findnz(sp)
            mk(v) = convert(
                SparseMatrixCSC{T, Int},
                cv(Matrix(sparse(rows, cols, v, n, n) + 4I)),
            )
            A1 = mk(randn(length(rows)))
            A2 = mk(randn(length(rows)))
            F = csr_qr(A1; ordering = :amd)
            csr_refactor!(F, A2)   # warm
            csr_refactor!(F, A1)
            @test (@allocated csr_refactor!(F, A2)) == 0
            @test (@allocated csr_refactor!(F, A1)) == 0
        end
    end
end
