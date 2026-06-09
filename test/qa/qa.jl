using SparseColumnPivotedQR
using Aqua
using JET
using Test

@testset "Quality Assurance" begin
    @testset "Aqua" begin
        Aqua.test_all(SparseColumnPivotedQR)
    end
    @testset "JET" begin
        JET.test_package(SparseColumnPivotedQR; target_defined_modules = true)
    end
end
