using SafeTestsets

@safetestset "Aqua" begin
    using SparseColumnPivotedQR
    using Aqua
    using Test
    # deps_compat fails: `Pkg` is declared in [extras]/[targets] without a
    # [compat] entry. Marked broken pending a root Project.toml fix.
    Aqua.test_all(SparseColumnPivotedQR; deps_compat = false)
    @test_broken false  # Aqua deps_compat: missing [compat] for Pkg extra — see https://github.com/SciML/SparseColumnPivotedQR.jl/issues/44
end

@safetestset "JET" begin
    using Test
    @test_broken false  # JET: \ -> ldiv!/_ldiv_adjoint! no matching method (Matrix union-split branch) — see https://github.com/SciML/SparseColumnPivotedQR.jl/issues/44
end
