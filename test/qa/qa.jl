using SciMLTesting, SparseColumnPivotedQR, Test

# ExplicitImports only sees an extension module once its trigger package is
# loaded (`Base.get_extension` returns `nothing` otherwise), so load every
# weakdep here to bring `SparseColumnPivotedQRAMDExt` and
# `SparseColumnPivotedQRSparseMatricesCSRExt` under QA.
using AMD
using SparseMatricesCSR

# ExplicitImports silently skips an extension that fails to load, so assert the
# extension modules actually exist rather than trusting a green run_qa.
@testset "Extensions loaded" begin
    @test Base.get_extension(SparseColumnPivotedQR, :SparseColumnPivotedQRAMDExt) !== nothing
    @test Base.get_extension(
        SparseColumnPivotedQR, :SparseColumnPivotedQRSparseMatricesCSRExt
    ) !== nothing
end

run_qa(
    SparseColumnPivotedQR;
    ei_kwargs = (;
        all_qualified_accesses_are_public = (;
            ignore = (
                # The extension's whole job is to fill in the host package's
                # AMD hooks. `Base.moduleroot` of an extension is the extension
                # itself, so ExplicitImports cannot see these as same-package
                # internal accesses and flags them.
                :_AMD_EXT_LOADED, :_amd_colperm,
            ),
        ),
    )
)
