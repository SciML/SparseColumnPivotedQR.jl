using SciMLTesting, SparseColumnPivotedQR

# ExplicitImports only sees an extension module once its trigger package is
# loaded (`Base.get_extension` returns `nothing` otherwise), so load every
# weakdep here to bring `SparseColumnPivotedQRAMDExt` under QA.
using AMD

run_qa(
    SparseColumnPivotedQR;
    ei_kwargs = (;
        all_qualified_accesses_are_public = (;
            ignore = (
                # AMD.jl exposes only the allocating high-level wrappers
                # (`colamd`, `amd`) as public API. The extension deliberately
                # drives the low-level colamd bindings so it can scatter the
                # CSR pattern straight into colamd's workspace with no
                # intermediate `SparseMatrixCSC`; these have no public spelling.
                :SS_Int, :colamd_l, :colamd_l_recommended, :colamd_set_defaults,
                # The extension's whole job is to fill in the host package's
                # AMD hooks. `Base.moduleroot` of an extension is the extension
                # itself, so ExplicitImports cannot see these as same-package
                # internal accesses and flags them.
                :_AMD_EXT_LOADED, :_amd_colperm,
            ),
        ),
    )
)
