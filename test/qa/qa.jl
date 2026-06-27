using SciMLTesting, SparseColumnPivotedQR, Test
using JET

run_qa(
    SparseColumnPivotedQR;
    explicit_imports = true,
    # JET report_package surfaces 4 union-split `no matching method` errors on the
    # `\` solve paths (`x = zeros(T, F.n)` infers a `Matrix` branch with no
    # `ldiv!`/`_ldiv_adjoint!` match) — tracked in
    # https://github.com/SciML/SparseColumnPivotedQR.jl/issues/44
    jet_broken = true,
    jet_kwargs = (; target_defined_modules = true),
    ei_kwargs = (
        # libblastrampoline is re-exported through LinearAlgebra.BLAS; its owner is
        # the libblastrampoline_jll stdlib.
        all_qualified_accesses_via_owners = (; ignore = (:libblastrampoline,)),
        # stdlib non-public names used by the allocation-free LAPACK geqp3 bindings:
        # @blasfunc/BlasFloat/BlasInt/chklapackerror/libblastrampoline (LinearAlgebra)
        # and RefValue (Base).
        all_qualified_accesses_are_public = (;
            ignore = (
                Symbol("@blasfunc"), :BlasFloat, :BlasInt,
                :RefValue, :chklapackerror, :libblastrampoline,
            ),
        ),
        # getcolptr is not public in the SparseArrays stdlib.
        all_explicit_imports_are_public = (; ignore = (:getcolptr,)),
    ),
)
