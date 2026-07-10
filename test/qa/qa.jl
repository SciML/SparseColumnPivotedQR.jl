using SciMLTesting, SparseColumnPivotedQR, Test
using JET

function _public_api_names()
    public_names = Set(names(SparseColumnPivotedQR; all = false, imported = false))
    delete!(public_names, :SparseColumnPivotedQR)
    return public_names
end

function _has_source_docstring(name::Symbol)
    doc = get(Docs.meta(SparseColumnPivotedQR), Docs.Binding(SparseColumnPivotedQR, name), nothing)
    doc === nothing && return false
    return !isempty(strip(sprint(show, MIME("text/plain"), doc)))
end

function _documented_api_names()
    docs_src = normpath(joinpath(@__DIR__, "..", "..", "docs", "src"))
    documented = Set{Symbol}()
    for path in sort(readdir(docs_src; join = true))
        endswith(path, ".md") || continue
        text = read(path, String)
        for block in eachmatch(r"(?s)```@docs\s+(.*?)```", text)
            for line in eachsplit(block.captures[1], '\n')
                entry = strip(replace(String(line), r"#.*$" => ""))
                isempty(entry) && continue
                entry = replace(entry, "SparseColumnPivotedQR." => "")
                m = match(r"^([A-Za-z_][A-Za-z_0-9!]*)", entry)
                m === nothing && continue
                push!(documented, Symbol(m.captures[1]))
            end
        end
    end
    return documented
end

@testset "public API documentation coverage" begin
    public_names = _public_api_names()
    @test sort(String.(public_names)) == [
        "SparseColumnPivotedQRFactorization",
        "SparseColumnPivotedQRSymbolic",
        "has_amd_extension",
        "scpqr",
        "scpqr_analyze",
        "scpqr_factor",
        "scpqr_refactor!",
    ]

    missing_docstrings = sort(collect(String(name) for name in public_names if !_has_source_docstring(name)))
    @test missing_docstrings == String[]

    documented_names = _documented_api_names()
    missing_docs_entries = sort(collect(String(name) for name in setdiff(public_names, documented_names)))
    @test missing_docs_entries == String[]
end

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
