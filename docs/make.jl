using Documenter, SparseColumnPivotedQR

include("pages.jl")

makedocs(
    sitename = "SparseColumnPivotedQR.jl",
    authors = "Chris Rackauckas",
    modules = [SparseColumnPivotedQR],
    clean = true, doctest = false, linkcheck = true,
    format = Documenter.HTML(
        canonical = "https://docs.sciml.ai/SparseColumnPivotedQR/stable/"
    ),
    pages = pages
)

deploydocs(
    repo = "github.com/SciML/SparseColumnPivotedQR.jl.git";
    push_preview = true
)
