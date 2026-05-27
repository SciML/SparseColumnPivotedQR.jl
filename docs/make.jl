using Documenter, SparseColumnPivotedQR

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml", force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force = true)

include("pages.jl")

makedocs(
    sitename = "SparseColumnPivotedQR.jl",
    authors = "Chris Rackauckas",
    modules = [SparseColumnPivotedQR],
    clean = true, doctest = false, linkcheck = true,
    format = Documenter.HTML(
        assets = ["assets/favicon.ico"],
        canonical = "https://docs.sciml.ai/SparseColumnPivotedQR/stable/"
    ),
    pages = pages
)

deploydocs(
    repo = "github.com/SciML/SparseColumnPivotedQR.jl.git";
    push_preview = true
)
