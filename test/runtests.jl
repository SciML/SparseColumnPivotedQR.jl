using Test
using SafeTestsets

# CI dispatches test groups through the GROUP env var (see test/test_groups.toml
# and the grouped-tests.yml caller). "All" runs everything; "QA" runs only the
# Aqua/JET metadata checks from their isolated test/qa environment.
const GROUP = get(ENV, "GROUP", "All")

if GROUP == "QA"
    using Pkg
    Pkg.activate(joinpath(@__DIR__, "qa"))
    Pkg.instantiate()
    include(joinpath(@__DIR__, "qa", "qa.jl"))
    exit()
end

# Core solver/factorization checks. Each independent unit runs in its own module
# via `@safetestset` (the body just `include`s a self-contained file so its
# top-level `using`s resolve before any macro is reached).
@safetestset "SparseColumnPivotedQR" begin
    include("scpqr_core.jl")
end

# Dedicated CSC-native core checks (the `SparseMatrixCSC` API is the native,
# allocation-free path).
@safetestset "CSC-native core" begin
    include("csc_core.jl")
end
