# Benchmark SparseColumnPivotedQR vs SPQR vs CXSparse vs dense LAPACK column-pivoted QR.
#
# Usage from the SparseColumnPivotedQR.jl directory:
#   julia --project=bench bench/bench.jl
#
# The bench environment dev-loads this package (..) and CXSparse from a sibling
# CXSparse.jl directory if available.

using Pkg
Pkg.activate(@__DIR__)
let
    parent_pkg = abspath(joinpath(@__DIR__, ".."))
    cxsparse_candidates = [
        abspath(joinpath(@__DIR__, "..", "..", "CXSparse.jl")),
        abspath(joinpath(@__DIR__, "..", "..", "..", "..", "..", "CXSparse.jl")),
        abspath(joinpath(@__DIR__, "..", "..", "..", "..", "..", "..", "CXSparse.jl")),
    ]
    # Dev-load the parent package first so resolution sees it as a path source.
    Pkg.develop(path = parent_pkg)
    cxsparse_found = false
    for cx in cxsparse_candidates
        if isdir(cx)
            try
                Pkg.develop(path = cx)
                cxsparse_found = true
            catch e
                @warn "Failed to dev CXSparse at $cx" exception = e
            end
            break
        end
    end
    if !cxsparse_found
        # Drop CXSparse from the Project.toml so resolution can proceed
        # without it (e.g. when CXSparse.jl isn't checked out alongside).
        try
            Pkg.rm("CXSparse")
        catch
        end
    end
    Pkg.add(["AMD", "BenchmarkTools", "SparseMatricesCSR"])
end

using LinearAlgebra, SparseArrays, SparseMatricesCSR, BenchmarkTools, Printf
using SparseColumnPivotedQR
using AMD  # trigger AMD extension

cxsparse_ok = try
    @eval using CXSparse
    true
catch
    false
end

const BUNDLED = abspath(joinpath(@__DIR__, "..", "test", "matrices"))
const UPLOADS = "/home/crackauc/.claude/uploads/d279ff12-71e6-4faf-b1ac-6715899a256b"
files = if isdir(BUNDLED)
    sort(filter(f -> endswith(f, ".txt"), readdir(BUNDLED; join = true)))
elseif isdir(UPLOADS)
    sort(readdir(UPLOADS; join = true))
else
    String[]
end

function load_case(f)
    text = read(f, String)
    lines = split(text, "\n"; keepempty = false)
    A = eval(Meta.parse(strip(lines[1])))
    b = eval(Meta.parse(strip(lines[2])))
    return A, b
end

function fmt_row(file, solver, t_us, res, nn)
    return @printf(
        "%-30s %-22s %10s   %12s   %6d\n",
        file, solver,
        isnan(t_us) ? "-" : @sprintf("%9.1f", t_us),
        isnan(res) ? "NaN" : @sprintf("%.3e", res),
        nn
    )
end

println()
println("="^95)
println("Benchmarks on user matrices (199x199, nnz≈979); minimum time of @benchmark seconds=1")
println("="^95)
@printf("%-30s %-22s %10s   %12s   %6s\n", "file", "solver", "time (μs)", "||Ax-b||", "nnan(x)")
println("-"^95)

for f in files
    A, b = load_case(f)
    Acsr = SparseMatrixCSR(transpose(sparse(transpose(A))))
    Adense = Matrix(A)
    short = first(basename(f), 28)

    # 0) SparseColumnPivotedQR — default ordering (:amd when AMD loaded,
    # which it is here). This is what the common convenience entry point
    # `csr_qr(A)` does without an explicit ordering=.
    F = csr_qr(Acsr); x = F \ b
    t = @benchmark begin
        F2 = csr_qr($Acsr); $x .= F2 \ $b
    end seconds = 1
    res = all(isfinite, x) ? norm(A * x - b) : NaN
    nn = count(!isfinite, x)
    fmt_row(short, "CSR-QR default", minimum(t.times) / 1000, res, nn)

    # 1) SparseColumnPivotedQR — natural ordering (opt-in), one-shot csr_qr
    F = csr_qr(Acsr; ordering = :natural); x = F \ b
    t = @benchmark begin
        F2 = csr_qr($Acsr; ordering = :natural); $x .= F2 \ $b
    end seconds = 1
    res = all(isfinite, x) ? norm(A * x - b) : NaN
    nn = count(!isfinite, x)
    fmt_row(short, "CSR-QR natural", minimum(t.times) / 1000, res, nn)

    # 2) SparseColumnPivotedQR — AMD ordering, one-shot csr_qr
    F = csr_qr(Acsr; ordering = :amd); x = F \ b
    t = @benchmark begin
        F2 = csr_qr($Acsr; ordering = :amd); $x .= F2 \ $b
    end seconds = 1
    res = all(isfinite, x) ? norm(A * x - b) : NaN
    nn = count(!isfinite, x)
    fmt_row(short, "CSR-QR amd", minimum(t.times) / 1000, res, nn)

    # 2b) SparseColumnPivotedQR — adaptive ordering, one-shot csr_qr
    F = csr_qr(Acsr; ordering = :adaptive); x = F \ b
    t = @benchmark begin
        F2 = csr_qr($Acsr; ordering = :adaptive); $x .= F2 \ $b
    end seconds = 1
    res = all(isfinite, x) ? norm(A * x - b) : NaN
    nn = count(!isfinite, x)
    fmt_row(short, "CSR-QR adaptive", minimum(t.times) / 1000, res, nn)

    # 3) SparseColumnPivotedQR — refactor! reusing natural symbolic
    sym = csr_analyze(Acsr; ordering = :natural)
    F0 = csr_factor(Acsr, sym); x = F0 \ b
    t = @benchmark begin
        F2 = csr_refactor!($F0, $Acsr); $x .= F2 \ $b
    end seconds = 1
    res = all(isfinite, x) ? norm(A * x - b) : NaN
    nn = count(!isfinite, x)
    fmt_row(short, "CSR-QR refactor! (nat)", minimum(t.times) / 1000, res, nn)

    # 3b) SparseColumnPivotedQR — refactor! reusing AMD symbolic. This is
    # the apples-to-apples comparison to CXSparse cs_qr: AMD ordering up front
    # and only the numeric phase running per solve call.
    sym_amd = csr_analyze(Acsr; ordering = :amd)
    F0_amd = csr_factor(Acsr, sym_amd); x = F0_amd \ b
    t = @benchmark begin
        F2 = csr_refactor!($F0_amd, $Acsr); $x .= F2 \ $b
    end seconds = 1
    res = all(isfinite, x) ? norm(A * x - b) : NaN
    nn = count(!isfinite, x)
    fmt_row(short, "CSR-QR refactor! (amd)", minimum(t.times) / 1000, res, nn)

    # 3c) SparseColumnPivotedQR — adaptive dense fallback on top of AMD.
    # Probes whether materializing the late-fill block + LAPACK geqp3 is
    # cheaper than continuing in sparse Householder land.
    sym_amd_ad = csr_analyze(Acsr; ordering = :amd)
    F0_ad = csr_factor(Acsr, sym_amd_ad; adaptive_dense = true); x = F0_ad \ b
    t = @benchmark begin
        F2 = csr_refactor!($F0_ad, $Acsr; adaptive_dense = true); $x .= F2 \ $b
    end seconds = 1
    res = all(isfinite, x) ? norm(A * x - b) : NaN
    nn = count(!isfinite, x)
    fmt_row(short, "CSR-QR refactor! (amd+ad)", minimum(t.times) / 1000, res, nn)

    # 3c) SparseColumnPivotedQR — refactor! with compact-WY blocking on the
    # AMD symbolic. block_size = 1 is the default (matches row 3b); we sweep
    # a few values to show where the block path stands. On the user matrices
    # the block path is not a win (the T-build + sort overhead exceeds the
    # per-Householder loop-overhead savings at this size/density); on larger
    # / denser inputs the two paths trend toward parity.
    for bs in (2, 4, 8, 16)
        F0_wy = csr_factor(Acsr, sym_amd; block_size = bs); x = F0_wy \ b
        t = @benchmark begin
            F2 = csr_refactor!($F0_wy, $Acsr; block_size = $bs); $x .= F2 \ $b
        end seconds = 1
        res = all(isfinite, x) ? norm(A * x - b) : NaN
        nn = count(!isfinite, x)
        fmt_row(short, "CSR-QR refactor! (amd, WY bs=$bs)", minimum(t.times) / 1000, res, nn)
    end

    # 4) SuiteSparseQR (SPQR) via qr(::SparseMatrixCSC)
    Fs = qr(A); xs = Fs \ b
    t = @benchmark begin
        F2 = qr($A); $xs .= F2 \ $b
    end seconds = 1
    res = all(isfinite, xs) ? norm(A * xs - b) : NaN
    nn = count(!isfinite, xs)
    fmt_row(short, "SPQR", minimum(t.times) / 1000, res, nn)

    # 5) CXSparse cs_qr (if available)
    if cxsparse_ok
        try
            Fcx = CXSparse.cs_qr(A); xcx = Fcx \ b
            t = @benchmark begin
                F2 = CXSparse.cs_qr($A); $xcx .= F2 \ $b
            end seconds = 1
            res = all(isfinite, xcx) ? norm(A * xcx - b) : NaN
            nn = count(!isfinite, xcx)
            fmt_row(short, "CXSparse cs_qr", minimum(t.times) / 1000, res, nn)
        catch e
            println("    CXSparse cs_qr failed: ", e)
        end
    end

    # 6) Dense LAPACK column-pivoted QR
    Fd = qr(Adense, ColumnNorm()); xd = Fd \ b
    t = @benchmark begin
        F2 = qr($Adense, ColumnNorm()); $xd .= F2 \ $b
    end seconds = 1
    res = all(isfinite, xd) ? norm(A * xd - b) : NaN
    nn = count(!isfinite, xd)
    fmt_row(short, "LAPACK xgeqp3", minimum(t.times) / 1000, res, nn)

    println()
end
