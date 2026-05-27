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
    cxsparse_pkg = abspath(joinpath(@__DIR__, "..", "..", "CXSparse.jl"))
    try
        Pkg.develop(path=parent_pkg)
    catch
    end
    if isdir(cxsparse_pkg)
        try
            Pkg.develop(path=cxsparse_pkg)
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
    sort(filter(f -> endswith(f, ".txt"), readdir(BUNDLED; join=true)))
elseif isdir(UPLOADS)
    sort(readdir(UPLOADS; join=true))
else
    String[]
end

function load_case(f)
    text = read(f, String)
    lines = split(text, "\n"; keepempty=false)
    A = eval(Meta.parse(strip(lines[1])))
    b = eval(Meta.parse(strip(lines[2])))
    return A, b
end

function fmt_row(file, solver, t_us, res, nn)
    @printf("%-30s %-22s %10s   %12s   %6d\n",
            file, solver,
            isnan(t_us) ? "-" : @sprintf("%9.1f", t_us),
            isnan(res) ? "NaN" : @sprintf("%.3e", res),
            nn)
end

println()
println("=" ^ 95)
println("Benchmarks on user matrices (199x199, nnz≈979); minimum time of @benchmark seconds=1")
println("=" ^ 95)
@printf("%-30s %-22s %10s   %12s   %6s\n", "file", "solver", "time (μs)", "||Ax-b||", "nnan(x)")
println("-" ^ 95)

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
    end seconds=1
    res = all(isfinite, x) ? norm(A*x - b) : NaN
    nn = count(!isfinite, x)
    fmt_row(short, "CSR-QR default", minimum(t.times)/1000, res, nn)

    # 1) SparseColumnPivotedQR — natural ordering (opt-in), one-shot csr_qr
    F = csr_qr(Acsr; ordering=:natural); x = F \ b
    t = @benchmark begin
        F2 = csr_qr($Acsr; ordering=:natural); $x .= F2 \ $b
    end seconds=1
    res = all(isfinite, x) ? norm(A*x - b) : NaN
    nn = count(!isfinite, x)
    fmt_row(short, "CSR-QR natural", minimum(t.times)/1000, res, nn)

    # 2) SparseColumnPivotedQR — AMD ordering, one-shot csr_qr
    F = csr_qr(Acsr; ordering=:amd); x = F \ b
    t = @benchmark begin
        F2 = csr_qr($Acsr; ordering=:amd); $x .= F2 \ $b
    end seconds=1
    res = all(isfinite, x) ? norm(A*x - b) : NaN
    nn = count(!isfinite, x)
    fmt_row(short, "CSR-QR amd", minimum(t.times)/1000, res, nn)

    # 2b) SparseColumnPivotedQR — adaptive ordering, one-shot csr_qr
    F = csr_qr(Acsr; ordering=:adaptive); x = F \ b
    t = @benchmark begin
        F2 = csr_qr($Acsr; ordering=:adaptive); $x .= F2 \ $b
    end seconds=1
    res = all(isfinite, x) ? norm(A*x - b) : NaN
    nn = count(!isfinite, x)
    fmt_row(short, "CSR-QR adaptive", minimum(t.times)/1000, res, nn)

    # 3) SparseColumnPivotedQR — refactor! reusing natural symbolic
    sym = csr_analyze(Acsr; ordering=:natural)
    F0 = csr_factor(Acsr, sym); x = F0 \ b
    t = @benchmark begin
        F2 = csr_refactor!($F0, $Acsr); $x .= F2 \ $b
    end seconds=1
    res = all(isfinite, x) ? norm(A*x - b) : NaN
    nn = count(!isfinite, x)
    fmt_row(short, "CSR-QR refactor! (nat)", minimum(t.times)/1000, res, nn)

    # 3b) SparseColumnPivotedQR — refactor! reusing AMD symbolic. This is
    # the apples-to-apples comparison to CXSparse cs_qr: AMD ordering up front
    # and only the numeric phase running per solve call.
    sym_amd = csr_analyze(Acsr; ordering=:amd)
    F0_amd = csr_factor(Acsr, sym_amd); x = F0_amd \ b
    t = @benchmark begin
        F2 = csr_refactor!($F0_amd, $Acsr); $x .= F2 \ $b
    end seconds=1
    res = all(isfinite, x) ? norm(A*x - b) : NaN
    nn = count(!isfinite, x)
    fmt_row(short, "CSR-QR refactor! (amd)", minimum(t.times)/1000, res, nn)

    # 4) SuiteSparseQR (SPQR) via qr(::SparseMatrixCSC)
    Fs = qr(A); xs = Fs \ b
    t = @benchmark begin
        F2 = qr($A); $xs .= F2 \ $b
    end seconds=1
    res = all(isfinite, xs) ? norm(A*xs - b) : NaN
    nn = count(!isfinite, xs)
    fmt_row(short, "SPQR", minimum(t.times)/1000, res, nn)

    # 5) CXSparse cs_qr (if available)
    if cxsparse_ok
        try
            Fcx = CXSparse.cs_qr(A); xcx = Fcx \ b
            t = @benchmark begin
                F2 = CXSparse.cs_qr($A); $xcx .= F2 \ $b
            end seconds=1
            res = all(isfinite, xcx) ? norm(A*xcx - b) : NaN
            nn = count(!isfinite, xcx)
            fmt_row(short, "CXSparse cs_qr", minimum(t.times)/1000, res, nn)
        catch e
            println("    CXSparse cs_qr failed: ", e)
        end
    end

    # 6) Dense LAPACK column-pivoted QR
    Fd = qr(Adense, ColumnNorm()); xd = Fd \ b
    t = @benchmark begin
        F2 = qr($Adense, ColumnNorm()); $xd .= F2 \ $b
    end seconds=1
    res = all(isfinite, xd) ? norm(A*xd - b) : NaN
    nn = count(!isfinite, xd)
    fmt_row(short, "LAPACK xgeqp3", minimum(t.times)/1000, res, nn)

    println()
end
