dir = @__DIR__
files = [
    "moo_multitrip_cvrp_budapest.jl",
    "osrm_client.jl",
    "nsga2_moo_backend.jl",
    "moo_pareto_visualization.jl",
    "robustness_tests.jl",
    "moo_result_manager.jl",
]

println("=== Julia 문법 체크 ===\n")
all_ok = true
for f in files
    path = joinpath(dir, f)
    try
        src = read(path, String)
        Meta.parse("begin\n$src\nend")
        println("  ✅ OK: $f")
    catch e
        println("  ❌ 오류: $f")
        println("     → $e")
        all_ok = false
    end
end

println()
if all_ok
    println("✅ 모든 Julia 파일 문법 정상!")
else
    println("❌ 위 파일에 문법 오류 있음 — 수정 필요")
end
