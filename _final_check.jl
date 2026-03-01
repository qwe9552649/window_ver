using Pkg
Pkg.activate(@__DIR__)

println("=== 1. Julia 패키지 로드 테스트 ===\n")
pkgs = ["PyCall","JSON3","DataFrames","CSV","HTTP",
        "Distances","Clustering","StatsBase","Combinatorics",
        "HypothesisTests","Plots","StatsPlots","Distributed"]

for pkg in pkgs
    try
        @eval using $(Symbol(pkg))
        println("  ✅ $pkg")
    catch e
        println("  ❌ $pkg → $e")
    end
end

println("\n=== 2. PyCall ↔ Python ↔ pymoo ===\n")
using PyCall
println("  Python 경로 : ", PyCall.python)
println("  Python 버전 : ", PyCall.pyversion)
try
    pm = pyimport("pymoo")
    println("  pymoo 버전  : ", pm.__version__)
    println("  ✅ PyCall ↔ pymoo 정상")
catch e
    println("  ❌ pymoo 임포트 실패: $e")
end

println("\n=== 3. 시스템 리소스 ===\n")
println("  논리 코어 : ", Sys.CPU_THREADS, " 개")
println("  전체 RAM  : ", round(Sys.total_memory()/1024^3, digits=1), " GB")
println("  여유 RAM  : ", round(Sys.free_memory()/1024^3, digits=1), " GB")

println("\n=== 4. 프로젝트 파일 ===\n")
dir = @__DIR__
for f in ["moo_multitrip_cvrp_budapest.jl","osrm_client.jl",
          "nsga2_moo_backend.jl","moo_pareto_visualization.jl",
          "robustness_tests.jl","moo_result_manager.jl",
          "pymoo_vrp_nsga2_hi.py","run_windows.ps1","setup_windows.ps1"]
    println("  $(isfile(joinpath(dir,f)) ? "✅" : "❌") $f")
end

println("\n=== 5. 자동 병렬 설정 시뮬레이션 ===\n")
logical  = Sys.CPU_THREADS
physical = max(1, logical ÷ 2)
free_gb  = Sys.free_memory() / 1024^3
workers  = min(max(1, physical - 1), max(1, floor(Int, (free_gb - 2.0) / 0.55)))
threads  = max(1, logical ÷ (workers + 1))
println("  물리 코어       : $physical 개")
println("  논리 코어       : $logical 개")
println("  ► 최종 워커 수  : $workers 개  (Distributed 프로세스)")
println("  ► 스레드/워커   : $threads 개  (Julia threads)")
println("  ► 총 Julia 스레드: $(threads*(workers+1)) / $logical 논리코어")
println("  ► Python 동시 실행: $workers 개")
