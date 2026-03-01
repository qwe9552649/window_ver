using Pkg

# 1. General 레지스트리 추가 (없을 경우)
println("레지스트리 확인 중...")
regs = Pkg.Registry.reachable_registries()
if isempty(regs)
    println("  General 레지스트리 추가 중...")
    Pkg.Registry.add("General")
    println("  레지스트리 추가 완료")
else
    println("  레지스트리 OK: ", join([r.name for r in regs], ", "))
end

# 2. 프로젝트 활성화
Pkg.activate(@__DIR__)
println("프로젝트 활성화: ", @__DIR__)

# 3. Python 경로 설정
python_exe = raw"C:\Users\idonghyeog\AppData\Local\Python\pythoncore-3.14-64\python.exe"
ENV["PYTHON"] = python_exe
println("Python 경로 설정: ", python_exe)

# 4. 필요한 패키지 한번에 설치 (이미 있으면 Pkg가 스킵)
required = [
    "PyCall", "JSON3", "DataFrames", "CSV",
    "HTTP", "Distances", "Clustering", "StatsBase",
    "Combinatorics", "HypothesisTests", "StatsPlots", "Plots"
]
println("\n패키지 설치/업데이트 중...")
Pkg.add(required)

# 5. PyCall을 현재 Python으로 재빌드
println("\nPyCall 재빌드 중 (Python 경로 반영)...")
ENV["PYTHON"] = python_exe
Pkg.build("PyCall")

println("\n✅ Julia 패키지 설치 완료!")
