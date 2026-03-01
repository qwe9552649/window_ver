using Pkg
Pkg.activate(@__DIR__)

python_exe = raw"C:\Users\idonghyeog\AppData\Local\Python\pythoncore-3.14-64\python.exe"
ENV["PYTHON"] = python_exe

println("PyCall 빌드 중 (Python: $python_exe)...")
Pkg.build("PyCall")

# 빌드 결과 확인
println("\n=== 확인 ===")
using PyCall
println("PyCall Python 경로: ", PyCall.python)
println("Python 버전: ", PyCall.pyversion)
println("✅ PyCall 정상!")
