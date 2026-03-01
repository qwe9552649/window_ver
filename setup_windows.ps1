# ============================================================
# setup_windows.ps1
# 어느 Windows PC에서나 한 번만 실행하면 전체 환경 자동 설치
#
# 설치 항목:
#   1. Julia  (없으면 winget으로 자동 설치)
#   2. Python (없으면 winget으로 자동 설치)
#   3. Python 패키지: pymoo, numpy
#   4. Julia  패키지: 필요한 패키지 전부
#   5. Gurobi 라이선스 상태 확인
#   6. OSRM   Docker 상태 확인 및 안내
# ============================================================

$ErrorActionPreference = "Continue"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

function Write-Step { param($msg) Write-Host "`n▶ $msg" -ForegroundColor Cyan }
function Write-OK   { param($msg) Write-Host "  ✅ $msg" -ForegroundColor Green }
function Write-Warn { param($msg) Write-Host "  ⚠️  $msg" -ForegroundColor Yellow }
function Write-Fail { param($msg) Write-Host "  ❌ $msg" -ForegroundColor Red }
function Write-Info { param($msg) Write-Host "  ℹ️  $msg" -ForegroundColor Gray }

$AllOk = $true

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════╗"
Write-Host "║     NSGA-II MOO CVRP Budapest  —  환경 자동 설치 스크립트  ║"
Write-Host "╚══════════════════════════════════════════════════════════════╝"

# ────────────────────────────────────────────────────────────
# 1. Julia 확인 / 설치
# ────────────────────────────────────────────────────────────
Write-Step "Julia 확인"
$juliaCmd = Get-Command julia -ErrorAction SilentlyContinue
if ($juliaCmd) {
    $juliaVer = julia --version 2>&1
    Write-OK "Julia 설치 확인: $juliaVer"
} else {
    Write-Warn "Julia가 설치되어 있지 않습니다. winget으로 설치를 시도합니다..."
    try {
        winget install --id Julialang.Julia -e --silent
        $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","Machine") + ";" +
                    [System.Environment]::GetEnvironmentVariable("PATH","User")
        $juliaCmd = Get-Command julia -ErrorAction SilentlyContinue
        if ($juliaCmd) {
            Write-OK "Julia 설치 완료"
        } else {
            Write-Fail "Julia 자동 설치 실패. https://julialang.org/downloads/ 에서 수동 설치 후 재실행하세요."
            $AllOk = $false
        }
    } catch {
        Write-Fail "winget 오류. https://julialang.org/downloads/ 에서 수동 설치하세요."
        $AllOk = $false
    }
}

# ────────────────────────────────────────────────────────────
# 2. Python 확인 / 설치
# ────────────────────────────────────────────────────────────
Write-Step "Python 확인"
$pythonCmd = Get-Command python  -ErrorAction SilentlyContinue
$python3Cmd = Get-Command python3 -ErrorAction SilentlyContinue

if ($pythonCmd -or $python3Cmd) {
    $pyExe = if ($python3Cmd) { "python3" } else { "python" }
    $pyVer = & $pyExe --version 2>&1
    Write-OK "Python 설치 확인: $pyVer"
} else {
    Write-Warn "Python이 설치되어 있지 않습니다. winget으로 설치를 시도합니다..."
    try {
        winget install --id Python.Python.3 -e --silent
        $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","Machine") + ";" +
                    [System.Environment]::GetEnvironmentVariable("PATH","User")
        $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
        if ($pythonCmd) {
            Write-OK "Python 설치 완료"
            $pyExe = "python"
        } else {
            Write-Fail "Python 자동 설치 실패. https://www.python.org/downloads/ 에서 수동 설치 후 재실행하세요."
            $AllOk = $false
        }
    } catch {
        Write-Fail "winget 오류. Python을 수동 설치하세요."
        $AllOk = $false
    }
}

# ────────────────────────────────────────────────────────────
# 3. Python 패키지 확인 / 설치
# ────────────────────────────────────────────────────────────
Write-Step "Python 패키지 확인 (pymoo, numpy)"
if ($pythonCmd -or $python3Cmd) {
    $pyExe = if (Get-Command python3 -ErrorAction SilentlyContinue) { "python3" } else { "python" }

    $pymooOk  = & $pyExe -c "import pymoo;  print(pymoo.__version__)"  2>&1
    $numpyOk  = & $pyExe -c "import numpy;  print(numpy.__version__)"  2>&1

    if ($LASTEXITCODE -eq 0 -and $pymooOk -notmatch "Error") {
        Write-OK "pymoo: $pymooOk"
    } else {
        Write-Warn "pymoo 없음. 설치 중..."
        & $pyExe -m pip install pymoo --quiet
        Write-OK "pymoo 설치 완료"
    }

    $numpyOk = & $pyExe -c "import numpy; print(numpy.__version__)" 2>&1
    if ($LASTEXITCODE -eq 0 -and $numpyOk -notmatch "Error") {
        Write-OK "numpy: $numpyOk"
    } else {
        Write-Warn "numpy 없음. 설치 중..."
        & $pyExe -m pip install numpy --quiet
        Write-OK "numpy 설치 완료"
    }
}

# ────────────────────────────────────────────────────────────
# 4. Julia 패키지 확인 / 설치
# ────────────────────────────────────────────────────────────
Write-Step "Julia 패키지 확인 및 설치"
if (Get-Command julia -ErrorAction SilentlyContinue) {
    Write-Info "필요한 Julia 패키지를 설치합니다 (첫 실행 시 수 분 소요)..."

    $juliaSetup = @"
using Pkg
Pkg.activate("$($ScriptDir.Replace('\','/'))")

required = [
    "JuMP", "Gurobi", "PyCall",
    "JSON3", "GeometryBasics", "DataFrames", "CSV",
    "Clustering", "Plots", "StatsPlots",
    "Combinatorics", "StatsBase",
    "Distances", "HTTP",
    "HypothesisTests"
]

installed = [p.name for p in values(Pkg.dependencies())]
to_install = filter(p -> !(p in installed), required)

if !isempty(to_install)
    println("설치할 패키지: ", join(to_install, ", "))
    Pkg.add(to_install)
else
    println("모든 Julia 패키지가 이미 설치되어 있습니다.")
end

# Distances는 Project.toml에 있지만 나머지는 환경에 추가
Pkg.resolve()
println("Julia 패키지 준비 완료!")
"@

    $juliaSetup | julia --project="$ScriptDir"
    if ($LASTEXITCODE -eq 0) {
        Write-OK "Julia 패키지 설치 완료"
    } else {
        Write-Warn "일부 Julia 패키지 설치 실패 (Gurobi는 라이선스 필요 — 아래 참고)"
    }
}

# ────────────────────────────────────────────────────────────
# 5. Gurobi 라이선스 확인
# ────────────────────────────────────────────────────────────
Write-Step "Gurobi 라이선스 확인"
$gurobiOk = $false
try {
    $grbCheck = julia --project="$ScriptDir" -e "using Gurobi; println(`"OK`")" 2>&1
    if ($grbCheck -match "OK") {
        Write-OK "Gurobi 사용 가능"
        $gurobiOk = $true
    } else {
        Write-Warn "Gurobi를 불러올 수 없습니다."
    }
} catch {
    Write-Warn "Gurobi 확인 실패"
}

if (-not $gurobiOk) {
    Write-Warn "Gurobi가 설치되어 있지 않거나 라이선스가 없습니다."
    Write-Info "➡ 무료 Academic 라이선스: https://www.gurobi.com/academia/academic-program-and-licenses/"
    Write-Info "➡ 설치 후 'grbgetkey <라이선스키>' 실행"
    $AllOk = $false
}

# ────────────────────────────────────────────────────────────
# 6. OSRM 서버 확인
# ────────────────────────────────────────────────────────────
Write-Step "OSRM 도로망 서버 확인 (localhost:5001, 5002)"
$osrmCar  = $false
$osrmFoot = $false
try {
    $r = Invoke-WebRequest -Uri "http://localhost:5001/health" -TimeoutSec 2 -ErrorAction Stop
    $osrmCar = $true
    Write-OK "OSRM 차량 서버 (포트 5001) 작동 중"
} catch { Write-Warn "OSRM 차량 서버 (포트 5001) 오프라인" }

try {
    $r = Invoke-WebRequest -Uri "http://localhost:5002/health" -TimeoutSec 2 -ErrorAction Stop
    $osrmFoot = $true
    Write-OK "OSRM 도보 서버 (포트 5002) 작동 중"
} catch { Write-Warn "OSRM 도보 서버 (포트 5002) 오프라인" }

if (-not ($osrmCar -and $osrmFoot)) {
    Write-Info "OSRM이 없으면 유클리드 거리로 자동 대체됩니다 (정확도 낮음)."
    Write-Info "Docker Desktop 설치 후 아래 명령어로 OSRM 실행:"
    Write-Info "  docker run -p 5001:5000 osrm/osrm-backend osrm-routed --algorithm mld /data/hungary-latest.osrm"
    Write-Info "  docker run -p 5002:5000 osrm/osrm-backend osrm-routed --algorithm mld /data/hungary-latest.osrm"
}

# ────────────────────────────────────────────────────────────
# 최종 결과
# ────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if ($AllOk) {
    Write-Host "  ✅ 모든 환경 준비 완료!  →  .\run_windows.ps1 으로 실행하세요." -ForegroundColor Green
} else {
    Write-Host "  ⚠️  일부 항목을 수동으로 설치해야 합니다. 위 안내를 참고하세요." -ForegroundColor Yellow
    Write-Host "      Gurobi 없이도 실행은 가능하지만 최적화 성능이 저하됩니다."  -ForegroundColor Yellow
}
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Write-Host ""
