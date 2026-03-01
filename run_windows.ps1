# ============================================================
# run_windows.ps1
# Windows 최적 성능 실행 스크립트 (NSGA-II MOO CVRP Budapest)
# 사용법: PowerShell에서  .\run_windows.ps1
# 특정 시나리오: .\run_windows.ps1 -Scenarios "1,2,3"
# ============================================================

param(
    [string]$Scenarios = "",       # 실행할 시나리오 (예: "1,2,3,5" / 빈 값=전체)
    [int]$Omega = 0,               # Monte Carlo 반복 횟수 (0=기본값 사용)
    [int]$Workers = 0,             # Distributed 워커 수 (0=자동)
    [int]$Threads = 0,             # Julia 스레드 수 (0=자동)
    [switch]$SkipRobustness        # Robustness 테스트 생략
)

# ────────────────────────────────────────────────────────────
# 1. 시스템 정보 수집
# ────────────────────────────────────────────────────────────
$LogicalCores  = [System.Environment]::ProcessorCount
$PhysicalCores = (Get-CimInstance Win32_Processor).NumberOfCores

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════╗"
Write-Host "║       NSGA-II MOO CVRP Budapest — Windows 실행      ║"
Write-Host "╚══════════════════════════════════════════════════════╝"
Write-Host ""
Write-Host "[ 시스템 정보 ]"
Write-Host "  물리 코어  : $PhysicalCores"
Write-Host "  논리 코어  : $LogicalCores  (하이퍼스레딩 포함)"

# ────────────────────────────────────────────────────────────
# 2. Julia 스레드 / Distributed 워커 수 결정
#
# 병목 분석:
#   - Python pymoo NSGA-II = CPU 집약적 (코어 1개 100% 사용)
#   - Julia 워커는 Python 실행 중 거의 대기 상태
#   → 워커 수를 최대화해서 Python 동시 실행 수를 늘리는 것이 핵심
#   → Julia 스레드는 최소화 (addprocs가 --threads를 워커에 그대로 상속하므로
#      스레드 수가 많으면 코어 × (워커+1)만큼 과잉 배정됨)
# ────────────────────────────────────────────────────────────

# Distributed 워커: 코어 - 1 (메인 Julia 프로세스용 코어 1개 확보)
if ($Workers -eq 0) {
    $NumWorkers = [Math]::Max(1, $LogicalCores - 1)
} else {
    $NumWorkers = $Workers
}

# Julia --threads (메인 프로세스 기준):
#   총 Julia 스레드 = (워커+1) × threads_per_process ≤ 논리 코어 수
#   → threads_per_process = 논리코어 ÷ (워커+1), 최소 1
if ($Threads -eq 0) {
    $JuliaThreads = [Math]::Max(1, [Math]::Floor($LogicalCores / ($NumWorkers + 1)))
} else {
    $JuliaThreads = $Threads
}

$TotalJuliaThreads = $JuliaThreads * ($NumWorkers + 1)

Write-Host "[ 성능 배분 계획 ]"
Write-Host "  Distributed 워커 수: $NumWorkers  → Python 동시 실행 $NumWorkers 개"
Write-Host "  Julia 스레드/프로세스: $JuliaThreads  → 총 Julia 스레드: $TotalJuliaThreads / $LogicalCores 코어"
Write-Host "  (Python이 병목이므로 Julia 스레드는 최소, 워커 수를 최대화)"
Write-Host ""

# ────────────────────────────────────────────────────────────
# 3. 환경변수 설정
# ────────────────────────────────────────────────────────────
$env:NUM_WORKERS = "$NumWorkers"

if ($Scenarios -ne "") {
    $env:SCENARIOS_TO_RUN = $Scenarios
    Write-Host "  실행 시나리오: $Scenarios"
} else {
    $env:SCENARIOS_TO_RUN = ""
    Write-Host "  실행 시나리오: 전체 (1,2,3,4,5)"
}

if ($Omega -gt 0) {
    $env:OMEGA_COUNT = "$Omega"
    Write-Host "  Monte Carlo 횟수 (omega): $Omega"
}

if ($SkipRobustness) {
    $env:SKIP_ROBUSTNESS = "1"
    Write-Host "  Robustness 테스트: 생략"
} else {
    $env:SKIP_ROBUSTNESS = ""
}

Write-Host ""

# ────────────────────────────────────────────────────────────
# 4. Julia 실행
# ────────────────────────────────────────────────────────────
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$MainScript = Join-Path $ScriptDir "moo_multitrip_cvrp_budapest.jl"

Write-Host "[ 실행 명령 ]"
Write-Host "  julia --threads $JuliaThreads --project=. moo_multitrip_cvrp_budapest.jl"
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Write-Host ""

$StartTime = Get-Date

julia --threads $JuliaThreads --project="$ScriptDir" "$MainScript"

$Elapsed = (Get-Date) - $StartTime
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Write-Host "  총 실행 시간: $($Elapsed.ToString('hh\:mm\:ss'))"
Write-Host "  결과 위치  : $env:USERPROFILE\Desktop\runs\"
Write-Host ""
