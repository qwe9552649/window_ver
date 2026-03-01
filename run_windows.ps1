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
# ────────────────────────────────────────────────────────────

# Julia --threads: 논리 코어 전부 활용 (Julia 내부 @threads 루프용)
if ($Threads -eq 0) {
    $JuliaThreads = $LogicalCores
} else {
    $JuliaThreads = $Threads
}

# Distributed 워커: Julia 코드가 pmap으로 omega를 병렬 처리
# 메인 프로세스 + OS 여유분(2코어) 제외
if ($Workers -eq 0) {
    $NumWorkers = [Math]::Max(1, $LogicalCores - 2)
} else {
    $NumWorkers = $Workers
}

Write-Host "  Julia 스레드: $JuliaThreads  (--threads $JuliaThreads)"
Write-Host "  Distributed 워커: $NumWorkers  (NUM_WORKERS=$NumWorkers)"
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
