# ============================================================
# run_windows.ps1  —  NSGA-II MOO CVRP Budapest
# PC 성능 자동 분석 후 최적 세팅으로 Julia 실행
#
# 사용법:
#   .\run_windows.ps1                      # 전체 자동
#   .\run_windows.ps1 -Scenarios "1,2,3"  # 특정 시나리오
#   .\run_windows.ps1 -SkipRobustness     # robustness 생략
#   .\run_windows.ps1 -Workers 4 -Threads 2  # 수동 오버라이드
# ============================================================
param(
    [string]$Scenarios      = "",
    [int]$Omega             = 0,
    [int]$Workers           = 0,   # 0 = 자동
    [int]$Threads           = 0,   # 0 = 자동
    [switch]$SkipRobustness
)

# ────────────────────────────────────────────────────────────
# 1. PC 하드웨어 자동 탐지
# ────────────────────────────────────────────────────────────
$CpuInfo      = Get-CimInstance Win32_Processor
$PhysicalCores = ($CpuInfo | Measure-Object -Property NumberOfCores         -Sum).Sum
$LogicalCores  = ($CpuInfo | Measure-Object -Property NumberOfLogicalProcessors -Sum).Sum
$CpuName       = ($CpuInfo | Select-Object -First 1).Name.Trim()

$RamInfo       = Get-CimInstance Win32_OperatingSystem
$TotalRamGB    = [Math]::Round($RamInfo.TotalVisibleMemorySize / 1MB, 1)
$FreeRamGB     = [Math]::Round($RamInfo.FreePhysicalMemory     / 1MB, 1)
$UsedRamGB     = [Math]::Round($TotalRamGB - $FreeRamGB, 1)

$HasHyperThread = ($LogicalCores -gt $PhysicalCores)

# ────────────────────────────────────────────────────────────
# 2. 최적 워커/스레드 수 계산
#
# 병목: Python pymoo NSGA-II (CPU 집약, 프로세스당 물리코어 1개 사용)
# Julia 워커는 Python 실행 중 거의 대기 → Julia 스레드 최소화
#
# CPU 제약: 워커 ≤ 물리코어 - 1  (메인 Julia용 1코어)
# RAM 제약: 워커당 ~550MB  (Julia 150 + Python/pymoo 400)
#           OS + 메인 Julia = 2GB 예약
# ────────────────────────────────────────────────────────────
$MemPerWorkerGB = 0.55
$ReservedGB     = 2.0
$UsableGB       = [Math]::Max(0, $FreeRamGB - $ReservedGB)

$WorkersByCpu   = [Math]::Max(1, $PhysicalCores - 1)
$WorkersByRam   = [Math]::Max(1, [Math]::Floor($UsableGB / $MemPerWorkerGB))

$AutoWorkers    = [Math]::Min($WorkersByCpu, $WorkersByRam)

# --threads: 총 Julia 스레드가 논리 코어를 초과하지 않도록
# (addprocs는 --threads를 워커에 그대로 상속)
$AutoThreads    = [Math]::Max(1, [Math]::Floor($LogicalCores / ($AutoWorkers + 1)))

# 수동 오버라이드 처리
if ($Workers -gt 0) { $NumWorkers   = $Workers } else { $NumWorkers   = $AutoWorkers }
if ($Threads -gt 0) { $JuliaThreads = $Threads } else { $JuliaThreads = $AutoThreads }

$TotalJuliaThreads = $JuliaThreads * ($NumWorkers + 1)
$PythonConcurrent  = $NumWorkers   # pmap으로 동시에 실행되는 Python 프로세스 수

# ────────────────────────────────────────────────────────────
# 3. 분석 결과 출력
# ────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════╗"
Write-Host "║         NSGA-II MOO CVRP Budapest  —  PC 자동 분석          ║"
Write-Host "╠══════════════════════════════════════════════════════════════╣"
Write-Host ("║  CPU  : {0,-53}║" -f $CpuName)
Write-Host ("║  코어 : 물리 {0}개  /  논리 {1}개{2,-33}║" -f $PhysicalCores, $LogicalCores, $(if ($HasHyperThread) { "  (하이퍼스레딩 ON)" } else { "" }))
Write-Host ("║  RAM  : 전체 {0} GB  /  사용 {1} GB  /  여유 {2} GB{3,-6}║" -f $TotalRamGB, $UsedRamGB, $FreeRamGB, "")
Write-Host "╠══════════════════════════════════════════════════════════════╣"
Write-Host ("║  CPU 제약 워커 : {0,2}개  (물리코어 {1} - 1){2,-25}║" -f $WorkersByCpu,  $PhysicalCores, "")
Write-Host ("║  RAM 제약 워커 : {0,2}개  (여유 {1}GB ÷ 0.55GB){2,-21}║" -f $WorkersByRam, $FreeRamGB, "")
Write-Host "╠══════════════════════════════════════════════════════════════╣"
Write-Host ("║  ✅ 최종 워커 수         : {0,2}개{1,-33}║" -f $NumWorkers,         "")
Write-Host ("║  ✅ Julia 스레드/프로세스 : {0,2}개{1,-33}║" -f $JuliaThreads,       "")
Write-Host ("║  ✅ 총 Julia 스레드      : {0,2}개 / {1}개 논리코어{2,-19}║" -f $TotalJuliaThreads, $LogicalCores, "")
Write-Host ("║  ✅ Python 동시 실행     : {0,2}개  → 코어 가동률 {1,3}%{2,-14}║" -f $PythonConcurrent, ([Math]::Round($PythonConcurrent / $PhysicalCores * 100)), "")
Write-Host "╚══════════════════════════════════════════════════════════════╝"
Write-Host ""

if ($Workers -gt 0 -or $Threads -gt 0) {
    Write-Host "  ⚠️  수동 오버라이드 적용됨 (Workers=$NumWorkers, Threads=$JuliaThreads)" -ForegroundColor Yellow
    Write-Host ""
}

# ────────────────────────────────────────────────────────────
# 4. 환경변수 설정
# ────────────────────────────────────────────────────────────
$env:NUM_WORKERS        = "$NumWorkers"
$env:THREADS_PER_WORKER = "$JuliaThreads"

if ($Scenarios -ne "")  { $env:SCENARIOS_TO_RUN = $Scenarios  ; Write-Host "  시나리오: $Scenarios" }
else                    { $env:SCENARIOS_TO_RUN = ""           ; Write-Host "  시나리오: 전체 (1~5)" }

if ($Omega -gt 0)       { $env:OMEGA_COUNT = "$Omega"         ; Write-Host "  Monte Carlo: $Omega 회" }
if ($SkipRobustness)    { $env:SKIP_ROBUSTNESS = "1"          ; Write-Host "  Robustness: 생략" }
else                    { $env:SKIP_ROBUSTNESS = "" }

Write-Host ""

# ────────────────────────────────────────────────────────────
# 5. Julia 실행
# ────────────────────────────────────────────────────────────
$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$MainScript = Join-Path $ScriptDir "moo_multitrip_cvrp_budapest.jl"

Write-Host "[ 실행 ]  julia --threads $JuliaThreads --project=. moo_multitrip_cvrp_budapest.jl"
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Write-Host ""

$StartTime = Get-Date

julia --threads $JuliaThreads --project="$ScriptDir" "$MainScript"

$Elapsed = (Get-Date) - $StartTime
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Write-Host ("  총 실행 시간 : {0}" -f $Elapsed.ToString('hh\:mm\:ss'))
Write-Host ("  결과 위치   : {0}\Desktop\runs\" -f $env:USERPROFILE)
Write-Host ""
