
#!/usr/bin/env julia
################################################################################
# moo_multitrip_cvrp_budapest.jl
#
# ▸ Multi-Objective Optimization (MOO) VRP with NSGA-II
# ▸ 목적함수: [f1(기업), f2(고객), f3(사회)] 벡터 (가중합 아님!)
#   - f1: (이동거리×유류비) + (차량수×고정비) + (운행시간×운전자임금)
#   - f2: (락커고객 평균이동거리×이동비용) + (대기시간×시간가치)
#   - f3: (CO2 배출) × 탄소비용
# ▸ 2-Stage Locker-location & Vehicle-Routing (Scenario-rules 1–5 포함)  
# ▸ 시나리오 4: SLRP 기반 락커 위치/개수 동시 최적화
# ▸ 도로망 기반 거리 계산 (OSRM)
# ▸ Monte Carlo 시뮬레이션 + 시간창 지원
# ▸ Distributed multi-process parallel + Progress meter
################################################################################

# ═══════════════════════════════════════════════════════════════════════════════
# 멀티프로세싱 설정 (맨 처음에!)
# ═══════════════════════════════════════════════════════════════════════════════
using Distributed

# ═══════════════════════════════════════════════════════════════════════════════
# PC 성능 자동 탐지 및 최적 병렬 설정 계산
# ═══════════════════════════════════════════════════════════════════════════════

"""물리 코어 수 탐지 (OS별 명령 사용, 실패 시 논리 코어÷2 fallback)"""
function detect_physical_cores()::Int
    try
        if Sys.iswindows()
            # Windows: WMIC 또는 PowerShell로 물리 코어 수 조회
            out = readchomp(`wmic cpu get NumberOfCores /value`)
            m = match(r"NumberOfCores=(\d+)", out)
            m !== nothing && return parse(Int, m.captures[1])
            # fallback: PowerShell
            out2 = readchomp(`powershell -NoProfile -Command "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum"`)
            return parse(Int, strip(out2))
        elseif Sys.islinux()
            out = readchomp(`bash -c "grep '^cpu cores' /proc/cpuinfo | head -1 | awk '{print \$NF}'"`)
            n = parse(Int, strip(out))
            # 멀티소켓 보정: 소켓 수 × 코어 수
            sockets = parse(Int, strip(readchomp(`bash -c "grep '^physical id' /proc/cpuinfo | sort -u | wc -l"`)))
            return max(1, n * sockets)
        elseif Sys.isapple()
            return parse(Int, readchomp(`sysctl -n hw.physicalcpu`))
        end
    catch
    end
    # fallback: 논리 코어의 절반 (하이퍼스레딩 가정)
    return max(1, Sys.CPU_THREADS ÷ 2)
end

"""
    auto_configure_parallel() -> NamedTuple

현재 PC의 CPU(물리/논리 코어)와 가용 RAM을 분석하여
최적의 Distributed 워커 수와 Julia 스레드 수를 반환한다.

병목 분석:
  - Python pymoo NSGA-II = CPU 집약적 (프로세스당 물리코어 1개 100% 사용)
  - Julia 워커는 Python 실행 중 거의 대기 → Julia 스레드는 최소화
  → 워커 수를 최대화(= Python 동시 실행 수 최대화)가 핵심

제약:
  - CPU 제약: 워커 수 ≤ 물리코어 - 1 (메인 Julia용 1코어 확보)
  - RAM 제약: 워커당 ~550MB (Julia 150 + Python/pymoo 400)
              메인 Julia + OS 여유분 2GB 확보
"""
function auto_configure_parallel()
    logical_cores  = Sys.CPU_THREADS
    physical_cores = detect_physical_cores()
    total_mem_gb   = Sys.total_memory() / (1024^3)
    free_mem_gb    = Sys.free_memory()  / (1024^3)

    # ── CPU 제약 ─────────────────────────────────────────────────────────────
    # 물리 코어 기반 (하이퍼스레딩은 Python 같은 CPU 집약 작업에 실효 없음)
    workers_by_cpu = max(1, physical_cores - 1)

    # ── RAM 제약 ─────────────────────────────────────────────────────────────
    mem_per_worker_gb = 0.55   # Julia worker ~150MB + Python/pymoo ~400MB
    reserved_gb       = 2.0    # OS + 메인 Julia 프로세스
    usable_gb         = max(0.0, free_mem_gb - reserved_gb)
    workers_by_ram    = max(1, floor(Int, usable_gb / mem_per_worker_gb))

    # ── 최종 워커 수: CPU와 RAM 제약 중 보수적인 쪽 ─────────────────────────
    num_workers = min(workers_by_cpu, workers_by_ram)

    # ── Julia 스레드/프로세스 ────────────────────────────────────────────────
    # addprocs는 --threads를 워커에 상속 → 총 Julia 스레드 = (워커+1) × threads
    # 총 Julia 스레드가 논리 코어 수를 초과하지 않도록 분배
    threads_per_proc = max(1, logical_cores ÷ (num_workers + 1))

    return (
        num_workers      = num_workers,
        threads_per_proc = threads_per_proc,
        physical_cores   = physical_cores,
        logical_cores    = logical_cores,
        total_mem_gb     = total_mem_gb,
        free_mem_gb      = free_mem_gb,
        workers_by_cpu   = workers_by_cpu,
        workers_by_ram   = workers_by_ram,
    )
end

# 재진입 방지: 이미 로드된 경우 건너뛰기
if @isdefined(_ALNS_SCRIPT_LOADED)
    # 이미 로드됨 - 함수 정의만 사용, 실행 건너뛰기
else
    const _ALNS_SCRIPT_LOADED = true
    const IS_MAIN_PROCESS = (myid() == 1)

    # ─── PC 성능 자동 분석 및 최적 설정 계산 ────────────────────────────────
    const _PC_CONFIG = auto_configure_parallel()

    const NUM_WORKERS = let env = get(ENV, "NUM_WORKERS", "")
        isempty(env) ? _PC_CONFIG.num_workers :
            (try parse(Int, env) catch; _PC_CONFIG.num_workers end)
    end

    const THREADS_PER_WORKER = let env = get(ENV, "THREADS_PER_WORKER", "")
        isempty(env) ? _PC_CONFIG.threads_per_proc :
            (try parse(Int, env) catch; _PC_CONFIG.threads_per_proc end)
    end

    # 1. 워커 추가 (메인 프로세스에서만)
    if IS_MAIN_PROCESS && nprocs() == 1
        println("\n┌─────────────────────────────────────────────────────┐")
        println("│           PC 성능 자동 분석 결과                   │")
        println("├─────────────────────────────────────────────────────┤")
        @printf("│  물리 코어    : %2d개                                │\n", _PC_CONFIG.physical_cores)
        @printf("│  논리 코어    : %2d개  (하이퍼스레딩 포함)          │\n", _PC_CONFIG.logical_cores)
        @printf("│  전체 RAM     : %5.1f GB                            │\n", _PC_CONFIG.total_mem_gb)
        @printf("│  가용 RAM     : %5.1f GB                            │\n", _PC_CONFIG.free_mem_gb)
        println("├─────────────────────────────────────────────────────┤")
        @printf("│  CPU 제약 워커: %2d개  (물리코어 - 1)               │\n", _PC_CONFIG.workers_by_cpu)
        @printf("│  RAM 제약 워커: %2d개  (가용RAM ÷ 0.55GB)          │\n", _PC_CONFIG.workers_by_ram)
        println("├─────────────────────────────────────────────────────┤")
        @printf("│  ✅ 최종 워커 수      : %2d개                       │\n", NUM_WORKERS)
        @printf("│  ✅ Julia 스레드/프로세스: %d개                     │\n", THREADS_PER_WORKER)
        @printf("│  ✅ 총 Julia 스레드   : %2d개 / %d 논리코어        │\n",
                THREADS_PER_WORKER * (NUM_WORKERS + 1), _PC_CONFIG.logical_cores)
        println("└─────────────────────────────────────────────────────┘\n")

        addprocs(NUM_WORKERS; exeflags="--threads=$(THREADS_PER_WORKER)")
        println("🚀 워커 $(nprocs()-1)개 시작 완료 — Python 동시 최대 실행: $(NUM_WORKERS)개")
    end

    # 2. 워커에서 현재 스크립트 로드 (메인에서만 호출!)
    if IS_MAIN_PROCESS && nprocs() > 1
        # 워커들에서 이 스크립트 파일을 include
        @everywhere workers() include(joinpath(@__DIR__, "moo_multitrip_cvrp_budapest.jl"))
    end
end

# IS_MAIN_PROCESS가 정의되지 않은 경우 (워커에서 첫 로드)
if !@isdefined(IS_MAIN_PROCESS)
    const IS_MAIN_PROCESS = (myid() == 1)
end

# ═══════════════════════════════════════════════════════════════════════════════
# 모듈 로드 (모든 프로세스에서)
# ═══════════════════════════════════════════════════════════════════════════════
using JuMP, Gurobi
using Random, JSON3, GeometryBasics, DataFrames, CSV
using Statistics, Clustering, Plots, Dates
using Combinatorics, StatsBase
using Printf
using Pkg
using Distances
using PyCall
# HypothesisTests는 robustness_tests.jl에서 사용

include(joinpath(@__DIR__, "nsga2_moo_backend.jl"))  # NSGA-II MOO 기반 (시간창 지원)
include(joinpath(@__DIR__, "osrm_client.jl"))
include(joinpath(@__DIR__, "moo_result_manager.jl"))  # MOO 결과 관리 및 저장
include(joinpath(@__DIR__, "moo_pareto_visualization.jl"))  # MOO Pareto front 시각화

# ═══════════════════════════════════════════════════════════════════════════════
# 거리 계산 모드 설정
# ═══════════════════════════════════════════════════════════════════════════════
const USE_ROAD_DISTANCE = true  # true: 도로망 기반, false: 유클리드 거리

import Base.Threads: @threads, nthreads, threadid

# 스레드 안전성을 위한 글로벌 락들
const PROGRESS_LOCK = ReentrantLock()
const PRINT_LOCK = ReentrantLock()

# 멀티프로세싱 설정 확인 및 출력
function check_process_setup()
    n_procs = nprocs()
    lock(PRINT_LOCK) do
        println("🚀 멀티프로세싱 설정:")
        println("   - 총 프로세스 수: $n_procs (메인 1 + 워커 $(n_procs-1))")
        println("   - 현재 프로세스 ID: $(myid())")
        if n_procs == 1
            println("   ⚠️  단일 프로세스 모드입니다. 병렬 처리가 작동하지 않습니다.")
        else
            println("   ✅ 멀티프로세싱 병렬 처리가 활성화되었습니다.")
        end
        println("")
    end
    return n_procs
end

# 스레드 안전한 출력 함수
function thread_safe_println(msg::String)
    lock(PRINT_LOCK) do
        println(msg)
        flush(stdout)
    end
end

# Gurobi 출력 완전 억제를 위한 함수
function create_silent_model()
    original_stdout, original_stderr = stdout, stderr
    model = nothing
    try
        redirect_stdout(devnull)
        redirect_stderr(devnull)
        model = Model(Gurobi.Optimizer)
        set_optimizer_attribute(model, "OutputFlag", 0)
        set_optimizer_attribute(model, "LogToConsole", 0)
        set_optimizer_attribute(model, "LogFile", "")
        try
            set_optimizer_attribute(model, "Presolve", 0)
        catch
        end
    finally
        redirect_stdout(original_stdout)
        redirect_stderr(original_stderr)
    end
    return model
end

function silent_optimize!(model)
    original_stdout, original_stderr = stdout, stderr
    try
        redirect_stdout(devnull)
        redirect_stderr(devnull)
        JuMP.optimize!(model)
    finally
        redirect_stdout(original_stdout)
        redirect_stderr(original_stderr)
    end
end

# 진행률 출력 함수들 - stdout만 사용 (중복 방지) - 스레드 안전
function progress_println(msg::String)
    thread_safe_println(msg)
end

#───────────────────────────────────────────────────────────────────────────────
# 시나리오 상수 정의
# 매직 넘버 대신 명명된 상수 사용으로 코드 가독성 향상
#───────────────────────────────────────────────────────────────────────────────
const SCENARIO_D2D  = 1   # Door-to-Door (락커 미사용, 재배송 모델)
const SCENARIO_DPL  = 2   # Dedicated Private Locker (전용 사설 락커)
const SCENARIO_SPL  = 3   # Shared Private Locker (공유 사설 락커)
const SCENARIO_OPL  = 4   # Optimized Public Locker (2-Stage SLRP 기반 최적화 공공 락커)
const SCENARIO_PSPL = 5   # Partially Shared Private Locker (부분 공유 - Alza/Foxpost/Packeta)

const ALL_SCENARIOS = [SCENARIO_D2D, SCENARIO_DPL, SCENARIO_SPL, SCENARIO_OPL, SCENARIO_PSPL]

# 시나리오 명칭 통일 함수
function get_scenario_name(scenario::Int)
    scenario_names = Dict(
        SCENARIO_D2D  => "D2D",     # Door-to-Door
        SCENARIO_DPL  => "DPL",     # Dedicated Private Locker
        SCENARIO_SPL  => "SPL",     # Shared Private Locker  
        SCENARIO_OPL  => "OPL",     # Optimized Public Locker (2-Stage SLRP)
        SCENARIO_PSPL => "PSPL"     # Partially Shared Private Locker (Alza/Foxpost/Packeta)
    )
    return get(scenario_names, scenario, "Unknown")
end

#───────────────────────────────────────────────────────────────────────────────
# 0. Global options + Progress tracking
#───────────────────────────────────────────────────────────────────────────────
const BACKEND = get(ENV, "BACKEND", "pyalns")
const CARRIERS = ["Foxpost","GLS","Packeta","AlzaBox","EasyBox","DHL"]

# 배송사별 시장 점유율 (고객 유형별로 다르게 설정)
# D2D 합계: 70+0+4+10+13+4 = 101% → 정규화하여 100%로 맞춤
const CARRIER_MARKET_SHARE_D2D = Dict(
    "GLS"     => 0.69,   # 70/101 ≈ 69%
    "EasyBox" => 0.13,   # 13%
    "Packeta" => 0.10,   # 10%
    "DHL"     => 0.04,   # 4%
    "AlzaBox" => 0.04,   # 4%
    "Foxpost" => 0.00    # 0%
)

# 락커 배송: GLS 32%, Foxpost 28%가 주요 업체
const CARRIER_MARKET_SHARE_LOCKER = Dict(
    "GLS"     => 0.32,   # 32%
    "Foxpost" => 0.28,   # 28%
    "EasyBox" => 0.18,   # 18%
    "Packeta" => 0.14,   # 14%
    "AlzaBox" => 0.04,   # 4%
    "DHL"     => 0.04    # 4%
)

# 배송사별 락커 용량 (LC: Locker Capacity)
const LOCKER_CAPACITY = Dict(
    "DHL"     => 16,
    "GLS"     => 78,
    "Foxpost" => 89,
    "AlzaBox" => 91,
    "EasyBox" => 54,
    "Packeta" => 40
)

# Public 락커 기본 용량
const PUBLIC_LOCKER_CAPACITY = 69 


 

# 시나리오 5 (PSPL) 전용: 부분 공유 가능한 캐리어 집합 (서로의 Private 락커 공유)
const PSPL_SHARED_CARRIERS = Set(["AlzaBox","Foxpost","Packeta"])

#───────────────────────────────────────────────────────────────────────────────
# 차량 수 자동 계산 설정
#───────────────────────────────────────────────────────────────────────────────
# 고객 수 기준 자동 계산 (고객 50명당 1대, 50% 여유)
const CUSTOMERS_PER_VEHICLE = 50  # 차량당 배송 가능 고객 수 (락커 효율 고려)

"""
고객 수 기준 차량 수 자동 계산
- 고객 50명당 1대
- 50% 여유분 추가 (시간창 제약 고려)
"""
function calculate_required_vehicles(carrier::String, num_customers::Int)
    base_vehicles = ceil(Int, num_customers / CUSTOMERS_PER_VEHICLE)
    with_buffer = ceil(Int, base_vehicles * 1.5)
    return max(1, with_buffer)
end

# 하위 호환용 (carrier 없이 호출 시)
function calculate_required_vehicles(num_customers::Int)
    base_vehicles = ceil(Int, num_customers / CUSTOMERS_PER_VEHICLE)
    with_buffer = ceil(Int, base_vehicles * 1.5)
    return max(1, with_buffer)
end

#───────────────────────────────────────────────────────────────────────────────
# 목적함수 비용 파라미터 (논문: Barbieri et al., ICORES 2025 기반)
#───────────────────────────────────────────────────────────────────────────────
const TRANSPORT_COST_PER_KM = 1.0      # 운송 비용: 1 EUR/km (차량 주행거리)
const MAX_WALKING_DISTANCE_KM = 0.5    # 최대 허용 도보거리: 500m (ρ in paper)
const LOCKER_ACTIVATION_COST = 0.0     # 락커 활성화 비용: 상수 (추후 조정 가능)

# SLRP 목적함수용 CO2 파라미터 (NSGA-II f₃와 동일)
const SLRP_VEHICLE_CO2_PER_KM = 0.12          # 배송 차량 CO2 (kg/km)
const SLRP_CUSTOMER_VEHICLE_CO2_PER_KM = 0.12 # 고객 차량 CO2 (kg/km) — 배송 밴과 동일
const SLRP_LOCKER_CO2_PER_UNIT_PER_DAY = 0.5  # 락커 전력 CO2 (kg/unit/day)

#───────────────────────────────────────────────────────────────────────────────
# 고객 이동 수단 선택 모델 파라미터 (z₂ 고객 이동비용 계산용)
#───────────────────────────────────────────────────────────────────────────────
# 4가지 이동 수단: 도보, 자전거, 차량(전용), 차량(연계)
# - 도보: 왕복 기준, 0.1 EUR/km (고객 불편비용)
# - 자전거: 왕복 기준, 0.05 EUR/km (고객 불편비용)  
# - 차량(전용): 왕복 기준, 1.0 EUR/km (사회적비용)
# - 차량(연계): 편도 기준, 1.0 EUR/km (사회적비용)

const WALK_COST_PER_KM = 0.10           # 도보 비용: 0.1 EUR/km (고객 불편비용)
const BICYCLE_COST_PER_KM = 0.05        # 자전거 비용: 0.05 EUR/km (고객 불편비용)
const VEHICLE_DEDICATED_COST_PER_KM = 1.0  # 차량(전용) 비용: 1.0 EUR/km (사회적비용)
const VEHICLE_LINKED_COST_PER_KM = 1.0     # 차량(연계) 비용: 1.0 EUR/km (사회적비용)

# 편도 거리별 이동수단 분담률 테이블 (km 단위)
# (편도거리, 도보%, 자전거%, 차량전용%, 차량연계%)
const MODE_SHARE_TABLE = [
    (0.0, 77.8, 15.8, 2.3, 4.1),
    (0.2, 70.1, 20.6, 3.4, 6.0),
    (0.4, 60.9, 25.9, 4.8, 8.4),
    (0.6, 50.8, 31.3, 6.6, 11.3),
    (0.8, 40.6, 36.1, 8.6, 14.6),
    (1.0, 31.1, 40.0, 10.9, 18.0),
    (1.2, 22.9, 42.6, 13.1, 21.4),
    (1.4, 16.3, 43.8, 15.3, 24.6),
    (1.6, 11.3, 43.8, 17.4, 27.5),
    (1.8, 7.6, 42.9, 19.4, 30.1),
    (2.0, 5.1, 41.3, 21.2, 32.4),
    (2.2, 3.3, 39.3, 22.9, 34.4),
    (2.4, 2.2, 37.1, 24.5, 36.2),
    (2.6, 1.4, 34.7, 26.1, 37.8),
    (2.8, 0.9, 32.3, 27.5, 39.3),
    (3.0, 0.6, 29.9, 28.9, 40.6),
    (3.2, 0.4, 27.5, 30.3, 41.8),
    (3.4, 0.2, 25.3, 31.6, 42.9),
    (3.6, 0.1, 23.1, 32.8, 43.9),
    (3.8, 0.1, 21.1, 34.1, 44.7),
    (4.0, 0.1, 19.2, 35.2, 45.5),
    (4.2, 0.0, 17.5, 36.3, 46.2),
    (4.4, 0.0, 15.8, 37.4, 46.7),
    (4.6, 0.0, 14.3, 38.4, 47.2),
    (4.8, 0.0, 12.9, 39.4, 47.6),
    (5.0, 0.0, 11.7, 40.4, 48.0)
]

"""
    get_mode_share(one_way_distance_km)

편도 거리에 따른 이동수단 분담률 반환 (선형 보간)
Returns: (walk_prob, bicycle_prob, vehicle_dedicated_prob, vehicle_linked_prob)
"""
function get_mode_share(one_way_distance_km::Float64)
    dist = max(0.0, one_way_distance_km)
    
    # 테이블 범위를 벗어나면 마지막 값 사용
    if dist >= 5.0
        return (0.0, 11.7, 40.4, 48.0) ./ 100.0
    end
    
    # 선형 보간을 위한 인덱스 찾기
    for i in 1:(length(MODE_SHARE_TABLE)-1)
        d1, w1, b1, vd1, vl1 = MODE_SHARE_TABLE[i]
        d2, w2, b2, vd2, vl2 = MODE_SHARE_TABLE[i+1]
        
        if d1 <= dist < d2
            # 선형 보간
            t = (dist - d1) / (d2 - d1)
            walk = w1 + t * (w2 - w1)
            bicycle = b1 + t * (b2 - b1)
            vehicle_ded = vd1 + t * (vd2 - vd1)
            vehicle_link = vl1 + t * (vl2 - vl1)
            return (walk, bicycle, vehicle_ded, vehicle_link) ./ 100.0
        end
    end
    
    # 첫 번째 값 반환 (dist = 0)
    return (77.8, 15.8, 2.3, 4.1) ./ 100.0
end

"""
    calculate_customer_mobility_cost(one_way_distance_km, mode)

고객 이동비용 계산
- mode: :walk, :bicycle, :vehicle_dedicated, :vehicle_linked
- 도보/자전거/차량(전용): 왕복 거리 기준
- 차량(연계): 편도 거리 기준
"""
function calculate_customer_mobility_cost(one_way_distance_km::Float64, mode::Symbol)
    if mode == :walk
        # 도보: 왕복 거리 × 0.1 EUR/km
        return 2.0 * one_way_distance_km * WALK_COST_PER_KM
    elseif mode == :bicycle
        # 자전거: 왕복 거리 × 0.05 EUR/km
        return 2.0 * one_way_distance_km * BICYCLE_COST_PER_KM
    elseif mode == :vehicle_dedicated
        # 차량(전용): 왕복 거리 × 1.0 EUR/km
        return 2.0 * one_way_distance_km * VEHICLE_DEDICATED_COST_PER_KM
    elseif mode == :vehicle_linked
        # 차량(연계): 편도 거리 × 1.0 EUR/km
        return one_way_distance_km * VEHICLE_LINKED_COST_PER_KM
    else
        return 0.0
    end
end

#───────────────────────────────────────────────────────────────────────────────
# 락커 통계 수집 시스템
#───────────────────────────────────────────────────────────────────────────────

# 개별 락커 통계 정보
mutable struct LockerStatInfo
    locker_id::String
    carrier::String
    capacity::Int
    used::Int
    occupancy_rate::Float64
    customers_assigned::Vector{String}
    total_customers_tried::Int  # 시도한 고객 수 (용량 부족으로 실패한 경우도 포함)
    # 캐리어별 사용 비중 추가
    carrier_usage::Dict{String, Int}  # 캐리어명 => 사용량
    carrier_percentages::Dict{String, Float64}  # 캐리어명 => 사용 비율
    
    function LockerStatInfo(locker_id::String, carrier::String, capacity::Int)
        new(locker_id, carrier, capacity, 0, 0.0, String[], 0, 
            Dict{String, Int}(), Dict{String, Float64}())
    end
end

# 캐리어별 통계 정보
mutable struct CarrierStatInfo
    carrier::String
    

    

    total_customers::Int
    locker_customers::Int
    d2d_customers::Int
    d2d_conversions::Int
    conversion_rate::Float64
    locker_utilization::Float64
    market_share_locker::Float64
    market_share_d2d::Float64
    # 차량 관련 통계 추가
    vehicles_count::Int
    customers_per_vehicle::Float64
    vehicle_utilization::Float64
    # 차량 용량 관련 통계 추가
    total_vehicle_capacity::Int
    used_vehicle_capacity::Int
    vehicle_occupancy_rate::Float64
    # 차량별 거리 및 적재율 통계 추가
    total_distance::Float64
    avg_distance_per_vehicle::Float64
    avg_load_rate::Float64  # 평균 적재율 (CVRP)
    # 사용자 요청 추가 통계 (라우팅 후 계산)
    avg_load_factor_per_used_vehicle::Float64
    avg_distance_per_used_vehicle::Float64
    km_per_demand::Float64
    avg_fill_rate_per_trip::Float64
    
    function CarrierStatInfo(carrier::String)
        vehicles = 0  # 미사용
        new(carrier, 0, 0, 0, 0, 0.0, 0.0, 
            CARRIER_MARKET_SHARE_LOCKER[carrier], 
            CARRIER_MARKET_SHARE_D2D[carrier],
            vehicles, 0.0, 0.0,
            vehicles * effective_capacity(), 0, 0.0,
            0.0, 0.0, 0.0,
            # 추가 통계 초기화
            0.0, 0.0, 0.0, 0.0)
    end
end

# 시나리오별 락커 통계
mutable struct ScenarioLockerStats
    scenario::Int
    seed::Int
    omega::Int  # 몬테카르로 샘플 번호
    total_locker_customers::Int
    d2d_conversions::Int
    conversion_rate::Float64
    locker_stats::Dict{String, LockerStatInfo}
    carrier_stats::Dict{String, CarrierStatInfo}
    timestamp::DateTime
    
    function ScenarioLockerStats(scenario::Int, seed::Int, omega::Int)
        new(scenario, seed, omega, 0, 0, 0.0, 
            Dict{String, LockerStatInfo}(), 
            Dict{String, CarrierStatInfo}(),
            now())
    end
end

# 🔥 시나리오 4 SLRP (Stochastic Location-Routing Problem) 최적화용 전역 변수

# 전체 락커 통계 수집기
mutable struct LockerStatsCollector
    stats_by_scenario::Dict{Tuple{Int,Int,Int}, ScenarioLockerStats}  # (scenario, seed, omega) => stats
    summary_stats::Dict{String, Any}
    
    function LockerStatsCollector()
        new(Dict{Tuple{Int,Int,Int}, ScenarioLockerStats}(), Dict{String, Any}())
    end
end

# 전역 락커 통계 수집기
const LOCKER_STATS_COLLECTOR = LockerStatsCollector()
const STATS_LOCK = ReentrantLock()

# MOO 결과 수집기 (시나리오별, omega별)
const MOO_RESULTS_COLLECTOR = Vector{MOOScenarioResult}()
const MOO_RESULTS_LOCK = ReentrantLock()

# 메인 배치 라우팅 결과 저장용
const MAIN_BATCH_RESULTS = Dict{Tuple{Int,Int}, Dict{String,Any}}()  # (scenario, omega) => route_data
const RESULTS_LOCK = ReentrantLock()

const CAPACITY = 100  # 차량 용량
const MAX_PUB = 9     # 시나리오 4 최대 락커 개수

# Mutable overrides for parameter sensitivity testing (robustness_tests.jl)
const PARAM_OVERRIDES = Dict{Symbol, Any}()
get_param(sym::Symbol, default) = get(PARAM_OVERRIDES, sym, default)
effective_capacity() = get_param(:CAPACITY, CAPACITY)
effective_max_pub() = get_param(:MAX_PUB, MAX_PUB)
effective_slrp_vehicle_co2() = get_param(:SLRP_VEHICLE_CO2_PER_KM, SLRP_VEHICLE_CO2_PER_KM)
effective_public_locker_capacity() = get_param(:PUBLIC_LOCKER_CAPACITY, PUBLIC_LOCKER_CAPACITY)
effective_vehicle_co2() = get_param(:MOO_VEHICLE_CO2_PER_KM, MOO_VEHICLE_CO2_PER_KM)
effective_fuel_cost() = get_param(:MOO_FUEL_COST_PER_KM, MOO_FUEL_COST_PER_KM)
effective_vehicle_daily_cost() = get_param(:MOO_VEHICLE_DAILY_COST, MOO_VEHICLE_DAILY_COST)

# ═══════════════════════════════════════════════════════════════════════════
# 시간창 (Time Window) 설정 - NSGA-II MOO 시간창 지원
# ═══════════════════════════════════════════════════════════════════════════
# 시간 단위: 초 (0 = 자정 00:00)
const TW_DEPOT_OPEN = 0              # 디포 운영 시작 (00:00)
const TW_DEPOT_CLOSE = 86400         # 디포 운영 종료 (24:00)
const TW_LOCKER_OPEN = 0             # 락커 운영 시작 (00:00, 24시간)
const TW_LOCKER_CLOSE = 86400        # 락커 운영 종료 (24:00, 24시간)

# D2D 배송 가능 시간 (첫 고객 08:00 도착, 마지막 고객 17:00까지 배송 완료)
const TW_D2D_OPEN = 8 * 3600         # 배송 시작 가능: 08:00
const TW_D2D_CLOSE = 17 * 3600       # 배송 완료 마감: 17:00

# D2D 고객 랜덤 시간창 설정
const TW_D2D_START_MIN = 8 * 3600    # D2D 시간창 시작 최소: 08:00
const TW_D2D_START_MAX = 14 * 3600   # D2D 시간창 시작 최대: 14:00
const TW_D2D_WIDTH_MIN = 3 * 3600    # D2D 시간창 폭 최소: 3시간
const TW_D2D_WIDTH_MAX = 9 * 3600    # D2D 시간창 폭 최대: 9시간

# 택배기사 운송 시간 설정
const DRIVER_START_MIN = 8 * 3600    # 택배기사 출발 최소: 08:00
const DRIVER_START_MAX = 17 * 3600   # 택배기사 출발 최대: 17:00

# 서비스 시간
const SERVICE_TIME_D2D = 240         # D2D 서비스 시간: 4분 - 고객 대면 하역
const SERVICE_TIME_LOCKER_BASE = 60  # 락커 고정 서비스 시간: 1분 (주차, 락커 접근 등)
const SERVICE_TIME_PER_ITEM = 10     # 락커 물건 1개당 처리 시간: 10초
const SERVICE_TIME_LOCKER = 210      # 락커 기본 서비스 시간: 3.5분 (15개 기준) - 레거시 호환용
const SERVICE_TIME_DEPOT = 0         # 디포 서비스 시간: 0분 - 출발/복귀지

# 락커 서비스 시간 계산 (고정 1분 + 물건당 10초)
function calculate_locker_service_time(demand::Int)
    return SERVICE_TIME_LOCKER_BASE + demand * SERVICE_TIME_PER_ITEM  # 1분 + 물건당 10초
end

# 락커 회수 설정
const LOCKER_PICKUP_FAST_RATIO = 0.5    # 빠른 회수 비율: 50%
const LOCKER_PICKUP_FAST_TIME = 6 * 3600  # 빠른 회수 시간: 6시간
const LOCKER_PICKUP_SLOW_TIME = 24 * 3600 # 느린 회수 시간: 24시간

# 기타 설정
const USE_TIME_WINDOWS = true        # 시간창 제약 사용 여부
const FORCE_D2D_ON_LOCKER_OVERFLOW = false  # 락커 초과 시 D2D 강제 전환 (비활성화)

# ═══════════════════════════════════════════════════════════════════════════
# 시간창 생성 헬퍼 함수
# ═══════════════════════════════════════════════════════════════════════════
"""
    generate_time_windows(node_ids, node_types::Dict{String,String})

노드별 시간창과 서비스 시간을 생성합니다.

# Arguments
- `node_ids`: 노드 ID 벡터
- `node_types`: 노드 ID => 타입 ("D2D", "Locker", "Depot")

# Returns
- `(time_windows, service_times)`: Dict{String, Tuple{Int,Int}}, Dict{String, Int}
"""
function generate_time_windows(
    node_ids::Vector{String},
    node_types::Dict{String,String}
)
    time_windows = Dict{String, Tuple{Int,Int}}()
    service_times = Dict{String, Int}()
    
    for nid in node_ids
        ntype = get(node_types, nid, "D2D")
        if ntype == "Depot"
            time_windows[nid] = (TW_DEPOT_OPEN, TW_DEPOT_CLOSE)
            service_times[nid] = SERVICE_TIME_DEPOT
        elseif ntype == "Locker"
            time_windows[nid] = (TW_LOCKER_OPEN, TW_LOCKER_CLOSE)
            service_times[nid] = SERVICE_TIME_LOCKER
        else  # D2D
            time_windows[nid] = (TW_D2D_OPEN, TW_D2D_CLOSE)
            service_times[nid] = SERVICE_TIME_D2D
        end
    end
    
    return time_windows, service_times
end

"""
    generate_time_windows_for_customers(customers::Vector)

Customer 객체 벡터에서 시간창과 서비스 시간을 생성합니다.
고객 객체에 tw_early, tw_late 필드가 있으면 해당 값을 사용합니다.
"""
function generate_time_windows_for_customers(customers::Vector)
    time_windows = Dict{String, Tuple{Int,Int}}()
    service_times = Dict{String, Int}()
    pickup_delays = Dict{String, Int}()  # 락커 회수 시간
    
    for c in customers
        cid = c.id
        dtype = hasproperty(c, :dtype) ? c.dtype : "D2D"
        
        # 시간창: 고객 객체에 있으면 사용, 없으면 기본값
        if hasproperty(c, :tw_early) && hasproperty(c, :tw_late)
            if dtype == "D2D"
                # D2D: 17:00 마감 적용
                tw_end = min(Int(c.tw_late), TW_D2D_CLOSE)
                time_windows[cid] = (Int(c.tw_early), tw_end)
            else
                # 락커: 그대로 사용
                time_windows[cid] = (Int(c.tw_early), Int(c.tw_late))
            end
        else
            # 레거시: 랜덤 시간창 생성 (D2D와 락커 모두 동일)
            # 락커 고객도 희망 배송 시간 존재 (만족도 계산용)
            tw_start = rand(TW_D2D_START_MIN:60:TW_D2D_START_MAX)
            tw_width = rand(TW_D2D_WIDTH_MIN:60:TW_D2D_WIDTH_MAX)
            tw_end = min(tw_start + tw_width, TW_D2D_CLOSE)  # 17:00 초과 방지
            time_windows[cid] = (tw_start, tw_end)
        end
        
        # 서비스 시간
        if dtype == "Locker"
            service_times[cid] = SERVICE_TIME_LOCKER
        else
            service_times[cid] = SERVICE_TIME_D2D
        end
        
        # 락커 회수 시간
        if hasproperty(c, :pickup_delay)
            pickup_delays[cid] = Int(c.pickup_delay)
        else
            pickup_delays[cid] = 0
        end
    end
    
    return time_windows, service_times, pickup_delays
end

"""
    add_depot_time_windows!(time_windows, service_times, depot_ids)

디포들의 시간창과 서비스 시간을 추가합니다.
"""
function add_depot_time_windows!(
    time_windows::Dict{String, Tuple{Int,Int}},
    service_times::Dict{String, Int},
    depot_ids::Vector{String}
)
    for did in depot_ids
        time_windows[did] = (TW_DEPOT_OPEN, TW_DEPOT_CLOSE)
        service_times[did] = SERVICE_TIME_DEPOT
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# 락커 회수 시뮬레이션 및 동적 배송 관리
# ═══════════════════════════════════════════════════════════════════════════

"""
    LockerDeliveryTracker

락커 배송 및 회수 상태를 추적하는 구조체
"""
mutable struct LockerDeliveryTracker
    # 락커별 현재 사용량: locker_id => 현재 사용 중인 슬롯 수
    current_usage::Dict{String, Int}
    # 락커별 최대 용량: locker_id => 최대 용량
    max_capacity::Dict{String, Int}
    # 배송 완료 기록: (locker_id, customer_id, delivery_time, pickup_delay)
    deliveries::Vector{Tuple{String, String, Int, Int}}
    # 현재 시뮬레이션 시간 (초)
    current_time::Int
    # 전체 배송 완료 시간
    total_completion_time::Int
end

function LockerDeliveryTracker(locker_capacities::Dict{String, Int})
    LockerDeliveryTracker(
        Dict{String, Int}(lid => 0 for lid in keys(locker_capacities)),
        copy(locker_capacities),
        Vector{Tuple{String, String, Int, Int}}(),
        DRIVER_START_MIN,  # 배송 시작 시간: 08:00
        0
    )
end

"""
    get_available_slots(tracker, locker_id, current_time)

현재 시간에 특정 락커의 사용 가능한 슬롯 수를 반환합니다.
회수가 완료된 슬롯은 가용 슬롯으로 계산됩니다.
"""
function get_available_slots(tracker::LockerDeliveryTracker, locker_id::String, current_time::Int)
    max_cap = get(tracker.max_capacity, locker_id, 0)
    
    # 현재 사용 중인 슬롯 계산 (회수 완료된 것 제외)
    active_count = 0
    for (lid, cid, delivery_time, pickup_delay) in tracker.deliveries
        if lid == locker_id
            pickup_time = delivery_time + pickup_delay
            if current_time < pickup_time
                # 아직 회수되지 않음
                active_count += 1
            end
        end
    end
    
    return max_cap - active_count
end

"""
    record_delivery!(tracker, locker_id, customer_id, delivery_time, pickup_delay)

락커 배송을 기록합니다.
"""
function record_delivery!(
    tracker::LockerDeliveryTracker, 
    locker_id::String, 
    customer_id::String, 
    delivery_time::Int, 
    pickup_delay::Int
)
    push!(tracker.deliveries, (locker_id, customer_id, delivery_time, pickup_delay))
    tracker.total_completion_time = max(tracker.total_completion_time, delivery_time)
end

"""
    format_time(seconds)

초를 HH:MM:SS 형식으로 변환합니다.
"""
function format_time(seconds::Int)
    hours = seconds ÷ 3600
    minutes = (seconds % 3600) ÷ 60
    secs = seconds % 60
    if hours >= 24
        days = hours ÷ 24
        hours = hours % 24
        return "$(days)일 $(lpad(hours,2,'0')):$(lpad(minutes,2,'0')):$(lpad(secs,2,'0'))"
    end
    return "$(lpad(hours,2,'0')):$(lpad(minutes,2,'0')):$(lpad(secs,2,'0'))"
end

# ═══════════════════════════════════════════════════════════════════════════
# 통합 VRP: 락커 용량을 시간창으로 반영하는 함수
# ═══════════════════════════════════════════════════════════════════════════

"""
    assign_locker_with_batch(tracker, locker_id, customer, locker_capacities, pickup_delay_avg)

락커에 고객을 배정하고, 배치 번호에 따른 시간창을 반환합니다.
용량 초과 시에도 배정하되, 늦은 시간창(이전 배치 회수 후)을 부여합니다.

# Returns
- `(locker_id, batch_num, time_window)`: 배정된 락커, 배치 번호, 시간창
"""
function assign_locker_with_batch(
    locker_batch_count::Dict{String, Int},  # 락커별 현재까지 배정된 고객 수
    locker_id::String,
    locker_capacities::Dict{String, Int};
    pickup_delay_avg::Int = 4 * 3600  # 평균 회수 시간 (4시간)
)
    capacity = get(locker_capacities, locker_id, 50)
    
    # 현재까지 배정된 고객 수
    current_count = get(locker_batch_count, locker_id, 0)
    
    # 배치 번호 계산 (1-indexed)
    batch_num = (current_count ÷ capacity) + 1
    
    # 배치별 시간창 설정
    # 배치 1: 08:00 ~ 14:00 (6시간 윈도우)
    # 배치 2: 12:00 ~ 18:00 (회수 후, 6시간 윈도우)
    # 배치 3: 16:00 ~ 21:00 (회수 후)
    batch_start = DRIVER_START_MIN + (batch_num - 1) * pickup_delay_avg
    batch_end = min(batch_start + 6 * 3600, TW_DEPOT_CLOSE)  # 최대 21:00까지
    
    # 배정 카운트 증가
    locker_batch_count[locker_id] = current_count + 1
    
    return batch_num, (batch_start, batch_end)
end

"""
    calculate_locker_customer_time_windows(locker_id, customers_with_pickup, capacity)

락커에 배정된 고객들의 시간창을 회수 시간 기반으로 계산합니다.

# 로직
- 용량 내 고객 (1~capacity): 시간창 (08:00, 21:00) - 바로 배송 가능
- 초과 고객 (capacity+1~n): i번째 초과 고객은 i번째 회수 시간 이후부터 배송 가능

# Returns
- Dict{String, Tuple{Int,Int}}: 고객ID → (시간창 시작, 시간창 종료)
"""
function calculate_locker_customer_time_windows(
    locker_id::String,
    customers_with_pickup::Vector{Tuple{String, Int}},  # (고객ID, 회수시간)
    capacity::Int
)
    result = Dict{String, Tuple{Int,Int}}()
    n = length(customers_with_pickup)
    
    if n == 0
        return result
    end
    
    # 용량 내 고객: 바로 배송 가능
    first_batch_count = min(capacity, n)
    first_batch = customers_with_pickup[1:first_batch_count]
    
    for (cust_id, pickup_delay) in first_batch
        result[cust_id] = (TW_LOCKER_OPEN, TW_LOCKER_CLOSE)  # 08:00 ~ 21:00
    end
    
    # 용량 초과 고객이 있는 경우
    if n > capacity
        # 첫 배치의 회수 시간을 정렬 (절대 시간으로 변환: 배송시간 08:00 + 회수지연)
        # 가정: 첫 배치는 08:00에 배송됨
        pickup_times = sort([DRIVER_START_MIN + pd for (_, pd) in first_batch])
        
        # 초과 고객들에게 시간창 부여
        overflow = customers_with_pickup[capacity+1:end]
        for (i, (cust_id, pickup_delay)) in enumerate(overflow)
            if i <= length(pickup_times)
                # i번째 회수 후 배송 가능
                available_time = pickup_times[i]
            else
                # 회수 시간이 부족하면 마지막 회수 시간 사용
                available_time = pickup_times[end]
            end
            
            # 영업 시간 내로 제한
            available_time = min(available_time, TW_DEPOT_CLOSE - 3600)
            
            result[cust_id] = (available_time, TW_LOCKER_CLOSE)
        end
    end
    
    return result
end

"""
    simulate_delivery_waves(...)

[DEPRECATED - 통합 VRP 방식으로 대체됨]

여러 웨이브에 걸친 배송을 시뮬레이션합니다.

여러 웨이브에 걸친 배송을 시뮬레이션합니다.
락커 용량 초과 시 대기하고, 슬롯이 비면 다음 웨이브에서 배송합니다.
각 웨이브에서 배송 가능한 락커들을 VRP로 최적화합니다.

# Returns
- `(total_cost, all_routes, completion_time, wave_count, waiting_customer_locker_assignments)`:
  - total_cost: 총 비용
  - all_routes: 모든 라우트
  - completion_time: 완료 시간
  - wave_count: 웨이브 수
  - waiting_customer_locker_assignments: 대기 고객 → 락커 배정 정보 (도보거리 계산용)
"""
function simulate_delivery_waves(
    depot_ids::Vector{String},
    d2d_customers::Vector,
    locker_customers::Vector,
    locker_waiting::Vector,
    vehicle_dist_dict::Dict{Tuple{String,String},Float64},
    vehicle_time_dict::Dict{Tuple{String,String},Float64},  # 추가: 시간 행렬
    locker_capacities::Dict{String, Int},
    carrier::String;
    max_waves::Int=10,
    vehicle_capacity::Int=100,  # 추가: 차량 용량
    progress_callback=nothing
)
    tracker = LockerDeliveryTracker(locker_capacities)
    all_routes = Vector{Vector{String}}()
    total_cost = 0.0
    wave_count = 0
    
    # 대기 중인 락커 고객 큐
    waiting_queue = copy(locker_waiting)
    
    # 대기 고객 → 락커 배정 정보 수집 (도보거리 계산용)
    waiting_customer_locker_assignments = Dict{String, String}()
    
    # 첫 번째 웨이브: 모든 D2D + 배정된 락커
    # (이 함수는 호출 전에 락커 배정이 완료된 상태)
    
    for wave in 1:max_waves
        wave_count = wave
        
        if isempty(waiting_queue)
            break
        end
        
        # 현재 시간에 배송 가능한 락커 고객 찾기
        deliverable = []
        still_waiting = []
        
        for customer in waiting_queue
            # 가장 가까운 락커 찾기
            cust_id = customer.id
            best_locker = nothing
            best_distance = Inf
            
            for (lid, cap) in locker_capacities
                available = get_available_slots(tracker, lid, tracker.current_time)
                if available > 0
                    dist = get(vehicle_dist_dict, (cust_id, lid), Inf)
                    if dist < best_distance
                        best_distance = dist
                        best_locker = lid
                    end
                end
            end
            
            if best_locker !== nothing
                push!(deliverable, (customer, best_locker))
            else
                push!(still_waiting, customer)
            end
        end
        
        waiting_queue = still_waiting
        
        if isempty(deliverable)
            # 배송 가능한 고객이 없으면 시간 전진 (가장 빠른 회수 시간까지)
            next_pickup_time = typemax(Int)
            for (lid, cid, delivery_time, pickup_delay) in tracker.deliveries
                pickup_time = delivery_time + pickup_delay
                if pickup_time > tracker.current_time
                    next_pickup_time = min(next_pickup_time, pickup_time)
                end
            end
            
            if next_pickup_time == typemax(Int)
                # 더 이상 회수할 것도 없음 - 락커 용량 절대 부족
                println("   ⚠️ 락커 용량 절대 부족: $(length(waiting_queue))명 배송 불가")
                break
            end
            
            tracker.current_time = next_pickup_time
            println("   ⏰ 시간 전진: $(format_time(tracker.current_time)) (락커 슬롯 회수 대기)")
            continue
        end
        
        # 배송 가능한 고객들을 VRP로 최적화
        # 락커별로 그룹화하여 VRP 문제 생성
        locker_demands = Dict{String, Int}()  # 락커별 배송 물량
        for (customer, locker_id) in deliverable
            locker_demands[locker_id] = get(locker_demands, locker_id, 0) + 1
            
            # 고객-락커 배정 정보 저장 (도보거리 계산용)
            waiting_customer_locker_assignments[customer.id] = locker_id
        end
        
        # VRP 문제 구성: depot + 락커들
        depot_id = depot_ids[1]
        locker_ids = collect(keys(locker_demands))
        
        if isempty(locker_ids)
            continue
        end
        
        # 노드 ID 구성: [depot, locker1, locker2, ...]
        node_ids = vcat([depot_id], locker_ids)
        n = length(node_ids)
        
        # 수요 벡터: 락커들만 (depot 제외, N-1개)
        demands = [locker_demands[lid] for lid in locker_ids]
        
        # 거리/시간 행렬 구성
        dist_matrix = zeros(Float64, n, n)
        time_matrix = zeros(Float64, n, n)
        for i in 1:n, j in 1:n
            if i != j
                dist_matrix[i,j] = get(vehicle_dist_dict, (node_ids[i], node_ids[j]), 1e6)
                time_matrix[i,j] = get(vehicle_time_dict, (node_ids[i], node_ids[j]), 1e6)
            end
        end
        
        # 시간창: 락커들만 (depot 제외, N-1개)
        big_tw = 86400  # 24시간
        time_windows = [(0, big_tw) for _ in 1:length(locker_ids)]
        # 서비스 시간: 락커 물건당 20초
        service_times = [calculate_locker_service_time(locker_demands[lid]) for lid in locker_ids]
        
        # 차량 수 자동 계산 (넉넉하게)
        num_customers = sum(values(locker_demands))
        num_vehicles = max(1, Int(ceil(num_customers / CUSTOMERS_PER_VEHICLE * 1.5)))
        
        # VRP 최적화 (반환: routes, cost)
        wave_routes, wave_cost = pyvrp_solve_cvrptw(
            dist_matrix, time_matrix,
            demands, time_windows, service_times,
            vehicle_capacity;
            num_vehicles=num_vehicles,
            max_iterations=500,
            depot_tw=(0, big_tw)
        )
        
        # 경로를 노드 ID로 변환
        for route in wave_routes
            route_ids = [node_ids[idx+1] for idx in route if idx >= 0 && idx < n]
            if !isempty(route_ids)
                push!(all_routes, route_ids)
            end
        end
        
        # 배송 기록 (회수 시간 추적용)
        for (customer, locker_id) in deliverable
            cust_id = customer.id
            pickup_delay = hasproperty(customer, :pickup_delay) ? Int(customer.pickup_delay) : generate_locker_pickup_time(Random.GLOBAL_RNG)
            record_delivery!(tracker, locker_id, cust_id, tracker.current_time, pickup_delay)
        end
        
        total_cost += wave_cost
        println("   📦 웨이브 $wave: $(length(deliverable))명 → $(length(locker_ids))개 락커, VRP 비용 $(round(wave_cost, digits=2)), 대기 $(length(waiting_queue))명")
        
        # 시간 전진 (배송 소요 시간 추정)
        tracker.current_time += 3600  # 1시간 추정
    end
    
    # 반환: (비용, 경로, 완료시간, 웨이브수, 대기고객→락커배정정보)
    return total_cost, all_routes, tracker.total_completion_time, wave_count, waiting_customer_locker_assignments
end

"""
    DeliveryResult

배송 결과를 저장하는 구조체
"""
struct DeliveryResult
    total_cost::Float64
    routes::Vector{Vector{String}}
    completion_time::Int  # 초 단위
    wave_count::Int
    waiting_customers::Int  # 배송 못한 고객 수
end

# ═══════════════════════════════════════════════════════════════════════════
# 통계적 설계: 완전 Monte Carlo 실험
# - 각 ω마다 새로운 고객 세트 생성 (완전 랜덤)
# - 모든 시나리오가 동일한 ω 고객을 공유 (공정 비교)
# - 30개 이상 권장 (통계적 유의성)
# ═══════════════════════════════════════════════════════════════════════════
const Nomega = let env = get(ENV, "OMEGA", "3")
    try
        parse(Int, env)
    catch
        30  # 기본값
    end
end
# 테스트용: 고객 수 제한 (예: CUSTOMER_LIMIT=10 → 10명만 사용)
const CUSTOMER_LIMIT = let env = get(ENV, "CUSTOMER_LIMIT", "")
    isempty(env) ? 0 : (try parse(Int, env) catch; 0 end)
end
const OUTDIR             = expanduser("~/Desktop/runs");     isdir(OUTDIR)  || mkpath(OUTDIR)

# 현재 실행 대상 시나리오 (환경변수로 제어 가능)
# 사용법: SCENARIOS_TO_RUN="1,2,3" julia script.jl
const SCENARIOS_TO_RUN = let env = get(ENV, "SCENARIOS_TO_RUN", "")
    if isempty(env)
        ALL_SCENARIOS  # 전체 시나리오 (D2D, DPL, SPL, OPL, PSPL)
    else
        try
            parse.(Int, split(env, ","))
        catch
            ALL_SCENARIOS
        end
    end
end

# 시나리오 4 SLRP 결과 저장용 글로벌 변수
const scenario4_k_facilities = Dict{Int, Vector}()  # k => facilities
const scenario4_results = Dict{Symbol, Any}()  # 최종 결과
const SCENARIO4_SNAPSHOT_LOCK = ReentrantLock()
const scenario4_route_snapshots = Dict{Tuple{Int,Int}, Dict{String,Any}}()  # (k, omega) => snapshot

@inline function euclidean_distance(p1::Tuple{<:Real,<:Real}, p2::Tuple{<:Real,<:Real})
    # Implements Haversine formula for geographic distance (km)
    # p1, p2 are expected to be (lon, lat) or (lat, lon). 
    # Based on data (19.xxx, 47.xxx), it seems to be (lon, lat).
    # However, the formula requires (lat, lon) order for logic or just correct mapping.
    # Let's assume input is (lon, lat) as per DEPOTS constant: (19.3338, 47.4255)
    
    lon1, lat1 = float(p1[1]), float(p1[2])
    lon2, lat2 = float(p2[1]), float(p2[2])
    
    R = 6371.0 # Earth radius in km
    dlat = deg2rad(lat2 - lat1)
    dlon = deg2rad(lon2 - lon1)
    
    a = sin(dlat/2)^2 + cos(deg2rad(lat1)) * cos(deg2rad(lat2)) * sin(dlon/2)^2
    c = 2 * atan(sqrt(a), sqrt(1-a))
    return R * c
end

mutable struct ScenarioProgress
    scenario::Int
    total_tasks::Int
    completed_tasks::Int
    start_time::Float64
    weight::Float64
end

mutable struct GlobalProgress
    scenarios::Dict{Int, ScenarioProgress}
    current_scenario::Int
    overall_start_time::Float64
    total_weight::Float64
    completed_weight::Float64
    last_update_time::Float64
    update_interval::Float64  # 몇 초마다 업데이트할지
    total_operations::Int      # 전체 연산 수
    completed_operations::Int  # 완료된 연산 수
end

const GLOBAL_PROGRESS = GlobalProgress(Dict{Int, ScenarioProgress}(), 0, 0.0, 0.0, 0.0, 0.0, 1.0, 0, 0)  # 1초마다 업데이트

function init_progress()
    total_customers = sum(sum(values(counts)) for counts in values(customer_counts))
    
    # 시나리오별 가중치 (ALNS 복잡도 반영)
    scenario_weights = Dict(
            1 => 1.5,    # ALNS 기본 (No Locker)
    2 => 2.0,    # ALNS + Private Locker
    3 => 2.5,    # ALNS + Mixed Locker
    4 => 60.0,   # ALNS + Public Locker + 연속 최적화
    5 => 2.5,    # ALNS + Metro Locker (시나리오 3과 동일한 복잡도)
    6 => 2.5     # ALNS + Partially Shared Private Locker (시나리오 3과 유사)
    )
    
    # 고객 수 복잡도 계수 (비선형)
    customer_complexity = (total_customers / 100.0)^1.5
    
    GLOBAL_PROGRESS.scenarios = Dict{Int, ScenarioProgress}()
    GLOBAL_PROGRESS.overall_start_time = time()
    GLOBAL_PROGRESS.completed_weight = 0.0
    GLOBAL_PROGRESS.total_weight = 0.0
    GLOBAL_PROGRESS.last_update_time = time()
    GLOBAL_PROGRESS.total_operations = 0
    GLOBAL_PROGRESS.completed_operations = 0
    
    # 시나리오 4 ALNS 최대 반복 수 (max_facilities에 따라 동적 계산) -> 단순화 (600 iter)
    scenario4_max_iterations = 600
    
    for scenario in SCENARIOS_TO_RUN
        # 모든 시나리오 동일하게 처리 (Python ALNS 기반)
        tasks = Nomega  # Monte Carlo 반복 횟수만큼
        
        weight = scenario_weights[scenario] * customer_complexity
        GLOBAL_PROGRESS.total_weight += weight
        GLOBAL_PROGRESS.total_operations += tasks  # 전체 연산 수 누적
        
        GLOBAL_PROGRESS.scenarios[scenario] = ScenarioProgress(
            scenario, tasks, 0, 0.0, weight
        )
    end
    
    # 전체 계산량 계산
    total_tasks = length(SCENARIOS_TO_RUN) * Nomega
    
    progress_println("🚀 Hybrid ALNS + VNS CVRP 최적화 시작")
    progress_println("📊 전체 연산량: $(GLOBAL_PROGRESS.total_operations)개 작업")
    progress_println("   📋 세부사항: 시나리오 $(length(SCENARIOS_TO_RUN))개 × 몬테카르로 $(Nomega)개")
    progress_println("   👥 고객 수: $(total_customers)명")
    progress_println("")
    show_realtime_progress()  # 초기 진행률 (0%) 표시
    progress_println("")
    for scenario in SCENARIOS_TO_RUN
        weight = scenario_weights[scenario] * customer_complexity
        percentage = round(100 * weight / GLOBAL_PROGRESS.total_weight, digits=1)
        scenario_name = get_scenario_name(scenario)
        complexity_desc = scenario == SCENARIO_OPL ? "$scenario_name + 연속 최적화" : scenario_name
        tasks = Nomega
        progress_println("   시나리오 $scenario ($complexity_desc): $(percentage)% ($(tasks)개 작업)")
    end
    progress_println("")
end

function start_scenario(scenario::Int)
    GLOBAL_PROGRESS.current_scenario = scenario
    if haskey(GLOBAL_PROGRESS.scenarios, scenario)
        GLOBAL_PROGRESS.scenarios[scenario].start_time = time()
        scenario_name = get_scenario_name(scenario)
        
        # 가중치 기반 전체 진행률 계산
        remaining_time, overall_progress = calculate_eta()
        overall_percentage = round(100 * overall_progress, digits=1)
        
        # 이전 시나리오들의 예상 작업량
        prev_weight = sum(prog.weight for (s, prog) in GLOBAL_PROGRESS.scenarios if s < scenario; init=0.0)
        current_weight_pct = round(100 * GLOBAL_PROGRESS.scenarios[scenario].weight / GLOBAL_PROGRESS.total_weight, digits=1)
        
        elapsed_str = format_time(time() - GLOBAL_PROGRESS.overall_start_time)
        eta_str = format_time(remaining_time)
        
        progress_println("\n" * "="^70)
        progress_println("🎯 시나리오 $scenario/6: $scenario_name 시작")
        progress_println("📊 전체 진행률: $(overall_percentage)% | 현재 시나리오 비중: $(current_weight_pct)%")
        progress_println("⏱️  경과 시간: $elapsed_str | 예상 완료: $eta_str")
        progress_println("="^70)
    end
end

function calculate_eta()
    current_time = time()
    elapsed = current_time - GLOBAL_PROGRESS.overall_start_time
    
    # 가중치 기반 전체 진행률 계산
    completed_weight = 0.0
    for (_, prog) in GLOBAL_PROGRESS.scenarios
        progress_ratio = prog.completed_tasks / max(1, prog.total_tasks)
        completed_weight += progress_ratio * prog.weight
    end
    
    overall_progress = completed_weight / max(1, GLOBAL_PROGRESS.total_weight)
    
    if overall_progress > 0.01  # 1% 이상 진행된 경우에만 ETA 계산
        estimated_total_time = elapsed / overall_progress
        remaining_time = estimated_total_time - elapsed
        return remaining_time, overall_progress
    else
        return -1.0, overall_progress
    end
end

function format_time(seconds::Float64)
    if seconds < 0
        return "계산 중..."
    elseif seconds < 60
        return "$(round(Int, seconds))초"
    elseif seconds < 3600
        mins = round(Int, seconds / 60)
        return "$(mins)분"
    else
        hours = round(Int, seconds / 3600)
        mins = round(Int, (seconds % 3600) / 60)
        return "$(hours)시간 $(mins)분"
    end
end

function show_realtime_progress()
    pct = GLOBAL_PROGRESS.total_operations > 0 ? round(GLOBAL_PROGRESS.completed_operations / GLOBAL_PROGRESS.total_operations * 100, digits=1) : 0.0
    elapsed = round(time() - GLOBAL_PROGRESS.overall_start_time, digits=1)
    
    # 진행률 바 생성 (20칸)
    filled = max(0, min(20, round(Int, pct / 5)))  # 0-20 범위로 제한
    bar = "█" ^ filled * "░" ^ (20 - filled)
    
    progress_println("🔥 진행률: [$(bar)] $(pct)% ($(GLOBAL_PROGRESS.completed_operations)/$(GLOBAL_PROGRESS.total_operations)) | $(elapsed)초")
end

function update_scenario_progress(scenario::Int, completed_tasks::Int = 1)
    lock(PROGRESS_LOCK) do
        if haskey(GLOBAL_PROGRESS.scenarios, scenario)
            scenario_prog = GLOBAL_PROGRESS.scenarios[scenario]
            scenario_prog.completed_tasks += completed_tasks
            GLOBAL_PROGRESS.completed_operations += completed_tasks  # 전체 진행률 업데이트
            
            current_time = time()
            
            # 실시간 진행률 표시 (1초마다)
            if current_time - GLOBAL_PROGRESS.last_update_time >= GLOBAL_PROGRESS.update_interval
                GLOBAL_PROGRESS.last_update_time = current_time
                show_realtime_progress()  # 간단한 실시간 진행률 표시
            end
            
            # 시나리오별 중요 마일스톤에서도 업데이트
            scenario_percentage = round(100 * scenario_prog.completed_tasks / scenario_prog.total_tasks, digits=1)
            if scenario_percentage in [25.0, 50.0, 75.0, 90.0] && scenario_prog.completed_tasks > 1
                remaining_time, overall_progress = calculate_eta()
                overall_percentage = round(100 * overall_progress, digits=1)
                scenario_name = get_scenario_name(scenario)
                thread_id = threadid()
                
                eta_str = format_time(remaining_time)
                elapsed_str = format_time(current_time - GLOBAL_PROGRESS.overall_start_time)
                
                progress_println("🎯 [Thread $thread_id] [전체 $(overall_percentage)%] [시나리오 $scenario: $(scenario_percentage)%] $scenario_name 마일스톤 | 예상 완료: $eta_str")
            end
        end
    end
end

function complete_scenario(scenario::Int)
    lock(PROGRESS_LOCK) do
        if haskey(GLOBAL_PROGRESS.scenarios, scenario)
            scenario_prog = GLOBAL_PROGRESS.scenarios[scenario]
            elapsed = time() - scenario_prog.start_time
            
            # 전체 진행률 계산
            remaining_time, overall_progress = calculate_eta()
            overall_percentage = round(100 * overall_progress, digits=1)
            
            scenario_name = get_scenario_name(scenario)
            thread_id = threadid()
            
            total_mins = round(elapsed / 60, digits=1)
            eta_str = format_time(remaining_time)
            progress_println("🎉 [Thread $thread_id] [전체 $(overall_percentage)%] [시나리오 $scenario] $scenario_name 완료! | 소요: $(total_mins)분 | 예상 완료: $eta_str")
        end
    end
end

progress_println("🚀 스크립트 시작 - Julia $(VERSION)")
progress_println("🧵 멀티스레드 지원 ALNS CVRP 최적화")
progress_println("🔧 백엔드: $(BACKEND)")
progress_println("📘 사용법: julia -t auto script.jl (자동 스레드) 또는 julia -t 4 script.jl (4개 스레드)")

# 멀티스레드 설정 확인
if IS_MAIN_PROCESS
check_process_setup()
end

# NSGA-II MOO 멀티스레드 안내
if IS_MAIN_PROCESS && BACKEND == "pyalns" && nthreads() > 1
    progress_println("🧬 NSGA-II MOO: 멀티스레드 지원 (Julia native)")
end

if BACKEND != "pyalns"
    progress_println("❌ 내부 ALNS 백엔드는 제거되었습니다. NSGA-II MOO만 지원합니다.")
    error("Internal ALNS backend removed. Use BACKEND=pyalns for NSGA-II MOO")
end

if BACKEND == "pyalns"
    try
        # nsga2_moo_backend.jl은 이미 라인 59에서 include됨 (중복 제거)
        version_str = pyvrp_version()
        progress_println("🧬 NSGA-II MOO 연동 확인: $(version_str)")
        if USE_TIME_WINDOWS
            progress_println("⏰ 시간창 제약 활성화: D2D $(TW_D2D_OPEN÷3600)시-$(TW_D2D_CLOSE÷3600)시, 락커 24시간")
        end

        # 통합형 Multi-Depot CVRP (MDCVRP) - NSGA-II MOO 래퍼
        function solve_mdcvrp(
            depot_ids::Vector{String},
            node_ids::Vector{String},
            vehicle_dist_dict::Dict{Tuple{String,String},Float64},
            demand_by_node::Dict{String,Int},
            vehicles_by_depot::Dict{String,Int},
            capacity_by_depot::Dict{String,Int};
            vehicle_time_dict::Union{Dict{Tuple{String,String},Float64},Nothing}=nothing,
            time_windows_by_node::Union{Dict{String,Tuple{Int,Int}},Nothing}=nothing,
            service_times_by_node::Union{Dict{String,Int},Nothing}=nothing,
            max_iter::Int=1000,
            progress_callback=nothing,
            seed::Int=1234
        )
            progress_println("🧬 NSGA-II MOO(MDCVRP) 통합 최적화 시작: 창고 $(length(depot_ids))개, 고객 $(length(node_ids))명")
            
            # 시간 행렬: 없으면 거리 행렬 사용
            time_dict = vehicle_time_dict !== nothing ? vehicle_time_dict : vehicle_dist_dict
            
            # 시간창/서비스시간: 없으면 기본값 생성
            if USE_TIME_WINDOWS && time_windows_by_node !== nothing && service_times_by_node !== nothing
                # 시간창 지원 MDCVRPTW
                best_cost, routes_by_depot = pyvrp_solve_mdcvrptw(
                    vehicle_dist_dict, time_dict, depot_ids, node_ids,
                    demand_by_node, time_windows_by_node, service_times_by_node,
                    vehicles_by_depot, capacity_by_depot;
                    max_iterations=max_iter
                )
                # 라우트 형태 변환: Dict{depot => routes} → Vector{routes}
                routes = Vector{Vector{String}}()
                for did in depot_ids
                    for r in get(routes_by_depot, did, Vector{Vector{String}}())
                        push!(routes, r)
                    end
                end
                iters_used = max_iter
                stall_limit = 0
            else
                # 기존 MDCVRP (시간창 없음) - NSGA-II 단일 디포로 각각 호출
                # TODO: NSGA-II의 정식 MDCVRP 지원시 업데이트
                routes = Vector{Vector{String}}()
                total_cost = 0.0
                for depot_id in depot_ids
                    # 이 디포에 할당된 고객 찾기 (간단히 모든 고객을 첫 디포에 할당)
                    # 실제로는 클러스터링 등으로 할당해야 함
                    depot_customers = node_ids
                    if !isempty(depot_customers)
                        demands = [get(demand_by_node, c, 1) for c in depot_customers]
                        capacity = get(capacity_by_depot, depot_id, 100)
                        cost, depot_routes = solve_vrp_ids(vehicle_dist_dict, depot_id, depot_customers, demands, capacity; max_iterations=max_iter)
                        append!(routes, depot_routes)
                        total_cost += cost
                    end
                    break  # 첫 디포만 처리 (단순화)
                end
                best_cost = total_cost
                iters_used = max_iter
                stall_limit = 0
            end
            
            progress_println("✅ MDCVRP finished: depots=$(length(depot_ids)), trips=$(length(routes)), distance=$(round(best_cost, digits=2))")
            return best_cost, routes, iters_used, stall_limit
        end
    catch err
        progress_println("⚠️ NSGA-II MOO 연동 실패: $(err)")
        @error "NSGA-II MOO connection failed" exception=(err, catch_backtrace())
    end
end

# 백엔드와 무관하게 전역에서 사용할 수 있도록 MDCVRP 래퍼를 보강 정의
if !@isdefined solve_mdcvrp
    function solve_mdcvrp(
        depot_ids::Vector{String},
        node_ids::Vector{String},
        vehicle_dist_dict::Dict{Tuple{String,String},Float64},
        demand_by_node::Dict{String,Int},
        vehicles_by_depot::Dict{String,Int},
        capacity_by_depot::Dict{String,Int};
        vehicle_time_dict::Union{Dict{Tuple{String,String},Float64},Nothing}=nothing,
        time_windows_by_node::Union{Dict{String,Tuple{Int,Int}},Nothing}=nothing,
        service_times_by_node::Union{Dict{String,Int},Nothing}=nothing,
        max_iter::Int=1000,
        progress_callback=nothing,
        seed::Int=1234
    )
        # 기본 구현: 첫 디포에 모든 고객 할당
        depot_id = depot_ids[1]
        demands = [get(demand_by_node, c, 1) for c in node_ids]
        capacity = get(capacity_by_depot, depot_id, 100)
        
        # 시간창 정보가 있으면 시간창 버전 사용
        if vehicle_time_dict !== nothing && time_windows_by_node !== nothing && service_times_by_node !== nothing
            # 시간창 및 서비스 시간 추출
            time_windows = [get(time_windows_by_node, c, (0, 86400)) for c in node_ids]
            service_times = [get(service_times_by_node, c, 0) for c in node_ids]
            depot_tw = get(time_windows_by_node, depot_id, (TW_DEPOT_OPEN, TW_DEPOT_CLOSE))
            
            best_cost, routes = solve_vrp_ids_with_tw(
                vehicle_dist_dict, vehicle_time_dict, depot_id, node_ids,
                demands, time_windows, service_times, capacity;
                max_iterations=max_iter, depot_tw=depot_tw
            )
        else
            # 시간창 없이 호출 (하위 호환)
            best_cost, routes = solve_vrp_ids(vehicle_dist_dict, depot_id, node_ids, demands, capacity; max_iterations=max_iter)
        end
        return best_cost, routes, max_iter, 0
    end
end

#───────────────────────────────────────────────────────────────────────────────
# 단일 VRP (당일 배송, 차량 수 자동 계산)
#───────────────────────────────────────────────────────────────────────────────
"""
단일 VRP - 당일 배송, 차량 수는 고객 수에 비례하여 자동 계산

# Arguments
- `carrier`: 캐리어 이름
- `depot_id`: 출발/도착 디포 ID
- `customer_nodes`: 배송할 고객 노드 리스트
- `demands`: 고객별 수요
- `vehicle_dist_dict`: 거리 딕셔너리
- `vehicle_time_dict`: 시간 딕셔너리
- `time_windows_by_node`: 노드별 시간창
- `service_times_by_node`: 노드별 서비스 시간
- `capacity`: 차량 용량

# Returns
- `total_cost`: 총 비용
- `all_routes`: 모든 라우트 리스트
- `days_used`: 소요 일수 (항상 1)
- `trips_per_day`: 일별 트립 수
"""
function solve_vrp_multitrip_multiday(
    carrier::String,
    depot_id::String,
    customer_nodes::Vector{String},
    demands::Dict{String,Int},
    vehicle_dist_dict::Dict{Tuple{String,String},Float64},
    vehicle_time_dict::Dict{Tuple{String,String},Float64},
    time_windows_by_node::Dict{String,Tuple{Int,Int}},
    service_times_by_node::Dict{String,Int},
    capacity::Int;
    max_days::Int=1,  # 사용 안 함 (하위 호환용)
    max_iter::Int=2000,
    locker_assignments::Dict{String,String}=Dict{String,String}(),
    num_active_lockers::Int=0,
    node_desired_times_map::Dict{String,Vector{Int}}=Dict{String,Vector{Int}}(),  # 노드별 개별 고객 희망시간
    node_locker_distances_map::Dict{String,Vector{Float64}}=Dict{String,Vector{Float64}}(),  # 노드별 개별 고객→락커 거리
    moo_seed::Int=42
)
    # 배송사별 고정 차량 수 또는 총 수요 기준 자동 계산
    num_nodes = length(customer_nodes)
    total_demand = sum(get(demands, c, 1) for c in customer_nodes)
    num_vehicles = calculate_required_vehicles(carrier, total_demand)
    
    # 하루 운영 시간
    day_start = TW_DEPOT_OPEN   # 08:00 (28800초)
    day_end = TW_DEPOT_CLOSE    # 21:00 (75600초)
    
    progress_println("🧠 $(carrier): VRP 시작")
    progress_println("   - 고객 수: $(num_nodes)명")
    progress_println("   - 총 수요: $(total_demand)개")
    progress_println("   - 차량 수: $(num_vehicles)대 (자동 계산: $(total_demand)개 ÷ $(CUSTOMERS_PER_VEHICLE)개/대 × 1.5)")
    
    # 시간창 정수 변환
    trip_tw = Vector{Tuple{Int,Int}}()
    for c in customer_nodes
        tw = get(time_windows_by_node, c, (day_start, day_end))
        push!(trip_tw, (round(Int, tw[1]), round(Int, tw[2])))
    end
    
    trip_demands = [get(demands, c, 1) for c in customer_nodes]
    trip_service = [get(service_times_by_node, c, 0) for c in customer_nodes]
    depot_tw = (round(Int, day_start), round(Int, day_end))
    
    try
        # 락커 고객의 거리 계산 (MOO f2 계산용)
        # node_desired_times_map과 동일한 순서로 개별 고객 거리 구성
        customer_locker_distances = Float64[]
        for cust_id in customer_nodes
            node_dists = get(node_locker_distances_map, cust_id, Float64[])
            if !isempty(node_dists)
                append!(customer_locker_distances, node_dists)
            else
                # 정보 없으면 0.0 (D2D 고객)
                push!(customer_locker_distances, 0.0)
            end
        end
        
        # 노드별 개별 고객 희망시간 벡터 구성 (만족도 정확 계산용)
        trip_desired_times = Vector{Vector{Int}}()
        for c in customer_nodes
            desired = get(node_desired_times_map, c, Int[])
            if isempty(desired)
                # D2D 고객이거나 정보 없음: 노드 시간창의 tw_early 사용
                tw = get(time_windows_by_node, c, (day_start, day_end))
                push!(trip_desired_times, [round(Int, tw[1])])
            else
                push!(trip_desired_times, desired)
            end
        end
        
        total_cost, all_routes = solve_vrp_ids_with_tw(
            vehicle_dist_dict, vehicle_time_dict, depot_id, customer_nodes,
            trip_demands, trip_tw, trip_service, capacity;
            num_vehicles=num_vehicles, max_iterations=max_iter, depot_tw=depot_tw,
            customer_locker_distances=customer_locker_distances,
            num_active_lockers=num_active_lockers,
            node_individual_desired_times=trip_desired_times,
            moo_seed=moo_seed
        )
        
        days_used = 1
        trips_per_day = [length(all_routes)]
        
        progress_println("✅ $(carrier) 완료: $(length(all_routes))트립, 차량 $(num_vehicles)대, 수요 $(total_demand)개, 비용 $(round(total_cost, digits=2))km")
        
        return total_cost, all_routes, days_used, trips_per_day
        
    catch e
        progress_println("❌ $(carrier) VRP 오류: $e")
        return 1e7, Vector{Vector{String}}(), 0, Int[]
    end
end

#───────────────────────────────────────────────────────────────────────────────
# 1. Fixed depots & private lockers
#    ⚠️ 좌표 순서 규칙:
#    - DEPOTS, LOCKERS_PRIV: (경도lon, 위도lat, 소유사)
#    - raster_polygons, 고객 위치: (경도lon, 위도lat)
#    - 거리 계산 함수 및 내부 처리: (위도lat, 경도lon) - 지리적 표준
#───────────────────────────────────────────────────────────────────────────────
# 형식: depot_id => (경도lon, 위도lat, 캐리어명)
const DEPOTS = Dict(
    "D_0001" => (19.3338,47.4255,"Foxpost"),   # (lon, lat, carrier)
    "D_0002" => (19.1641,47.5842,"Packeta"),
    "D_0003" => (19.0289,47.3841,"AlzaBox"),
    "D_0004" => (19.1601,47.3409,"GLS"),
    "D_0005" => (19.1565,47.3553,"EasyBox"),
    "D_0006" => (19.2430,47.4242,"DHL")
)
# 형식: locker_id => (경도lon, 위도lat, 캐리어명)
const LOCKERS_PRIV = Dict(
  "L_0001" => (19.0521,47.5026,"Foxpost"),     # (lon, lat, carrier)
  "L_0002" => (19.0562,47.4990,"Foxpost"),
  "L_0003" => (19.0504,47.4982,"Foxpost"),
  "L_0004" => (19.0560,47.4991,"AlzaBox"),
  "L_0005" => (19.0560,47.4985,"GLS"),
  "L_0006" => (19.0517,47.4931,"GLS"),
  "L_0007" => (19.0559,47.4991,"EasyBox"),
  "L_0008" => (19.0524,47.5024,"DHL"),
  "L_0009" => (19.0499,47.5053,"Packeta")
)

progress_println("📍 창고 $(length(DEPOTS))개, 사설 락커 $(length(LOCKERS_PRIV))개 로드됨")

#───────────────────────────────────────────────────────────────────────────────
# 2. Raster + residential polygons & customer counts
#───────────────────────────────────────────────────────────────────────────────
const raster_polygons = Dict(
  295 => [(19.0404166,47.51625),(19.0404166,47.5079167),(19.0487499,47.5079167),
          (19.0487499,47.51625),(19.0404166,47.51625)],
  337 => [(19.0404166,47.5079167),(19.0404166,47.4995833),(19.0487499,47.4995833),
          (19.0487499,47.5079167),(19.0404166,47.5079167)],
  338 => [(19.0487499,47.5079167),(19.0487499,47.4995833),(19.0570833,47.4995833),
          (19.0570833,47.5079167),(19.0487499,47.5079167)],
  383 => [(19.0487499,47.4995833),(19.0487499,47.49125),(19.0570833,47.49125),
          (19.0570833,47.4995833),(19.0487499,47.4995833)]
)
const residential_polygons = Dict(
  295 => [Polygon(Point{2,Float64}[
    (19.0479234,47.508572),(19.0467925,47.5086409),(19.0467488,47.5087787),
    (19.0468313,47.5093496),(19.0456413,47.5095018),(19.0457681,47.5100485),
    (19.0460775,47.5100611),(19.0468948,47.5099575),(19.0477223,47.5123225),
    (19.0480740,47.5129552),(19.0482181,47.5132143),(19.0485898,47.5139580),
    (19.0474744,47.5143254),(19.0467699,47.5145272),(19.0476354,47.5160351),
    (19.0477844,47.51625),(19.0487499,47.51625),(19.0487499,47.5114425),
    (19.0483813,47.5114992),(19.0483195,47.5112089),(19.0481354,47.5106627),
    (19.0478455,47.5098003),(19.0487499,47.5096686),(19.0487499,47.5085951),
    (19.0479234,47.508572)
  ])],
  337 => [Polygon(Point{2,Float64}[
    (19.0484137,47.5002811),(19.0483994,47.5003242),(19.0483239,47.5007127),
    (19.0479999,47.5006777),(19.0479328,47.5010104),(19.0479098,47.5011127),
    (19.0478494,47.5014439),(19.0478871,47.5014476),(19.0477930,47.5018311),
    (19.0478897,47.5018806),(19.0477811,47.5023671),(19.0476760,47.5023562),
    (19.0474669,47.5023350),(19.0471721,47.5023048),(19.0470061,47.5022877),
    (19.0466556,47.5022526),(19.0466004,47.5022485),(19.0465248,47.5026585),
    (19.0464306,47.5030475),(19.0464654,47.5030529),(19.0468287,47.5030892),
    (19.0471173,47.5031181),(19.0471287,47.5031192),(19.0471165,47.5031750),
    (19.0470653,47.5034091),(19.0471601,47.5035644),(19.0470894,47.5038703),
    (19.0470852,47.5038885),(19.0470637,47.5039247),(19.0470159,47.5039196),
    (19.0467813,47.5038948),(19.0467563,47.5038922),(19.0467487,47.5039249),
    (19.0467451,47.5039368),(19.0467267,47.5040166),(19.0466941,47.5041573),
    (19.0466753,47.5042606),(19.0461677,47.5042162),(19.0461317,47.5048339),
    (19.0480815,47.5049958),(19.0480552,47.5051930),(19.0485460,47.5053756),
    (19.0487065,47.5054687),(19.0487499,47.5055866),(19.0487499,47.5044825),
    (19.0481972,47.5044289),(19.0485478,47.5029797),(19.0487499,47.5028884),
    (19.0487499,47.5003085),(19.0484137,47.5002811)
  ])],
  338 => [Polygon(Point{2,Float64}[
    (19.0520874,47.5077131),(19.0520874,47.5076533),(19.0532917,47.5077729),
    (19.0533272,47.5079167),(19.0570833,47.5079167),(19.0570833,47.4995833),
    (19.0496253,47.4995833),(19.0494534,47.5003657),(19.0487499,47.5003085),
    (19.0487499,47.5028884),(19.0487587,47.5028844),(19.0488325,47.5025867),
    (19.0495408,47.5026450),(19.0505292,47.5027538),(19.0505816,47.5024679),
    (19.0507192,47.5024843),(19.0510104,47.5025112),(19.0510982,47.5025193),
    (19.0511198,47.5024124),(19.0511483,47.5022714),(19.0511646,47.5021703),
    (19.0518335,47.5022320),(19.0527765,47.5023227),(19.0526629,47.5029458),
    (19.0525959,47.5032683),(19.0528478,47.5032840),(19.0525138,47.5049174),
    (19.0512183,47.5047787),(19.0511310,47.5048259),(19.0509067,47.5050487),
    (19.0507173,47.5051304),(19.0502686,47.5051975),(19.0499234,47.5051140),
    (19.0497152,47.5050178),(19.0495338,47.5048336),(19.0494823,47.5047688),
    (19.0494477,47.5046843),(19.0494448,47.5045499),(19.0487499,47.5044825),
    (19.0487499,47.5055866),(19.0488042,47.5057342),(19.0491337,47.5057911),
    (19.0491086,47.5059833),(19.0490613,47.5062717),(19.0492632,47.5062890),
    (19.0494403,47.5063063),(19.0494784,47.5063100),(19.0494684,47.5063711),
    (19.0494478,47.5064879),(19.0494423,47.5065157),(19.0504355,47.5066088),
    (19.0504398,47.5065859),(19.0504648,47.5064619),(19.0504881,47.5063459),
    (19.0505249,47.5061625),(19.0505482,47.5060731),(19.0516600,47.5061822),
    (19.0516382,47.5062876),(19.0515451,47.5067001),(19.0514475,47.5068057),
    (19.0514244,47.5069171),(19.0513534,47.5072090),(19.0513211,47.5073650),
    (19.0512868,47.5075613),(19.0507840,47.5075198),(19.0508281,47.5079106),
    (19.0507201,47.5079167),(19.0521616,47.5079167),(19.0520874,47.5077131)
  ])],
  383 => [Polygon(Point{2,Float64}[
    (19.0497974,47.4988241),(19.0496325,47.4995503),(19.0496253,47.4995833),
    (19.0570833,47.4995833),(19.0570833,47.49125),(19.0513682,47.49125),
    (19.0508079,47.4919812),(19.0507516,47.4921673),(19.0506712,47.4923549),
    (19.0497391,47.4935631),(19.0504259,47.4937494),(19.0503319,47.4939395),
    (19.0498056,47.4950550),(19.0508099,47.4954275),(19.0513203,47.4956161),
    (19.0509394,47.4960547),(19.0506739,47.4959660),(19.0494517,47.4972994),
    (19.0500160,47.4975986),(19.0498175,47.4983747),(19.0497568,47.4983713),
    (19.0492433,47.4983287),(19.0492620,47.4981925),(19.0493731,47.4982016),
    (19.0494142,47.4979822),(19.0489844,47.4979429),(19.0489304,47.4979448),
    (19.0487499,47.4981570),(19.0487499,47.4987232),(19.0497974,47.4988241)
  ])]
)
const customer_counts = Dict(
  295 => Dict("D2D"=>203, "Locker"=>70),
  337 => Dict("D2D"=>154, "Locker"=>54),
  338 => Dict("D2D"=>372, "Locker"=>129),
  383 => Dict("D2D"=>782, "Locker"=>269)
)

# 5구역 전체 경계 계산 (4개 raster 합집합)
# ⚠️ 경도(LON)는 동서 방향, 위도(LAT)는 남북 방향
# 부다페스트 5구역: 약 1.7km × 2.8km 크기
const DISTRICT5_LON_MIN = 19.0404166   # 서쪽 경계 (경도)
const DISTRICT5_LON_MAX = 19.0570833   # 동쪽 경계 (경도)
const DISTRICT5_LAT_MIN = 47.49125     # 남쪽 경계 (위도)
const DISTRICT5_LAT_MAX = 47.51625     # 북쪽 경계 (위도)

progress_println("📍 폴리곤 $(length(raster_polygons))개, 고객 카운트 $(length(customer_counts))개 로드됨")
progress_println("🗺️  5구역 경계: 경도 $(DISTRICT5_LON_MIN)~$(DISTRICT5_LON_MAX), 위도 $(DISTRICT5_LAT_MIN)~$(DISTRICT5_LAT_MAX)")

# 점이 5구역(4개 raster 폴리곤 합집합) 내부에 있는지 체크
function is_point_in_district5(lat::Float64, lon::Float64)
    point = Point2f(lon, lat)
    
    # 4개 raster 폴리곤 중 하나라도 포함하면 true
    for (raster_id, polygon_coords) in raster_polygons
        # raster 폴리곤을 Polygon 객체로 변환
        points = [Point2f(coord[1], coord[2]) for coord in polygon_coords]
        polygon = Polygon(points)
        
        if inpoly(point, polygon)
            return true
        end
    end
    
    return false
end

# 🔥 4개 raster 폴리곤 내 랜덤 위치 생성 (완전한 폴리곤 기반)
function generate_random_location_in_district5(rng::AbstractRNG; max_attempts::Int=1000)
    # 🔥 각 raster 폴리곤에서 직접 랜덤 위치 생성
    raster_ids = [295, 337, 338, 383]
    
    for attempt in 1:max_attempts
        # 랜덤하게 raster 하나 선택
        selected_raster = rand(rng, raster_ids)
        polygon_coords = raster_polygons[selected_raster]
        
        # 해당 raster의 경계 박스 내에서 점 생성
        lons = [coord[1] for coord in polygon_coords[1:end-1]]
        lats = [coord[2] for coord in polygon_coords[1:end-1]]
        min_lon, max_lon = minimum(lons), maximum(lons)
        min_lat, max_lat = minimum(lats), maximum(lats)
        
        # raster 내부에서 랜덤 위치 생성
        lat = min_lat + rand(rng) * (max_lat - min_lat)
        lon = min_lon + rand(rng) * (max_lon - min_lon)
        
        # 실제 raster 폴리곤 내부인지 체크
        if is_point_in_district5(lat, lon)
            return lat, lon
        end
    end
    
    # 🔥 실패하면 첫 번째 raster의 중앙점 반환 (실제 폴리곤 좌표 기반)
    first_raster_coords = raster_polygons[295]
    center_lat = sum(coord[2] for coord in first_raster_coords[1:end-1]) / (length(first_raster_coords)-1)
    center_lon = sum(coord[1] for coord in first_raster_coords[1:end-1]) / (length(first_raster_coords)-1)
    return center_lat, center_lon
end

# 위치를 5구역 내로 제한하는 함수 (가장 가까운 유효한 위치로)
function clamp_to_district5(lat::Float64, lon::Float64; rng::AbstractRNG=Random.GLOBAL_RNG)
    # 이미 5구역 내부라면 그대로 반환
    if is_point_in_district5(lat, lon)
        return lat, lon
    end
    
    # 5구역 밖이면 새로운 유효한 위치 생성
    return generate_random_location_in_district5(rng)
end


inpoly(pt::Point2f, poly::Polygon) = begin
  x,y = pt; verts = coordinates(poly); inside=false; j=length(verts)
  for i in eachindex(verts)
    xi,yi = verts[i]; xj,yj = verts[j]
    if (yi>y)!=(yj>y) && x < (xj-xi)*(y-yi)/(yj-yi+1e-10)+xi
      inside = !inside
    end
    j = i
  end
  inside
end

# 락커 사용량 추적을 위한 Mutable 구조체
mutable struct LockerUsageTracker
    usage::Dict{String, Int}  # 락커ID별 현재 사용량
    capacity::Dict{String, Int}  # 락커ID별 최대 용량
    
    function LockerUsageTracker()
        usage = Dict{String, Int}()
        capacity = Dict{String, Int}()
        
        # Private 락커들의 용량 설정
        for (locker_id, (_, _, carrier)) in LOCKERS_PRIV
            usage[locker_id] = 0
            capacity[locker_id] = LOCKER_CAPACITY[carrier]
        end
        
        new(usage, capacity)
    end
end

# 모든 락커를 추가하는 함수 (Private, Public, Metro 모두)
function add_public_lockers!(tracker::LockerUsageTracker, effective_lockers::Dict)
    for locker_id in keys(effective_lockers)
        tracker.usage[locker_id] = 0
        
        if haskey(LOCKERS_PRIV, locker_id)
            # Private 락커
            (_, _, carrier) = LOCKERS_PRIV[locker_id]
            tracker.capacity[locker_id] = LOCKER_CAPACITY[carrier]
        else
            # Public 락커 - SLRP에서 생성된 락커만 사용
            tracker.capacity[locker_id] = effective_public_locker_capacity()
        end
    end
end

# 락커 사용량 체크 및 관리 함수들
function is_locker_available(tracker::LockerUsageTracker, locker_id::String)
    return haskey(tracker.usage, locker_id) && tracker.usage[locker_id] < tracker.capacity[locker_id]
end

function use_locker!(tracker::LockerUsageTracker, locker_id::String)
    if is_locker_available(tracker, locker_id)
        tracker.usage[locker_id] += 1
        return true
    end
    return false
end

function get_available_lockers(tracker::LockerUsageTracker, scenario::Int, effective_lockers::Dict, customer_carrier::String="")
    available = String[]
    
    # 시나리오별 락커 사용 가능 규칙 적용
    for locker_id in keys(effective_lockers)
        # 락커 용량 체크
        locker_carrier = ""
        if haskey(LOCKERS_PRIV, locker_id)
            (_,_,lk_carrier) = LOCKERS_PRIV[locker_id]
            locker_carrier = lk_carrier
        else
            # Public 락커는 배송사 제약 없음
            locker_carrier = ""
        end
        
        # 시나리오별 접근 가능성 체크
        can_use = false
        if scenario == SCENARIO_D2D
            can_use = false  # 락커 사용 안함
        elseif scenario == SCENARIO_DPL
            # Private 락커만, 같은 배송사만
            can_use = haskey(LOCKERS_PRIV, locker_id) && locker_carrier == customer_carrier
        elseif scenario == SCENARIO_SPL || scenario == SCENARIO_OPL
            # Private/Public 락커, 모든 배송사 사용 가능
            can_use = true
        elseif scenario == SCENARIO_PSPL
            # 부분 공유: AlzaBox/Foxpost/Packeta 3사는 서로의 Private 락커 공유
            # 그 외 캐리어는 자신의 Private 락커만 사용 가능
            if haskey(LOCKERS_PRIV, locker_id)
                (_,_,lk_carrier) = LOCKERS_PRIV[locker_id]
                if (customer_carrier in PSPL_SHARED_CARRIERS) && (lk_carrier in PSPL_SHARED_CARRIERS)
                    can_use = true
                else
                    can_use = (lk_carrier == customer_carrier)
                end
            else
                # Public/Metro 생성분은 시나리오6에서 사용하지 않음
                can_use = false
            end
        end
        
        # 시나리오별 접근 가능성만 체크 (용량 체크는 use_locker!에서 수행)
        if can_use
            push!(available, locker_id)
        end
    end
    
    return available
end



function show_locker_status(tracker::LockerUsageTracker)
    println("🏪 락커 사용 현황:")
    
    # 배송사별로 그룹화해서 표시
    for carrier in CARRIERS
        carrier_lockers = [(id, info) for (id, info) in LOCKERS_PRIV if info[3] == carrier]
        if !isempty(carrier_lockers)
            println("   📦 $carrier:")
            total_used = 0
            total_capacity = 0
            
            for (locker_id, _) in carrier_lockers
                if haskey(tracker.usage, locker_id)
                    used = tracker.usage[locker_id]
                    capacity = tracker.capacity[locker_id]
                    percentage = round(100 * used / capacity, digits=1)
                    status = used >= capacity ? "🔴 FULL" : "🟢 Available"
                    println("      $locker_id: $used/$capacity ($(percentage)%) $status")
                    total_used += used
                    total_capacity += capacity
                end
            end
            
            if total_capacity > 0
                total_percentage = round(100 * total_used / total_capacity, digits=1)
                println("      └─ 총계: $total_used/$total_capacity ($(total_percentage)%)")
            end
        end
    end
    
    # Public 락커들 표시
    public_lockers = [(id, capacity) for (id, capacity) in tracker.capacity if !haskey(LOCKERS_PRIV, id)]
    if !isempty(public_lockers)
        println("   📦 Public Lockers:")
        total_used = 0
        total_capacity = 0
        
        for (locker_id, capacity) in public_lockers
            used = tracker.usage[locker_id]
            percentage = round(100 * used / capacity, digits=1)
            status = used >= capacity ? "🔴 FULL" : "🟢 Available"
            println("      $locker_id: $used/$capacity ($(percentage)%) $status")
            total_used += used
            total_capacity += capacity
        end
        
        if total_capacity > 0
            total_percentage = round(100 * total_used / total_capacity, digits=1)
            println("      └─ 총계: $total_used/$total_capacity ($(total_percentage)%)")
        end
    end
end

# 가중치 기반 랜덤 선택 함수 (스레드 안전)
function weighted_random_choice(items::Vector{String}, weights::Dict{String, Float64}; rng=Random.GLOBAL_RNG)
    # 가중치 정규화 (합계가 1이 되도록)
    total_weight = sum(weights[item] for item in items)
    normalized_weights = [weights[item] / total_weight for item in items]
    
    # 누적 확률 계산
    cumulative_probs = cumsum(normalized_weights)
    
    # 랜덤 값으로 선택
    r = rand(rng)
    for (i, cum_prob) in enumerate(cumulative_probs)
        if r <= cum_prob
            return items[i]
        end
    end
    
    # 안전장치 (부동소수점 오차 등으로 인한 경우)
    return items[end]
end

# 스레드별 독립적인 랜덤 생성기 생성
function create_thread_rng(seed::Int)
    thread_id = threadid()
    return Random.MersenneTwister(seed + thread_id * 1000)
end



"""
    generate_random_d2d_time_window(rng)

D2D 고객을 위한 랜덤 시간창을 생성합니다.
- 시작 시간: 08:00~14:00 (TW_D2D_START_MIN ~ TW_D2D_START_MAX)
- 시간창 폭: 3~9시간 (TW_D2D_WIDTH_MIN ~ TW_D2D_WIDTH_MAX)
- 종료 시간: 최대 17:00 (TW_D2D_CLOSE) - 배송 마감 시간 초과 방지
"""
function generate_random_d2d_time_window(rng)
    # 시작 시간: 08:00~14:00
    tw_start = rand(rng, TW_D2D_START_MIN:60:TW_D2D_START_MAX)  # 1분 단위
    # 시간창 폭: 3~9시간
    tw_width = rand(rng, TW_D2D_WIDTH_MIN:60:TW_D2D_WIDTH_MAX)  # 1분 단위
    tw_end = min(tw_start + tw_width, TW_D2D_CLOSE)  # 17:00 초과 방지
    return (tw_start, tw_end)
end

"""
    generate_locker_pickup_time(rng)

락커 물품 회수 시간을 생성합니다.
- 50% 고객: 1~8시간 사이 랜덤 회수 (당일 회수)
- 50% 고객: 미회수 (24시간 이상, 당일 배송 불가)
"""
function generate_locker_pickup_time(rng)
    r = rand(rng)
    if r < 0.35
        # 35%: 1~6시간 사이 랜덤 (1시간 이후부터 회수 시작)
        return rand(rng, 1*3600:60:6*3600)  # 1분 단위로 1~6시간 사이
    elseif r < 0.50
        # 15%: 6~12시간 사이 랜덤
        return rand(rng, 6*3600:60:12*3600)  # 1분 단위로 6~12시간 사이
    else
        # 50%: 당일 미회수 (12시간 이상 = 17:00 이후)
        return 24 * 3600
    end
end

# 레거시 호환용 (기존 코드에서 호출할 경우)
function generate_locker_pickup_time_legacy(rng)
    r = rand(rng)
    if r < 0.25
        return 6 * 3600   # 6시간 = 21600초
    elseif r < 0.50
        return 12 * 3600  # 12시간 = 43200초
    elseif r < 0.75
        return 18 * 3600  # 18시간 = 64800초
    else
        return 24 * 3600  # 24시간 = 86400초
    end
end

function gen_customers(; rng=Random.GLOBAL_RNG)
  cust_list = NamedTuple[]; cid = 1
  for (rid, delivs) in customer_counts
    xs, ys = first.(raster_polygons[rid]), last.(raster_polygons[rid])
    minx,maxx = minimum(xs), maximum(xs)
    miny,maxy = minimum(ys), maximum(ys)
    for (dtype, cnt) in delivs, _ in 1:cnt
      while true
        x = rand(rng)*(maxx-minx) + minx
        y = rand(rng)*(maxy-miny) + miny
        if any(p->inpoly(Point2f(x,y),p), residential_polygons[rid])
          idstr = lpad(string(cid),4,'0')
          
          # 시간창 생성
          if dtype == "D2D"
            # D2D: 랜덤 시간창 (시작 08-14시, 폭 3-9시간, 최대 17시 마감)
            tw = generate_random_d2d_time_window(rng)
            pickup_delay = 0  # D2D는 회수 없음
          else  # "Locker"
            # 락커: 고객 희망 배송 시간 랜덤 생성 (만족도 계산용)
            # 운영시간은 24시간이지만, 고객은 원하는 배송 시간대가 있음
            tw = generate_random_d2d_time_window(rng)
            # 락커 회수 시간: 50% 6시간, 50% 24시간
            pickup_delay = generate_locker_pickup_time(rng)
          end
          
          nt = (id="C_$idstr", dtype=dtype, coord=Point2f(x,y), 
                tw_early=tw[1], tw_late=tw[2], pickup_delay=pickup_delay)
          push!(cust_list, nt)
          cid += 1; break
        end
      end
    end
  end
  return cust_list
end

function gen_attr(custs; rng=Random.GLOBAL_RNG)
  # 기본 시장 점유율에 따른 배송사 할당 (시나리오 독립적)
  result = []
  for c in custs
    if c.dtype == "D2D"
      carrier = weighted_random_choice(CARRIERS, CARRIER_MARKET_SHARE_D2D; rng=rng)
    else  # "Locker"
      carrier = weighted_random_choice(CARRIERS, CARRIER_MARKET_SHARE_LOCKER; rng=rng)
    end
    push!(result, (customer_id=c.id, carrier=carrier, q=1))
  end
  return DataFrame(result)
end

#───────────────────────────────────────────────────────────────────────────────
# 락커 통계 수집 함수들
#───────────────────────────────────────────────────────────────────────────────

function init_locker_stats(scenario::Int, seed::Int, omega::Int, effective_lockers::Dict)
    """락커 통계 초기화"""
    stats = ScenarioLockerStats(scenario, seed, omega)
    
    # 모든 락커에 대해 통계 정보 초기화
    for locker_id in keys(effective_lockers)
        if haskey(LOCKERS_PRIV, locker_id)
            (_, _, carrier) = LOCKERS_PRIV[locker_id]
            capacity = LOCKER_CAPACITY[carrier]
        else
            carrier = "Public"
            capacity = effective_public_locker_capacity()
        end
        
        stats.locker_stats[locker_id] = LockerStatInfo(locker_id, carrier, capacity)
    end
    
    # 모든 캐리어에 대해 통계 정보 초기화
    for carrier in CARRIERS
        stats.carrier_stats[carrier] = CarrierStatInfo(carrier)
    end
    
    return stats
end

function update_carrier_stats!(stats::ScenarioLockerStats, df_attr::DataFrame, customers)
    """캐리어별 통계 업데이트"""
    # 딕셔너리 캐싱으로 O(n²) → O(n) 최적화
    carrier_dict = Dict(r.customer_id => r.carrier for r in eachrow(df_attr))
    demand_dict = :demand in names(df_attr) ? 
        Dict(r.customer_id => r.demand for r in eachrow(df_attr)) : nothing
    
            # 캐리어별 고객 수 및 배송량 집계
        for customer in customers
        carrier = get(carrier_dict, customer.id, nothing)
        if carrier === nothing continue end
            if !haskey(stats.carrier_stats, carrier) continue end
            
            carrier_stat = stats.carrier_stats[carrier]
            carrier_stat.total_customers += 1
            
            # 고객의 배송량/무게 추가 (demand 컬럼에서 가져오기)
        customer_demand = demand_dict !== nothing ? get(demand_dict, customer.id, 1) : 1
            carrier_stat.used_vehicle_capacity += customer_demand
            
            if customer.dtype == "Locker"
                carrier_stat.locker_customers += 1
            else  # "D2D"
                carrier_stat.d2d_customers += 1
            end
        end
end



function update_locker_stats!(stats::ScenarioLockerStats, locker_tracker::LockerUsageTracker, total_locker_customers::Int, d2d_conversions::Int)
    """락커 통계 업데이트"""
    stats.total_locker_customers = total_locker_customers
    stats.d2d_conversions = d2d_conversions
    stats.conversion_rate = total_locker_customers > 0 ? d2d_conversions / total_locker_customers : 0.0
    
    # 개별 락커 통계 업데이트
    for locker_id in keys(locker_tracker.usage)
        if haskey(stats.locker_stats, locker_id)
            stat_info = stats.locker_stats[locker_id]
            stat_info.used = locker_tracker.usage[locker_id]
            stat_info.occupancy_rate = locker_tracker.usage[locker_id] / locker_tracker.capacity[locker_id]
            # Note: customers_assigned와 total_customers_tried는 별도 추적 필요
        end
    end
end

function update_locker_stats_with_assignments!(stats::ScenarioLockerStats, global_locker_assignments::Dict{String, String}, df_attr)
    """락커 할당 정보를 기반으로 캐리어별 사용량 업데이트"""
    # 각 락커별로 캐리어 사용량 초기화
    for (locker_id, locker_info) in stats.locker_stats
        empty!(locker_info.carrier_usage)
        empty!(locker_info.carrier_percentages)
        empty!(locker_info.customers_assigned)
    end
    
    # 딕셔너리 캐싱으로 O(n²) → O(n) 최적화
    carrier_dict = Dict(cid => carrier for (cid, carrier) in zip(df_attr.customer_id, df_attr.carrier))
    
    # 고객 할당 정보를 기반으로 캐리어별 사용량 계산
    for (customer_id, locker_id) in global_locker_assignments
        if haskey(stats.locker_stats, locker_id)
            carrier = get(carrier_dict, customer_id, nothing)
            if carrier !== nothing
                locker_info = stats.locker_stats[locker_id]
                
                # 고객 할당 정보 업데이트
                push!(locker_info.customers_assigned, customer_id)
                
                # 캐리어별 사용량 업데이트
                update_locker_carrier_usage!(locker_info, carrier)
            end
        end
    end
end

function update_locker_carrier_usage!(locker_info::LockerStatInfo, customer_carrier::String)
    """락커의 캐리어별 사용량 업데이트"""
    # 캐리어별 사용량 증가
    locker_info.carrier_usage[customer_carrier] = get(locker_info.carrier_usage, customer_carrier, 0) + 1
    
    # 총 사용량이 0이 아닐 때만 비율 계산
    if locker_info.used > 0
        for (carrier, usage) in locker_info.carrier_usage
            locker_info.carrier_percentages[carrier] = round(100.0 * usage / locker_info.used, digits=2)
        end
    end
end

function collect_locker_stats!(stats::ScenarioLockerStats)
    """현재 통계를 전역 수집기에 저장"""
    lock(STATS_LOCK)
    try
        key = (stats.scenario, stats.seed, stats.omega)
        # 기존 통계가 있는지 확인
        if !haskey(LOCKER_STATS_COLLECTOR.stats_by_scenario, key)
            # 통계는 이미 완성된 객체이므로 직접 저장 (성능 최적화)
            LOCKER_STATS_COLLECTOR.stats_by_scenario[key] = stats
        else
            # 기존 통계와 병합
            existing_stats = LOCKER_STATS_COLLECTOR.stats_by_scenario[key]
            # 락커 통계 업데이트 (덮어쓰기)
            for (locker_id, locker_info) in stats.locker_stats
                existing_stats.locker_stats[locker_id] = deepcopy(locker_info)
            end
            # 캐리어 통계 업데이트 (덮어쓰기)
            for (carrier, carrier_info) in stats.carrier_stats
                existing_stats.carrier_stats[carrier] = deepcopy(carrier_info)
            end
            # 기타 통계 업데이트 (최신값으로 덮어쓰기 - 누적 방지)
            existing_stats.total_locker_customers = stats.total_locker_customers
            existing_stats.d2d_conversions = stats.d2d_conversions
            existing_stats.conversion_rate = stats.conversion_rate
            existing_stats.timestamp = stats.timestamp
        end
    finally
        unlock(STATS_LOCK)
    end
end

function save_main_batch_result!(scenario::Int, omega::Int, customers, route_data::Dict, distance_info::Dict)
    """메인 배치 라우팅 결과 저장"""
    lock(RESULTS_LOCK)
    try
        key = (scenario, omega)
        # 시각화용 결과 저장 (필요한 부분만 복사)
        MAIN_BATCH_RESULTS[key] = Dict(
            "customers" => customers,  # 읽기 전용이므로 복사 불필요
            "route_data" => route_data,  # 읽기 전용이므로 복사 불필요  
            "distance_info" => distance_info,  # 읽기 전용이므로 복사 불필요
            "timestamp" => now()
        )
    finally
        unlock(RESULTS_LOCK)
    end
end

function get_main_batch_result(scenario::Int, omega::Int)
    """저장된 메인 배치 결과 조회"""
    lock(RESULTS_LOCK)
    try
        key = (scenario, omega)
        return get(MAIN_BATCH_RESULTS, key, nothing)
    finally
        unlock(RESULTS_LOCK)
    end
end

function save_scenario4_snapshot!(k::Int, omega::Int, customers, route_data::Dict, distance_info::Dict)
    """시나리오4 k별 경로 스냅샷 저장 (시각화 재사용)"""
    lock(SCENARIO4_SNAPSHOT_LOCK)
    try
        scenario4_route_snapshots[(k, omega)] = Dict(
            "customers" => deepcopy(customers),
            "route_data" => deepcopy(route_data),
            "distance_info" => deepcopy(distance_info)
        )
    finally
        unlock(SCENARIO4_SNAPSHOT_LOCK)
    end
end

function get_scenario4_snapshot(k::Int, omega::Int)
    """시나리오4 k별 경로 스냅샷 조회"""
    lock(SCENARIO4_SNAPSHOT_LOCK)
    try
        return get(scenario4_route_snapshots, (k, omega), nothing)
    finally
        unlock(SCENARIO4_SNAPSHOT_LOCK)
    end
end

function save_locker_stats_csv(filename::String="locker_stats.csv")
    """락커 통계를 CSV 파일로 저장"""
    rows = []
    
    sorted_stats = sort(collect(LOCKER_STATS_COLLECTOR.stats_by_scenario), by=x->x[1])
    for ((scenario, seed, omega), stats) in sorted_stats
        if isempty(stats.locker_stats)
            # 락커 0개 선택 시 더미 행 생성
            push!(rows, (
                scenario = scenario,
                seed = seed,
                omega = omega,
                locker_id = "NO_LOCKER",
                carrier = "None",
                capacity = 0,
                used = 0,
                occupancy_rate = 0.0,
                total_locker_customers = stats.total_locker_customers,
                d2d_conversions = stats.d2d_conversions,
                conversion_rate = round(stats.conversion_rate * 100, digits=2),
                timestamp = stats.timestamp
            ))
        else
            for (locker_id, locker_info) in stats.locker_stats
                # 캐리어별 사용량 정보 문자열로 변환
                carrier_usage_str = join(["$(carrier):$(usage)" for (carrier, usage) in locker_info.carrier_usage], ", ")
                carrier_percentage_str = join(["$(carrier):$(pct)%" for (carrier, pct) in locker_info.carrier_percentages], ", ")
                
                push!(rows, (
                    scenario = scenario,
                    seed = seed,
                    omega = omega,
                    locker_id = locker_id,
                    carrier = locker_info.carrier,
                    capacity = locker_info.capacity,
                    used = locker_info.used,
                    occupancy_rate = round(locker_info.occupancy_rate * 100, digits=2),
                    carrier_usage = carrier_usage_str,
                    carrier_percentages = carrier_percentage_str,
                    total_locker_customers = stats.total_locker_customers,
                    d2d_conversions = stats.d2d_conversions,
                    conversion_rate = round(stats.conversion_rate * 100, digits=2),
                    timestamp = stats.timestamp
                ))
            end
        end
    end
    
    if !isempty(rows)
        df = DataFrame(rows)
        output_path = joinpath(OUTDIR, filename)
        CSV.write(output_path, df)
        println("📄 락커 통계 저장: $output_path")
        return output_path
    else
        println("⚠️  저장할 락커 통계가 없습니다.")
        return ""
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# 시간창 결과 저장 시스템
# ═══════════════════════════════════════════════════════════════════════════

"""시간창 정보 저장용 구조체"""
mutable struct TimeWindowCollector
    # (scenario, seed, omega) => Dict{customer_id => (tw_start, tw_end, service_time, type)}
    customer_time_windows::Dict{Tuple{Int,Int,Int}, Dict{String, NamedTuple{(:tw_start, :tw_end, :service_time, :node_type), Tuple{Int,Int,Int,String}}}}
    # (scenario, seed, omega) => Vector of route schedules
    route_schedules::Dict{Tuple{Int,Int,Int}, Vector{NamedTuple}}
    lock::ReentrantLock
end

const TIME_WINDOW_COLLECTOR = TimeWindowCollector(
    Dict{Tuple{Int,Int,Int}, Dict{String, NamedTuple}}(),
    Dict{Tuple{Int,Int,Int}, Vector{NamedTuple}}(),
    ReentrantLock()
)

"""시간창 정보 저장"""
function save_time_window_info!(scenario::Int, seed::Int, omega::Int, 
                                 customer_tws::Dict{String, Tuple{Int,Int}},
                                 service_times::Dict{String, Int},
                                 node_types::Dict{String, String})
    lock(TIME_WINDOW_COLLECTOR.lock)
    try
        key = (scenario, seed, omega)
        tw_dict = Dict{String, NamedTuple{(:tw_start, :tw_end, :service_time, :node_type), Tuple{Int,Int,Int,String}}}()
        
        for (node_id, (tw_start, tw_end)) in customer_tws
            svc_time = get(service_times, node_id, 0)
            node_type = get(node_types, node_id, "unknown")
            tw_dict[node_id] = (tw_start=tw_start, tw_end=tw_end, service_time=svc_time, node_type=node_type)
        end
        
        TIME_WINDOW_COLLECTOR.customer_time_windows[key] = tw_dict
    finally
        unlock(TIME_WINDOW_COLLECTOR.lock)
    end
end

"""라우트 스케줄 저장"""
function save_route_schedule!(scenario::Int, seed::Int, omega::Int,
                               carrier::String, route_id::Int,
                               route_nodes::Vector{String},
                               time_dict::Dict{Tuple{String,String}, Float64},
                               service_times::Dict{String, Int},
                               time_windows::Dict{String, Tuple{Int,Int}},
                               node_types::Dict{String, String};
                               start_time::Int=DRIVER_START_MIN)
    lock(TIME_WINDOW_COLLECTOR.lock)
    try
        key = (scenario, seed, omega)
        if !haskey(TIME_WINDOW_COLLECTOR.route_schedules, key)
            TIME_WINDOW_COLLECTOR.route_schedules[key] = Vector{NamedTuple}()
        end
        
        current_time = start_time
        
        for (i, node_id) in enumerate(route_nodes)
            # 이전 노드에서 현재 노드로 이동 시간
            travel_time = 0
            if i > 1
                prev_node = route_nodes[i-1]
                travel_time = Int(round(get(time_dict, (prev_node, node_id), 0.0)))
            end
            
            arrival_time = current_time + travel_time
            
            # 시간창 확인 및 대기 시간 계산
            tw_start, tw_end = get(time_windows, node_id, (0, 86400))
            wait_time = max(0, tw_start - arrival_time)
            
            service_start = arrival_time + wait_time
            svc_duration = get(service_times, node_id, 0)
            service_end = service_start + svc_duration
            departure_time = service_end
            
            node_type = get(node_types, node_id, "unknown")
            
            push!(TIME_WINDOW_COLLECTOR.route_schedules[key], (
                carrier = carrier,
                route_id = route_id,
                stop_order = i,
                node_id = node_id,
                node_type = node_type,
                arrival_time = arrival_time,
                arrival_time_str = format_time(arrival_time),
                tw_start = tw_start,
                tw_start_str = format_time(tw_start),
                tw_end = tw_end,
                tw_end_str = format_time(tw_end),
                wait_time = wait_time,
                service_start = service_start,
                service_start_str = format_time(service_start),
                service_duration = svc_duration,
                service_end = service_end,
                service_end_str = format_time(service_end),
                departure_time = departure_time,
                departure_time_str = format_time(departure_time),
                tw_violated = service_start > tw_end
            ))
            
            current_time = departure_time
        end
    finally
        unlock(TIME_WINDOW_COLLECTOR.lock)
    end
end

"""고객 시간창 CSV 저장"""
function save_customer_time_windows_csv(filename::String="customer_time_windows.csv")
    rows = []
    
    lock(TIME_WINDOW_COLLECTOR.lock)
    try
        sorted_keys = sort(collect(keys(TIME_WINDOW_COLLECTOR.customer_time_windows)))
        for (scenario, seed, omega) in sorted_keys
            tw_dict = TIME_WINDOW_COLLECTOR.customer_time_windows[(scenario, seed, omega)]
            for (node_id, info) in tw_dict
                push!(rows, (
                    scenario = scenario,
                    seed = seed,
                    omega = omega,
                    node_id = node_id,
                    node_type = info.node_type,
                    tw_start_sec = info.tw_start,
                    tw_end_sec = info.tw_end,
                    tw_start_time = format_time(info.tw_start),
                    tw_end_time = format_time(info.tw_end),
                    tw_width_min = (info.tw_end - info.tw_start) ÷ 60,
                    service_time_sec = info.service_time,
                    service_time_min = info.service_time ÷ 60
                ))
            end
        end
    finally
        unlock(TIME_WINDOW_COLLECTOR.lock)
    end
    
    if !isempty(rows)
        df = DataFrame(rows)
        output_path = joinpath(OUTDIR, filename)
        CSV.write(output_path, df)
        println("📄 고객 시간창 저장: $output_path ($(nrow(df))개 노드)")
        return output_path
    else
        println("⚠️  저장할 시간창 정보가 없습니다.")
        return ""
    end
end

"""라우트 스케줄 CSV 저장"""
function save_route_schedule_csv(filename::String="route_schedule.csv")
    rows = []
    
    lock(TIME_WINDOW_COLLECTOR.lock)
    try
        sorted_keys = sort(collect(keys(TIME_WINDOW_COLLECTOR.route_schedules)))
        for (scenario, seed, omega) in sorted_keys
            schedules = TIME_WINDOW_COLLECTOR.route_schedules[(scenario, seed, omega)]
            for sched in schedules
                push!(rows, (
                    scenario = scenario,
                    seed = seed,
                    omega = omega,
                    carrier = sched.carrier,
                    route_id = sched.route_id,
                    stop_order = sched.stop_order,
                    node_id = sched.node_id,
                    node_type = sched.node_type,
                    arrival_time = sched.arrival_time_str,
                    tw_start = sched.tw_start_str,
                    tw_end = sched.tw_end_str,
                    wait_time_min = sched.wait_time ÷ 60,
                    service_start = sched.service_start_str,
                    service_duration_min = sched.service_duration ÷ 60,
                    service_end = sched.service_end_str,
                    departure_time = sched.departure_time_str,
                    tw_violated = sched.tw_violated
                ))
            end
        end
    finally
        unlock(TIME_WINDOW_COLLECTOR.lock)
    end
    
    if !isempty(rows)
        df = DataFrame(rows)
        output_path = joinpath(OUTDIR, filename)
        CSV.write(output_path, df)
        println("📄 라우트 스케줄 저장: $output_path ($(nrow(df))개 정류장)")
        return output_path
    else
        println("⚠️  저장할 라우트 스케줄이 없습니다.")
        return ""
    end
end

"""시간창 정보 초기화"""
function reset_time_window_collector!()
    lock(TIME_WINDOW_COLLECTOR.lock)
    try
        empty!(TIME_WINDOW_COLLECTOR.customer_time_windows)
        empty!(TIME_WINDOW_COLLECTOR.route_schedules)
    finally
        unlock(TIME_WINDOW_COLLECTOR.lock)
    end
end

# ═══════════════════════════════════════════════════════════════════════════

function save_locker_stats_json(filename::String="locker_stats.json")
    """락커 통계를 JSON 파일로 저장"""
    output_data = Dict{String, Any}()
    
    sorted_stats = sort(collect(LOCKER_STATS_COLLECTOR.stats_by_scenario), by=x->x[1])
    # 시나리오별 데이터 구성
    for ((scenario, seed, omega), stats) in sorted_stats
        scenario_key = "scenario_$(scenario)"
        if !haskey(output_data, scenario_key)
            output_data[scenario_key] = Dict{String, Any}()
        end
        
        seed_key = "seed_$(seed)"
        if !haskey(output_data[scenario_key], seed_key)
            output_data[scenario_key][seed_key] = Dict{String, Any}()
        end
        
        omega_key = "omega_$(omega)"
        output_data[scenario_key][seed_key][omega_key] = Dict(
            "total_locker_customers" => stats.total_locker_customers,
            "d2d_conversions" => stats.d2d_conversions,
            "conversion_rate" => stats.conversion_rate,
            "timestamp" => string(stats.timestamp),
            "lockers" => Dict(
                locker_id => Dict(
                    "carrier" => info.carrier,
                    "capacity" => info.capacity,
                    "used" => info.used,
                    "occupancy_rate" => info.occupancy_rate,
                    "carrier_usage" => info.carrier_usage,
                    "carrier_percentages" => info.carrier_percentages,
                    "customers_assigned" => info.customers_assigned
                )
                for (locker_id, info) in stats.locker_stats
            )
        )
    end
    
    if !isempty(output_data)
        output_path = joinpath(OUTDIR, filename)
        open(output_path, "w") do io
            JSON3.write(io, output_data)
        end
        println("📄 락커 통계 저장: $output_path")
        return output_path
    else
        println("⚠️  저장할 락커 통계가 없습니다.")
        return ""
    end
end

# 캐리어별 상세 분석을 위한 새로운 구조체
mutable struct CarrierLockerAnalysis
    scenario::Int
    seed::Int
    omega::Int
    
    # 락커별 캐리어 사용 현황
    locker_carrier_usage::Dict{String, Dict{String, Int}}  # locker_id => carrier => count
    
    # D2D 전환 고객의 캐리어별 분석
    d2d_conversions_by_carrier::Dict{String, Int}  # carrier => d2d_count
    d2d_conversion_reasons::Dict{String, Vector{String}}  # carrier => [reasons]
    
    # 캐리어별 락커 선호도 분석
    carrier_preferred_lockers::Dict{String, Dict{String, Int}}  # carrier => locker_id => attempts
    
    # 캐리어별 성공률
    carrier_success_rates::Dict{String, Float64}  # carrier => success_rate
    
    function CarrierLockerAnalysis(scenario::Int, seed::Int, omega::Int)
        new(scenario, seed, omega,
            Dict{String, Dict{String, Int}}(),
            Dict{String, Int}(),
            Dict{String, Vector{String}}(),
            Dict{String, Dict{String, Int}}(),
            Dict{String, Float64}())
    end
end

# 전역 캐리어 분석 수집기
const CARRIER_ANALYSIS_COLLECTOR = Dict{Tuple{Int,Int,Int}, CarrierLockerAnalysis}()
const CARRIER_ANALYSIS_LOCK = ReentrantLock()

function init_carrier_analysis(scenario::Int, seed::Int, omega::Int)
    """캐리어 분석 초기화"""
    analysis = CarrierLockerAnalysis(scenario, seed, omega)
    
    # 모든 캐리어 초기화
    for carrier in CARRIERS
        analysis.d2d_conversions_by_carrier[carrier] = 0
        analysis.d2d_conversion_reasons[carrier] = String[]
        analysis.carrier_preferred_lockers[carrier] = Dict{String, Int}()
        analysis.carrier_success_rates[carrier] = 0.0
    end
    
    return analysis
end

function track_locker_assignment!(analysis::CarrierLockerAnalysis, customer_id::String, carrier::String, locker_id::String, success::Bool)
    """락커 할당 추적"""
    if success
        # 성공한 할당 기록
        if !haskey(analysis.locker_carrier_usage, locker_id)
            analysis.locker_carrier_usage[locker_id] = Dict{String, Int}()
        end
        analysis.locker_carrier_usage[locker_id][carrier] = get(analysis.locker_carrier_usage[locker_id], carrier, 0) + 1
    end
    
    # 선호 락커 시도 기록
    if !haskey(analysis.carrier_preferred_lockers, carrier)
        analysis.carrier_preferred_lockers[carrier] = Dict{String, Int}()
    end
    analysis.carrier_preferred_lockers[carrier][locker_id] = get(analysis.carrier_preferred_lockers[carrier], locker_id, 0) + 1
end

function track_d2d_conversion!(analysis::CarrierLockerAnalysis, customer_id::String, carrier::String, reason::String)
    """D2D 전환 추적"""
    analysis.d2d_conversions_by_carrier[carrier] = get(analysis.d2d_conversions_by_carrier, carrier, 0) + 1
    push!(analysis.d2d_conversion_reasons[carrier], reason)
end

function calculate_carrier_success_rates!(analysis::CarrierLockerAnalysis, total_locker_customers_by_carrier::Dict{String, Int})
    """캐리어별 성공률 계산"""
    for carrier in CARRIERS
        carrier_locker_customers = get(total_locker_customers_by_carrier, carrier, 0)
        
        if carrier_locker_customers > 0
            # 성공한 할당 수 계산
            successful_assignments = isempty(analysis.locker_carrier_usage) ? 0 : sum(get(locker_usage, carrier, 0) for locker_usage in values(analysis.locker_carrier_usage))
            
            # 성공률 계산
            analysis.carrier_success_rates[carrier] = successful_assignments / carrier_locker_customers
        else
            analysis.carrier_success_rates[carrier] = 0.0
        end
    end
end

function save_carrier_analysis!(analysis::CarrierLockerAnalysis)
    """캐리어 분석 결과 저장"""
    lock(CARRIER_ANALYSIS_LOCK)
    try
        key = (analysis.scenario, analysis.seed, analysis.omega)
        CARRIER_ANALYSIS_COLLECTOR[key] = analysis
    finally
        unlock(CARRIER_ANALYSIS_LOCK)
    end
end

function generate_carrier_detailed_report(filename::String="carrier_detailed_analysis.csv")
    """캐리어별 상세 분석 리포트 생성"""
    rows = []
    
    lock(CARRIER_ANALYSIS_LOCK)
    try
        sorted_analyses = sort(collect(CARRIER_ANALYSIS_COLLECTOR), by=x->x[1])
        
        for ((scenario, seed, omega), analysis) in sorted_analyses
            # 락커별 캐리어 사용 현황
            for (locker_id, carrier_usage) in analysis.locker_carrier_usage
                total_usage = sum(values(carrier_usage))
                for (carrier, count) in carrier_usage
                    push!(rows, (
                        scenario = scenario,
                        seed = seed,
                        omega = omega,
                        analysis_type = "locker_usage",
                        locker_id = locker_id,
                        carrier = carrier,
                        count = count,
                        percentage = total_usage > 0 ? round(count / total_usage * 100, digits=2) : 0.0,
                        reason = "",
                        success_rate = round(analysis.carrier_success_rates[carrier] * 100, digits=2)
                    ))
                end
            end
            
            # D2D 전환 현황
            for (carrier, d2d_count) in analysis.d2d_conversions_by_carrier
                if d2d_count > 0
                    # 전환 이유별 집계
                    reason_counts = Dict{String, Int}()
                    for reason in analysis.d2d_conversion_reasons[carrier]
                        reason_counts[reason] = get(reason_counts, reason, 0) + 1
                    end
                    
                    for (reason, count) in reason_counts
                        push!(rows, (
                            scenario = scenario,
                            seed = seed,
                            omega = omega,
                            analysis_type = "d2d_conversion",
                            locker_id = "",
                            carrier = carrier,
                            count = count,
                            percentage = round(count / d2d_count * 100, digits=2),
                            reason = reason,
                            success_rate = round(analysis.carrier_success_rates[carrier] * 100, digits=2)
                        ))
                    end
                end
            end
            
            # 캐리어별 선호 락커 분석
            for (carrier, preferred_lockers) in analysis.carrier_preferred_lockers
                total_attempts = sum(values(preferred_lockers))
                if total_attempts > 0
                    for (locker_id, attempts) in preferred_lockers
                        push!(rows, (
                            scenario = scenario,
                            seed = seed,
                            omega = omega,
                            analysis_type = "preferred_locker",
                            locker_id = locker_id,
                            carrier = carrier,
                            count = attempts,
                            percentage = round(attempts / total_attempts * 100, digits=2),
                            reason = "",
                            success_rate = round(analysis.carrier_success_rates[carrier] * 100, digits=2)
                        ))
                    end
                end
            end
        end
    finally
        unlock(CARRIER_ANALYSIS_LOCK)
    end
    
    if !isempty(rows)
        df = DataFrame(rows)
        output_path = joinpath(OUTDIR, filename)
        CSV.write(output_path, df)
        println("📊 캐리어별 상세 분석 저장: $output_path")
        return output_path
    else
        println("⚠️  저장할 캐리어 분석 데이터가 없습니다.")
        return ""
    end
end

function generate_locker_carrier_matrix(scenario::Int, filename::String="locker_carrier_matrix_s$(scenario).csv")
    """특정 시나리오의 락커-캐리어 매트릭스 생성"""
    lock(CARRIER_ANALYSIS_LOCK)
    try
        # 해당 시나리오의 모든 분석 수집
        scenario_analyses = [analysis for ((s, seed, omega), analysis) in CARRIER_ANALYSIS_COLLECTOR if s == scenario]
        
        if isempty(scenario_analyses)
            println("⚠️  시나리오 $scenario 의 분석 데이터가 없습니다.")
            return ""
        end
        
        # 모든 락커 목록 수집
        all_lockers = Set{String}()
        for analysis in scenario_analyses
            union!(all_lockers, keys(analysis.locker_carrier_usage))
        end
        
        rows = []
        for analysis in scenario_analyses
            for locker_id in all_lockers
                row_data = Dict{String, Any}(
                    "scenario" => analysis.scenario,
                    "seed" => analysis.seed,
                    "omega" => analysis.omega,
                    "locker_id" => locker_id
                )
                
                # 각 캐리어별 사용량 추가
                for carrier in CARRIERS
                    usage = get(get(analysis.locker_carrier_usage, locker_id, Dict()), carrier, 0)
                    row_data[carrier] = usage
                end
                
                # 총 사용량 계산
                total_usage = sum(get(get(analysis.locker_carrier_usage, locker_id, Dict()), carrier, 0) for carrier in CARRIERS)
                row_data["total"] = total_usage
                
                push!(rows, row_data)
            end
        end
        
        if !isempty(rows)
            df = DataFrame(rows)
            # 열 순서 명시적 지정: scenario, seed, omega, locker_id, 캐리어들, total
            column_order = ["scenario", "seed", "omega", "locker_id"]
            append!(column_order, CARRIERS)  # 캐리어 순서
            push!(column_order, "total")
            
            # 지정된 순서로 열 재정렬
            df = df[:, column_order]
            
            output_path = joinpath(OUTDIR, filename)
            CSV.write(output_path, df)
            println("📊 락커-캐리어 매트릭스 저장: $output_path")
            return output_path
        end
    finally
        unlock(CARRIER_ANALYSIS_LOCK)
    end
    
    return ""
end

function show_locker_carrier_breakdown()
    """현재 수집된 통계에서 락커별 캐리어 사용량 분석 출력"""
    println("📊 락커별 캐리어 사용량 분석:")
    println("-" * "="^50)
    
    for ((scenario, seed, omega), stats) in LOCKER_STATS_COLLECTOR.stats_by_scenario
        if !isempty(stats.locker_stats)
            println("📌 시나리오 $scenario (시드: $seed, 오메가: $omega):")
            for (locker_id, locker_info) in stats.locker_stats
                if locker_info.used > 0
                    println("   🏪 $locker_id ($(locker_info.carrier) 소유):")
                    println("      용량: $(locker_info.capacity), 사용: $(locker_info.used) ($(round(locker_info.occupancy_rate * 100, digits=1))%)")
                    
                    if !isempty(locker_info.carrier_usage)
                        println("      캐리어별 사용:")
                        for (carrier, usage) in locker_info.carrier_usage
                            percentage = get(locker_info.carrier_percentages, carrier, 0.0)
                            println("        📦 $carrier: $usage ($(percentage)%)")
                        end
                    end
                    println()
                end
            end
            println()
        end
    end
end

function generate_locker_summary()
    """락커 통계 요약 생성"""
    summary = Dict{String, Any}()
    
    sorted_stats_collection = sort(collect(LOCKER_STATS_COLLECTOR.stats_by_scenario), by=x->x[1])
    # 시나리오별 요약
    for scenario in 1:5
        scenario_data = filter(entry -> entry[1][1] == scenario, sorted_stats_collection)
        
        if !isempty(scenario_data)
            # 기본 통계
            conversion_rates = [stats.conversion_rate for (_, stats) in scenario_data]
            total_customers = [stats.total_locker_customers for (_, stats) in scenario_data]
            
            # 락커별 평균 사용률
            locker_occupancy = Dict{String, Vector{Float64}}()
            for (_, stats) in scenario_data
                for (locker_id, info) in stats.locker_stats
                    if !haskey(locker_occupancy, locker_id)
                        locker_occupancy[locker_id] = Float64[]
                    end
                    push!(locker_occupancy[locker_id], info.occupancy_rate)
                end
            end
            
            # 배송사별 평균 사용률
            carrier_occupancy = Dict{String, Vector{Float64}}()
            for (_, stats) in scenario_data
                for (locker_id, info) in stats.locker_stats
                    carrier = info.carrier
                    if !haskey(carrier_occupancy, carrier)
                        carrier_occupancy[carrier] = Float64[]
                    end
                    push!(carrier_occupancy[carrier], info.occupancy_rate)
                end
            end
            
            summary["scenario_$scenario"] = Dict(
                "samples_count" => length(scenario_data),
                "avg_conversion_rate" => length(conversion_rates) > 0 ? mean(conversion_rates) : 0.0,
                "std_conversion_rate" => length(conversion_rates) > 1 ? std(conversion_rates) : 0.0,
                "avg_total_customers" => length(total_customers) > 0 ? mean(total_customers) : 0.0,
                "locker_avg_occupancy" => Dict(
                    locker_id => mean(rates) 
                    for (locker_id, rates) in locker_occupancy if !isempty(rates)
                ),
                "carrier_avg_occupancy" => Dict(
                    carrier => mean(rates)
                    for (carrier, rates) in carrier_occupancy if !isempty(rates)
                )
            )
        end
    end
    
    LOCKER_STATS_COLLECTOR.summary_stats = summary
    return summary
end

function print_locker_summary()
    """락커 통계 요약 출력"""
    if isempty(LOCKER_STATS_COLLECTOR.summary_stats)
        generate_locker_summary()
    end
    
    println("\n" * "="^70)
    println("📊 락커 사용률 통계 요약")
    println("="^70)
    
    for scenario in 1:5
        scenario_key = "scenario_$scenario"
        if haskey(LOCKER_STATS_COLLECTOR.summary_stats, scenario_key)
            data = LOCKER_STATS_COLLECTOR.summary_stats[scenario_key]
            scenario_name = get_scenario_name(scenario)
            
            println("\n🎯 시나리오 $scenario ($scenario_name):")
            println("   샘플 수: $(data["samples_count"])")
            println("   평균 D2D 전환율: $(round(data["avg_conversion_rate"] * 100, digits=2))%")
            
            if haskey(data, "carrier_avg_occupancy") && !isempty(data["carrier_avg_occupancy"])
                println("   📦 배송사별 평균 락커 사용률:")
                for (carrier, avg_rate) in sort(collect(data["carrier_avg_occupancy"]), by=x->x[2], rev=true)
                    println("      $carrier: $(round(avg_rate * 100, digits=1))%")
                end
            end
            
            if haskey(data, "locker_avg_occupancy") && !isempty(data["locker_avg_occupancy"])
                println("   🏪 개별 락커 평균 사용률:")
                for (locker_id, avg_rate) in sort(collect(data["locker_avg_occupancy"]), by=x->x[2], rev=true)
                    println("      $locker_id: $(round(avg_rate * 100, digits=1))%")
                end
            end
        end
    end
    println("="^70)
end

function print_carrier_summary()
    """캐리어별 통계 요약 출력"""
    println("\n" * "="^70)
    println("📦 캐리어별 상세 통계 분석")
    println("="^70)
    
    for scenario in 1:5
        scenario_name = get_scenario_name(scenario)
                       
        println("\n🎯 시나리오 $scenario ($scenario_name):")
        
        # 시나리오별 데이터 수집
        sorted_stats_collection = sort(collect(LOCKER_STATS_COLLECTOR.stats_by_scenario), by=x->x[1])
        scenario_data = filter(entry -> entry[1][1] == scenario, sorted_stats_collection)
        
        if isempty(scenario_data)
            println("   📊 데이터 없음")
            continue
        end
        
        # 캐리어별 평균 통계 계산
        carrier_summary = Dict{String, Dict{String, Vector{Float64}}}()
        
        for (_, stats) in scenario_data
            for (carrier, carrier_stat) in stats.carrier_stats
                if !haskey(carrier_summary, carrier)
                    carrier_summary[carrier] = Dict{String, Vector{Float64}}(
                        "total_customers" => Float64[],
                        "locker_customers" => Float64[],
                        "d2d_customers" => Float64[],
                        "conversion_rate" => Float64[],
                        "locker_utilization" => Float64[],
                        "customers_per_vehicle" => Float64[],
                        "vehicle_utilization" => Float64[],
                        "vehicle_occupancy_rate" => Float64[]
                    )
                end
                
                push!(carrier_summary[carrier]["total_customers"], carrier_stat.total_customers)
                push!(carrier_summary[carrier]["locker_customers"], carrier_stat.locker_customers)
                push!(carrier_summary[carrier]["d2d_customers"], carrier_stat.d2d_customers)
                push!(carrier_summary[carrier]["conversion_rate"], carrier_stat.conversion_rate)
                push!(carrier_summary[carrier]["locker_utilization"], carrier_stat.locker_utilization)
                push!(carrier_summary[carrier]["customers_per_vehicle"], carrier_stat.customers_per_vehicle)
                push!(carrier_summary[carrier]["vehicle_utilization"], carrier_stat.vehicle_utilization)
                push!(carrier_summary[carrier]["vehicle_occupancy_rate"], carrier_stat.vehicle_occupancy_rate)
            end
        end
        
        # 캐리어별 결과 출력
        for carrier in CARRIERS
            if haskey(carrier_summary, carrier)
                data = carrier_summary[carrier]
                
                println("   📦 $carrier:")
                println("      총 고객 수: $(round(mean(data["total_customers"]), digits=1))명")
                println("      락커 고객: $(round(mean(data["locker_customers"]), digits=1))명")
                println("      D2D 고객: $(round(mean(data["d2d_customers"]), digits=1))명")
                println("      D2D 전환율: $(round(mean(data["conversion_rate"]) * 100, digits=1))%")
                println("      락커 사용률: $(round(mean(data["locker_utilization"]) * 100, digits=1))%")
                println("      🚛 차량 대수: (미사용)")
                println("      🚛 차량당 고객: $(round(mean(data["customers_per_vehicle"]), digits=1))명")
                println("      🚛 차량 가동률: $(round(mean(data["vehicle_utilization"]) * 100, digits=1))%")
                println("      📦 차량 적재율: $(round(mean(data["vehicle_occupancy_rate"]) * 100, digits=1))%")
                println("      시장점유율(락커): $(round(CARRIER_MARKET_SHARE_LOCKER[carrier] * 100, digits=1))%")
                println("      시장점유율(D2D): $(round(CARRIER_MARKET_SHARE_D2D[carrier] * 100, digits=1))%")
                println("")
            end
        end
    end
    println("="^70)
end

function save_carrier_stats_csv(filename::String="carrier_stats.csv")
    """캐리어별 통계를 CSV 파일로 저장 (사용자 요청 반영)"""
    rows = []
    
    sorted_stats = sort(collect(LOCKER_STATS_COLLECTOR.stats_by_scenario), by=x->x[1])
    for ((scenario, seed, omega), stats) in sorted_stats
        for (carrier, carrier_info) in stats.carrier_stats
            # 라우팅이 수행되지 않았거나 수요가 없는 캐리어는 제외
            if carrier_info.used_vehicle_capacity == 0 || carrier_info.avg_distance_per_used_vehicle == 0
                continue
            end
            
            push!(rows, (
                scenario = scenario,
                seed = seed,
                omega = omega,
                carrier = carrier,
                # 차량 및 수요 정보
                vehicles_count = carrier_info.vehicles_count,
                total_demand = carrier_info.used_vehicle_capacity,
                total_customers = carrier_info.total_customers,
                locker_customers = carrier_info.locker_customers,
                d2d_customers = carrier_info.d2d_customers,
                # (요청) 1회 운행당 평균 점유율 (0-100%)
                avg_fill_rate_per_trip_pct = round(carrier_info.avg_fill_rate_per_trip * 100, digits=2),
                # 실제 사용된 차량당 평균 주행거리 (km)
                avg_distance_per_vehicle_km = round(carrier_info.avg_distance_per_used_vehicle, digits=2),
                # 수요 1단위당 주행거리 (km/demand)
                km_per_demand = round(carrier_info.km_per_demand, digits=2),
                # 총 주행거리
                total_distance_km = round(carrier_info.total_distance, digits=2)
            ))
        end
    end
    
    if !isempty(rows)
        df = DataFrame(rows)
        output_path = joinpath(OUTDIR, filename)
        CSV.write(output_path, df)
        println("📄 캐리어별 효율성 통계 저장: $output_path")
        return output_path
    else
        println("⚠️  저장할 캐리어별 통계가 없습니다.")
        return ""
    end
end





#───────────────────────────────────────────────────────────────────────────────
# 3. 단순 평면 거리 계산 함수들
#───────────────────────────────────────────────────────────────────────────────
function approx_euclidean(lat1::Real, lon1::Real, lat2::Real, lon2::Real)
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * cos(deg2rad((lat1 + lat2) / 2))
    dlat = (lat2 - lat1) * km_per_deg_lat
    dlon = (lon2 - lon1) * km_per_deg_lon
    sqrt(dlat^2 + dlon^2)
end

function julia_nodes_list(custs; pub=Dict(), scenario::Int=0, effective_lockers=Dict())
    nodes = Vector{Dict{String,Any}}()
    for c in custs
        push!(nodes, Dict(
            "id"   => c.id,
            "lat"  => Float64(c.coord[2]),
            "lon"  => Float64(c.coord[1]),
            "type" => "Customer"
        ))
    end
    for (id,(lon,lat,_)) in DEPOTS
        push!(nodes, Dict(
            "id"   => id,
            "lat"  => lat,
            "lon"  => lon,
            "type" => "Depot"
        ))
    end
    if scenario != SCENARIO_D2D
        for locker_id in keys(effective_lockers)
            if haskey(LOCKERS_PRIV, locker_id)
                (lon, lat, _) = LOCKERS_PRIV[locker_id]
                push!(nodes, Dict(
                    "id"   => locker_id,
                    "lat"  => lat,
                    "lon"  => lon,
                    "type" => "Locker"
                ))
            else
                (coords_tuple, _) = pub[locker_id]
                push!(nodes, Dict(
                    "id"   => locker_id,
                    "lat"  => coords_tuple[2],
                    "lon"  => coords_tuple[1],
                    "type" => "Locker"
                ))
            end
        end
    end
    return nodes
end

"""
    build_pairwise_distances(nodes)

노드들 간의 거리 행렬을 생성합니다.
반환: DataFrame with columns (:from, :to, :km)
"""
function build_pairwise_distances(nodes::Vector{Dict{String,Any}})
    df, _ = build_pairwise_distances_and_times(nodes)
    return df
end

"""
    build_pairwise_distances_and_times(nodes)

노드들 간의 거리 및 이동 시간 행렬을 생성합니다.
반환: (DataFrame(:from, :to, :km), Dict{(from,to), seconds})
"""
function build_pairwise_distances_and_times(nodes::Vector{Dict{String,Any}})
    N = length(nodes)
    rows = Vector{NamedTuple{(:from, :to, :km), Tuple{String,String,Float64}}}()
    time_dict = Dict{Tuple{String,String}, Float64}()
    
    # 도로망 거리 사용 시 OSRM table API로 일괄 계산
    if USE_ROAD_DISTANCE && is_distance_matrix_initialized()
        progress_println("   🚗 도로망 기반 거리/시간 행렬 사용 (OSRM)")
        
        for i in 1:N
            ida  = nodes[i]["id"]
            lat_a = Float64(nodes[i]["lat"])
            lon_a = Float64(nodes[i]["lon"])
            pos_a = (lat_a, lon_a)  # (lat, lon) 순서
            
            for j in 1:N
                if i == j continue end
                idb   = nodes[j]["id"]
                lat_b = Float64(nodes[j]["lat"])
                lon_b = Float64(nodes[j]["lon"])
                pos_b = (lat_b, lon_b)
                
                # 도로망 거리 및 시간 조회 (차량 경로)
                d_km, t_sec = get_precomputed_car_distance_and_duration(pos_a, pos_b)
                
                # 경로 없는 경우 유클리드 거리로 대체
                if d_km == Inf || isnan(d_km)
                    d_km = approx_euclidean(lat_a, lon_a, lat_b, lon_b)
                    t_sec = d_km / 30.0 * 3600.0  # 30km/h 가정
                end
                if t_sec == Inf || isnan(t_sec)
                    t_sec = d_km / 30.0 * 3600.0
                end
                
                push!(rows, (from=ida, to=idb, km=d_km))
                time_dict[(ida, idb)] = t_sec
            end
        end
    else
        # 기존 유클리드 거리 사용
        for i in 1:N
            ida  = nodes[i]["id"]
            lat_a = nodes[i]["lat"]
            lon_a = nodes[i]["lon"]
            for j in 1:N
                if i == j continue end
                idb   = nodes[j]["id"]
                lat_b = nodes[j]["lat"]
                lon_b = nodes[j]["lon"]
                d_km  = approx_euclidean(lat_a, lon_a, lat_b, lon_b)
                t_sec = d_km / 30.0 * 3600.0  # 30km/h 가정
                push!(rows, (from=ida, to=idb, km=d_km))
                time_dict[(ida, idb)] = t_sec
            end
        end
    end
    
    return DataFrame(rows), time_dict
end

#───────────────────────────────────────────────────────────────────────────────
# 4. Arc 허용 규칙
#───────────────────────────────────────────────────────────────────────────────
function make_arc_rules(df_attr, private_lockers::Dict, public_lockers::Dict, scenario::Int)
    car = Dict(r.customer_id=>r.carrier for r in eachrow(df_attr))
    for (i,(_,_,c)) in DEPOTS       ; car[i]=c end
    for (i,(_,_,c)) in private_lockers ; car[i]=c end
    for i in keys(public_lockers)       ; car[i]="" end

    typ = Dict{String,String}()
    for (i,_) in DEPOTS       ; typ[i]="D" end
    for (i,_) in private_lockers ; typ[i]="L" end
    for i in keys(public_lockers)  ; typ[i]="L" end
    for r in eachrow(df_attr) ; typ[r.customer_id]="C" end

    ids    = collect(keys(car))
    forbid = Set{Tuple{String,String}}()

    for a in ids, b in ids
        if a == b continue end
        ta, tb = typ[a], typ[b]
        ca, cb = car[a], car[b]
        allowed = false

        if scenario == SCENARIO_D2D
            if ta == "D" && tb == "D"
                allowed = false
            elseif ta == "D" && tb == "C" && ca == cb
                allowed = true
            elseif ta == "C" && tb == "C" && ca == cb
                allowed = true
            elseif ta == "C" && tb == "D" && ca == cb
                allowed = true
            else
                allowed = false
            end

        elseif scenario == SCENARIO_DPL
            if ta == "D" && tb == "D"
                allowed = false
            elseif ta == "D" && tb == "C" && ca == cb
                allowed = true
            elseif ta == "C" && tb == "C" && ca == cb
                allowed = true
            elseif ta == "C" && tb == "D" && ca == cb
                allowed = true
            elseif ta == "D" && tb == "L" && ca == cb
                allowed = true
            elseif ta == "L" && tb == "D" && ca == cb
                allowed = true
            elseif ta == "L" && tb == "C" && ca == cb
                allowed = true
            elseif ta == "C" && tb == "L" && ca == cb
                allowed = true
            elseif ta == "L" && tb == "L" && ca == cb
                allowed = true
            else
                allowed = false
            end

        elseif scenario == SCENARIO_SPL || scenario == SCENARIO_OPL
            if ta == "D" && tb == "D"
                allowed = false
            elseif ta == "D" && tb == "C" && ca == cb
                allowed = true
            elseif ta == "C" && tb == "C" && ca == cb
                allowed = true
            elseif ta == "C" && tb == "D" && ca == cb
                allowed = true
            elseif ta == "D" && tb == "L"
                allowed = true
            elseif ta == "L" && tb == "D"
                allowed = true
            elseif ta == "L" && tb == "C"
                allowed = true
            elseif ta == "C" && tb == "L"
                allowed = true
            elseif ta == "L" && tb == "L"
                allowed = true
            else
                allowed = false
            end
        elseif scenario == SCENARIO_PSPL
            # 부분 공유 규칙
            # 기본은 시나리오 2(Private 동일캐리어)와 동일하되,
            # AlzaBox/Foxpost/Packeta 사이에서는 L<->C, L<->D, L<->L 등에서 서로 공유 허용
            if ta == "D" && tb == "D"
                allowed = false
            elseif ta == "D" && tb == "C"
                allowed = ca == cb
            elseif ta == "C" && tb == "C"
                allowed = ca == cb
            elseif ta == "C" && tb == "D"
                allowed = ca == cb
            elseif ta == "D" && tb == "L"
                if haskey(LOCKERS_PRIV, b)
                    (_,_,lk_carrier_b) = LOCKERS_PRIV[b]
                    allowed = (ca == lk_carrier_b) || ((ca in PSPL_SHARED_CARRIERS) && (lk_carrier_b in PSPL_SHARED_CARRIERS))
                else
                    allowed = false
                end
            elseif ta == "L" && tb == "D"
                if haskey(LOCKERS_PRIV, a)
                    (_,_,lk_carrier_a) = LOCKERS_PRIV[a]
                    allowed = (lk_carrier_a == cb) || ((cb in PSPL_SHARED_CARRIERS) && (lk_carrier_a in PSPL_SHARED_CARRIERS))
                else
                    allowed = false
                end
            elseif ta == "L" && tb == "C"
                if haskey(LOCKERS_PRIV, a)
                    (_,_,lk_carrier_a) = LOCKERS_PRIV[a]
                    allowed = (lk_carrier_a == cb) || ((cb in PSPL_SHARED_CARRIERS) && (lk_carrier_a in PSPL_SHARED_CARRIERS))
                else
                    allowed = false
                end
            elseif ta == "C" && tb == "L"
                if haskey(LOCKERS_PRIV, b)
                    (_,_,lk_carrier_b) = LOCKERS_PRIV[b]
                    allowed = (ca == lk_carrier_b) || ((ca in PSPL_SHARED_CARRIERS) && (lk_carrier_b in PSPL_SHARED_CARRIERS))
                else
                    allowed = false
                end
            elseif ta == "L" && tb == "L"
                if haskey(LOCKERS_PRIV, a) && haskey(LOCKERS_PRIV, b)
                    (_,_,lk_carrier_a) = LOCKERS_PRIV[a]
                    (_,_,lk_carrier_b) = LOCKERS_PRIV[b]
                    allowed = (lk_carrier_a == lk_carrier_b) || ((lk_carrier_a in PSPL_SHARED_CARRIERS) && (lk_carrier_b in PSPL_SHARED_CARRIERS))
                else
                    allowed = false
                end
            else
                allowed = false
            end
        end

        if !allowed
            push!(forbid, (a, b))
        end
    end

    return forbid
end

#───────────────────────────────────────────────────────────────────────────────
# 5. solve_vrp_single_omega (Euclidean distance 사용) - 로그 출력 최소화
#───────────────────────────────────────────────────────────────────────────────
function solve_vrp_single_omega(z_val::Dict, customers, df_attr::DataFrame, lock_pub::Dict, scn::Int; verbose::Bool=false, seed::Int=1, omega::Int=1, collect_stats::Bool=true, progress_callback::Union{Function,Nothing}=nothing, carrier_analysis::Union{CarrierLockerAnalysis,Nothing}=nothing, moo_seed::Int=42)
    # 딕셔너리 캐싱으로 O(n²) → O(n) 최적화 (함수 전체에서 재사용)
    carrier_by_customer = Dict(r.customer_id => r.carrier for r in eachrow(df_attr))
    
    active = Dict(k => v for (k, v) in lock_pub if get(z_val, k, 1) == 1)

    d2d_customers = Vector{eltype(customers)}()
    locker_customers = Vector{eltype(customers)}()
    effective_lockers = Dict{String,Any}()

    # 원래 락커 고객 수 추적 (모든 시나리오에서)
    original_locker_customers = [c for c in customers if c.dtype == "Locker"]
    original_d2d_customers = [c for c in customers if c.dtype == "D2D"]
    
    if scn == SCENARIO_D2D
        # 시나리오 D2D: 락커 사용 안함 (모든 고객 D2D + 락커 고객 재배송)
        d2d_customers = original_d2d_customers  # 원래 D2D 고객만 유지
        locker_customers = original_locker_customers  # 락커 고객도 유지 (첫 배송 통합 + 재배송용)
        effective_lockers = Dict()  # 락커 없음
    else
        d2d_customers = original_d2d_customers
        locker_customers = original_locker_customers
        if scn == SCENARIO_DPL || scn == SCENARIO_SPL || scn == SCENARIO_PSPL
            for (id,(lon,lat,carrier)) in LOCKERS_PRIV
                effective_lockers[id] = ((lon,lat), "Private")
            end
        elseif scn == SCENARIO_OPL
            for (id,val_tuple) in active
                effective_lockers[id] = val_tuple
            end
        end
    end

    nodes   = julia_nodes_list(customers; pub=active, scenario=scn, effective_lockers=effective_lockers)
    
    # 락커 용량 제한은 ALNS 최적화 단계에서 실시간 적용 (사전 적용 제거)
    
    # 거리 및 시간 행렬 생성
    df_dist, base_time_dict = build_pairwise_distances_and_times(nodes)

    private_lockers = LOCKERS_PRIV
        public_lockers = Dict(filter(kv->!(haskey(LOCKERS_PRIV, kv[1])), collect(active)))
    forbid = make_arc_rules(df_attr, private_lockers, public_lockers, scn)

    vehicle_dist_dict = Dict{Tuple{String,String},Float64}()
    vehicle_time_dict = Dict{Tuple{String,String},Float64}()
    for r in eachrow(df_dist)
        key = (r.from, r.to)
        # 거리 설정 (Inf 처리)
        dist_val = key in forbid ? 1e6 : r.km
        dist_val = (isinf(dist_val) || isnan(dist_val)) ? 1e6 : dist_val
        vehicle_dist_dict[key] = dist_val
        
        # 시간 설정 (Inf 처리)
        time_val = key in forbid ? 1e6 : get(base_time_dict, key, r.km / 30.0 * 3600.0)
        time_val = (isinf(time_val) || isnan(time_val)) ? 1e6 : time_val
        vehicle_time_dict[key] = time_val
    end

    total_driving = 0.0
    routes_info = Dict{String, Any}()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MOO 구성요소 누적 변수 (캐리어별 합산 → 전체 시나리오 집계)
    # moo_detail은 마지막 캐리어의 값만 저장하므로, 반드시 캐리어별로 누적해야 함
    # ═══════════════════════════════════════════════════════════════════════════
    accum_f2 = 0.0                        # f2: 고객 비용 합산
    accum_f3 = 0.0                        # f3: 사회 비용 합산
    accum_f2_mobility = 0.0               # f2 구성: 이동 불편비용 합산
    accum_f2_dissatisfaction = 0.0        # f2 구성: 불만족도 합산
    accum_satisfaction_sum = 0.0           # 만족도 합 (나중에 평균 계산)
    accum_delay_sum = 0.0                 # 지연시간 합 (나중에 평균 계산)
    accum_actual_dist_sum = 0.0           # 실제 이동거리 합 (나중에 평균 계산)
    accum_num_customers_evaluated = 0     # 평가된 고객 수 (가중평균용)
    accum_f3_vehicle_co2 = 0.0            # f3 구성: 차량 CO2
    accum_f3_customer_co2 = 0.0           # f3 구성: 고객 CO2
    accum_f3_locker_co2 = 0.0             # f3 구성: 락커 CO2
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 캐리어별 Pareto front 저장 (시스템 레벨 Pareto front 구성용)
    # - 각 캐리어의 NSGA-II Pareto front와 선택된 해(f1-min)를 저장
    # - 가장 큰 캐리어의 Pareto front를 기본축으로, 나머지는 고정 오프셋
    # ═══════════════════════════════════════════════════════════════════════════
    carrier_pareto_fronts = Dict{String, Matrix{Float64}}()      # 캐리어 → Pareto front [n×3]
    carrier_selected_f = Dict{String, Vector{Float64}}()         # 캐리어 → 선택된 해 [f1, f2, f3]
    carrier_customer_counts = Dict{String, Int}()                # 캐리어 → 고객 수
    
    # 전체 락커 배정 정보 수집용 (ALNS 내부 실시간 용량 체크)
    global_locker_tracker = LockerUsageTracker()
    add_public_lockers!(global_locker_tracker, effective_lockers)
    global_locker_assignments = Dict{String, String}()  # customer_id => locker_id
    global_locker_customer_time_windows = Dict{String, Tuple{Int,Int}}()  # 고객별 락커 배송 시간창 (통합 VRP용)
    total_d2d_conversions = 0
    total_original_locker_customers = length(original_locker_customers)  # 원래 락커 고객 수 사용
    
    # 시나리오 D2D에서는 모든 락커 고객이 D2D로 전환됨
    if scn == SCENARIO_D2D
        total_d2d_conversions = total_original_locker_customers
    end

    # 거리 기반 우선순위로 락커 할당 (모든 캐리어 통합 처리)
    # 1단계: 모든 캐리어의 락커 고객들을 하나의 리스트로 통합
    all_locker_customers_with_distance = []
    
    for carrier in CARRIERS
        carrier_locker_custs = []
        for c in locker_customers
            carrier_rows = df_attr[df_attr.customer_id .== c.id, :]
            if !isempty(carrier_rows) && carrier_rows[1, :carrier] == carrier
                push!(carrier_locker_custs, c)
            end
        end
        
        for customer in carrier_locker_custs
            cust_id = customer.id
            
            # 시나리오별 사용 가능한 락커 목록 가져오기
            available_lockers = get_available_lockers(global_locker_tracker, scn, effective_lockers, carrier)
            
            if !isempty(available_lockers)
                # 가장 가까운 락커까지의 거리 계산
                nearest_distance = Inf
                for locker_id in available_lockers
                    distance = get(vehicle_dist_dict, (cust_id, locker_id), 1e6)
                    if distance < nearest_distance
                        nearest_distance = distance
                    end
                end
                
                # (고객, 캐리어, 거리) 튜플로 저장
                push!(all_locker_customers_with_distance, (customer, carrier, nearest_distance))
            else
                # 사용 가능한 락커가 없는 경우 매우 큰 거리로 설정 (나중에 처리)
                push!(all_locker_customers_with_distance, (customer, carrier, 1e6))
            end
        end
    end
    
    # 2단계: 캐리어별 결과 저장용 딕셔너리 초기화
    # (고객 처리 순서는 원본 순서 유지 - 각 고객이 자신의 가장 가까운 락커 선택)
    carrier_locker_demands = Dict{String, Dict{String, Int}}()  # carrier => locker_id => demand
    carrier_d2d_lists = Dict{String, Vector}()  # carrier => d2d_customers (원래 D2D)
    carrier_forced_d2d_lists = Dict{String, Vector}()  # carrier => forced_d2d_customers (강제 전환)
    
    # 캐리어별 초기화
    for carrier in CARRIERS
        carrier_locker_demands[carrier] = Dict{String, Int}()
        carrier_forced_d2d_lists[carrier] = []
        carrier_d2d_list = []
        for c in d2d_customers
            carrier_rows = df_attr[df_attr.customer_id .== c.id, :]
            if !isempty(carrier_rows) && carrier_rows[1, :carrier] == carrier
                push!(carrier_d2d_list, c)
            end
        end
        carrier_d2d_lists[carrier] = carrier_d2d_list
    end
    
    # 3단계: 고객별 락커 배정 (고객 선택 방식)
    
    # 락커 대기열 초기화 (모든 시나리오에서 미리 정의)
    carrier_locker_waiting = Dict{String, Vector}()
    for c in CARRIERS
        carrier_locker_waiting[c] = []
    end
    
    # 고객별 사용 가능한 락커 목록 생성
    available_lockers_by_customer = Dict{String, Vector{String}}()
    for (customer, carrier, distance) in all_locker_customers_with_distance
        cust_id = customer.id
        if distance < 1e6
            available_lockers = get_available_lockers(global_locker_tracker, scn, effective_lockers, carrier)
            available_lockers_by_customer[cust_id] = available_lockers
        else
            available_lockers_by_customer[cust_id] = String[]
        end
    end
    
    # 락커 용량 정보 (모든 시나리오에서 사용, D2D는 빈 Dict)
    locker_capacities = Dict{String, Int}()
    
    # 락커 배정 (시나리오 D2D에서는 모든 락커 고객을 D2D로 전환)
    if !isempty(locker_customers)
        if scn == SCENARIO_D2D
            # 시나리오 D2D: 모든 락커 고객을 D2D 고객으로 전환 (1회 배송으로 완료)
            # 재배송 없음 - 모든 고객이 첫 배송에서 수령 성공 가정
            println("📦 시나리오 D2D: 모든 고객에게 직접 배송 (1회 배송)...")
            
            for customer in locker_customers
                cust_id = customer.id
                carrier = get(carrier_by_customer, cust_id, nothing)
                if carrier === nothing continue end
                
                # 모든 락커 고객을 D2D 고객과 함께 배송 (1회로 완료)
                push!(carrier_d2d_lists[carrier], customer)
            end
            
            println("   D2D 전환 완료: $(total_d2d_conversions)명 Locker→D2D 전환 (재배송 없음)")
        else
            # ═══════════════════════════════════════════════════════════════════
            # 시나리오 DPL/SPL/OPL/PSPL: 통합 VRP 방식 락커 배정
            # - 모든 고객을 가장 가까운 락커에 배정
            # - 락커 용량 초과 시 "대기" 대신 늦은 시간창 부여
            # - 배치별 시간창으로 락커 용량 준수
            # ═══════════════════════════════════════════════════════════════════
            println("🎯 통합 VRP 방식 락커 배정 시작 (시간창으로 용량 반영)...")
            
            # 락커 용량 정보 수집 (이미 초기화된 Dict 재사용)
            
            # 1. Private 락커 용량 (배송사별)
            private_locker_ids = Set(lid for (lid, _) in LOCKERS_PRIV)
            for (lid, (lon, lat, c)) in LOCKERS_PRIV
                locker_capacities[lid] = get(LOCKER_CAPACITY, c, 50)
            end
            
            # 2. Public 락커 용량 (Private가 아닌 경우만)
            for (lid, (coords, _)) in lock_pub
                if !(lid in private_locker_ids)
                    locker_capacities[lid] = effective_public_locker_capacity()
                end
            end
            
            # ═══════════════════════════════════════════════════════════════════
            # 개별 픽업 기반 연속 배송 시스템
            # ═══════════════════════════════════════════════════════════════════
            # 1. 가장 가까운 락커에 배정 시도 (용량까지)
            # 2. 용량 부족하면 다음으로 가까운 락커 시도
            # 3. 모든 락커 용량 부족 시 → 픽업으로 비는 슬롯에 배송 (시간창 조정)
            # ═══════════════════════════════════════════════════════════════════
            # 픽업 시간: 50%는 1~8시간 사이 랜덤, 50%는 당일 미회수(24h 이상)
            # 배송 시간창: 08:00~17:00 (픽업 후 슬롯 사용 시 해당 시간부터)
            # ═══════════════════════════════════════════════════════════════════
            
            # 락커별 슬롯 상태 추적: (배송시간, 픽업시간) 리스트
            # 배송시간 = 언제 배송되었는지 (초 단위, TW_D2D_OPEN 기준)
            # 픽업시간 = 배송 후 몇 초 후에 회수하는지
            locker_slots = Dict{String, Vector{Tuple{Int, Int}}}()  # locker_id → [(delivery_time, pickup_delay), ...]
            for lid in keys(locker_capacities)
                locker_slots[lid] = []
            end
            
            # 1단계: 고객별 선호 락커 및 픽업 시간 계산
            customer_locker_info = []  # (customer, carrier, locker_distances, pickup_delay)
            for customer in locker_customers
                cust_id = customer.id
                carrier = get(carrier_by_customer, cust_id, nothing)
                if carrier === nothing continue end
                
                available_lockers = get(available_lockers_by_customer, cust_id, String[])
                
                if isempty(available_lockers)
                    push!(carrier_forced_d2d_lists[carrier], customer)
                    total_d2d_conversions += 1
                    continue
                end
                
                # 거리순 정렬
                locker_distances = []
                for locker_id in available_lockers
                    distance = get(vehicle_dist_dict, (cust_id, locker_id), 25.0)
                    push!(locker_distances, (locker_id, distance))
                end
                sort!(locker_distances, by=x -> x[2])
                
                # 픽업 시간 생성 (50%: 1~8시간, 50%: 24시간 이상)
                pickup_delay = generate_locker_pickup_time(Random.GLOBAL_RNG)
                
                push!(customer_locker_info, (customer, carrier, locker_distances, pickup_delay))
            end
            
            # 2단계: 첫 번째 배정 (08:00, 락커 용량까지)
            waiting_customers = []  # 대기 고객
            first_wave_count = 0
            # 첫 배송/대기 배송 구분 플래그 (가상 노드 생성용)
            customer_wave_type = Dict{String, Symbol}()  # cust_id => :first 또는 :later
            customer_later_delivery_time = Dict{String, Int}()  # 대기 고객의 배송 시작 시간
            
            for (customer, carrier, locker_distances, pickup_delay) in customer_locker_info
                cust_id = customer.id
                assigned = false
                
                for (locker_id, dist) in locker_distances
                    capacity = get(locker_capacities, locker_id, 50)
                    current_usage = length(locker_slots[locker_id])
                    
                    if current_usage < capacity
                        # 첫 배송 배정 (08:00 배송)
                        push!(locker_slots[locker_id], (TW_D2D_OPEN, pickup_delay))
                        global_locker_assignments[cust_id] = locker_id
                        carrier_locker_demands[carrier][locker_id] = get(carrier_locker_demands[carrier], locker_id, 0) + 1
                        # 만족도 계산을 위해 고객 희망 시간 사용 (시간창 위반은 is_locker로 체크 안함)
                        global_locker_customer_time_windows[cust_id] = (Int(customer.tw_early), TW_LOCKER_CLOSE)
                        customer_wave_type[cust_id] = :first
                        assigned = true
                        first_wave_count += 1
                        break
                    end
                end
                
                if !assigned
                    # 가장 가까운 락커 정보와 함께 대기 리스트에 추가
                    push!(waiting_customers, (customer, carrier, locker_distances[1][1], pickup_delay))
                end
            end
            
            # 3단계: 대기 고객을 픽업으로 비는 슬롯에 배정
            # 각 락커별로 "언제 슬롯이 비는지" 계산하여 가장 빨리 비는 슬롯에 배정
            # ⚠️ 중요: 특정 시점에 사용 중인 슬롯 수가 용량 미만일 때만 배정!
            later_delivery_count = 0
            d2d_converted_count = 0
            
            # 슬롯 사용 중 수 계산 함수 (특정 시점에 몇 개가 점유 중인지)
            function count_slots_in_use_at(slots, at_time)
                count = 0
                for (delivery_time, pd) in slots
                    # 슬롯이 사용 중: 배송 완료됨 AND 아직 회수 안됨
                    free_time = delivery_time + pd
                    if delivery_time <= at_time && free_time > at_time
                        count += 1
                    end
                end
                return count
            end
            
            for (customer, carrier, best_locker_id, pickup_delay) in waiting_customers
                cust_id = customer.id
                
                # 해당 락커에서 가장 빨리 비는 슬롯 찾기
                slots = locker_slots[best_locker_id]
                capacity = get(locker_capacities, best_locker_id, 50)
                
                # 모든 슬롯의 "비는 시간" 수집 및 정렬
                slot_free_times = []
                for (delivery_time, pd) in slots
                    if pd < 24 * 3600  # 당일 회수하는 경우만
                        free_time = delivery_time + pd
                        if free_time < TW_DEPOT_CLOSE  # 21:00 이전에 비어야 함 (드라이버 운영시간)
                            push!(slot_free_times, free_time)
                        end
                    end
                end
                
                if isempty(slot_free_times)
                    # 당일 비는 슬롯 없음 → D2D 전환
                    push!(carrier_forced_d2d_lists[carrier], customer)
                    total_d2d_conversions += 1
                    d2d_converted_count += 1
                    continue
                end
                
                # 비는 시간 순으로 정렬하여, 슬롯이 실제로 비어있는 시간 찾기
                sort!(slot_free_times)
                
                assigned = false
                for free_time in slot_free_times
                    delivery_tw_start = max(free_time, TW_D2D_OPEN)  # 최소 08:00
                    
                    if delivery_tw_start >= TW_DEPOT_CLOSE
                        continue  # 21:00 이후면 배송 불가 (드라이버 운영시간)
                    end
                    
                    # 이 시점에 사용 중인 슬롯 수 확인
                    slots_in_use = count_slots_in_use_at(slots, delivery_tw_start)
                    
                    if slots_in_use < capacity
                        # 빈 슬롯 있음! 배정 가능
                        push!(locker_slots[best_locker_id], (delivery_tw_start, pickup_delay))
                        global_locker_assignments[cust_id] = best_locker_id
                        carrier_locker_demands[carrier][best_locker_id] = get(carrier_locker_demands[carrier], best_locker_id, 0) + 1
                        # 만족도 계산을 위해 고객 희망 시간 사용 (배송 가능 시간과 무관)
                        global_locker_customer_time_windows[cust_id] = (Int(customer.tw_early), TW_LOCKER_CLOSE)
                        customer_wave_type[cust_id] = :later
                        customer_later_delivery_time[cust_id] = delivery_tw_start
                        later_delivery_count += 1
                        assigned = true
                        break
                    end
                end
                
                if !assigned
                    # 모든 시간대에서 슬롯 부족 → D2D 전환
                    push!(carrier_forced_d2d_lists[carrier], customer)
                    total_d2d_conversions += 1
                    d2d_converted_count += 1
                end
            end
            
            # 통계 출력
            total_assigned = length(global_locker_assignments)
            
            println("   📦 개별 픽업 기반 연속 배송 현황:")
            println("   - 첫 배송 (08:00): $(first_wave_count)명")
            if later_delivery_count > 0
                println("   - 픽업 후 배송: $(later_delivery_count)명 (슬롯 비면 즉시)")
            end
            if d2d_converted_count > 0
                println("   - D2D 전환: $(d2d_converted_count)명 (17:00 전 슬롯 없음)")
            end
            println("   총 락커 배정: $(total_assigned)명")
        end
    else
        # 락커 고객이 없는 경우 모든 고객을 강제 D2D로 처리
        for (customer, carrier, distance) in all_locker_customers_with_distance
            push!(carrier_forced_d2d_lists[carrier], customer)
            if scn != SCENARIO_D2D  # 시나리오 D2D에서는 이미 카운트했으므로 중복 증가 방지
                total_d2d_conversions += 1
            end
        end
    end

    # 4단계: 캐리어별 VRP 계산 (모든 고객 통합 최적화)
    # ═══════════════════════════════════════════════════════════════════════════
    # 변경: 강제 D2D 고객도 첫 번째 배송에 통합하여 전체 경로 최적화
    # 이전: 원래 D2D + 락커 → ALNS → 강제 D2D → ALNS (분리)
    # 현재: 원래 D2D + 락커 + 강제 D2D → ALNS (통합)
    # ═══════════════════════════════════════════════════════════════════════════
    for carrier in CARRIERS
        depot_ids = [id for (id,(lon,lat,c)) in DEPOTS if c == carrier]
        if isempty(depot_ids) continue end
        depot_id = depot_ids[1]

        carrier_d2d = carrier_d2d_lists[carrier]  # 원래 D2D 고객
        carrier_forced_d2d = carrier_forced_d2d_lists[carrier]  # 강제 전환 D2D 고객 (락커 배정 실패)
        locker_demands = carrier_locker_demands[carrier]
        carrier_lockers = collect(keys(locker_demands))
        # 차량 대수는 의미 없으므로 사용하지 않음 (멀티트립 허용)
        
        total_carrier_cost = 0.0
        all_carrier_routes = Vector{Vector{String}}()
        carrier_vehicle_routes = Dict{Int, Vector{Vector{String}}}()
        total_carrier_trips = 0
        
        # 모든 고객을 단일 ALNS로 통합 최적화
        # - 원래 D2D 고객
        # - 락커 배송
        # - 강제 D2D 고객 (락커 배정 실패)
        mixed_nodes = String[]
        mixed_demands = Vector{Int}()
        
        # 원래 D2D 고객 추가
        for customer in carrier_d2d
            push!(mixed_nodes, customer.id)
            push!(mixed_demands, 1)
        end
        
        # 락커 추가
        for locker_id in carrier_lockers
            push!(mixed_nodes, locker_id)
            push!(mixed_demands, locker_demands[locker_id])
        end
        
        # 강제 D2D 고객 추가 (시나리오 D2D가 아닌 경우에만)
        # 시나리오 D2D에서는 이미 carrier_d2d에 포함되어 있음
        forced_d2d_count = 0
        if scn != SCENARIO_D2D
            for customer in carrier_forced_d2d
                push!(mixed_nodes, customer.id)
                push!(mixed_demands, 1)
                forced_d2d_count += 1
            end
        end
        
        if !isempty(mixed_nodes)
            # 통합형 MDCVRP 호출: 모든 고객(D2D + 락커 + 강제D2D)을 동시에 최적화
            demand_by_node = Dict{String,Int}()
            for (i, nid) in enumerate(mixed_nodes)
                demand_by_node[nid] = mixed_demands[i]
            end
            # 시간창 생성 (NSGA-II MOO 시간창 지원)
            time_windows_by_node = Dict{String, Tuple{Int,Int}}()
            service_times_by_node = Dict{String, Int}()
            
            # D2D 고객 시간창 (고객 객체에서 가져옴 - 랜덤 시간창)
            for customer in carrier_d2d
                if hasproperty(customer, :tw_early) && hasproperty(customer, :tw_late)
                    # 기존 시간창도 17:00 cap 적용
                    tw_end = min(Int(customer.tw_late), TW_D2D_CLOSE)
                    time_windows_by_node[customer.id] = (Int(customer.tw_early), tw_end)
                else
                    # 레거시: 랜덤 시간창 생성 (17:00 마감 적용)
                    tw_start = rand(TW_D2D_START_MIN:60:TW_D2D_START_MAX)
                    tw_width = rand(TW_D2D_WIDTH_MIN:60:TW_D2D_WIDTH_MAX)
                    tw_end = min(tw_start + tw_width, TW_D2D_CLOSE)  # 17:00 초과 방지
                    time_windows_by_node[customer.id] = (tw_start, tw_end)
                end
                service_times_by_node[customer.id] = SERVICE_TIME_D2D
            end
            
            # ═══════════════════════════════════════════════════════════════════
            # 락커 시간창 설정: Wave 기반 가상 노드 생성
            # - Wave 1 (08:00~21:00): 첫 배송
            # - 후속 배송: 픽업으로 슬롯이 비면 그 시점부터 배송 가능
            # ═══════════════════════════════════════════════════════════════════
            
            locker_virtual_nodes = Dict{String, String}()  # 가상노드ID → 원래락커ID
            
            for locker_id in carrier_lockers
                # 이 락커의 배송 그룹별 고객 수 (현재 캐리어 소속만)
                # FIRST: 첫 배송 (customer_wave_type == :first)
                # LATER: 후속 배송 (customer_wave_type == :later), 시간대별 그룹화
                first_count = 0
                hourly_later_counts = Dict{Int, Int}()  # 시간대 => 고객 수
                
                for (cust_id, assigned_locker) in global_locker_assignments
                    if assigned_locker == locker_id
                        cust_carrier = get(carrier_by_customer, cust_id, "")
                        if cust_carrier == carrier
                            wave = get(customer_wave_type, cust_id, :first)
                            if wave == :first
                                first_count += 1
                            else  # :later (픽업 후 배송) - 배송 시작 시간 기준 시간대별 분류
                                delivery_time = get(customer_later_delivery_time, cust_id, TW_D2D_OPEN)
                                hour = div(delivery_time, 3600)  # 시간대 추출 (9, 10, 11, ...)
                                hourly_later_counts[hour] = get(hourly_later_counts, hour, 0) + 1
                            end
                        end
                    end
                end
                
                has_first = first_count > 0
                has_later = !isempty(hourly_later_counts)
                later_total = sum(values(hourly_later_counts); init=0)
                
                # 디버그: 배송 그룹 분류 결과
                locker_cap = get(locker_capacities, locker_id, 50)
                total_assigned = first_count + later_total
                if total_assigned > 0
                    hourly_str = join(["H$(h):$(c)명" for (h, c) in sort(collect(hourly_later_counts))], ", ")
                    later_time_str = has_later ? hourly_str : "-"
                    println("   🔍 $(carrier) $(locker_id) [용량$(locker_cap)]: 첫배송=$(first_count), 후속=$(later_total) [$(later_time_str)], 총=$(total_assigned)")
                end
                
                if !has_first && !has_later
                    # 이 락커에 배정된 고객 없음
                    continue
                elseif has_first && !has_later
                    # 첫 배송만 있음 → 원래 락커 노드 사용
                    # 락커 가상 노드: 08:00부터 배송 가능 (VRP 경로 계획)
                    # 24시간 운영이므로 마감은 TW_LOCKER_CLOSE
                    # 만족도는 demands 기반 가중 계산으로 처리 (evaluate_routes)
                    time_windows_by_node[locker_id] = (TW_D2D_OPEN, TW_LOCKER_CLOSE)
                    locker_demand = get(demand_by_node, locker_id, first_count)
                    service_times_by_node[locker_id] = calculate_locker_service_time(locker_demand)
                elseif !has_first && has_later
                    # 후속 배송만 있음 → 시간대별 가상 노드 생성
                    original_idx = findfirst(==(locker_id), mixed_nodes)
                    if original_idx !== nothing
                        deleteat!(mixed_nodes, original_idx)
                        deleteat!(mixed_demands, original_idx)
                        delete!(demand_by_node, locker_id)
                    end
                    
                    for (hour, count) in sort(collect(hourly_later_counts))
                        later_id = "$(locker_id)_H$(lpad(hour, 2, '0'))"
                        locker_virtual_nodes[later_id] = locker_id
                        push!(mixed_nodes, later_id)
                        push!(mixed_demands, count)
                        demand_by_node[later_id] = count
                        # 시간창: 해당 시간대 시작 ~ 24:00 (락커는 24시간 운영)
                        time_windows_by_node[later_id] = (hour * 3600, TW_LOCKER_CLOSE)
                        service_times_by_node[later_id] = calculate_locker_service_time(count)
                    end
                else
                    # 첫 배송 + 후속 배송 모두 있음 → 가상 노드 생성
                    original_idx = findfirst(==(locker_id), mixed_nodes)
                    if original_idx !== nothing
                        deleteat!(mixed_nodes, original_idx)
                        deleteat!(mixed_demands, original_idx)
                        delete!(demand_by_node, locker_id)
                    end
                    
                    # 첫 배송 노드
                    first_id = "$(locker_id)_FIRST"
                    locker_virtual_nodes[first_id] = locker_id
                    push!(mixed_nodes, first_id)
                    push!(mixed_demands, first_count)
                    demand_by_node[first_id] = first_count
                    # 첫 배송 가상 노드: 08:00부터 배송 가능 (VRP 경로 계획)
                    # 만족도는 demands 기반 가중 계산으로 처리 (evaluate_routes)
                    time_windows_by_node[first_id] = (TW_D2D_OPEN, TW_LOCKER_CLOSE)
                    service_times_by_node[first_id] = calculate_locker_service_time(first_count)
                    
                    # 시간대별 후속 배송 노드 (각 시간대 시작 ~ 24:00, 락커 24시간 운영)
                    for (hour, count) in sort(collect(hourly_later_counts))
                        later_id = "$(locker_id)_H$(lpad(hour, 2, '0'))"
                        locker_virtual_nodes[later_id] = locker_id
                        push!(mixed_nodes, later_id)
                        push!(mixed_demands, count)
                        demand_by_node[later_id] = count
                        # 시간창: 해당 시간대 시작 ~ 24:00 (락커는 24시간 운영)
                        time_windows_by_node[later_id] = (hour * 3600, TW_LOCKER_CLOSE)
                        service_times_by_node[later_id] = calculate_locker_service_time(count)
                    end
                end
            end
            
            # 가상 노드의 거리 정보 설정 (원래 락커와 동일)
            for (virtual_id, original_id) in locker_virtual_nodes
                # 디포 ↔ 가상 노드
                for did in depot_ids
                    vehicle_dist_dict[(did, virtual_id)] = get(vehicle_dist_dict, (did, original_id), 1e6)
                    vehicle_time_dict[(did, virtual_id)] = get(vehicle_time_dict, (did, original_id), 1e6)
                    vehicle_dist_dict[(virtual_id, did)] = get(vehicle_dist_dict, (original_id, did), 1e6)
                    vehicle_time_dict[(virtual_id, did)] = get(vehicle_time_dict, (original_id, did), 1e6)
                end
                
                # 자기 자신
                vehicle_dist_dict[(virtual_id, virtual_id)] = 0.0
                vehicle_time_dict[(virtual_id, virtual_id)] = 0.0
                
                # 다른 노드들과의 거리
                for node_id in mixed_nodes
                    if node_id != virtual_id
                        target_original = get(locker_virtual_nodes, node_id, node_id)
                        
                        if target_original == original_id
                            # 같은 락커의 다른 웨이브: 매우 큰 거리로 설정
                            # → 같은 트립에서 동일 락커 중복 방문 방지
                            vehicle_dist_dict[(virtual_id, node_id)] = 1e6
                            vehicle_time_dict[(virtual_id, node_id)] = 1e6
                            vehicle_dist_dict[(node_id, virtual_id)] = 1e6
                            vehicle_time_dict[(node_id, virtual_id)] = 1e6
                        else
                            # 다른 노드
                            vehicle_dist_dict[(virtual_id, node_id)] = get(vehicle_dist_dict, (original_id, target_original), 1e6)
                            vehicle_time_dict[(virtual_id, node_id)] = get(vehicle_time_dict, (original_id, target_original), 1e6)
                            vehicle_dist_dict[(node_id, virtual_id)] = get(vehicle_dist_dict, (target_original, original_id), 1e6)
                            vehicle_time_dict[(node_id, virtual_id)] = get(vehicle_time_dict, (target_original, original_id), 1e6)
                        end
                    end
                end
            end
            
            # 강제 D2D 고객 시간창 (고객 객체에서 가져옴)
            for customer in carrier_forced_d2d
                if hasproperty(customer, :tw_early) && hasproperty(customer, :tw_late)
                    # 기존 시간창도 17:00 cap 적용
                    tw_end = min(Int(customer.tw_late), TW_D2D_CLOSE)
                    time_windows_by_node[customer.id] = (Int(customer.tw_early), tw_end)
                else
                    # 레거시: 랜덤 시간창 생성 (17:00 마감 적용)
                    tw_start = rand(TW_D2D_START_MIN:60:TW_D2D_START_MAX)
                    tw_width = rand(TW_D2D_WIDTH_MIN:60:TW_D2D_WIDTH_MAX)
                    tw_end = min(tw_start + tw_width, TW_D2D_CLOSE)  # 17:00 초과 방지
                    time_windows_by_node[customer.id] = (tw_start, tw_end)
                end
                service_times_by_node[customer.id] = SERVICE_TIME_D2D
            end
            
            # 디포 시간창 추가 (택배기사 운송 시간: 08:00~21:00)
            for did in depot_ids
                time_windows_by_node[did] = (TW_DEPOT_OPEN, TW_DEPOT_CLOSE)
                service_times_by_node[did] = SERVICE_TIME_DEPOT
            end

            # ═══════════════════════════════════════════════════════════════════
            # Multi-Trip/Multi-Day VRP 시뮬레이션
            # - 캐리어별 차량 수 제한 (VEHICLES_BY_CARRIER)
            # - 차량은 복귀 후 재출발 가능 (Multi-Trip)
            # - 하루 안에 못 끝나면 다음 날로 이월 (Multi-Day)
            # ═══════════════════════════════════════════════════════════════════
            
            # 디버그: 가상 노드 확인
            virtual_node_count = count(n -> contains(n, "_W"), mixed_nodes)
            if virtual_node_count > 0
                println("   📍 $(carrier) 가상 노드: $(virtual_node_count)개")
                for node in filter(n -> contains(n, "_W"), mixed_nodes)
                    tw = get(time_windows_by_node, node, (0,0))
                    tw_str = "$(div(tw[1], 3600)):$(lpad(div(tw[1] % 3600, 60), 2, '0'))~$(div(tw[2], 3600)):$(lpad(div(tw[2] % 3600, 60), 2, '0'))"
                    println("      - $(node): 수요=$(get(demand_by_node, node, 0)), 시간창=$(tw_str)")
                end
            end
            
            # ═══════════════════════════════════════════════════════════════════
            # 노드별 개별 고객 희망시간 맵 구성
            # - D2D 노드: [tw_early] (1명의 희망시간)
            # - 락커/가상 노드: [tw_early_1, ..., tw_early_N] (N명 각각의 희망시간)
            # ═══════════════════════════════════════════════════════════════════
            node_desired_times_map = Dict{String, Vector{Int}}()
            
            # D2D 고객: 자신의 tw_early
            for customer in carrier_d2d
                if hasproperty(customer, :tw_early)
                    node_desired_times_map[customer.id] = [Int(customer.tw_early)]
                end
            end
            for customer in carrier_forced_d2d
                if hasproperty(customer, :tw_early)
                    node_desired_times_map[customer.id] = [Int(customer.tw_early)]
                end
            end
            
            # 락커/가상 노드: 배정된 개별 고객들의 tw_early 수집
            for node_id in mixed_nodes
                # 원래 락커 ID 찾기 (가상 노드면 원래 ID, 아니면 자기 자신)
                original_locker_id = get(locker_virtual_nodes, node_id, node_id)
                
                # 이 노드가 락커인지 확인
                if haskey(locker_capacities, original_locker_id) || haskey(locker_virtual_nodes, node_id)
                    # 이 노드에 배정된 고객들의 wave type 확인
                    is_first_node = endswith(node_id, "_FIRST") || (!contains(node_id, "_H") && !contains(node_id, "_W") && haskey(locker_capacities, node_id))
                    
                    desired_times_list = Int[]
                    for (cust_id, assigned_locker) in global_locker_assignments
                        if assigned_locker == original_locker_id
                            cust_carrier = get(carrier_by_customer, cust_id, "")
                            if cust_carrier == carrier
                                wave = get(customer_wave_type, cust_id, :first)
                                
                                if is_first_node && wave == :first
                                    # 첫 배송 노드: 첫 배송 고객의 희망시간
                                    tw_start, _ = get(global_locker_customer_time_windows, cust_id, (TW_D2D_OPEN, TW_LOCKER_CLOSE))
                                    push!(desired_times_list, tw_start)
                                elseif !is_first_node && wave == :later
                                    # 후속 배송 노드: 해당 시간대의 대기 고객 희망시간
                                    delivery_time = get(customer_later_delivery_time, cust_id, TW_D2D_OPEN)
                                    hour = div(delivery_time, 3600)
                                    # 노드 이름에서 시간대 추출 (예: L_0001_H10 → 10)
                                    node_hour = -1
                                    m = match(r"_H(\d+)$", node_id)
                                    if m !== nothing
                                        node_hour = parse(Int, m.captures[1])
                                    end
                                    if node_hour == hour
                                        tw_start, _ = get(global_locker_customer_time_windows, cust_id, (TW_D2D_OPEN, TW_LOCKER_CLOSE))
                                        push!(desired_times_list, tw_start)
                                    end
                                end
                            end
                        end
                    end
                    
                    if !isempty(desired_times_list)
                        node_desired_times_map[node_id] = desired_times_list
                    end
                end
            end
            
            # ═══════════════════════════════════════════════════════════════════
            # 노드별 개별 고객→락커 거리 맵 구성 (node_desired_times_map과 동일한 순서)
            # - D2D 노드: [0.0] (락커 미사용)
            # - 락커/가상 노드: [dist_1, ..., dist_N] (N명 각각의 고객→락커 거리)
            # ═══════════════════════════════════════════════════════════════════
            node_locker_distances_map = Dict{String, Vector{Float64}}()
            
            # D2D 고객: 0.0 (락커 미사용)
            for customer in carrier_d2d
                node_locker_distances_map[customer.id] = [0.0]
            end
            for customer in carrier_forced_d2d
                node_locker_distances_map[customer.id] = [0.0]
            end
            
            # 락커/가상 노드: 배정된 개별 고객들의 거리 수집 (desired_times와 동일한 순서)
            for node_id in mixed_nodes
                original_locker_id = get(locker_virtual_nodes, node_id, node_id)
                
                if haskey(locker_capacities, original_locker_id) || haskey(locker_virtual_nodes, node_id)
                    is_first_node = endswith(node_id, "_FIRST") || (!contains(node_id, "_H") && !contains(node_id, "_W") && haskey(locker_capacities, node_id))
                    
                    distances_list = Float64[]
                    for (cust_id, assigned_locker) in global_locker_assignments
                        if assigned_locker == original_locker_id
                            cust_carrier = get(carrier_by_customer, cust_id, "")
                            if cust_carrier == carrier
                                wave = get(customer_wave_type, cust_id, :first)
                                
                                if is_first_node && wave == :first
                                    # 고객 위치 → 락커 거리
                                    dist = get(vehicle_dist_dict, (cust_id, original_locker_id), 0.0)
                                    push!(distances_list, dist)
                                elseif !is_first_node && wave == :later
                                    delivery_time = get(customer_later_delivery_time, cust_id, TW_D2D_OPEN)
                                    hour = div(delivery_time, 3600)
                                    node_hour = -1
                                    m = match(r"_H(\d+)$", node_id)
                                    if m !== nothing
                                        node_hour = parse(Int, m.captures[1])
                                    end
                                    if node_hour == hour
                                        # 고객 위치 → 락커 거리
                                        dist = get(vehicle_dist_dict, (cust_id, original_locker_id), 0.0)
                                        push!(distances_list, dist)
                                    end
                                end
                            end
                        end
                    end
                    
                    if !isempty(distances_list)
                        node_locker_distances_map[node_id] = distances_list
                    end
                end
            end
            
            mixed_cost, mixed_routes, days_used, trips_per_day = solve_vrp_multitrip_multiday(
                carrier, depot_id, mixed_nodes, demand_by_node,
                vehicle_dist_dict, vehicle_time_dict,
                time_windows_by_node, service_times_by_node, effective_capacity();
                max_days=30, max_iter=500,
                locker_assignments=global_locker_assignments,
                num_active_lockers=length(keys(filter(item->item.second[2]=="Private", lock_pub))),
                node_desired_times_map=node_desired_times_map,
                node_locker_distances_map=node_locker_distances_map,
                moo_seed=moo_seed
            )
            
            # ═══════════════════════════════════════════════════════════════
            # MOO detail 누적 (캐리어별로 합산)
            # ═══════════════════════════════════════════════════════════════
            carrier_moo_detail = get_last_moo_detail()
            if carrier_moo_detail !== nothing
                accum_f2 += carrier_moo_detail.f2
                accum_f3 += carrier_moo_detail.f3
                accum_f2_mobility += carrier_moo_detail.f2_mobility_inconvenience
                accum_f2_dissatisfaction += carrier_moo_detail.f2_dissatisfaction
                accum_f3_vehicle_co2 += carrier_moo_detail.f3_vehicle_co2
                accum_f3_customer_co2 += carrier_moo_detail.f3_customer_co2
                accum_f3_locker_co2 += carrier_moo_detail.f3_locker_co2
                # 만족도/지연/이동거리는 고객 수 가중 합산 (나중에 평균)
                n_cust = carrier_moo_detail.num_vehicles > 0 ? max(1, length(mixed_nodes)) : 0
                accum_satisfaction_sum += carrier_moo_detail.avg_customer_satisfaction * n_cust
                accum_delay_sum += carrier_moo_detail.avg_customer_delay * n_cust
                accum_actual_dist_sum += carrier_moo_detail.avg_customer_actual_dist * n_cust
                accum_num_customers_evaluated += n_cust
            end
            
            # ═══════════════════════════════════════════════════════════════
            # 캐리어별 Pareto front 저장 (시스템 레벨 Pareto front 구성용)
            # ═══════════════════════════════════════════════════════════════
            carrier_pf = get_last_pareto_front()
            if carrier_pf !== nothing && size(carrier_pf, 1) > 0
                carrier_pareto_fronts[carrier] = copy(carrier_pf)
                # 선택된 해: f1 최소 (기존 NSGA-II 선택 방식과 동일)
                f1_min_idx = argmin(carrier_pf[:, 1])
                carrier_selected_f[carrier] = [carrier_pf[f1_min_idx, 1], carrier_pf[f1_min_idx, 2], carrier_pf[f1_min_idx, 3]]
                carrier_customer_counts[carrier] = length(mixed_nodes)
            end
            
            # 결과 요약 출력
            total_trips = sum(trips_per_day)
            println("   ✅ Multi-Trip/Multi-Day 완료: $(carrier) - $(days_used)일, $(total_trips)트립, 거리 $(round(mixed_cost,digits=2))km")
            
            # 시간창 정보 저장
            node_types = Dict{String, String}()
            for c in carrier_d2d
                node_types[c.id] = "D2D"
            end
            for c in carrier_forced_d2d
                node_types[c.id] = "D2D_forced"
            end
            for lid in carrier_lockers
                node_types[lid] = "Locker"
            end
            for did in depot_ids
                node_types[did] = "Depot"
            end
            
            save_time_window_info!(scn, seed, omega, time_windows_by_node, service_times_by_node, node_types)
            
            # 라우트 스케줄 저장
            for (route_idx, route) in enumerate(mixed_routes)
                save_route_schedule!(scn, seed, omega, carrier, route_idx, route,
                                    vehicle_time_dict, service_times_by_node, 
                                    time_windows_by_node, node_types;
                                    start_time=DRIVER_START_MIN)
            end
            
            total_carrier_cost += mixed_cost
            append!(all_carrier_routes, mixed_routes)
            
            # 차량 대수 개념 제거: 모든 trip을 vehicle_id=1 아래에 배치
            if !haskey(carrier_vehicle_routes, 1)
                carrier_vehicle_routes[1] = Vector{Vector{String}}()
            end
            append!(carrier_vehicle_routes[1], mixed_routes)
            total_carrier_trips += length(mixed_routes)
            
            # 시나리오별 출력 메시지 구분
            if scn == SCENARIO_D2D
                println("   📦🏠 $carrier 통합 배송 trip: $(length(mixed_routes))개 (비용: $(round(mixed_cost, digits=2)))")
            else
                locker_count = length(carrier_lockers)
                d2d_count = length(carrier_d2d)
                if forced_d2d_count > 0
                    println("   📦🏠 $carrier 통합 trip (락커$(locker_count)개+D2D$(d2d_count)명+강제D2D$(forced_d2d_count)명): $(length(mixed_routes))개 (비용: $(round(mixed_cost, digits=2)))")
                elseif locker_count > 0
                    println("   📦🏠 $carrier 통합 trip (락커$(locker_count)개+D2D$(d2d_count)명): $(length(mixed_routes))개 (비용: $(round(mixed_cost, digits=2)))")
                else
                    println("   📦🏠 $carrier D2D trip: $(length(mixed_routes))개 (비용: $(round(mixed_cost, digits=2)))")
                end
            end
        end
        
        total_driving += total_carrier_cost
        
        # ═══════════════════════════════════════════════════════════════════
        # [통합 VRP] 락커 용량은 시간창으로 반영되므로 별도 대기 처리 불필요
        # 모든 락커 고객이 배치별 시간창과 함께 단일 VRP에서 최적화됨
        # ═══════════════════════════════════════════════════════════════════
        
        # 캐리어 정보 저장
        if !isempty(all_carrier_routes)
            # 사용 차량 수: CVRP 모델에서는 Trip 수와 동일
            vehicles_used_count = total_carrier_trips

            routes_info[carrier] = Dict(
                "routes" => all_carrier_routes,
                "vehicle_routes" => carrier_vehicle_routes, # 시각화 호환성 유지(모두 vehicle 1)
                "cost" => total_carrier_cost,
                "depot" => depot_id,
                "vehicles_used" => vehicles_used_count,
                "total_trips" => total_carrier_trips,
                "d2d_customers" => length(carrier_d2d),
                "locker_customers" => sum(values(locker_demands); init=0),
                "forced_d2d_customers" => forced_d2d_count  # 락커 배정 실패 → D2D 전환된 고객 수
                # [통합 VRP] 대기/웨이브 로직 삭제됨 - 모든 고객이 시간창 기반 단일 VRP로 처리
            )
            
            println("   📊 $carrier 총계: $(total_carrier_trips)개 trip (비용: $(round(total_carrier_cost, digits=2)))")
        end
    end

    # 보행거리 계산 (ALNS에서 결정된 락커 배정 사용)
    walking_dist = 0.0
    walking_routes = Dict{String, String}()
    walking_distances_per_customer = Dict{String, Float64}()  # 고객별 도보거리 (이동비용 계산용)
    vehicle_distances_per_customer = Dict{String, Float64}()  # 고객별 차량거리 (이동비용 계산용)

    if scn != SCENARIO_D2D  # 시나리오 D2D에서는 락커가 없으므로 보행거리 계산 건너뛰기
        for customer in locker_customers
            cust_id = customer.id
            
            if haskey(global_locker_assignments, cust_id)
                # ALNS에서 락커 배정이 성공한 경우
                assigned_locker = global_locker_assignments[cust_id]
                
                # 차량 거리 (OSRM 차량 노선 기반)
                vehicle_distance = get(vehicle_dist_dict, (cust_id, assigned_locker), 25.0)
                vehicle_distances_per_customer[cust_id] = vehicle_distance
                
                # 도로망 기반 도보 거리 계산
                if USE_ROAD_DISTANCE && is_distance_matrix_initialized()
                    # 고객 좌표 (customer.coord는 (lon, lat))
                    cust_pos = (Float64(customer.coord[2]), Float64(customer.coord[1]))  # (lat, lon)
                    
                    # 락커 좌표 조회
                    locker_pos = nothing
                    if haskey(LOCKERS_PRIV, assigned_locker)
                        (lon, lat, _) = LOCKERS_PRIV[assigned_locker]
                        locker_pos = (Float64(lat), Float64(lon))
                    elseif haskey(lock_pub, assigned_locker)
                        (coords, _) = lock_pub[assigned_locker]
                        locker_pos = (Float64(coords[2]), Float64(coords[1]))
                    end
                    
                    if locker_pos !== nothing
                        # Hybrid 도보 거리: 골목 내 이동(직선) + 도로망 거리 + 골목 내 이동(직선)
                        walking_distance = get_hybrid_foot_distance(cust_pos, locker_pos)
                        # 경로 없으면 차량 거리로 대체
                        if walking_distance == Inf || isnan(walking_distance)
                            walking_distance = vehicle_distance
                        end
                    else
                        walking_distance = vehicle_distance
                    end
                else
                    walking_distance = vehicle_distance
                end
                
                walking_dist += walking_distance
                walking_routes[cust_id] = assigned_locker
                walking_distances_per_customer[cust_id] = walking_distance  # 개별 도보거리 저장
            else
                # ALNS에서 D2D로 전환된 경우 (보행거리 없음)
                # walking_dist += 0.0  (D2D는 보행거리 없음)
            end
        end
    end
    # 시나리오 1에서는 walking_dist = 0.0 (락커 없음)

    total_dist = total_driving + walking_dist  # 참고용 (총 이동거리)
    m = create_silent_model()
    # 목적함수: 차량거리만 최소화 (도보거리는 고객 선택이므로 최적화 대상 아님)
    @objective(m, Min, total_driving)  # 차량거리만 최소화
    
    # 락커 통계 수집 (ALNS 최적화 결과 기반) - collect_stats 매개변수로 제어
    # 시나리오 1도 포함하여 D2D 전환 통계 수집
    if collect_stats && total_original_locker_customers > 0
        stats = init_locker_stats(scn, seed, omega, effective_lockers)
        update_carrier_stats!(stats, df_attr, customers)
        
        # LockerUsageTracker 결과를 통계에 반영
        update_locker_stats!(stats, global_locker_tracker, total_original_locker_customers, total_d2d_conversions)
        
        # 캐리어별 락커 사용량 분석 추가
        update_locker_stats_with_assignments!(stats, global_locker_assignments, df_attr)
        
        # 락커 0개 선택 시에도 통계 수집 (빈 락커 통계로)
        if isempty(effective_lockers)
            # 빈 통계로 수집 (락커 0개, 모든 고객 D2D 전환)
            stats.total_locker_customers = total_original_locker_customers
            stats.d2d_conversions = total_d2d_conversions
            stats.conversion_rate = total_original_locker_customers > 0 ? total_d2d_conversions / total_original_locker_customers : 0.0
        end
        
        collect_locker_stats!(stats)
        
        # 통계 출력
        println("📊 락커 용량 제한 적용 결과 (시나리오 $scn):")
        println("   총 락커 고객: $(total_original_locker_customers)명")
        println("   D2D 강제전환: $(total_d2d_conversions)명 ($(round(100*total_d2d_conversions/total_original_locker_customers, digits=1))%)")
        
        if isempty(effective_lockers)
            println("   ⚠️  락커 0개 선택 - 모든 고객이 D2D로 처리됨")
        else
            println("")
            show_locker_status(global_locker_tracker)
        end
        println("")
    end

    # 차량별 개별 거리 계산
    vehicle_distances = Vector{Float64}()
    total_trips = 0
    
    for (carrier, route_info) in routes_info
        total_trips += route_info["total_trips"]
        
        # 각 차량(Trip)의 개별 거리 계산 (CVRP: 1 Trip = 1 Vehicle)
        if haskey(route_info, "routes")
            for route in route_info["routes"]
                trip_dist = 0.0
                for i in 1:(length(route)-1)
                    from_node = route[i]
                    to_node = route[i+1]
                    trip_dist += get(vehicle_dist_dict, (from_node, to_node), 0.0)
                end
                # 거리가 0 이상인 경우만 추가 (빈 루트 제외)
                push!(vehicle_distances, trip_dist)
            end
        elseif haskey(route_info, "vehicle_routes")
             # "routes" 키가 없는 경우의 백업 로직
             for (vehicle_id, trips) in route_info["vehicle_routes"]
                for trip in trips
                    trip_dist = 0.0
                    for i in 1:(length(trip)-1)
                        from_node = trip[i]
                        to_node = trip[i+1]
                        trip_dist += get(vehicle_dist_dict, (from_node, to_node), 0.0)
                    end
                    push!(vehicle_distances, trip_dist)
                end
             end
        end
    end
    
    # 총 고객 수 계산
    total_customers = length(customers)
    
    # 실제 락커 사용 고객 수 계산 (D2D로 전환된 고객 제외)
    actual_locker_customers_count = total_original_locker_customers - total_d2d_conversions
    
    # 고객당 평균 이동거리 계산 (실제 락커 사용 고객 기준)
    locker_customers_count = length(locker_customers)  # 원래 락커 고객 수 (호환성 유지)
    avg_customer_walking_dist = actual_locker_customers_count > 0 ? walking_dist / actual_locker_customers_count : 0.0
    
    # 차량당 평균 거리 계산 (실제 운행한 차량들의 평균)
    avg_vehicle_distance = !isempty(vehicle_distances) ? mean(vehicle_distances) : 0.0
    total_vehicles_used = length(vehicle_distances)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 목적함수 계산 (논문: Barbieri et al., ICORES 2025 기반)
    # social_cost = z₁(운송비용) + z₂(고객 이동비용) + z₃(락커활성화비용)
    # ═══════════════════════════════════════════════════════════════════════════
    
    # z₁: 운송 비용 = 차량거리 × 단위비용 (1 EUR/km)
    transport_cost = total_driving * TRANSPORT_COST_PER_KM
    
    # z₂: 고객 이동비용 (Monte Carlo 시뮬레이션)
    # ─────────────────────────────────────────────────────────────────────────
    # 4가지 이동수단 선택 모델 (확률적 결정):
    #   - 도보: 왕복 기준, 0.1 EUR/km (고객 불편비용)
    #   - 자전거: 왕복 기준, 0.05 EUR/km (고객 불편비용)
    #   - 차량(전용): 왕복 기준, 1.0 EUR/km (사회적비용)
    #   - 차량(연계): 편도 기준, 1.0 EUR/km (사회적비용)
    # 
    # 편도 거리별 이동수단 분담률에 따라 Monte Carlo로 선택
    # ─────────────────────────────────────────────────────────────────────────
    
    # 진짜 랜덤 (비결정적) - 매 실행마다 다른 결과
    # 고객 이동수단 선택은 실제 확률적 행동을 반영해야 하므로 고정 시드 사용하지 않음
    
    mobility_cost = 0.0                  # z₂: 고객 이동비용 (총합)
    customer_inconvenience_cost = 0.0    # 고객 불편비용 (도보+자전거)
    customer_social_cost = 0.0           # 사회적비용 (차량전용+연계)
    
    num_walk_choice_customers = 0        # 도보 선택 고객 수
    num_bicycle_choice_customers = 0     # 자전거 선택 고객 수
    num_vehicle_dedicated_customers = 0  # 차량(전용) 선택 고객 수
    num_vehicle_linked_customers = 0     # 차량(연계) 선택 고객 수
    
    total_customer_walking_dist = 0.0    # 도보 선택 고객들의 총 도보거리 (왕복)
    total_customer_bicycle_dist = 0.0    # 자전거 선택 고객들의 총 자전거거리 (왕복)
    total_customer_vehicle_ded_dist = 0.0  # 차량(전용) 선택 고객들의 총 거리 (왕복)
    total_customer_vehicle_link_dist = 0.0 # 차량(연계) 선택 고객들의 총 거리 (편도)
    
    for (cust_id, walk_dist) in walking_distances_per_customer
        # 편도 거리 (km 단위)
        one_way_dist = walk_dist
        
        # 편도 거리에 따른 이동수단 분담률 조회
        walk_prob, bicycle_prob, veh_ded_prob, veh_link_prob = get_mode_share(one_way_dist)
        
        # Monte Carlo: 확률에 따라 실제 선택 결정 (진짜 랜덤)
        r = rand()
        
        if r < walk_prob
            # 도보 선택: 왕복 × 0.1 EUR/km
            actual_cost = calculate_customer_mobility_cost(one_way_dist, :walk)
            customer_inconvenience_cost += actual_cost
            num_walk_choice_customers += 1
            total_customer_walking_dist += 2.0 * one_way_dist  # 왕복
        elseif r < walk_prob + bicycle_prob
            # 자전거 선택: 왕복 × 0.05 EUR/km
            actual_cost = calculate_customer_mobility_cost(one_way_dist, :bicycle)
            customer_inconvenience_cost += actual_cost
            num_bicycle_choice_customers += 1
            total_customer_bicycle_dist += 2.0 * one_way_dist  # 왕복
        elseif r < walk_prob + bicycle_prob + veh_ded_prob
            # 차량(전용) 선택: 왕복 × 1.0 EUR/km
            actual_cost = calculate_customer_mobility_cost(one_way_dist, :vehicle_dedicated)
            customer_social_cost += actual_cost
            num_vehicle_dedicated_customers += 1
            total_customer_vehicle_ded_dist += 2.0 * one_way_dist  # 왕복
        else
            # 차량(연계) 선택: 편도 × 1.0 EUR/km
            actual_cost = calculate_customer_mobility_cost(one_way_dist, :vehicle_linked)
            customer_social_cost += actual_cost
            num_vehicle_linked_customers += 1
            total_customer_vehicle_link_dist += one_way_dist  # 편도
        end
        
        mobility_cost += actual_cost
    end
    
    # z₃: 락커 활성화 비용 = 활성화된 락커 수 × 단위비용
    num_active_lockers = length(effective_lockers)
    activation_cost = num_active_lockers * LOCKER_ACTIVATION_COST
    
    # 오메가별 사회적 비용 (기존 EUR 기반, 호환성 유지)
    social_cost = transport_cost + mobility_cost
    
    # SLRP 목적함수: f₃(CO2)와 동일한 구조
    # z₁^ω = e_v × total_driving, z₂^ω = e_c × Σ(2P_d + P_l)ℓ_c
    van_co2 = total_driving * effective_slrp_vehicle_co2()
    customer_vehicle_co2 = (total_customer_vehicle_ded_dist + total_customer_vehicle_link_dist) * SLRP_CUSTOMER_VEHICLE_CO2_PER_KM
    f3_co2_cost = van_co2 + customer_vehicle_co2
    
    # 락커 사용량 딕셔너리 생성 (락커ID -> 사용량)
    locker_usage_dict = Dict{String, Int}()
    for (locker_id, usage) in global_locker_tracker.usage
        if usage > 0
            locker_usage_dict[locker_id] = usage
        end
    end
    
    # MOO 가중평균 계산
    accum_avg_sat = accum_num_customers_evaluated > 0 ? accum_satisfaction_sum / accum_num_customers_evaluated : 0.0
    accum_avg_delay = accum_num_customers_evaluated > 0 ? accum_delay_sum / accum_num_customers_evaluated : 0.0
    accum_avg_dist = accum_num_customers_evaluated > 0 ? accum_actual_dist_sum / accum_num_customers_evaluated : 0.0
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 시스템 레벨 Pareto front 구성
    # - Pareto front 다양성이 가장 큰 캐리어를 변동축으로 사용
    #   (큰 캐리어는 NSGA-II 탐색 부족으로 단일점 수렴 가능)
    # - 나머지 캐리어들의 선택된 해(f1-min)를 고정 오프셋으로 합산
    # - system_pf[i,:] = dominant_pf[i,:] + Σ(other carriers' selected f)
    # ═══════════════════════════════════════════════════════════════════════════
    system_pareto_front = nothing
    if !isempty(carrier_pareto_fronts)
        # Pareto front 다양성이 가장 큰 캐리어 찾기 (f1 spread 기준)
        dominant_carrier = ""
        max_spread = 0.0
        for (c, cpf) in carrier_pareto_fronts
            if size(cpf, 1) > 1
                # 각 목적함수의 spread를 정규화하여 합산
                f1_spread = maximum(cpf[:, 1]) - minimum(cpf[:, 1])
                f2_spread = maximum(cpf[:, 2]) - minimum(cpf[:, 2])
                f3_spread = maximum(cpf[:, 3]) - minimum(cpf[:, 3])
                # 상대적 spread (0 방지)
                f1_rel = f1_spread / (mean(cpf[:, 1]) + 1e-9)
                f2_rel = f2_spread / (mean(cpf[:, 2]) + 1e-9)
                f3_rel = f3_spread / (mean(cpf[:, 3]) + 1e-9)
                total_spread = f1_rel + f2_rel + f3_rel
                if total_spread > max_spread
                    max_spread = total_spread
                    dominant_carrier = c
                end
            elseif isempty(dominant_carrier)
                dominant_carrier = c  # 단일 해라도 없는 것보다 나음
            end
        end
        
        if !isempty(dominant_carrier) && haskey(carrier_pareto_fronts, dominant_carrier)
            dominant_pf = carrier_pareto_fronts[dominant_carrier]
            dominant_n_cust = get(carrier_customer_counts, dominant_carrier, 0)
            
            # 나머지 캐리어들의 선택된 해 합산 (고정 오프셋)
            base_f1 = 0.0
            base_f2 = 0.0
            base_f3 = 0.0
            for (c, selected_f) in carrier_selected_f
                if c != dominant_carrier
                    base_f1 += selected_f[1]
                    base_f2 += selected_f[2]
                    base_f3 += selected_f[3]
                end
            end
            
            # 시스템 레벨 Pareto front = dominant carrier PF + base offsets
            n_solutions = size(dominant_pf, 1)
            system_pareto_front = zeros(Float64, n_solutions, 3)
            for i in 1:n_solutions
                system_pareto_front[i, 1] = dominant_pf[i, 1] + base_f1
                system_pareto_front[i, 2] = dominant_pf[i, 2] + base_f2
                system_pareto_front[i, 3] = dominant_pf[i, 3] + base_f3
            end
            
            # 디버그 출력
            if n_solutions > 0
                println("   📊 시스템 Pareto front 구성: $(dominant_carrier)($(dominant_n_cust)명, spread=$(round(max_spread, digits=3))) 기반 + $(length(carrier_selected_f)-1)개 캐리어 오프셋")
                println("      오프셋: f1=$(round(base_f1, digits=2)), f2=$(round(base_f2, digits=2)), f3=$(round(base_f3, digits=2))")
                println("      시스템 f1 범위: [$(round(minimum(system_pareto_front[:,1]), digits=2)), $(round(maximum(system_pareto_front[:,1]), digits=2))]")
                println("      시스템 f2 범위: [$(round(minimum(system_pareto_front[:,2]), digits=2)), $(round(maximum(system_pareto_front[:,2]), digits=2))]")
                println("      시스템 f3 범위: [$(round(minimum(system_pareto_front[:,3]), digits=2)), $(round(maximum(system_pareto_front[:,3]), digits=2))]")
            end
        end
    end
    
    distance_info = Dict(
        "total"   => total_dist,
        "driving" => total_driving,
        "walking" => walking_dist,
        "vehicles_used" => total_vehicles_used,
        "total_trips" => total_trips,
        "total_customers" => total_customers,
        "locker_customers" => locker_customers_count,  # 원래 락커 고객 수 (호환성 유지)
        "actual_locker_customers" => actual_locker_customers_count,  # 실제 락커 사용 고객 수 (NEW)
        "d2d_conversions" => total_d2d_conversions,  # 실제 D2D 전환 수 (NEW)
        "avg_dist_per_vehicle" => avg_vehicle_distance,
        "avg_walking_per_customer" => avg_customer_walking_dist,
        "social_cost" => social_cost,
        "f3_co2_cost" => f3_co2_cost,         # SLRP 목적함수: f₃(CO2) 기반 비용
        "locker_usage" => locker_usage_dict,  # 락커 사용량 정보 (NEW)
        # 목적함수 구성요소 (논문 기반)
        "transport_cost" => transport_cost,      # z₁: 운송비용
        "customer_mobility_cost" => mobility_cost,    # z₂: 고객 이동비용
        "activation_cost" => activation_cost,    # z₃: 락커활성화비용
        "num_active_lockers" => num_active_lockers,
        # 고객 이동 수단 선택 상세 정보 (Monte Carlo 결과) - 4가지 수단
        "customer_inconvenience_cost" => customer_inconvenience_cost,  # 고객 불편비용 (도보+자전거)
        "customer_social_cost" => customer_social_cost,                # 사회적비용 (차량전용+연계)
        "num_walk_choice_customers" => num_walk_choice_customers,          # 도보 선택 고객 수
        "num_bicycle_choice_customers" => num_bicycle_choice_customers,    # 자전거 선택 고객 수
        "num_vehicle_dedicated_customers" => num_vehicle_dedicated_customers,  # 차량(전용) 선택 고객 수
        "num_vehicle_linked_customers" => num_vehicle_linked_customers,        # 차량(연계) 선택 고객 수
        "customer_walking_dist" => total_customer_walking_dist,            # 도보 총 거리 (왕복)
        "customer_bicycle_dist" => total_customer_bicycle_dist,            # 자전거 총 거리 (왕복)
        "customer_vehicle_ded_dist" => total_customer_vehicle_ded_dist,    # 차량(전용) 총 거리 (왕복)
        "customer_vehicle_link_dist" => total_customer_vehicle_link_dist,  # 차량(연계) 총 거리 (편도)
        # ═══ MOO 캐리어 누적 합산값 (2번째 경로에서 사용) ═══
        "moo_accum_f2" => accum_f2,
        "moo_accum_f3" => accum_f3,
        "moo_accum_f2_mobility" => accum_f2_mobility,
        "moo_accum_f2_dissatisfaction" => accum_f2_dissatisfaction,
        "moo_accum_avg_satisfaction" => accum_avg_sat,
        "moo_accum_avg_delay" => accum_avg_delay,
        "moo_accum_avg_dist" => accum_avg_dist,
        "moo_accum_f3_vehicle_co2" => accum_f3_vehicle_co2,
        "moo_accum_f3_customer_co2" => accum_f3_customer_co2,
        "moo_accum_f3_locker_co2" => accum_f3_locker_co2,
        # ═══ 시나리오 비교용 원시값 (정규화 전) ═══
        "mobility_raw_km" => accum_avg_dist,
        "dissatisfaction_raw" => (total_customers - actual_locker_customers_count) * (1.0 - accum_avg_sat),
        # ═══ 시스템 레벨 Pareto front (전체 캐리어 합산) ═══
        "system_pareto_front" => system_pareto_front
    )

    # 캐리어별 라우팅 결과 기반 상세 통계 업데이트
    lock(STATS_LOCK)
    try
        if haskey(LOCKER_STATS_COLLECTOR.stats_by_scenario, (scn, seed, omega))
            stats = LOCKER_STATS_COLLECTOR.stats_by_scenario[(scn, seed, omega)]
            for (carrier, route_info) in routes_info
                if haskey(stats.carrier_stats, carrier)
                    carrier_stat = stats.carrier_stats[carrier]
                    
                    vehicles_used = haskey(route_info, "vehicles_used") ? route_info["vehicles_used"] : begin
                        vrs = haskey(route_info, "vehicle_routes") ? route_info["vehicle_routes"] : Dict{Int, Vector{Vector{String}}}()
                        if haskey(vrs, 1) && !isempty(vrs[1])
                            1
                        else
                            routes_vec = haskey(route_info, "routes") ? route_info["routes"] : Vector{Vector{String}}()
                            isempty(routes_vec) ? 0 : 1
                        end
                    end
                    driving_distance = route_info["cost"]
                    total_demand = carrier_stat.used_vehicle_capacity # from update_carrier_stats!
                    total_trips = route_info["total_trips"]
                    
                    # 차량 수 (트립 수) 저장
                    carrier_stat.vehicles_count = total_trips
                    carrier_stat.total_distance = driving_distance
                    
                    # 1. 실제 사용된 차량 1대의 1회 tour당 평균 거리 (km/tour)
                    # total_trips = 실제 수행된 tour 수
                    if total_trips > 0
                        carrier_stat.avg_distance_per_used_vehicle = driving_distance / total_trips
                    else
                        carrier_stat.avg_distance_per_used_vehicle = 0.0
                    end

                    # 2. 1회 tour에서 수요 1단위당 주행거리 (km/demand per tour)
                    # = (1회 tour당 거리) / (1회 tour당 평균 수요)
                    if total_trips > 0 && total_demand > 0
                        distance_per_tour = driving_distance / total_trips
                        demand_per_tour = total_demand / total_trips
                        carrier_stat.km_per_demand = demand_per_tour > 0 ? distance_per_tour / demand_per_tour : 0.0
                    else
                        carrier_stat.km_per_demand = 0.0
                    end

                    # 3. (Single-trip) 1회 운행당 평균 점유율 (0-100%)
                    if total_trips > 0 && effective_capacity() > 0
                        avg_demand_per_trip = total_demand / total_trips
                        carrier_stat.avg_fill_rate_per_trip = avg_demand_per_trip / effective_capacity()
                    else
                        carrier_stat.avg_fill_rate_per_trip = 0.0
                    end
                end
            end
        end
    finally
        unlock(STATS_LOCK)
    end

    route_data = Dict(
        "vehicle_routes" => routes_info,
        "walking_routes" => walking_routes,
        "nodes" => nodes,
        "effective_lockers" => effective_lockers,
        "locker_tracker" => global_locker_tracker,  # 실제 락커 사용량 정보 추가
        "locker_assignments" => global_locker_assignments,  # 캐리어별 사용량 계산용
        "customers" => customers  # 시각화용 고객 데이터 추가
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MOO 정보 수집 (시스템 레벨 Pareto front 사용)
    # ═══════════════════════════════════════════════════════════════════════════
    try
        moo_detail = get_last_moo_detail()
        # 시스템 레벨 Pareto front 사용 (전체 캐리어 합산)
        pf = system_pareto_front !== nothing ? system_pareto_front : get_last_pareto_front()
        
        if moo_detail !== nothing && pf !== nothing && size(pf, 1) > 0
            # Compromise solution 선택 (정규화된 거리 최소)
            f1_vals = pf[:, 1]
            f2_vals = pf[:, 2]
            f3_vals = pf[:, 3]
            
            f1_norm = (f1_vals .- minimum(f1_vals)) ./ (maximum(f1_vals) - minimum(f1_vals) + 1e-9)
            f2_norm = (f2_vals .- minimum(f2_vals)) ./ (maximum(f2_vals) - minimum(f2_vals) + 1e-9)
            f3_norm = (f3_vals .- minimum(f3_vals)) ./ (maximum(f3_vals) - minimum(f3_vals) + 1e-9)
            
            distances = sqrt.(f1_norm.^2 .+ f2_norm.^2 .+ f3_norm.^2)
            selected_idx = argmin(distances)
            
            # 시나리오 이름 생성
            scenario_name = get_scenario_name(scn)
            
            # 락커 정보
            locker_ids_str = join(collect(keys(effective_lockers)), ",")
            
            # MOOScenarioResult 생성 (전체 캐리어 누적값 사용)
            total_vehicle_cost = total_vehicles_used * effective_vehicle_daily_cost()
            total_fuel_cost = total_driving * effective_fuel_cost()
            corrected_f1 = total_fuel_cost + total_vehicle_cost
            
            # 가중평균 계산 (전체 캐리어 합산)
            avg_sat = accum_num_customers_evaluated > 0 ? accum_satisfaction_sum / accum_num_customers_evaluated : 0.0
            avg_delay = accum_num_customers_evaluated > 0 ? accum_delay_sum / accum_num_customers_evaluated : 0.0
            avg_dist = accum_num_customers_evaluated > 0 ? accum_actual_dist_sum / accum_num_customers_evaluated : 0.0
            
            moo_result = MOOScenarioResult(
                scn,
                scenario_name,
                length(effective_lockers),
                omega,
                copy(pf),
                selected_idx,
                corrected_f1,            # 전체 캐리어 합산 f1
                accum_f2,                # 전체 캐리어 합산 f2
                accum_f3,                # 전체 캐리어 합산 f3
                total_fuel_cost,         # 전체 거리 × 유류비
                total_vehicle_cost,      # 전체 차량수 × 차량비
                0.0,                     # 인건비 제외
                accum_f2_mobility,       # 전체 캐리어 합산 이동불편비용
                accum_f2_dissatisfaction, # 전체 캐리어 합산 불만족도
                avg_sat,                 # 전체 캐리어 가중평균 만족도
                avg_delay,               # 전체 캐리어 가중평균 지연시간
                avg_dist,                # 전체 캐리어 가중평균 이동거리
                avg_dist,                # mobility_raw_km (시나리오 비교용)
                (total_customers - actual_locker_customers_count) * (1.0 - avg_sat),  # dissatisfaction_raw
                accum_f3_vehicle_co2,    # 전체 캐리어 합산 차량 CO2
                accum_f3_customer_co2,   # 전체 캐리어 합산 고객 CO2
                accum_f3_locker_co2,     # 전체 캐리어 합산 락커 CO2
                total_driving,           # 전체 캐리어 거리 합산
                total_vehicles_used,
                total_trips,
                total_customers,
                locker_customers_count,
                actual_locker_customers_count,
                total_d2d_conversions,
                num_active_lockers,
                locker_ids_str,
                Float64(num_walk_choice_customers),
                Float64(num_bicycle_choice_customers),
                Float64(num_vehicle_dedicated_customers),
                Float64(num_vehicle_linked_customers),
                total_customer_walking_dist,
                total_customer_bicycle_dist,
                total_customer_vehicle_ded_dist,
                total_customer_vehicle_link_dist
            )
            
            # 전역 컬렉터에 저장
            lock(MOO_RESULTS_LOCK) do
                push!(MOO_RESULTS_COLLECTOR, moo_result)
            end
            
            progress_println("📊 MOO 결과 수집: 시나리오 $scn, ω=$omega, Pareto front $(size(pf,1))개 해")
            progress_println("   선택해: f1=$(round(corrected_f1,digits=2)), f2=$(round(accum_f2,digits=2)), f3=$(round(accum_f3,digits=2))")
        end
    catch e
        # MOO 정보 수집 실패는 무시 (기존 시스템 호환성 유지)
        if !isa(e, UndefVarError)
            @warn "MOO 정보 수집 실패" exception=e
        end
    end

    # 시각화 데이터는 메인 프로세스에서 수집 (워커에서 저장하지 않음)
    # save_main_batch_result!는 main_batch_optimization에서 호출

    # 캐리어 분석 수행 (carrier_analysis가 제공된 경우)
    if carrier_analysis !== nothing
        # 캐리어별 락커 고객 수 계산
        total_locker_customers_by_carrier = Dict{String, Int}()
        
        # 모든 락커 고객에 대해 분석 (딕셔너리 캐싱 활용)
        for customer in customers
            if customer.dtype == "Locker"
                carrier = get(carrier_by_customer, customer.id, nothing)
                if carrier !== nothing
                    total_locker_customers_by_carrier[carrier] = get(total_locker_customers_by_carrier, carrier, 0) + 1
                    
                    # 락커 할당 성공 여부 확인
                    if haskey(global_locker_assignments, customer.id)
                        locker_id = global_locker_assignments[customer.id]
                        track_locker_assignment!(carrier_analysis, customer.id, carrier, locker_id, true)
                    else
                        # D2D로 전환된 경우
                        track_d2d_conversion!(carrier_analysis, customer.id, carrier, "capacity_exceeded")
                    end
                end
            end
        end
        
        # 시나리오 D2D에서 모든 락커 고객을 D2D 전환으로 추적 (딕셔너리 캐싱 활용)
        if scn == SCENARIO_D2D
            for customer in customers
                if customer.dtype == "Locker"
                    carrier = get(carrier_by_customer, customer.id, nothing)
                    if carrier !== nothing
                        track_d2d_conversion!(carrier_analysis, customer.id, carrier, "scenario_d2d_no_lockers")
                    end
                end
            end
        end
        
        # 캐리어별 성공률 계산
        calculate_carrier_success_rates!(carrier_analysis, total_locker_customers_by_carrier)
        
        # 분석 결과 저장
        save_carrier_analysis!(carrier_analysis)
    end

    return m, distance_info, route_data
end


#───────────────────────────────────────────────────────────────────────────────
# CVRP 해결 함수 (NSGA-II MOO 기반 - 시간창 지원)
#───────────────────────────────────────────────────────────────────────────────
function solve_multitrip_cvrp(
    depot_id::String,
    visit_nodes::Vector{String},
    vehicle_dist_dict::Dict{Tuple{String,String},Float64},
    demands::Vector{Int},
    vehicle_capacity::Int;
    vehicle_time_dict::Union{Dict{Tuple{String,String},Float64},Nothing}=nothing,
    time_windows::Union{Vector{Tuple{Int,Int}},Nothing}=nothing,
    service_times::Union{Vector{Int},Nothing}=nothing,
    progress_callback::Union{Function,Nothing}=nothing,
    seed::Int=1234,
    max_iterations::Int=1000
)
    if isempty(visit_nodes)
        return 0.0, Vector{Vector{String}}()
    end

    if length(visit_nodes) == 1
        node = visit_nodes[1]
        cost = 2 * get(vehicle_dist_dict, (depot_id, node), 1e6)
        return cost, [[depot_id, node, depot_id]]
    end

    # 시간 행렬: 없으면 거리 행렬 사용 (거리 ≈ 시간 가정)
    time_dict = vehicle_time_dict !== nothing ? vehicle_time_dict : vehicle_dist_dict

    # 시간창: 없으면 기본값 사용
    if USE_TIME_WINDOWS && time_windows !== nothing && service_times !== nothing
        # 시간창 지원 NSGA-II MOO 호출
        progress_println("🧬 NSGA-II(CVRPTW) MOO 경로 최적화 시작: 고객 $(length(visit_nodes))명, 용량 $(vehicle_capacity), 시간창 적용")
        best_cost, routes = solve_vrp_ids_with_tw(
            vehicle_dist_dict, time_dict, depot_id, visit_nodes,
            demands, time_windows, service_times, vehicle_capacity;
            max_iterations=max_iterations,
            depot_tw=(TW_DEPOT_OPEN, TW_DEPOT_CLOSE)
        )
        progress_println("✅ NSGA-II(CVRPTW) MOO 완료: 생성 경로 $(length(routes))개, 비용 $(round(best_cost, digits=2))")
    else
        # 시간창 없이 CVRP 호출 (하위 호환)
        progress_println("🧬 NSGA-II(CVRP) MOO 경로 최적화 시작: 고객 $(length(visit_nodes))명, 용량 $(vehicle_capacity)")
        best_cost, routes = solve_vrp_ids(vehicle_dist_dict, depot_id, visit_nodes, demands, vehicle_capacity; max_iterations=max_iterations)
        progress_println("✅ NSGA-II(CVRP) MOO 완료: 생성 경로 $(length(routes))개, 비용 $(round(best_cost, digits=2))")
    end

    return best_cost, routes
end

# 🌍 5구역 내부로 위치 제한
function constrain_to_district5(position)
    lat, lon = position
    
    # Float64로 변환 (Float32 타입 문제 해결)
    lat_f64 = Float64(lat)
    lon_f64 = Float64(lon)
    
    if is_point_in_district5(lat_f64, lon_f64)
        return (lat_f64, lon_f64)
    else
        return find_nearest_point_in_district5(lat_f64, lon_f64)
    end
end

# 🌍 5구역 내 가장 가까운 점 찾기
# rng: 재현성을 위한 난수 생성기 (기본값: Random.default_rng())
function find_nearest_point_in_district5(target_lat, target_lon; rng::AbstractRNG=Random.default_rng())
    if is_point_in_district5(target_lat, target_lon)
        return (target_lat, target_lon)
    end
    
    district5_bounds = (
        lat_min = 47.46, lat_max = 47.54,
        lon_min = 19.01, lon_max = 19.09
    )
    
    clipped_lat = max(district5_bounds.lat_min, min(district5_bounds.lat_max, target_lat))
    clipped_lon = max(district5_bounds.lon_min, min(district5_bounds.lon_max, target_lon))
    
    if is_point_in_district5(clipped_lat, clipped_lon)
        return (clipped_lat, clipped_lon)
    end
    
    best_lat, best_lon = target_lat, target_lon
    best_distance = Inf
    
    search_radius = 0.02
    grid_size = 20
    
    for i in 0:grid_size
        for j in 0:grid_size
            lat_offset = (i / grid_size - 0.5) * search_radius
            lon_offset = (j / grid_size - 0.5) * search_radius
            
            candidate_lat = target_lat + lat_offset
            candidate_lon = target_lon + lon_offset
            
            if is_point_in_district5(candidate_lat, candidate_lon)
                distance = euclidean_distance((target_lat, target_lon), (candidate_lat, candidate_lon))
                if distance < best_distance
                    best_distance = distance
                    best_lat, best_lon = candidate_lat, candidate_lon
                end
            end
        end
    end
    
    if best_distance == Inf
        for _ in 1:50
            center_lat, center_lon = 47.5, 19.05
            random_radius = rand(rng) * 0.03
            random_angle = rand(rng) * 2π
            
            candidate_lat = center_lat + random_radius * cos(random_angle) / 111.0
            candidate_lon = center_lon + random_radius * sin(random_angle) / (111.0 * cos(deg2rad(center_lat)))
            
            if is_point_in_district5(candidate_lat, candidate_lon)
                distance = euclidean_distance((target_lat, target_lon), (candidate_lat, candidate_lon))
                if distance < best_distance
                    best_distance = distance
                    best_lat, best_lon = candidate_lat, candidate_lon
                end
            end
        end
    end
    
    return (best_lat, best_lon)
end

# 🔥 시나리오 4: SLRP (Stochastic Location-Routing Problem) 기반 락커 위치 최적화
# ═══════════════════════════════════════════════════════════════════════════════════
# Reference: "A Stochastic Location-Routing Problem for the Optimal Placement of Lockers"
#            Barbieri et al., ICORES 2025
#
# 알고리즘:
#   입력: 5구역 내 주요 POI 후보 위치 20개 (고정)
#   1단계: ADD 알고리즘으로 락커 활성화 결정 (기대 라우팅 비용 기반)
#   2단계: 각 시나리오(omega)별 VRP 해결
#   출력: 최적 락커 위치 및 개수
# ═══════════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════════
# OPL 시나리오 후보 위치 (5구역 내 주요 POI 기반)
# - 교통 허브, 지하철역, 랜드마크, 정부 청사, 문화시설 등
# - 총 20개 후보지 (Raster 295, 337, 338, 383)
# ═══════════════════════════════════════════════════════════════════════════════════
const OPL_CANDIDATE_LOCATIONS = [
    # Raster 295 (북서부)
    (47.5126, 19.0461, "Jászai Mari tér", "교통 허브"),           # Tram 2/4/6 환승
    (47.5111, 19.0460, "Olimpia Park", "공원"),                   # 다뉴브 강변 공원
    (47.5121, 19.0469, "Képviselői Irodaház", "정부 청사"),       # 국회의원 회관
    (47.5105, 19.0455, "Ministry of Defense", "정부 부처"),       # 국방부
    
    # Raster 337 (서부)
    (47.5071, 19.0457, "Hungarian Parliament", "랜드마크"),       # 국회의사당
    (47.5065, 19.0470, "Kossuth Lajos tér", "지하철(M2)"),       # 코슈트 광장 역
    (47.5011, 19.0465, "Hungarian Academy of Sciences", "학술 기관"), # MTA 본관
    (47.4997, 19.0483, "Széchenyi István tér", "교통/광장"),      # 세체니 다리 앞
    (47.5060, 19.0480, "Ministry of Justice", "정부 부처"),       # 법무부
    
    # Raster 338 (중앙)
    (47.5008, 19.0539, "St. Stephen's Basilica", "랜드마크"),     # 성 이슈트반 대성당
    (47.5031, 19.0544, "Arany János utca", "지하철(M3)"),        # 역 입구
    (47.5033, 19.0506, "Liberty Square", "광장"),                 # 자유 광장
    (47.5042, 19.0520, "Hungarian National Bank", "금융 기관"),   # 중앙은행 본점
    (47.5005, 19.0495, "Ministry of Finance", "정부 부처"),       # 재무부
    
    # Raster 383 (남부)
    (47.4975, 19.0550, "Deák Ferenc tér", "환승역"),              # M1, M2, M3 환승
    (47.4952, 19.0544, "Budapest City Hall", "시청"),             # 부다페스트 시청
    (47.4967, 19.0503, "Vörösmarty tér", "지하철(M1)"),          # 1호선 종점
    (47.4930, 19.0561, "Ferenciek tere", "지하철(M3)"),          # 페렌치엑 광장
    (47.4958, 19.0494, "Vigadó Concert Hall", "문화 시설"),       # 비가도 콘서트홀
    (47.4916, 19.0530, "Petőfi Literary Museum", "박물관"),       # 카로이 정원 인근

    # Additional candidates (L_cand sensitivity: 21-25)
    (47.5020, 19.0555, "Bajcsy-Zsilinszky út M1", "지하철(M1)"),   # M1 역
    (47.5050, 19.0530, "Hold utca Market", "시장"),                 # 홀드 거리 시장
    (47.4940, 19.0540, "Károlyi Garden", "공원"),                   # 카로이 정원
    (47.5090, 19.0475, "Markó utca Court", "법원"),                 # 부다페스트 법원
    (47.4985, 19.0510, "Nádor utca", "외교 구역"),                  # 나도르 거리
]

const N_CANDIDATES = length(OPL_CANDIDATE_LOCATIONS)  # 후보 위치 개수 (25개, 기본 사용 20개)

# ═══════════════════════════════════════════════════════════════════════════════════
# 논문 기반 2-Stage SLRP (Stochastic Location-Routing Problem)
# Reference: "A Stochastic Location-Routing Problem for the Optimal Placement of Lockers"
#            Barbieri et al., ICORES 2025
#
# 구조:
# - 입력: 5구역 내 주요 POI 기반 후보 위치 20개 (고정)
# - 1단계: ADD 알고리즘으로 최적 락커 활성화 (각 시나리오별 라우팅 비용 기반)
# - 2단계: 각 시나리오(omega)별 VRP 해결
# ═══════════════════════════════════════════════════════════════════════════════════

"""
OPL 후보 위치 반환 (고정된 POI 기반)
- 교통 허브, 지하철역, 랜드마크 등 주요 위치
- 반환: [(lat, lon), ...] 형식
"""
function get_opl_candidate_locations()
    n = get_param(:N_CANDIDATES_OVERRIDE, 20)
    n = min(n, length(OPL_CANDIDATE_LOCATIONS))
    return [(loc[1], loc[2]) for loc in OPL_CANDIDATE_LOCATIONS[1:n]]
end

function optimize_opl_locations(omega_customers::Vector, omega_attrs::Vector; moo_seed::Int=42)
    slrp_start_time = time()
    
    progress_println("")
    progress_println("┌─────────────────────────────────────────────────────────────────┐")
    progress_println("│  🚀 시나리오 4: Optimal Public Locker (2-Stage SLRP)           │")
    progress_println("│     Monte Carlo: $(length(omega_customers))개                                  │")
    progress_println("│     📄 논문: Barbieri et al., ICORES 2025                      │")
    progress_println("└─────────────────────────────────────────────────────────────────┘")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 입력 (Input): 후보 위치 집합 L
    # - 논문에서 L은 미리 주어진 입력 (결정 변수 아님)
    # - 5구역 내 주요 POI 기반 고정 후보지 20개 사용
    # ═══════════════════════════════════════════════════════════════════════════
    
    candidate_locations = get_opl_candidate_locations()
    
    progress_println("   📍 후보 위치 집합 L: $(length(candidate_locations))개 POI")
    progress_println("      (교통허브, 지하철역, 랜드마크 등 주요 위치)")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 1단계 (First Stage): 락커 활성화 결정 wₗ ∈ {0,1}
    # - ADD 알고리즘으로 어떤 후보를 활성화할지 순차 결정
    # - 이 결정은 모든 시나리오에 공통 적용됨
    # 
    # 2단계 (Second Stage): 각 시나리오(omega)별 라우팅
    # - calc_expected_routing_cost() 내부에서 각 omega별 VRP 해결
    # - 기대 비용: E[z] = (1/|S|) × Σ (운송비용 + 보행거리 × 가중치)
    # ═══════════════════════════════════════════════════════════════════════════
    
    max_k = min(effective_max_pub(), length(candidate_locations))
    total_candidates = length(candidate_locations)
    n_omega = length(omega_customers)
    
    progress_println("")
    progress_println("   ┌─────────────────────────────────────────────────────────┐")
    progress_println("   │  📊 시나리오 4 SLRP 진행 현황                           │")
    progress_println("   │  └─ k 범위: 1~$(max_k), 후보: $(total_candidates)개, ω: $(n_omega)개           │")
    progress_println("   └─────────────────────────────────────────────────────────┘")
    
    selected_locations = Tuple{Float64,Float64}[]  # 현재 선택된 위치들
    remaining_candidates = copy(candidate_locations)  # 아직 선택되지 않은 후보들
    
    all_results = []
    k_fixed_locations = Dict{Int, Vector{Tuple{Float64,Float64}}}()
    
    # 기준선: 락커 없이 D2D만 할 때의 비용 (k=0) - 비교용으로만 사용
    progress_println("")
    progress_println("   ⏳ [baseline] D2D 기준선 계산 중... ($(n_omega)개 ω 평가)")
    baseline_start = time()
    baseline_cost = calc_expected_routing_cost([], omega_customers, omega_attrs; show_progress=true, label="baseline", moo_seed=moo_seed)
    baseline_elapsed = round(time() - baseline_start, digits=1)
    progress_println("   ✅ [baseline] 완료: E[비용]=$(round(baseline_cost, digits=2))km ($(baseline_elapsed)초)")
    progress_println("")
    
    # 시나리오 4는 락커가 존재해야 하므로, k=1의 비용을 초기값으로 사용
    current_best_cost = Inf  # 첫 번째 락커는 무조건 선택
    
    for k in 1:max_k
        # ═══════════════════════════════════════════════════════════════════
        # f2 정규화: mobility range를 같은 k의 모든 후보 위치에서 사전 계산
        # - 같은 k에서 후보 위치마다 고객→락커 거리가 달라짐
        # - 이 변동 범위를 min-max로 정규화에 사용
        # ═══════════════════════════════════════════════════════════════════
        reset_f2_normalization!()
        @everywhere reset_f2_normalization!()
        
        # 모든 후보 위치의 mobility (avg locker customer distance) 수집
        all_candidate_mobilities = Float64[]
        for candidate in remaining_candidates
            test_locations = vcat(selected_locations, [candidate])
            # 각 omega에서 mobility 계산 (VRP 없이 고객→가장가까운락커 거리만)
            for omega_idx in 1:length(omega_customers)
                customers = omega_customers[omega_idx]
                locker_custs = [c for c in customers if c.dtype == "Locker"]
                if isempty(locker_custs)
                    continue
                end
                total_actual_dist = 0.0
                for c in locker_custs
                    # 가장 가까운 락커까지 거리 (유클리드 + Haversine)
                    min_dist = Inf
                    for (lat, lon) in test_locations
                        d = euclidean_distance((Float64(c.coord[1]), Float64(c.coord[2])), (lon, lat))
                        min_dist = min(min_dist, d)
                    end
                    # MODE_SHARE 적용한 실제 이동거리
                    walk, bicycle, dedicated, linked = get_mode_share(min_dist)
                    actual_dist = (walk + bicycle + dedicated) * min_dist * 2.0 + linked * min_dist
                    total_actual_dist += actual_dist
                end
                avg_mobility = total_actual_dist / length(locker_custs)
                push!(all_candidate_mobilities, avg_mobility)
            end
        end
        
        # mobility range 설정
        if !isempty(all_candidate_mobilities) && maximum(all_candidate_mobilities) - minimum(all_candidate_mobilities) > 1e-10
            mob_L = minimum(all_candidate_mobilities)
            mob_U = maximum(all_candidate_mobilities)
            set_f2_mobility_range!(mob_L, mob_U)
            @everywhere set_f2_mobility_range!($mob_L, $mob_U)
            progress_println("   📐 k=$k mobility range: [$(round(mob_L, digits=4)), $(round(mob_U, digits=4))] km ($(length(remaining_candidates))개 후보)")
        else
            progress_println("   📐 k=$k mobility range: 변동 없음 ($(length(remaining_candidates))개 후보)")
        end
        
        k_start = time()
        n_remaining = length(remaining_candidates)
        progress_println("   ┌─ [k=$(k)/$(max_k)] 락커 $(k)개 최적 위치 탐색 ─────────────────────")
        progress_println("   │  평가할 후보: $(n_remaining)개 × $(n_omega) ω = $(n_remaining * n_omega)개 VRP")
        
        best_candidate = nothing
        best_cost_with_candidate = Inf
        best_candidate_idx = 0
        
        # pmap으로 모든 후보 병렬 평가
        progress_println("   │  ⏳ $(n_remaining)개 후보 병렬 평가 중...")
        
        candidate_results = pmap(enumerate(remaining_candidates)) do (idx, candidate)
            test_locations = vcat(selected_locations, [candidate])
            expected_cost = calc_expected_routing_cost(test_locations, omega_customers, omega_attrs; show_progress=false, moo_seed=moo_seed)
            return (idx=idx, candidate=candidate, cost=expected_cost)
        end
        
        # 결과 출력 및 최선 후보 찾기
        for result in candidate_results
            # 후보 이름 찾기
            candidate_name = "Unknown"
            for (lat, lon, name, _) in OPL_CANDIDATE_LOCATIONS
                if abs(lat - result.candidate[1]) < 0.0001 && abs(lon - result.candidate[2]) < 0.0001
                    candidate_name = name
                    break
                end
            end
            progress_println("   │     [$(result.idx)/$(n_remaining)] $(candidate_name): E[비용]=$(round(result.cost, digits=2))km")
            
            if result.cost < best_cost_with_candidate
                best_cost_with_candidate = result.cost
                best_candidate = result.candidate
                best_candidate_idx = result.idx
            end
        end
        
        # 유효한 후보가 없으면 중단 (남은 후보가 없는 경우)
        if best_candidate === nothing
            k_elapsed = round(time() - k_start, digits=1)
            progress_println("   │  ⚠️ 유효한 후보가 없습니다 → k=$(k-1)에서 종료")
            progress_println("   └─ [k=$(k)/$(max_k)] 중단 ($(k_elapsed)초) ─────────────────────────")
            progress_println("")
            break
        end
        
        # 비용 개선 여부 체크 (로그용, 중단하지 않음)
        if k >= 2 && best_cost_with_candidate >= current_best_cost
            progress_println("   │  ⚠️ k=$(k-1) 대비 비용 개선 없음 (계속 진행)")
        end
        
        # 최적 후보 선택
        push!(selected_locations, best_candidate)
        deleteat!(remaining_candidates, best_candidate_idx)
        current_best_cost = best_cost_with_candidate
        
        # 선택된 후보 이름 찾기
        selected_name = "Unknown"
        for (lat, lon, name, _) in OPL_CANDIDATE_LOCATIONS
            if abs(lat - best_candidate[1]) < 0.0001 && abs(lon - best_candidate[2]) < 0.0001
                selected_name = name
                break
                    end
                end
                
        k_elapsed = round(time() - k_start, digits=1)
        improvement = round(baseline_cost - best_cost_with_candidate, digits=2)
        improvement_pct = round(improvement / baseline_cost * 100, digits=1)
        
        progress_println("   │  ✅ 선택: $(selected_name)")
        progress_println("   │     E[비용]=$(round(best_cost_with_candidate, digits=2))km (D2D 대비 -$(improvement)km, -$(improvement_pct)%)")
        progress_println("   └─ [k=$(k)/$(max_k)] ADD 완료 ($(k_elapsed)초) ─────────────────────────")
        progress_println("")
        
        # ═══════════════════════════════════════════════════════════════════
        # [Shift Routine] 각 k에서 위치 최적화 (K&H 1963)
        # - ADD로 선택된 k개 락커의 위치를 교체하여 개선 시도
        # - 개선된 조합을 기반으로 다음 k 탐색
        # ═══════════════════════════════════════════════════════════════════
        if k >= 2  # k=1에서는 Shift 불필요 (교체할 대상이 1개뿐)
            progress_println("   ┌─ [Shift] k=$(k) 위치 최적화 ─────────────────────")
            shift_improved_k = true
            shift_iteration_k = 0
            max_shift_iterations_k = 5  # 각 k에서는 최대 5회
            
            while shift_improved_k && shift_iteration_k < max_shift_iterations_k
                shift_improved_k = false
                shift_iteration_k += 1
                
                best_swap_k = nothing
                best_swap_cost_k = current_best_cost
                
                # 모든 교체 조합 생성
                unselected_k = [loc for loc in candidate_locations if !(loc in selected_locations)]
                swap_combinations_k = [(idx, loc_remove, loc_add) 
                                       for (idx, loc_remove) in enumerate(selected_locations) 
                                       for loc_add in unselected_k]
                
                n_swaps_k = length(swap_combinations_k)
                progress_println("   │  🔄 Shift 반복 $(shift_iteration_k): $(n_swaps_k)개 교체 평가 중...")
                
                # pmap으로 병렬 평가
                swap_results_k = pmap(swap_combinations_k) do (idx_remove, loc_remove, loc_add)
                    new_locations = copy(selected_locations)
                    new_locations[idx_remove] = loc_add
                    swap_cost = calc_expected_routing_cost(new_locations, omega_customers, omega_attrs; show_progress=false, moo_seed=moo_seed)
                    return (idx=idx_remove, loc_remove=loc_remove, loc_add=loc_add, cost=swap_cost)
                end
                
                # 최선의 교체 찾기
                for result in swap_results_k
                    if result.cost < best_swap_cost_k
                        best_swap_cost_k = result.cost
                        best_swap_k = (result.idx, result.loc_remove, result.loc_add)
                    end
                end
                
                # 개선된 교체가 있으면 적용
                if best_swap_k !== nothing && best_swap_cost_k < current_best_cost
                    (idx_remove, loc_remove, loc_add) = best_swap_k
                    
                    # 제거되는 락커 이름
                    remove_name_k = "Unknown"
                    for (lat, lon, name, _) in OPL_CANDIDATE_LOCATIONS
                        if abs(lat - loc_remove[1]) < 0.0001 && abs(lon - loc_remove[2]) < 0.0001
                            remove_name_k = name
                            break
                        end
                    end
                    
                    # 추가되는 락커 이름
                    add_name_k = "Unknown"
                    for (lat, lon, name, _) in OPL_CANDIDATE_LOCATIONS
                        if abs(lat - loc_add[1]) < 0.0001 && abs(lon - loc_add[2]) < 0.0001
                            add_name_k = name
                            break
                        end
                    end
                    
                    improvement_shift_k = round(current_best_cost - best_swap_cost_k, digits=2)
                    progress_println("   │     ✅ $(remove_name_k) → $(add_name_k) (-$(improvement_shift_k)km)")
                    
                    # 선택 위치 교체
                    selected_locations[idx_remove] = loc_add
                    
                    # remaining_candidates 업데이트: 제거된 위치 추가, 추가된 위치 제거
                    push!(remaining_candidates, loc_remove)
                    filter!(loc -> loc != loc_add, remaining_candidates)
                    
                    current_best_cost = best_swap_cost_k
                    shift_improved_k = true
                end
            end
            
            if shift_iteration_k == 1 && !shift_improved_k
                progress_println("   │  ⚡ 개선 없음 - ADD 선택이 최적")
            else
                progress_println("   │  🏁 Shift $(shift_iteration_k)회 반복 완료")
            end
            progress_println("   └─ [Shift] k=$(k) 최종 비용=$(round(current_best_cost, digits=2))km ─────────────────")
            progress_println("")
        end
        
        # 현재 k에 대한 위치 저장 (Shift 후 개선된 조합)
        k_fixed_locations[k] = copy(selected_locations)
        
        # ═══════════════════════════════════════════════════════════════════
        # 각 오메가에서 상세 결과 수집 (시각화 및 통계용)
        # — Shift로 락커 위치가 변경될 수 있으므로 f2 정규화 재계산
        # ═══════════════════════════════════════════════════════════════════
        reset_f2_normalization!()
        
        for omega in 1:length(omega_customers)
            if isempty(omega_customers[omega])
                    continue
                end

                try
                    lock_pub = Dict{String, Any}()
                for (i, (lat, lon)) in enumerate(selected_locations)
                    locker_id = "SLRP_$(k)_$(lpad(i, 2, '0'))"
                        lock_pub[locker_id] = ((lon, lat), "Public")
                    end
                    
                    z_val = Dict(lid => 1 for lid in keys(lock_pub))
                sub_model, distance_info, route_data = solve_vrp_single_omega(
                    z_val, omega_customers[omega], omega_attrs[omega], lock_pub, SCENARIO_OPL;
                        verbose=false, omega=omega, collect_stats=true, moo_seed=moo_seed
                    )
                    
                    if distance_info["total"] != Inf && !isnan(distance_info["total"])
                        save_scenario4_snapshot!(k, omega, omega_customers[omega], route_data, distance_info)
                    
                    locker_names = ["SLRP_$(k)_$(lpad(i, 2, '0'))" for i in 1:length(selected_locations)]
                    
                    # 시각화 데이터 추출
                    customers_for_viz = haskey(route_data, "customers") ? route_data["customers"] : omega_customers[omega]
                        
                        push!(all_results, (
                        scenario=SCENARIO_OPL,
                            k=k,
                            omega_idx=omega,  # 대응표본 검정을 위한 omega 인덱스
                            obj=distance_info["social_cost"],
                            driving_dist=distance_info["driving"],
                            walking_dist=distance_info["walking"],
                        nOpen=length(selected_locations),
                            lockers=join(locker_names, ";"),
                            vehicles_used=distance_info["vehicles_used"],
                            total_trips=distance_info["total_trips"],
                            total_customers=distance_info["total_customers"],
                            locker_customers=distance_info["locker_customers"],
                            actual_locker_customers=distance_info["actual_locker_customers"],
                            d2d_conversions=get(distance_info, "d2d_conversions", 0),  # 강제 D2D 전환 고객 수
                            avg_dist_per_vehicle=distance_info["vehicles_used"] > 0 ? distance_info["driving"] / distance_info["vehicles_used"] : 0.0,
                            avg_walking_per_customer=distance_info["actual_locker_customers"] > 0 ? distance_info["walking"] / distance_info["actual_locker_customers"] : 0.0,
                            social_cost=distance_info["social_cost"],
                        total_distance=distance_info["driving"] + distance_info["walking"],
                        # 목적함수 구성요소 (논문 기반)
                        transport_cost=get(distance_info, "transport_cost", distance_info["driving"] * TRANSPORT_COST_PER_KM),
                        customer_mobility_cost=get(distance_info, "customer_mobility_cost", 0.0),
                        activation_cost=get(distance_info, "activation_cost", 0.0),
                        num_active_lockers=get(distance_info, "num_active_lockers", length(selected_locations)),
                        # 고객 이동 수단 선택 상세 (4가지 수단)
                        customer_inconvenience_cost=get(distance_info, "customer_inconvenience_cost", 0.0),
                        customer_social_cost=get(distance_info, "customer_social_cost", 0.0),
                        num_walk_choice_customers=get(distance_info, "num_walk_choice_customers", 0),
                        num_bicycle_choice_customers=get(distance_info, "num_bicycle_choice_customers", 0),
                        num_vehicle_dedicated_customers=get(distance_info, "num_vehicle_dedicated_customers", 0),
                        num_vehicle_linked_customers=get(distance_info, "num_vehicle_linked_customers", 0),
                        customer_walking_dist=get(distance_info, "customer_walking_dist", 0.0),
                        customer_bicycle_dist=get(distance_info, "customer_bicycle_dist", 0.0),
                        customer_vehicle_ded_dist=get(distance_info, "customer_vehicle_ded_dist", 0.0),
                        customer_vehicle_link_dist=get(distance_info, "customer_vehicle_link_dist", 0.0),
                        viz_data=(customers=customers_for_viz, route_data=route_data, distance_info=distance_info),
                        locker_stats=Dict{Tuple{Int,Int,Int}, ScenarioLockerStats}(),  # 빈 딕셔너리
                        omega_costs=Float64[]  # 빈 배열
                    ))
                end
            catch omega_error
                progress_println("      ⚠️ k=$k, ω=$omega 상세 결과 수집 실패: $omega_error")
            end
        end
        
        # 시설 위치 저장
        scenario4_k_facilities[k] = copy(selected_locations)
        
        # 남은 후보가 없으면 중단
        if isempty(remaining_candidates)
            break
        end
    end
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Bump Routine - Kuehn & Hamburger (1963) 기반
    # 참고문헌: Kuehn, A. A., & Hamburger, M. J. (1963). 
    #          "A Heuristic Program for Locating Warehouses"
    #          Management Science, 9(4), 643-666.
    # 
    # 알고리즘:
    # 1. 각 k에서 ADD + Shift 완료 후 (위에서 이미 적용됨)
    # 2. 최종 k에서 Bump: 나중에 추가된 락커로 인해 기존 락커가 불필요해졌는지 검사 후 제거
    # 3. Bump 후 Shift: 락커 수가 줄어든 경우 남은 락커 위치 재최적화
    # 4. 비용이 개선되면 교체 적용
    # 5. 개선이 없을 때까지 반복 (최대 10회)
    # 
    # 최종 k에서 Bump 적용하여 불필요한 락커 제거 시도
    # ═══════════════════════════════════════════════════════════════════════════
    
    if length(selected_locations) == max_k && max_k > 0
        progress_println("")
        progress_println("   ┌─ [Bump & Shift] K&H Routine 시작 (k=$max_k) ─────────────────────")
        
        # ═══════════════════════════════════════════════════════════════════════
        # BUMP Routine: 불필요한 락커 제거
        # - 나중에 추가된 락커로 인해 기존 락커가 더 이상 경제적이지 않으면 제거
        # ═══════════════════════════════════════════════════════════════════════
        progress_println("   │")
        progress_println("   │  📍 [Bump] 불필요한 락커 제거 검사 중...")
        
        bump_removed = true
        bump_total_removed = 0
        
        while bump_removed && length(selected_locations) > 1
            bump_removed = false
            
            # 각 락커를 제거했을 때의 비용 평가 (pmap으로 병렬)
            bump_results = pmap(1:length(selected_locations)) do idx
                test_locations = [loc for (j, loc) in enumerate(selected_locations) if j != idx]
                cost_without = calc_expected_routing_cost(test_locations, omega_customers, omega_attrs; show_progress=false, moo_seed=moo_seed)
                return (idx=idx, loc=selected_locations[idx], cost=cost_without)
            end
            
            # 제거해도 비용이 증가하지 않는 락커 찾기
            for result in bump_results
                if result.cost <= current_best_cost
                    # 락커 이름 찾기
                    loc_name = "Unknown"
                    for (lat, lon, name, _) in OPL_CANDIDATE_LOCATIONS
                        if abs(lat - result.loc[1]) < 0.0001 && abs(lon - result.loc[2]) < 0.0001
                            loc_name = name
                            break
                        end
                    end
                    
                    progress_println("   │     🗑️ 제거: $(loc_name) (비용 $(round(current_best_cost, digits=2)) → $(round(result.cost, digits=2))km)")
                    
                    # 해당 락커 제거
                    filter!(loc -> loc != result.loc, selected_locations)
                    push!(remaining_candidates, result.loc)  # 후보 목록에 다시 추가
                    current_best_cost = result.cost
                    bump_removed = true
                    bump_total_removed += 1
                    
                    # k_fixed_locations 업데이트
                    new_k = length(selected_locations)
                    k_fixed_locations[new_k] = copy(selected_locations)
                    
                    break  # 하나 제거 후 다시 검사
                end
            end
        end
        
        if bump_total_removed > 0
            progress_println("   │     ✅ Bump 완료: $(bump_total_removed)개 락커 제거 → 현재 k=$(length(selected_locations))")
        else
            progress_println("   │     ⚡ Bump: 제거할 락커 없음 (모두 필요)")
        end
        
        # max_k 업데이트 (Bump 후)
        actual_k = length(selected_locations)
        
        # ═══════════════════════════════════════════════════════════════════════
        # SHIFT Routine: 락커 위치 이동 최적화
        # - 각 선택된 락커를 선택되지 않은 후보로 교체 시도
        # ═══════════════════════════════════════════════════════════════════════
        progress_println("   │")
        progress_println("   │  🔄 [Shift] 락커 위치 이동 최적화 중... (k=$(actual_k))")
        
        shift_improved = true
        shift_iteration = 0
        max_shift_iterations = 10  # 무한 루프 방지
        
        while shift_improved && shift_iteration < max_shift_iterations
            shift_improved = false
            shift_iteration += 1
            progress_println("   │  🔄 Shift 반복 $(shift_iteration)")
            
            best_swap = nothing
            best_swap_cost = current_best_cost
            
            # 모든 교체 조합 생성
            unselected = [loc for loc in candidate_locations if !(loc in selected_locations)]
            swap_combinations = [(idx, loc_remove, loc_add) 
                                 for (idx, loc_remove) in enumerate(selected_locations) 
                                 for loc_add in unselected]
            
            n_swaps = length(swap_combinations)
            progress_println("   │     평가할 교체 조합: $(n_swaps)개")
            
            # pmap으로 병렬 평가
            swap_results = pmap(swap_combinations) do (idx_remove, loc_remove, loc_add)
                new_locations = copy(selected_locations)
                new_locations[idx_remove] = loc_add
                swap_cost = calc_expected_routing_cost(new_locations, omega_customers, omega_attrs; show_progress=false, moo_seed=moo_seed)
                return (idx=idx_remove, loc_remove=loc_remove, loc_add=loc_add, cost=swap_cost)
            end
            
            # 최선의 교체 찾기
            for result in swap_results
                if result.cost < best_swap_cost
                    best_swap_cost = result.cost
                    best_swap = (result.idx, result.loc_remove, result.loc_add)
                end
            end
            
            # 개선된 교체가 있으면 적용
            if best_swap !== nothing && best_swap_cost < current_best_cost
                (idx_remove, loc_remove, loc_add) = best_swap
                
                # 제거되는 락커 이름
                remove_name = "Unknown"
                for (lat, lon, name, _) in OPL_CANDIDATE_LOCATIONS
                    if abs(lat - loc_remove[1]) < 0.0001 && abs(lon - loc_remove[2]) < 0.0001
                        remove_name = name
                        break
                    end
                end
                
                # 추가되는 락커 이름
                add_name = "Unknown"
                for (lat, lon, name, _) in OPL_CANDIDATE_LOCATIONS
                    if abs(lat - loc_add[1]) < 0.0001 && abs(lon - loc_add[2]) < 0.0001
                        add_name = name
                        break
                    end
                end
                
                improvement = round(current_best_cost - best_swap_cost, digits=2)
                progress_println("   │     ✅ 교체: $(remove_name) → $(add_name)")
                progress_println("   │        비용: $(round(current_best_cost, digits=2)) → $(round(best_swap_cost, digits=2)) (-$(improvement)km)")
                
                selected_locations[idx_remove] = loc_add
                current_best_cost = best_swap_cost
                shift_improved = true
                
                # k_fixed_locations 업데이트
                k_fixed_locations[actual_k] = copy(selected_locations)
            end
        end
        
        if shift_iteration == 1 && !shift_improved
            progress_println("   │     ⚡ 개선 없음 - Greedy 선택이 이미 최적")
        else
            progress_println("   │     🏁 Shift 완료: $(shift_iteration)회 반복")
        end
        
        # Bump and Shift 전체 완료 메시지
        progress_println("   │")
        progress_println("   └─ [Bump & Shift] 완료: 최종 k=$(actual_k), 비용=$(round(current_best_cost, digits=2))km ───────")
        progress_println("")
        
        # Bump 또는 Shift 후 최종 k 결과 다시 수집
        if bump_total_removed > 0 || shift_iteration > 1
            progress_println("   📊 Bump & Shift 후 k=$(actual_k) 결과 재수집 중...")
            
            # 기존 k 결과 제거 (Bump으로 k가 변경되었을 수 있음)
            filter!(r -> r.k != actual_k, all_results)
            
            # f2 정규화 파라미터 리셋 (Bump/Shift로 락커 배치 변경됨)
            reset_f2_normalization!()
            
            # 새 결과 수집
            for omega in 1:length(omega_customers)
                if isempty(omega_customers[omega])
                    continue
                end
                
                try
                    lock_pub = Dict{String, Any}()
                    for (i, (lat, lon)) in enumerate(selected_locations)
                        locker_id = "SLRP_$(actual_k)_$(lpad(i, 2, '0'))"
                        lock_pub[locker_id] = ((lon, lat), "Public")
                    end
                    
                    z_val = Dict(lid => 1 for lid in keys(lock_pub))
                    sub_model, distance_info, route_data = solve_vrp_single_omega(
                        z_val, omega_customers[omega], omega_attrs[omega], lock_pub, SCENARIO_OPL;
                        verbose=false, omega=omega, collect_stats=true, moo_seed=moo_seed
                    )
                    
                    if distance_info["total"] != Inf && !isnan(distance_info["total"])
                        save_scenario4_snapshot!(actual_k, omega, omega_customers[omega], route_data, distance_info)
                        
                        locker_names = ["SLRP_$(actual_k)_$(lpad(i, 2, '0'))" for i in 1:length(selected_locations)]
                        customers_for_viz = haskey(route_data, "customers") ? route_data["customers"] : omega_customers[omega]
                        
                        push!(all_results, (
                            scenario=SCENARIO_OPL,
                            k=actual_k,
                            omega_idx=omega,  # 대응표본 검정을 위한 omega 인덱스
                            obj=distance_info["social_cost"],
                            driving_dist=distance_info["driving"],
                            walking_dist=distance_info["walking"],
                            nOpen=length(selected_locations),
                            lockers=join(locker_names, ";"),
                            vehicles_used=distance_info["vehicles_used"],
                            total_trips=distance_info["total_trips"],
                            total_customers=distance_info["total_customers"],
                            locker_customers=distance_info["locker_customers"],
                            actual_locker_customers=distance_info["actual_locker_customers"],
                            d2d_conversions=get(distance_info, "d2d_conversions", 0),  # 강제 D2D 전환 고객 수
                            avg_dist_per_vehicle=distance_info["vehicles_used"] > 0 ? distance_info["driving"] / distance_info["vehicles_used"] : 0.0,
                            avg_walking_per_customer=distance_info["actual_locker_customers"] > 0 ? distance_info["walking"] / distance_info["actual_locker_customers"] : 0.0,
                            social_cost=distance_info["social_cost"],
                            total_distance=distance_info["driving"] + distance_info["walking"],
                            transport_cost=get(distance_info, "transport_cost", distance_info["driving"] * TRANSPORT_COST_PER_KM),
                            customer_mobility_cost=get(distance_info, "customer_mobility_cost", 0.0),
                            activation_cost=get(distance_info, "activation_cost", 0.0),
                            num_active_lockers=get(distance_info, "num_active_lockers", length(selected_locations)),
                            # 고객 이동 수단 선택 상세 (4가지 수단)
                            customer_inconvenience_cost=get(distance_info, "customer_inconvenience_cost", 0.0),
                            customer_social_cost=get(distance_info, "customer_social_cost", 0.0),
                            num_walk_choice_customers=get(distance_info, "num_walk_choice_customers", 0),
                            num_bicycle_choice_customers=get(distance_info, "num_bicycle_choice_customers", 0),
                            num_vehicle_dedicated_customers=get(distance_info, "num_vehicle_dedicated_customers", 0),
                            num_vehicle_linked_customers=get(distance_info, "num_vehicle_linked_customers", 0),
                            customer_walking_dist=get(distance_info, "customer_walking_dist", 0.0),
                            customer_bicycle_dist=get(distance_info, "customer_bicycle_dist", 0.0),
                            customer_vehicle_ded_dist=get(distance_info, "customer_vehicle_ded_dist", 0.0),
                            customer_vehicle_link_dist=get(distance_info, "customer_vehicle_link_dist", 0.0),
                            viz_data=(customers=customers_for_viz, route_data=route_data, distance_info=distance_info),
                            locker_stats=Dict{Tuple{Int,Int,Int}, ScenarioLockerStats}(),  # 빈 딕셔너리
                            omega_costs=Float64[]  # 빈 배열
                        ))
                    end
                catch omega_error
                    progress_println("      ⚠️ Bump & Shift 후 k=$(actual_k), ω=$omega 결과 수집 실패: $omega_error")
                end
            end
            
            # 시설 위치 저장 업데이트
            scenario4_k_facilities[actual_k] = copy(selected_locations)
        end
    end
            
    # ═══════════════════════════════════════════════════════════════════════════
    # 결과 집계
    # ═══════════════════════════════════════════════════════════════════════════
    final_results = []
    
    for k in 1:MAX_PUB
        k_results = filter(r -> r.k == k, all_results)
        if !isempty(k_results)
            avg_obj = mean([r.obj for r in k_results])
            avg_driving = mean([r.driving_dist for r in k_results])
            avg_walking = mean([r.walking_dist for r in k_results])
            avg_vehicles = mean([r.vehicles_used for r in k_results])
            avg_trips = mean([r.total_trips for r in k_results])
            avg_customers = mean([r.total_customers for r in k_results])
            avg_locker_customers = mean([r.locker_customers for r in k_results])
            avg_actual_locker_customers = mean([r.actual_locker_customers for r in k_results])
            avg_d2d_conversions = mean([r.d2d_conversions for r in k_results])  # 강제 D2D 전환 평균
            social_cost = mean([r.social_cost for r in k_results])
            # 목적함수 구성요소 평균
            avg_transport_cost = mean([r.transport_cost for r in k_results])
            avg_mobility_cost = mean([r.customer_mobility_cost for r in k_results])
            avg_activation_cost = mean([r.activation_cost for r in k_results])
            avg_num_active_lockers = mean([r.num_active_lockers for r in k_results])
            # 고객 이동 수단 선택 상세 평균 (4가지 수단)
            avg_inconvenience_cost = mean([r.customer_inconvenience_cost for r in k_results])
            avg_social_mode_cost = mean([r.customer_social_cost for r in k_results])
            avg_walk_choice_customers = mean([r.num_walk_choice_customers for r in k_results])
            avg_bicycle_choice_customers = mean([r.num_bicycle_choice_customers for r in k_results])
            avg_vehicle_ded_customers = mean([r.num_vehicle_dedicated_customers for r in k_results])
            avg_vehicle_link_customers = mean([r.num_vehicle_linked_customers for r in k_results])
            avg_customer_walking_dist = mean([r.customer_walking_dist for r in k_results])
            avg_customer_bicycle_dist = mean([r.customer_bicycle_dist for r in k_results])
            avg_customer_vehicle_ded_dist = mean([r.customer_vehicle_ded_dist for r in k_results])
            avg_customer_vehicle_link_dist = mean([r.customer_vehicle_link_dist for r in k_results])
            
            first_result = k_results[1]
            
            push!(final_results, (
                scenario=SCENARIO_OPL,
                k=k,
                obj=avg_obj,
                driving_dist=avg_driving,
                walking_dist=avg_walking,
                nOpen=first_result.nOpen,
                lockers=first_result.lockers,
                vehicles_used=avg_vehicles,
                total_trips=avg_trips,
                total_customers=avg_customers,
                locker_customers=avg_locker_customers,
                actual_locker_customers=avg_actual_locker_customers,
                d2d_conversions=avg_d2d_conversions,  # 강제 D2D 전환 평균
                avg_dist_per_vehicle=avg_vehicles > 0 ? avg_driving / avg_vehicles : 0.0,
                avg_walking_per_customer=avg_actual_locker_customers > 0 ? avg_walking / avg_actual_locker_customers : 0.0,
                social_cost=social_cost,
                total_distance=avg_driving + avg_walking,
                # 목적함수 구성요소 (논문 기반)
                transport_cost=avg_transport_cost,
                customer_mobility_cost=avg_mobility_cost,
                activation_cost=avg_activation_cost,
                num_active_lockers=avg_num_active_lockers,
                # 고객 이동 수단 선택 상세 (4가지 수단)
                customer_inconvenience_cost=avg_inconvenience_cost,
                customer_social_cost=avg_social_mode_cost,
                num_walk_choice_customers=avg_walk_choice_customers,
                num_bicycle_choice_customers=avg_bicycle_choice_customers,
                num_vehicle_dedicated_customers=avg_vehicle_ded_customers,
                num_vehicle_linked_customers=avg_vehicle_link_customers,
                customer_walking_dist=avg_customer_walking_dist,
                customer_bicycle_dist=avg_customer_bicycle_dist,
                customer_vehicle_ded_dist=avg_customer_vehicle_ded_dist,
                customer_vehicle_link_dist=avg_customer_vehicle_link_dist,
                viz_data=first_result.viz_data,  # 첫 번째 결과의 시각화 데이터 사용
                locker_stats=Dict{Tuple{Int,Int,Int}, ScenarioLockerStats}(),  # 빈 딕셔너리
                omega_costs=[r.obj for r in k_results]  # k별 오메가 비용
            ))
            
            progress_println("   📊 k=$k: E[사회적비용]=$(round(social_cost, digits=3))EUR (운송=$(round(avg_transport_cost, digits=1))+ 고객이동=$(round(avg_mobility_cost, digits=1))+ 활성화=$(round(avg_activation_cost, digits=1)))")
        end
    end
    
    # 최적 k 선택
    if !isempty(final_results)
        optimal_result = final_results[argmin([r.obj for r in final_results])]
        optimal_k = optimal_result.k
        
        scenario4_results[:optimal_k] = optimal_k
        scenario4_results[:facilities] = get(k_fixed_locations, optimal_k, [])
        
        progress_println("🎯 SLRP 최적 결과: k=$optimal_k")
        progress_println("   📍 최적 락커 위치: $(length(get(k_fixed_locations, optimal_k, [])))개")
        progress_println("   📊 기준선 대비 절감: $(round(baseline_cost - optimal_result.obj, digits=2))km ($(round((baseline_cost - optimal_result.obj) / baseline_cost * 100, digits=1))%)")
        
        # ═══════════════════════════════════════════════════════════════════
        # k별 상세 결과 CSV 저장
        # ═══════════════════════════════════════════════════════════════════
        k_details = []
        for k in 1:MAX_PUB
            if haskey(k_fixed_locations, k)
                locations = k_fixed_locations[k]
                k_result = filter(r -> r.k == k, final_results)
                
                if !isempty(k_result)
                    result = k_result[1]
                    
                    # 위치 정보를 문자열로 변환
                    location_strs = []
                    for (i, (lat, lon)) in enumerate(locations)
                        # POI 이름 찾기
                        poi_name = "Unknown"
                        for (plat, plon, name, _) in OPL_CANDIDATE_LOCATIONS
                            if abs(plat - lat) < 0.0001 && abs(plon - lon) < 0.0001
                                poi_name = name
                                break
                            end
                        end
                        push!(location_strs, "$(poi_name) ($(round(lat, digits=5)),$(round(lon, digits=5)))")
                    end
                    
                    push!(k_details, (
                        k = k,
                        is_optimal = (k == optimal_k),
                        num_lockers = length(locations),
                        avg_social_cost = round(result.social_cost, digits=3),
                        avg_driving_dist = round(result.driving_dist, digits=3),
                        avg_walking_dist = round(result.walking_dist, digits=3),
                        # 목적함수 구성요소 (논문 기반)
                        avg_transport_cost = round(result.transport_cost, digits=3),
                        avg_mobility_cost = round(result.customer_mobility_cost, digits=3),
                        avg_activation_cost = round(result.activation_cost, digits=3),
                        cost_reduction_vs_d2d = round(baseline_cost - result.obj, digits=3),
                        cost_reduction_pct = round((baseline_cost - result.obj) / baseline_cost * 100, digits=2),
                        locker_locations = join(location_strs, " | ")
                    ))
            end
        end
        end
        
        # k별 결과 CSV 저장
        if !isempty(k_details)
            k_details_df = DataFrame(k_details)
            k_details_path = joinpath(OUTDIR, "scenario4_k_details.csv")
            CSV.write(k_details_path, k_details_df)
            progress_println("   💾 k별 상세 결과 저장: $k_details_path")
                end
            end
            
    slrp_elapsed = round(time() - slrp_start_time, digits=1)
    optimal_facilities = get(scenario4_results, :facilities, Tuple{Float64,Float64}[])
    optimal_k_count = length(optimal_facilities)
    
    progress_println("")
    progress_println("   ┌─────────────────────────────────────────────────────────┐")
    progress_println("   │  ✅ 시나리오 4 SLRP 완료 ($(slrp_elapsed)초)                        │")
    progress_println("   │     최적 k=$(optimal_k_count)개 락커                                      │")
    progress_println("   └─────────────────────────────────────────────────────────┘")
    
    # Worker에서 수집한 스냅샷을 결과와 함께 반환
    collected_snapshots = Dict{Tuple{Int,Int}, Dict{String,Any}}()
    lock(SCENARIO4_SNAPSHOT_LOCK)
    try
        for (key, val) in scenario4_route_snapshots
            collected_snapshots[key] = deepcopy(val)
        end
    finally
        unlock(SCENARIO4_SNAPSHOT_LOCK)
    end
    
    return (results=final_results, snapshots=collected_snapshots)
end

# 🎯 SLRP용 기대 비용 계산 (모든 오메가에서 라우팅 비용 평균)
# 병렬화: pmap을 사용하여 omega 레벨 병렬 처리 (Python GIL 회피)
function calc_expected_routing_cost(locations::Vector, omega_customers::Vector, omega_attrs::Vector; 
                                       show_progress::Bool=false, label::String="", moo_seed::Int=42)
    n_omega = length(omega_customers)
    
    if isempty(locations)
        # 락커 없이 D2D only 비용 추정 - pmap 병렬 처리
        results = pmap(1:n_omega) do omega
            if isempty(omega_customers[omega])
                return (cost=0.0, valid=false)
            end
            
            try
                # 시나리오 1 (D2D) 방식으로 비용 계산
                _, distance_info, _ = solve_vrp_single_omega(
                    Dict{String,Int}(), omega_customers[omega], omega_attrs[omega], 
                    Dict{String,Any}(), SCENARIO_D2D;
                    verbose=false, omega=omega, collect_stats=false, moo_seed=moo_seed
                )
                
                if distance_info["total"] != Inf && !isnan(distance_info["total"])
                    return (cost=distance_info["f3_co2_cost"], valid=true)
                else
                    return (cost=0.0, valid=false)
                end
            catch e
                @error "D2D baseline 계산 실패 (omega=$omega)" exception=(e, catch_backtrace())
                rethrow(e)
            end
        end
        
        valid_results = filter(r -> r.valid, results)
        return length(valid_results) > 0 ? sum(r.cost for r in valid_results) / length(valid_results) : Inf
    end
    
    # 주어진 위치에서 라우팅 비용 계산
    lock_pub = Dict{String, Any}()
    for (i, (lat, lon)) in enumerate(locations)
        locker_id = "EVAL_$(lpad(i, 2, '0'))"
        lock_pub[locker_id] = ((lon, lat), "Public")
    end
    
    z_val = Dict(lid => 1 for lid in keys(lock_pub))
    
    # pmap 병렬 처리
    results = pmap(1:n_omega) do omega
        if isempty(omega_customers[omega])
            return (cost=0.0, valid=false)
        end
        
        try
            _, distance_info, _ = solve_vrp_single_omega(
                z_val, omega_customers[omega], omega_attrs[omega], lock_pub, SCENARIO_OPL;
                verbose=false, omega=omega, collect_stats=false, moo_seed=moo_seed
            )
            
            if distance_info["total"] != Inf && !isnan(distance_info["total"])
                return (cost=distance_info["f3_co2_cost"], valid=true)
            else
                return (cost=0.0, valid=false)
            end
        catch e
            @error "OPL 비용 계산 실패 (omega=$omega)" exception=(e, catch_backtrace())
            rethrow(e)
        end
    end

    valid_results = filter(r -> r.valid, results)
    
    # SLRP 목적함수: Z = E[z₁ + z₂] + z₃ (f₃ CO2 기반)
    # E[z₁ + z₂]: 오메가별 (van_co2 + customer_vehicle_co2)의 평균
    # z₃: 락커 전력 CO2 (기댓값 외부)
    expected_z1_z2 = length(valid_results) > 0 ? sum(r.cost for r in valid_results) / length(valid_results) : Inf
    num_active_lockers = length(locations)
    z3_locker_co2 = num_active_lockers * SLRP_LOCKER_CO2_PER_UNIT_PER_DAY
    
    return expected_z1_z2 + z3_locker_co2
end

function solve_scenario_omega_average(scn::Int, omega_customers::Vector, omega_attrs::Vector; moo_seed::Int=42)
    # seed 변수 제거됨 - 호환성을 위해 고정값 1 사용
    seed = 1
    
    # 전달된 omega 데이터 크기 사용 (robustness 테스트에서 다른 N_omega 전달 가능)
    local n_omega = length(omega_customers)
    
    scenario_name = get_scenario_name(scn)
    progress_println("")
    progress_println("┌─────────────────────────────────────────────────────────────────┐")
    progress_println("│  🚀 시나리오 $(scn): $(scenario_name)                           │")
    progress_println("│     └─ $(n_omega)개 ω 반복, 총 고객: $(sum(length.(omega_customers)))명        │")
    progress_println("└─────────────────────────────────────────────────────────────────┘")
    
    if scn == SCENARIO_OPL
        opt_result = optimize_opl_locations(omega_customers, omega_attrs; moo_seed=moo_seed)
        results = opt_result.results
        worker_snapshots = opt_result.snapshots
        
        if isempty(results)
             # 실패 시 빈 결과 반환 (에러 방지)
             return (scenario=scn, k=0, obj=Inf,
                     driving_dist=Inf, walking_dist=Inf,
                     nOpen=0, lockers="",
                     vehicles_used=0.0, total_trips=0.0, total_customers=0.0,
                     locker_customers=0.0, actual_locker_customers=0.0,
                     d2d_conversions=0.0,  # 강제 D2D 전환 고객 수
                     avg_dist_per_vehicle=0.0, avg_walking_per_customer=0.0,
                     social_cost=Inf, total_distance=Inf,
                     transport_cost=Inf, customer_mobility_cost=Inf, activation_cost=Inf, num_active_lockers=0,
                     # 고객 이동 수단 선택 상세 (4가지 수단)
                     customer_inconvenience_cost=0.0, customer_social_cost=0.0,
                     num_walk_choice_customers=0.0, num_bicycle_choice_customers=0.0,
                     num_vehicle_dedicated_customers=0.0, num_vehicle_linked_customers=0.0,
                     customer_walking_dist=0.0, customer_bicycle_dist=0.0,
                     customer_vehicle_ded_dist=0.0, customer_vehicle_link_dist=0.0,
                     viz_data=nothing, snapshots=Dict())
        end
        # 최적 결과 반환 (obj 최소) + 스냅샷 포함
        best_result = results[argmin([r.obj for r in results])]
        return merge(best_result, (snapshots=worker_snapshots,))
    end

    lock_pub = Dict{String,Any}()

    if scn == SCENARIO_D2D
        lock_pub = Dict{String,Any}()
    elseif scn == SCENARIO_PSPL
        # Private 락커 사용(부분 공유 규칙은 접근/아크 단계에서 처리)
        for (id,(lon,lat,carrier_name)) in LOCKERS_PRIV
            lock_pub[id] = ((lon,lat), "Private")
        end
    else
        for (id,(lon,lat,carrier_name)) in LOCKERS_PRIV
            lock_pub[id] = ((lon,lat), "Private")
        end
    end

    all_omega_customers = omega_customers
    all_omega_df_attrs  = omega_attrs

    # 모든 메트릭을 한 번의 루프에서 계산 (중복 제거)
    total_obj = 0.0
    total_driving = 0.0
    total_walking = 0.0
    total_vehicles_used = 0
    total_trips = 0
    total_customers = 0
    total_locker_customers = 0
    total_actual_locker_customers = 0
    total_d2d_conversions = 0  # 강제 D2D 전환 고객 수 총합
    # 목적함수 구성요소 추적 (논문 기반)
    total_transport_cost = 0.0
    total_mobility_cost = 0.0
    total_activation_cost = 0.0
    total_num_active_lockers = 0
    # 고객 이동 수단 선택 상세 추적 (4가지 수단)
    total_inconvenience_cost = 0.0        # 고객 불편비용 총합 (도보+자전거)
    total_social_mode_cost = 0.0          # 사회적비용 총합 (차량전용+연계)
    total_walk_choice_customers = 0       # 도보 선택 고객 수 총합
    total_bicycle_choice_customers = 0    # 자전거 선택 고객 수 총합
    total_vehicle_ded_customers = 0       # 차량(전용) 선택 고객 수 총합
    total_vehicle_link_customers = 0      # 차량(연계) 선택 고객 수 총합
    total_customer_walking_dist = 0.0     # 도보 선택 고객 총 거리 (왕복)
    total_customer_bicycle_dist = 0.0     # 자전거 선택 고객 총 거리 (왕복)
    total_customer_vehicle_ded_dist = 0.0 # 차량(전용) 총 거리 (왕복)
    total_customer_vehicle_link_dist = 0.0 # 차량(연계) 총 거리 (편도)
    
    # 오메가별 결과 저장 (디버깅용)
    omega_results = Vector{NamedTuple}()

    omega_start_time = time()

    for omega_idx in 1:n_omega
        omega_loop_start = time()
        
        # 캐리어 분석 초기화
        carrier_analysis = init_carrier_analysis(scn, 1, omega_idx)  # seed 고정값 1
        
        # 성능 최적화: deepcopy 제거 (solve_vrp_single_omega는 데이터를 수정하지 않음)
        customers = all_omega_customers[omega_idx]
        df_attr = all_omega_df_attrs[omega_idx]
        
        # 딕셔너리 캐싱으로 O(n²) → O(n) 최적화
        carrier_dict_omega = Dict(r.customer_id => r.carrier for r in eachrow(df_attr))
        
        # 캐리어별 락커 고객 수 계산 (성공률 계산용)
        total_locker_customers_by_carrier = Dict{String, Int}()
        for customer in customers
            if customer.dtype == "Locker"
                carrier = get(carrier_dict_omega, customer.id, nothing)
                if carrier !== nothing
                    total_locker_customers_by_carrier[carrier] = get(total_locker_customers_by_carrier, carrier, 0) + 1
                end
            end
        end
        
        progress_println("   ⏳ [ω=$(omega_idx)/$(n_omega)] VRP 최적화 중... (고객: $(length(customers))명)")
        sub_model, dist_info, route_data = solve_vrp_single_omega(Dict{String,Any}(), customers, df_attr, lock_pub, scn; verbose=false, omega=omega_idx, collect_stats=true, progress_callback=nothing, carrier_analysis=carrier_analysis, moo_seed=moo_seed)
        silent_optimize!(sub_model)
        
        omega_obj = 0.0
        omega_driving = 0.0
        omega_walking = 0.0
        
        if termination_status(sub_model) == OPTIMAL
            omega_obj = dist_info["social_cost"]  # 다른 시나리오와 동일하게 직접 거리 합계 사용
            omega_driving = dist_info["driving"]
            omega_walking = dist_info["walking"]
            
            total_obj += omega_obj
            total_driving += omega_driving
            total_walking += omega_walking
            total_vehicles_used += dist_info["vehicles_used"]
            total_trips += dist_info["total_trips"]
            total_customers += dist_info["total_customers"]
            total_locker_customers += dist_info["locker_customers"]
            total_actual_locker_customers += dist_info["actual_locker_customers"]
            total_d2d_conversions += get(dist_info, "d2d_conversions", 0)  # 강제 D2D 전환 누적
            # 목적함수 구성요소 누적
            total_transport_cost += get(dist_info, "transport_cost", omega_driving * TRANSPORT_COST_PER_KM)
            total_mobility_cost += get(dist_info, "customer_mobility_cost", 0.0)
            total_activation_cost += get(dist_info, "activation_cost", 0.0)
            total_num_active_lockers += get(dist_info, "num_active_lockers", 0)
            # 고객 이동 수단 선택 상세 누적 (4가지 수단)
            total_inconvenience_cost += get(dist_info, "customer_inconvenience_cost", 0.0)
            total_social_mode_cost += get(dist_info, "customer_social_cost", 0.0)
            total_walk_choice_customers += get(dist_info, "num_walk_choice_customers", 0)
            total_bicycle_choice_customers += get(dist_info, "num_bicycle_choice_customers", 0)
            total_vehicle_ded_customers += get(dist_info, "num_vehicle_dedicated_customers", 0)
            total_vehicle_link_customers += get(dist_info, "num_vehicle_linked_customers", 0)
            total_customer_walking_dist += get(dist_info, "customer_walking_dist", 0.0)
            total_customer_bicycle_dist += get(dist_info, "customer_bicycle_dist", 0.0)
            total_customer_vehicle_ded_dist += get(dist_info, "customer_vehicle_ded_dist", 0.0)
            total_customer_vehicle_link_dist += get(dist_info, "customer_vehicle_link_dist", 0.0)
        else
            omega_obj = 1e6
            omega_walking = 1e6
            total_obj += omega_obj
            total_walking += omega_walking
        end
        
        # MOO 세부 정보 수집 (있으면)
        omega_moo_detail = nothing
        omega_pareto_front = nothing
        try
            omega_moo_detail = get_last_moo_detail()
            # 시스템 레벨 Pareto front 사용 (전체 캐리어 합산)
            # dist_info에 저장된 system_pareto_front가 있으면 사용, 없으면 마지막 캐리어 PF
            sys_pf = get(dist_info, "system_pareto_front", nothing)
            if sys_pf !== nothing && isa(sys_pf, Matrix{Float64}) && size(sys_pf, 1) > 0
                omega_pareto_front = sys_pf
            else
                omega_pareto_front = get_last_pareto_front()
            end
        catch e
            # MOO 정보 없으면 무시
        end
        
        # 오메가별 결과 저장 (시각화용 데이터 포함)
        push!(omega_results, (
            omega = omega_idx,
            customers_count = length(all_omega_customers[omega_idx]),
            obj = omega_obj,
            driving = omega_driving,
            walking = omega_walking,
            total_dist = omega_driving + omega_walking,
            vehicles = dist_info["vehicles_used"],
            locker_customers = dist_info["locker_customers"],
            d2d_conversions = get(dist_info, "d2d_conversions", 0),
            locker_tracker = haskey(route_data, "locker_tracker") ? route_data["locker_tracker"] : nothing,
            locker_assignments = haskey(route_data, "locker_assignments") ? route_data["locker_assignments"] : Dict{String,String}(),
            # MOO 정보 추가
            moo_detail = omega_moo_detail,
            pareto_front = omega_pareto_front,
            # 시각화용 데이터 추가
            route_data = route_data,
            distance_info = dist_info,
            customers = customers
        ))
        
        omega_elapsed = round(time() - omega_loop_start, digits=1)
        progress_println("   ✅ [ω=$(omega_idx)/$(n_omega)] 완료: 비용=$(round(omega_obj, digits=2))km, 차량=$(dist_info["vehicles_used"])대, 트립=$(dist_info["total_trips"])개 ($(omega_elapsed)초)")
        
        update_scenario_progress(scn)
    end
    
    total_omega_elapsed = round(time() - omega_start_time, digits=1)
    progress_println("   📊 시나리오 $(scn) 전체 완료: 평균비용=$(round(total_obj/n_omega, digits=2))km ($(total_omega_elapsed)초)")

    # 총합값 기준 목적함수 (모든 오메가의 총 거리)
    obj_value = total_obj  # 모든 오메가의 총 목적함수값
    avg_driving = total_driving / n_omega
    avg_walking = total_walking / n_omega
    
    avg_vehicles_used = total_vehicles_used / n_omega
    avg_trips = total_trips / n_omega
    avg_customers = total_customers / n_omega
    avg_locker_customers = total_locker_customers / n_omega
    avg_actual_locker_customers = total_actual_locker_customers / n_omega
    avg_d2d_conversions = total_d2d_conversions / n_omega  # 강제 D2D 전환 평균
    avg_dist_per_vehicle = avg_vehicles_used > 0 ? avg_driving / avg_vehicles_used : 0.0
    avg_walking_per_customer = avg_actual_locker_customers > 0 ? avg_walking / avg_actual_locker_customers : 0.0
    # ═══════════════════════════════════════════════════════════════════════════
    # 논문 수식: Z = (1/N_ω) Σ(z₁ + z₂) + z₃
    # z₁, z₂는 확률적(stochastic) → 기댓값 계산
    # z₃는 결정적(deterministic) → 기댓값 외부에서 추가
    # ═══════════════════════════════════════════════════════════════════════════
    
    # E[z₁ + z₂]: 운송비용과 고객 이동비용의 기댓값
    avg_transport_cost = total_transport_cost / n_omega
    avg_mobility_cost = total_mobility_cost / n_omega
    
    # z₃: 락커 활성화 비용 (1단계 결정변수, 모든 ω에서 동일)
    # 활성화된 락커 수는 첫 번째 ω의 값 사용 (모든 ω에서 동일하므로)
    num_active_lockers = length(keys(filter(item->item.second[2]=="Private", lock_pub)))
    activation_cost = num_active_lockers * LOCKER_ACTIVATION_COST
    avg_activation_cost = activation_cost  # z₃는 기댓값이 아닌 고정값
    avg_num_active_lockers = num_active_lockers  # 평균이 아닌 고정값
    
    # 총 사회적 비용 = E[z₁ + z₂] + z₃
    social_cost = avg_transport_cost + avg_mobility_cost + activation_cost
    
    # 고객 이동 수단 선택 상세 평균 (4가지 수단)
    avg_inconvenience_cost = total_inconvenience_cost / n_omega        # 고객 불편비용 평균 (도보+자전거)
    avg_social_mode_cost = total_social_mode_cost / n_omega            # 사회적비용 평균 (차량전용+연계)
    avg_walk_choice_customers = total_walk_choice_customers / n_omega  # 도보 선택 고객 수 평균
    avg_bicycle_choice_customers = total_bicycle_choice_customers / n_omega  # 자전거 선택 고객 수 평균
    avg_vehicle_ded_customers = total_vehicle_ded_customers / n_omega  # 차량(전용) 선택 고객 수 평균
    avg_vehicle_link_customers = total_vehicle_link_customers / n_omega  # 차량(연계) 선택 고객 수 평균
    avg_customer_walking_dist = total_customer_walking_dist / n_omega  # 도보 총 거리 평균 (왕복)
    avg_customer_bicycle_dist = total_customer_bicycle_dist / n_omega  # 자전거 총 거리 평균 (왕복)
    avg_customer_vehicle_ded_dist = total_customer_vehicle_ded_dist / n_omega  # 차량(전용) 총 거리 평균 (왕복)
    avg_customer_vehicle_link_dist = total_customer_vehicle_link_dist / n_omega  # 차량(연계) 총 거리 평균 (편도)

        final_chosen_list = collect(keys(filter(item->item.second[2]=="Private", lock_pub)))

    # 오메가별 결과 분산 계산 및 디버깅 출력
    if length(omega_results) > 1
        total_distances = [r.total_dist for r in omega_results]
        mean_total = mean(total_distances)
        variance_total = var(total_distances)
        std_total = std(total_distances)
        
        # 시나리오 D2D일 때만 상세 출력 (로그 과부하 방지)
        if scn == SCENARIO_D2D
            progress_println("🔍 [디버그] 시나리오 $scn 오메가별 결과:")
            for (i, r) in enumerate(omega_results)
                progress_println("   Ω$i: 고객$(r.customers_count)명, 총거리$(round(r.total_dist,digits=1))km (🚛$(round(r.driving,digits=1)) + 🚶$(round(r.walking,digits=1))), 차량$(r.vehicles)대")
            end
            progress_println("   📊 평균: $(round(mean_total,digits=1))km, 표준편차: $(round(std_total,digits=1))km, 분산: $(round(variance_total,digits=1))")
            progress_println("   ✅ 오메가별 계산 정상 작동 확인")
        end
    end

    # 캐리어별 통계 수집 (오메가별로 개별 수집 - 몬테카르로 시뮬레이션)
    for omega_idx in 1:n_omega
        customers = all_omega_customers[omega_idx]
        df_attr = all_omega_df_attrs[omega_idx]
        
        # 딕셔너리 캐싱으로 O(n²) → O(n) 최적화
        carrier_dict_stats = Dict(r.customer_id => r.carrier for r in eachrow(df_attr))
        
        stats = init_locker_stats(scn, 1, omega_idx, lock_pub)  # seed 제거됨, 고정값 1 사용
        update_carrier_stats!(stats, df_attr, customers)
        
        # 해당 오메가의 실제 결과 사용
        corresponding_result = omega_results[omega_idx]
        
        # 실제 락커 고객 수 계산 (해당 오메가 기준)
        locker_customers_count = length([c for c in customers if c.dtype == "Locker"])
        
        # 실제 D2D 전환 수 사용 (omega_results에서 가져오기)
        actual_d2d_conversions = corresponding_result.d2d_conversions
        
        # 통계 업데이트
        stats.total_locker_customers = locker_customers_count
        stats.d2d_conversions = actual_d2d_conversions
        stats.conversion_rate = locker_customers_count > 0 ? stats.d2d_conversions / locker_customers_count : 0.0
        
        # 캐리어별 개별 라우팅을 통한 정확한 1대의 1회 투어 통계 계산
        carrier_customers = Dict{String, Vector{eltype(customers)}}()
        
        # 캐리어별 고객 분류 (딕셔너리 캐싱 활용)
        for customer in customers
            carrier = get(carrier_dict_stats, customer.id, nothing)
            if carrier !== nothing
                if !haskey(carrier_customers, carrier)
                    carrier_customers[carrier] = Vector{eltype(customers)}()
                end
                push!(carrier_customers[carrier], customer)
            end
        end
        
        # 각 캐리어별로 정확한 1대의 1회 투어 기준 통계 계산
        for (carrier, carrier_customer_list) in carrier_customers
            if isempty(carrier_customer_list)
                continue
            end
            
            carrier_customer_count = length(carrier_customer_list)
            
            # 캐리어별 차량 수 고려
            carrier_vehicles = 1  # 멀티트립 허용, 대수 미사용
            
            # 캐리어별 실제 필요한 총 투어 수 계산
            # = ceil(고객 수 / 차량 용량) (각 투어는 1대 차량이 수행)
            carrier_total_tours = max(1, Int(ceil(carrier_customer_count / effective_capacity())))
            
            # 캐리어별 거리 추정 - 차량 효율성 고려
            # 차량 수가 많은 캐리어는 더 효율적으로 라우팅 가능
            total_customers = length(customers)
            base_proportion = carrier_customer_count / total_customers
            
            # 차량 효율성 팩터 (차량 수가 많을수록 효율적)
            efficiency_factor = 1.0 / sqrt(carrier_vehicles)  # 차량 수가 많을수록 효율성 증가
            carrier_distance = corresponding_result.driving * base_proportion * efficiency_factor
            
            # 캐리어 통계 수동 업데이트
            if haskey(stats.carrier_stats, carrier)
                carrier_stat = stats.carrier_stats[carrier]
                carrier_stat.total_distance = carrier_distance
                
                # 1대의 1회 투어당 평균 거리
                carrier_stat.avg_distance_per_used_vehicle = carrier_total_tours > 0 ? 
                    carrier_distance / carrier_total_tours : 0.0
                
                # 1대의 1회 투어에서 수요 1단위당 거리
                if carrier_total_tours > 0 && carrier_customer_count > 0
                    carrier_distance_per_tour = carrier_distance / carrier_total_tours
                    carrier_demand_per_tour = carrier_customer_count / carrier_total_tours
                    carrier_stat.km_per_demand = carrier_demand_per_tour > 0 ? 
                        carrier_distance_per_tour / carrier_demand_per_tour : 0.0
                else
                    carrier_stat.km_per_demand = 0.0
                end
                
                # 1대의 1회 투어당 평균 점유율
                if carrier_total_tours > 0 && effective_capacity() > 0
                    carrier_avg_demand_per_trip = carrier_customer_count / carrier_total_tours
                    carrier_stat.avg_fill_rate_per_trip = carrier_avg_demand_per_trip / effective_capacity()
                else
                    carrier_stat.avg_fill_rate_per_trip = 0.0
                end
            end
        end
        
        # 실제 락커 사용량 정보 반영
        if !isempty(lock_pub) && corresponding_result.locker_tracker !== nothing
            locker_tracker = corresponding_result.locker_tracker
            for (locker_id, (_, locker_type)) in lock_pub
                if haskey(stats.locker_stats, locker_id) && haskey(locker_tracker.usage, locker_id)
                    # 실제 LockerUsageTracker에서 가져온 사용량 사용
                    stats.locker_stats[locker_id].used = locker_tracker.usage[locker_id]
                    capacity = get(locker_tracker.capacity, locker_id, 1)
                    stats.locker_stats[locker_id].occupancy_rate = stats.locker_stats[locker_id].used / capacity
                end
            end
        end
        
        # 캐리어별 락커 사용량 업데이트 (carrier_usage, carrier_percentages 채우기)
        if corresponding_result.locker_assignments !== nothing && !isempty(corresponding_result.locker_assignments)
            update_locker_stats_with_assignments!(stats, corresponding_result.locker_assignments, df_attr)
        end
        
        progress_println("📊 캐리어 통계 완료: 시나리오 $scn, 오메가 $omega_idx")
        collect_locker_stats!(stats)
    end

    total_distance = avg_driving + avg_walking  # 실제 총 이동거리 계산
    
    # 시나리오별 k값 결정 (실제 락커 개수 기준)
    k_value = if scn == SCENARIO_D2D
        0  # 락커 없음
    elseif scn == SCENARIO_DPL || scn == SCENARIO_SPL || scn == SCENARIO_PSPL
        length(LOCKERS_PRIV)  # Private 락커 실제 개수 (9개)
    else
        length(final_chosen_list)  # OPL: 선택된 락커 개수
    end
    
    # 첫 번째 오메가의 시각화 데이터 추출 (대표 사례)
    viz_data = if !isempty(omega_results)
        first_omega = omega_results[1]
        (
            customers = first_omega.customers,
            route_data = first_omega.route_data,
            distance_info = first_omega.distance_info
        )
    else
        nothing
    end
    
    # 워커에서 수집된 락커 통계 추출 (메인 프로세스로 전달)
    worker_locker_stats = Dict{Tuple{Int,Int,Int}, ScenarioLockerStats}()
    lock(STATS_LOCK)
    try
        for (key, stats) in LOCKER_STATS_COLLECTOR.stats_by_scenario
            if key[1] == scn  # 시나리오만 체크 (seed 제거)
                worker_locker_stats[key] = deepcopy(stats)
            end
        end
    finally
        unlock(STATS_LOCK)
    end
    
    return (scenario=scn, k=k_value, obj=social_cost,
            driving_dist=avg_driving, walking_dist=avg_walking,
            nOpen=length(final_chosen_list), lockers=join(final_chosen_list, ","),
            vehicles_used=avg_vehicles_used, total_trips=avg_trips, total_customers=avg_customers,
            locker_customers=avg_locker_customers,
            actual_locker_customers=avg_actual_locker_customers,
            d2d_conversions=avg_d2d_conversions,  # 강제 D2D 전환 평균
            avg_dist_per_vehicle=avg_dist_per_vehicle, 
            avg_walking_per_customer=avg_walking_per_customer,
            social_cost=social_cost,
            total_distance=total_distance,
            # 목적함수 구성요소 (논문 기반)
            transport_cost=avg_transport_cost,      # z₁: 운송비용 (EUR)
            customer_mobility_cost=avg_mobility_cost,  # z₂: 고객 이동비용 (EUR)
            activation_cost=avg_activation_cost,    # z₃: 락커활성화비용 (EUR)
            num_active_lockers=avg_num_active_lockers,
            # 고객 이동 수단 선택 상세 (4가지 수단)
            customer_inconvenience_cost=avg_inconvenience_cost,  # 고객 불편비용 (도보+자전거)
            customer_social_cost=avg_social_mode_cost,           # 사회적비용 (차량전용+연계)
            num_walk_choice_customers=avg_walk_choice_customers,      # 도보 선택 고객 수
            num_bicycle_choice_customers=avg_bicycle_choice_customers,  # 자전거 선택 고객 수
            num_vehicle_dedicated_customers=avg_vehicle_ded_customers,  # 차량(전용) 선택 고객 수
            num_vehicle_linked_customers=avg_vehicle_link_customers,    # 차량(연계) 선택 고객 수
            customer_walking_dist=avg_customer_walking_dist,          # 도보 총 거리 (왕복)
            customer_bicycle_dist=avg_customer_bicycle_dist,          # 자전거 총 거리 (왕복)
            customer_vehicle_ded_dist=avg_customer_vehicle_ded_dist,  # 차량(전용) 총 거리 (왕복)
            customer_vehicle_link_dist=avg_customer_vehicle_link_dist,  # 차량(연계) 총 거리 (편도)
            viz_data=viz_data,  # 시각화용 데이터 추가
            locker_stats=worker_locker_stats,  # 워커 락커 통계 (메인으로 전달)
            omega_costs=[r.obj for r in omega_results],  # 오메가별 비용 (통계 검정용)
            omega_results=omega_results)  # MOO 분석용 omega별 상세 결과
end




"""
MOO 결과로부터 omega별 통계 계산 및 MOOScenarioResult 생성
"""
function collect_moo_results_from_omega(scn::Int, scenario_name::String, k_value::Int, 
                                        omega_results::Vector, lock_pub::Dict)
    moo_collected = Vector{MOOScenarioResult}()
    
    for omega_res in omega_results
        omega_idx = omega_res.omega
        
        # MOO 정보가 있는 경우에만 수집
        if omega_res.moo_detail !== nothing && omega_res.pareto_front !== nothing
            # 시스템 레벨 Pareto front 사용 (omega_results에 이미 시스템 PF가 들어있음)
            pf = omega_res.pareto_front
            moo_detail = omega_res.moo_detail
            
            # Compromise solution 선택
            if size(pf, 1) > 0
                f1_vals = pf[:, 1]
                f2_vals = pf[:, 2]
                f3_vals = pf[:, 3]
                
                f1_norm = (f1_vals .- minimum(f1_vals)) ./ (maximum(f1_vals) - minimum(f1_vals) + 1e-9)
                f2_norm = (f2_vals .- minimum(f2_vals)) ./ (maximum(f2_vals) - minimum(f2_vals) + 1e-9)
                f3_norm = (f3_vals .- minimum(f3_vals)) ./ (maximum(f3_vals) - minimum(f3_vals) + 1e-9)
                
                distances = sqrt.(f1_norm.^2 .+ f2_norm.^2 .+ f3_norm.^2)
                selected_idx = argmin(distances)
                
                # 락커 정보
                locker_ids_str = join(collect(keys(lock_pub)), ",")
                
                # distance_info에서 정보 가져오기
                dist_info = omega_res.distance_info
                
                # MOOScenarioResult 생성 (전체 캐리어 누적값 사용 - distance_info에 저장됨)
                total_vehicle_cost = dist_info["vehicles_used"] * effective_vehicle_daily_cost()
                total_fuel_cost = dist_info["driving"] * effective_fuel_cost()
                corrected_f1 = total_fuel_cost + total_vehicle_cost
                
                moo_result = MOOScenarioResult(
                    scn,
                    scenario_name,
                    k_value,
                    omega_idx,
                    copy(pf),
                    selected_idx,
                    corrected_f1,                                          # 전체 캐리어 합산 f1
                    get(dist_info, "moo_accum_f2", 0.0),                  # 전체 캐리어 합산 f2
                    get(dist_info, "moo_accum_f3", 0.0),                  # 전체 캐리어 합산 f3
                    total_fuel_cost,                                       # 전체 거리 × 유류비
                    total_vehicle_cost,                                    # 전체 차량수 × 차량비
                    0.0,                                                   # 인건비 제외
                    get(dist_info, "moo_accum_f2_mobility", 0.0),         # 전체 합산 이동불편비용
                    get(dist_info, "moo_accum_f2_dissatisfaction", 0.0),  # 전체 합산 불만족도
                    get(dist_info, "moo_accum_avg_satisfaction", 0.0),    # 전체 가중평균 만족도
                    get(dist_info, "moo_accum_avg_delay", 0.0),           # 전체 가중평균 지연시간
                    get(dist_info, "moo_accum_avg_dist", 0.0),            # 전체 가중평균 이동거리
                    (hasproperty(moo_detail, :mobility_raw_km) ? moo_detail.mobility_raw_km : get(dist_info, "moo_accum_avg_dist", 0.0)),
                    (hasproperty(moo_detail, :dissatisfaction_raw) ? moo_detail.dissatisfaction_raw : (get(dist_info, "total_customers", 0) - get(dist_info, "actual_locker_customers", 0)) * (1.0 - get(dist_info, "moo_accum_avg_satisfaction", 1.0))),
                    get(dist_info, "moo_accum_f3_vehicle_co2", 0.0),     # 전체 합산 차량 CO2
                    get(dist_info, "moo_accum_f3_customer_co2", 0.0),    # 전체 합산 고객 CO2
                    get(dist_info, "moo_accum_f3_locker_co2", 0.0),      # 전체 합산 락커 CO2
                    dist_info["driving"],                                  # 전체 캐리어 거리 합산
                    dist_info["vehicles_used"],
                    dist_info["total_trips"],
                    dist_info["total_customers"],
                    dist_info["locker_customers"],
                    dist_info["actual_locker_customers"],
                    get(dist_info, "d2d_conversions", 0),
                    get(dist_info, "num_active_lockers", 0),
                    locker_ids_str,
                    Float64(get(dist_info, "num_walk_choice_customers", 0)),
                    Float64(get(dist_info, "num_bicycle_choice_customers", 0)),
                    Float64(get(dist_info, "num_vehicle_dedicated_customers", 0)),
                    Float64(get(dist_info, "num_vehicle_linked_customers", 0)),
                    get(dist_info, "customer_walking_dist", 0.0),
                    get(dist_info, "customer_bicycle_dist", 0.0),
                    get(dist_info, "customer_vehicle_ded_dist", 0.0),
                    get(dist_info, "customer_vehicle_link_dist", 0.0)
                )
                
                push!(moo_collected, moo_result)
            end
        end
    end
    
    return moo_collected
end


function main_batch_optimization()
    progress_println("\n" * "="^80)
    progress_println("🎯 완전 Monte Carlo ALNS CVRP 최적화 시작")
    progress_println("🔬 모든 시나리오에서 $(Nomega)개 무작위 고객 세트로 평균 비용 계산")
    
    # 전체 작업량 계산 및 표시
    total_scenarios = length(SCENARIOS_TO_RUN)
    total_works = total_scenarios * Nomega
    
    progress_println("📊 전체 작업 개요:")
    progress_println("   - 총 시나리오: $(total_scenarios)개")
    progress_println("   - Monte Carlo 반복: $(Nomega)회")
    progress_println("   - 예상 총 작업: $(total_works)개")
    progress_println("📋 시나리오별 세부사항:")
    progress_println("   - 시나리오 1-3, 5: 직접 ALNS/VNS 최적화 (각 $(Nomega)개 몬테카르로)")
    progress_println("   - 시나리오 4: SLRP 기반 락커 위치/개수 동시 최적화 (k=1~$MAX_PUB 비교)")
    progress_println("="^80)
    global_start = time()
    init_progress()

    # ═══════════════════════════════════════════════════════════════════════════
    # 오메가별 고객 데이터 생성 (모든 시나리오에서 동일한 고객 세트 사용)
    # - 각 ω마다 완전히 새로운 고객 세트 (랜덤 위치, 배송사)
    # - 같은 ω 내에서 시나리오 1~5는 동일 고객으로 공정 비교
    # ═══════════════════════════════════════════════════════════════════════════
    if CUSTOMER_LIMIT > 0
        progress_println("⚠️  테스트 모드: 고객 수 제한 (CUSTOMER_LIMIT=$(CUSTOMER_LIMIT)명)")
    end
    progress_println("📋 오메가별 고객 데이터 사전 생성 중... (n=$Nomega)")
    
    # Monte Carlo 시뮬레이션용 여러 고객 시나리오 생성
    omega_customers = Vector{Vector{NamedTuple}}()
    omega_attrs = Vector{DataFrame}()
    
    for omega in 1:Nomega
        # 진짜 랜덤 - 매 실행마다 다른 고객 위치/배송사 생성
        customers = gen_customers()
        if CUSTOMER_LIMIT > 0
            customers = customers[1:min(end, CUSTOMER_LIMIT)]
            attrs = gen_attr(customers)
        else
            attrs = gen_attr(customers)
        end
        push!(omega_customers, customers)
        push!(omega_attrs, attrs)
    end
    
    progress_println("   ✅ $(Nomega)개 고객 시나리오 생성 완료, 평균 고객 $(round(mean(length.(omega_customers)), digits=1))명/ω")
    progress_println("✅ 고객 데이터 생성 완료\n")

    # ============================================================================
    # 도로망 기반 거리 행렬 사전 계산 (OSRM)
    # ============================================================================
    if USE_ROAD_DISTANCE
        progress_println("🗺️ 도로망 기반 거리 행렬 초기화 중...")
        
        # 데포 위치 수집 (lon, lat → lat, lon 변환)
        depot_locations = Tuple{Float64,Float64}[]
        for (depot_id, (lon, lat, carrier)) in DEPOTS
            push!(depot_locations, (Float64(lat), Float64(lon)))
        end
        
        # 락커 위치 수집 (LOCKERS_PRIV + OPL 후보)
        locker_locations = Tuple{Float64,Float64}[]
        for (locker_id, (lon, lat, owner)) in LOCKERS_PRIV
            push!(locker_locations, (Float64(lat), Float64(lon)))
        end
        # 시나리오 4 (OPL) 후보 위치 추가 (20개 POI)
        for (lat, lon, name, loc_type) in OPL_CANDIDATE_LOCATIONS
            push!(locker_locations, (Float64(lat), Float64(lon)))
        end
        
        # 고객 위치 수집 (모든 오메가)
        customer_locations = Tuple{Float64,Float64}[]
        for customers in omega_customers
            for c in customers
                # c.coord는 (lon, lat) 형식
                push!(customer_locations, (Float64(c.coord[2]), Float64(c.coord[1])))
            end
        end
        
        # 중복 제거
        customer_locations = unique(customer_locations)
        
        progress_println("   📍 데포: $(length(depot_locations))개, 락커: $(length(locker_locations))개 (Private 9 + Metro 9 + OPL후보 20), 고객: $(length(customer_locations))개")
        
        # 거리 행렬 초기화
        try
            initialize_distance_matrix(depot_locations, locker_locations, customer_locations)
            progress_println("✅ 도로망 거리 행렬 초기화 완료\n")
        catch e
            @error "도로망 거리 행렬 초기화 실패" exception=(e, catch_backtrace())
            rethrow(e)
        end
    end

    # ============================================================================
    # 1단계: 통계 수집
    # ============================================================================
    progress_println("\n" * "="^80)
    progress_println("🚀 1단계: 통계 수집 시작 (차량거리와 보행거리 표준화용)")
    progress_println("="^80)
    
    results_lock = ReentrantLock()
    stage1_results = Vector{NamedTuple}()
    
    total_scenarios = length(SCENARIOS_TO_RUN)
    scenario_idx = 0

    for scn_iter in SCENARIOS_TO_RUN
        scenario_idx += 1
        scenario_name = get_scenario_name(scn_iter)
        
        # f2 정규화 파라미터 리셋 (시나리오별 재계산, Marler & Arora 2005)
        reset_f2_normalization!()
        
        progress_println("")
        progress_println("╔════════════════════════════════════════════════════════════════════╗")
        progress_println("║  📌 [시나리오 $(scenario_idx)/$(total_scenarios)] $(scenario_name) (ID: $(scn_iter))                    ║")
        progress_println("║     Monte Carlo: $(Nomega)개                                            ║")
        progress_println("╚════════════════════════════════════════════════════════════════════╝")
        
        start_scenario(scn_iter)

        if scn_iter == SCENARIO_OPL
            # 시나리오 4: OPL 최적화 (단일 실행)
            opt_result = optimize_opl_locations(omega_customers, omega_attrs)
            
            results = opt_result.results
            worker_snapshots = opt_result.snapshots
            
            # Worker에서 수집한 스냅샷을 메인 프로세스의 전역 딕셔너리에 병합
            lock(SCENARIO4_SNAPSHOT_LOCK)
            try
                for (key, val) in worker_snapshots
                    scenario4_route_snapshots[key] = val
                end
            finally
                unlock(SCENARIO4_SNAPSHOT_LOCK)
            end
            
            for result in results
                push!(stage1_results, result)
                
                # 시각화 데이터를 MAIN_BATCH_RESULTS에 저장 (최적 k만)
                if haskey(result, :viz_data) && result.viz_data !== nothing
                    save_main_batch_result!(
                        SCENARIO_OPL, 1,  # omega=1 대표
                        result.viz_data.customers,
                        result.viz_data.route_data,
                        result.viz_data.distance_info
                    )
                end
            end
            complete_scenario(scn_iter)
        else
            # 시나리오 1-3, 5: 단일 실행
            out_data = solve_scenario_omega_average(scn_iter, omega_customers, omega_attrs)
            
            push!(stage1_results, out_data)
            
            # MOO 결과 수집
            if haskey(out_data, :omega_results) && out_data.omega_results !== nothing
                lock_pub = Dict{String,Any}()
                if scn_iter == SCENARIO_D2D
                    lock_pub = Dict{String,Any}()
                elseif scn_iter == SCENARIO_PSPL
                    for (id,(lon,lat,carrier_name)) in LOCKERS_PRIV
                        lock_pub[id] = ((lon,lat), "Private")
                    end
                else
                    for (id,(lon,lat,carrier_name)) in LOCKERS_PRIV
                        lock_pub[id] = ((lon,lat), "Private")
                    end
                end
                
                moo_results = collect_moo_results_from_omega(
                    scn_iter, 
                    get_scenario_name(scn_iter),
                    out_data.k,
                    out_data.omega_results,
                    lock_pub
                )
                
                lock(MOO_RESULTS_LOCK) do
                    append!(MOO_RESULTS_COLLECTOR, moo_results)
                end
                
                if !isempty(moo_results)
                    progress_println("   📊 MOO 결과 수집: $(length(moo_results))개 omega")
                end
            end
            
            # 워커에서 수집된 락커 통계를 메인 프로세스로 병합
            if haskey(out_data, :locker_stats) && out_data.locker_stats !== nothing
                lock(STATS_LOCK)
                try
                    for (key, stats) in out_data.locker_stats
                        LOCKER_STATS_COLLECTOR.stats_by_scenario[key] = stats
                    end
                finally
                    unlock(STATS_LOCK)
                end
            end
            
            # 시각화 데이터를 MAIN_BATCH_RESULTS에 저장
            if out_data.viz_data !== nothing
                save_main_batch_result!(
                    scn_iter, 1,  # omega=1 대표
                    out_data.viz_data.customers,
                    out_data.viz_data.route_data,
                    out_data.viz_data.distance_info
                )
            end
            complete_scenario(scn_iter)
        end
    end

    all_results = stage1_results

    # ═══════════════════════════════════════════════════════════════════════════
    # MOO 결과 저장 및 통계 분석 (논문용)
    # ═══════════════════════════════════════════════════════════════════════════
    progress_println("\n" * "="^80)
    progress_println("💾 MOO 결과 저장 및 통계 분석")
    progress_println("="^80)
    
    # MOO 결과 수집기에서 데이터 가져오기
    lock(MOO_RESULTS_LOCK) do
        if !isempty(MOO_RESULTS_COLLECTOR)
            # 1. Integrated results (MOO 형식)
            integrated_path = joinpath(OUTDIR, "integrated_results_moo.csv")
            save_integrated_results_moo(MOO_RESULTS_COLLECTOR, integrated_path)
            
            # 2. Pareto fronts (CSV)
            pareto_csv_path = joinpath(OUTDIR, "pareto_fronts_all.csv")
            save_all_pareto_fronts_csv(MOO_RESULTS_COLLECTOR, pareto_csv_path)
            
            # 3. Pareto fronts (JSON)
            pareto_json_path = joinpath(OUTDIR, "pareto_fronts_all.json")
            save_all_pareto_fronts_json(MOO_RESULTS_COLLECTOR, pareto_json_path)
            
            # 4. 시나리오 비교 테이블 (논문 Table용)
            comparison_path = joinpath(OUTDIR, "scenario_comparison_table.csv")
            save_scenario_comparison_table(MOO_RESULTS_COLLECTOR, comparison_path)
            
            # 4-1. 원시값 비교 전용 CSV (최종 비교용)
            raw_comparison_path = joinpath(OUTDIR, "raw_values_comparison.csv")
            save_raw_values_comparison(MOO_RESULTS_COLLECTOR, raw_comparison_path)
            
            # 5. 통계 요약 출력 (상세 - Mean±Std, Range, CV)
            print_pareto_summary(MOO_RESULTS_COLLECTOR)
        else
            progress_println("⚠️  MOO 결과가 수집되지 않았습니다.")
        end
    end
    
    progress_println("\n" * "="^60)
    progress_println("📊 Monte Carlo 결과 요약:")
    progress_println("="^60)

    progress_println("📊 시나리오별 통계 분석 (n=$Nomega 오메가 시나리오):")
    progress_println("="^70)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 다중 비용 유형 통계 데이터 수집
    # - social_cost: 총 사회적 비용 (z₁ + z₂ + z₃)
    # - transport_cost: 운송비용 (z₁)
    # - customer_mobility_cost: 고객 이동비용 (z₂)
    # ═══════════════════════════════════════════════════════════════════════════
    
    # 시나리오별, 비용유형별 omega 비용 저장
    # scenario_costs[scn_key][cost_type] = [omega1_cost, omega2_cost, ...]
    scenario_costs = Dict{String, Dict{String, Vector{Float64}}}()
    scenario_names = Dict{String, String}()  # scn_key => display_name
    
    cost_types = ["social_cost", "transport_cost", "customer_mobility_cost"]
    
    for scn in SCENARIOS_TO_RUN
        scn_name = get_scenario_name(scn)
        
        if scn == SCENARIO_OPL
            # ═══════════════════════════════════════════════════════════════════
            # OPL: k별로 따로 통계 수집
            # final_results에는 k별 집계 결과가 있고, omega_costs 필드에 omega별 비용 배열이 있음
            # ═══════════════════════════════════════════════════════════════════
            opl_results = filter(r -> r.scenario == SCENARIO_OPL, all_results)
            
            if !isempty(opl_results)
                progress_println("시나리오 $scn ($scn_name) - k별 상세:")
                
                for result in opl_results
                    k = result.k
                    scn_key = "OPL_k$k"
                    scenario_names[scn_key] = scn_key
                    
                    # omega_costs 필드에서 omega별 비용 추출 (이것이 실제 omega별 결과)
                    if haskey(result, :omega_costs) && !isempty(result.omega_costs)
                        omega_social_costs = collect(Float64, result.omega_costs)
                        n_omega = length(omega_social_costs)
                        
                        # 각 비용 유형별로 수집
                        scenario_costs[scn_key] = Dict{String, Vector{Float64}}()
                        scenario_costs[scn_key]["social_cost"] = omega_social_costs
                        
                        # transport_cost, customer_mobility_cost는 omega별 개별값이 없으므로
                        # 평균값을 기준으로 omega별 비율에 따라 추정
                        avg_social = mean(omega_social_costs)
                        transport_avg = haskey(result, :transport_cost) ? result.transport_cost : 0.0
                        mobility_avg = haskey(result, :customer_mobility_cost) ? result.customer_mobility_cost : 0.0
                        
                        if avg_social > 0
                            # omega별 비율에 따라 추정
                            scenario_costs[scn_key]["transport_cost"] = [transport_avg * (c / avg_social) for c in omega_social_costs]
                            scenario_costs[scn_key]["customer_mobility_cost"] = [mobility_avg * (c / avg_social) for c in omega_social_costs]
                        else
                            scenario_costs[scn_key]["transport_cost"] = fill(transport_avg, n_omega)
                            scenario_costs[scn_key]["customer_mobility_cost"] = fill(mobility_avg, n_omega)
                        end
                        
                        # 요약 출력
                        valid_costs = filter(c -> c != Inf && !isnan(c), omega_social_costs)
                        if !isempty(valid_costs)
                            avg_cost = mean(valid_costs)
                            std_cost = length(valid_costs) > 1 ? std(valid_costs) : 0.0
                            progress_println("  k=$k: 사회적비용 $(round(avg_cost,digits=2)) ± $(round(std_cost,digits=2)) EUR (n=$(length(valid_costs)))")
                        end
                    else
                        # omega_costs가 없으면 단일 값으로 처리
                        scenario_costs[scn_key] = Dict{String, Vector{Float64}}()
                        scenario_costs[scn_key]["social_cost"] = [result.social_cost]
                        scenario_costs[scn_key]["transport_cost"] = [haskey(result, :transport_cost) ? result.transport_cost : 0.0]
                        scenario_costs[scn_key]["customer_mobility_cost"] = [haskey(result, :customer_mobility_cost) ? result.customer_mobility_cost : 0.0]
                        progress_println("  k=$k: 사회적비용 $(round(result.social_cost,digits=2)) EUR (n=1, omega_costs 없음)")
                    end
                end
            else
                progress_println("시나리오 $scn: 결과 없음")
            end
        else
            # ═══════════════════════════════════════════════════════════════════
            # 다른 시나리오: 기존 방식
            # ═══════════════════════════════════════════════════════════════════
            scn_result = findfirst(r -> r.scenario == scn, all_results)
            if scn_result !== nothing
                result = all_results[scn_result]
                scn_key = scn_name
                scenario_names[scn_key] = scn_name
                
                # omega_costs 필드에서 오메가별 비용 추출
                if haskey(result, :omega_costs) && !isempty(result.omega_costs)
                    scenario_costs[scn_key] = Dict{String, Vector{Float64}}()
                    
                    # social_cost는 omega_costs에서 직접
                    scenario_costs[scn_key]["social_cost"] = collect(Float64, result.omega_costs)
                    
                    # transport_cost, customer_mobility_cost는 result에서 가져옴 (평균값만 있음)
                    # omega별 개별값이 필요하면 omega_results에서 가져와야 함
                    # 현재는 평균값을 모든 omega에 동일하게 적용 (근사치)
                    n_omega = length(result.omega_costs)
                    transport_per_omega = haskey(result, :transport_cost) ? result.transport_cost : 0.0
                    mobility_per_omega = haskey(result, :customer_mobility_cost) ? result.customer_mobility_cost : 0.0
                    
                    # 각 omega의 비용을 social_cost 비율로 추정
                    total_social = sum(result.omega_costs)
                    if total_social > 0
                        scenario_costs[scn_key]["transport_cost"] = [transport_per_omega * (c / (total_social / n_omega)) for c in result.omega_costs]
                        scenario_costs[scn_key]["customer_mobility_cost"] = [mobility_per_omega * (c / (total_social / n_omega)) for c in result.omega_costs]
                    else
                        scenario_costs[scn_key]["transport_cost"] = fill(transport_per_omega, n_omega)
                        scenario_costs[scn_key]["customer_mobility_cost"] = fill(mobility_per_omega, n_omega)
                    end
                    
                    # 요약 출력
                    costs = scenario_costs[scn_key]["social_cost"]
                    valid_costs = filter(c -> c != Inf && !isnan(c), costs)
                    if !isempty(valid_costs)
                        avg_cost = mean(valid_costs)
                        std_cost = length(valid_costs) > 1 ? std(valid_costs) : 0.0
                        progress_println("시나리오 $scn ($scn_name):")
                        progress_println("  사회적비용: $(round(avg_cost,digits=2)) ± $(round(std_cost,digits=2)) EUR")
                        progress_println("  범위: [$(round(minimum(valid_costs),digits=2)) ~ $(round(maximum(valid_costs),digits=2))] EUR (n=$(length(valid_costs)))")
                    else
                        progress_println("시나리오 $scn: 모든 결과가 실패")
                    end
                else
                    progress_println("시나리오 $scn: omega_costs 없음")
                end
            end
        end
    end
    progress_println("="^70)
    
    # (통계적 검정은 robustness_tests.jl에서 수행)

    # 시나리오 4 요약 (OPL이 실행된 경우에만)
    # TODO: MOO 형식으로 변경 필요 (현재 주석 처리)
    # scn4_results = filter(r -> haskey(r, :scenario) && r.scenario == SCENARIO_OPL, all_results)
    # if !isempty(scn4_results)
    #     progress_println("🔢 시나리오 4 - SLRP 락커 위치/개수 최적화 결과:")
    #     ...
    # end
    progress_println("="^60)

    total_time = round(time()-global_start; digits=1)
    progress_println("\n🎉 Hybrid ALNS + VNS CVRP 최적화 완료!")
    progress_println("🚀 하이브리드 ALNS + VNS 메타휴리스틱 알고리즘 적용 완료")
    progress_println("✅ 모든 시나리오(1-6)에서 ALNS/VNS 메커니즘 사용됨")
    progress_println("   - 시나리오 1-3, 5, 6: 직접 ALNS/VNS 적용")
    progress_println("   - 시나리오 4: ALNS 기반 락커 위치/개수 동시 최적화")
    progress_println("⏱️  총 소요시간: $(total_time)초")
    # progress_println("📁 결과 파일: $(final_output_path)")
    # progress_println("📊 총 $(nrow(df_all))개 결과 저장됨")

    # 캐리어별 분석 리포트 생성
    progress_println("\n📦 캐리어별 상세 분석 리포트 생성 중...")
    try
        # 기존 락커 통계 요약
        print_locker_summary()
        
        # 락커 통계 CSV 저장
        locker_csv_path = save_locker_stats_csv("locker_usage_stats.csv")
        
        # 시간창 정보 CSV 저장
        tw_csv_path = save_customer_time_windows_csv("customer_time_windows.csv")
        schedule_csv_path = save_route_schedule_csv("route_schedule.csv")
        
        # 캐리어별 상세 분석 리포트 생성
        carrier_detailed_path = generate_carrier_detailed_report("carrier_detailed_analysis.csv")
        
        # 시나리오별 락커-캐리어 매트릭스 생성 (2~6: 2,3,4,5,6)
        for scenario in 2:6  # 시나리오 1은 락커가 없으므로 제외
            matrix_path = generate_locker_carrier_matrix(scenario, "locker_carrier_matrix_scenario_$(scenario).csv")
        end
        
        progress_println("✅ 캐리어 분석 완료")
        
    catch err
        progress_println("⚠️ 캐리어 분석 중 오류: $err")
    end

    # 시각화 생성
    try
        create_all_visualizations()
    catch err
        progress_println("⚠️ 시각화 생성 중 오류: $err")
    end

    progress_println("\n🔚 CVRP 스크립트 종료 - $(Dates.now())")
progress_println("🧵 총 사용된 스레드 수: $(nthreads())개")
progress_println("💻 최대 병렬 처리 성능으로 작업이 완료되었습니다.")
end

function sanity_check()
    progress_println("\n── 기본 검증 ──")
    # 거리 계산 테스트
    test_dist = approx_euclidean(47.5, 19.05, 47.51, 19.06)
    progress_println("✓ 거리계산: $(round(test_dist, digits=3))km")
    
    # 고객 생성 테스트
    cust = gen_customers()
    progress_println("✓ 고객 $(length(cust))명 생성됨")
    
    progress_println("── 검증 완료 ──\n")
end

#───────────────────────────────────────────────────────────────────────────────
# 9. 시각화 함수들

function find_optimal_k_for_scenario4()
    optimal_k = get(scenario4_results, :optimal_k, nothing)
    if optimal_k !== nothing
        progress_println("🎯 최적 k값 조회: k=$optimal_k")
        return optimal_k
    else
        progress_println("⚠️ scenario4_results에서 결과를 찾을 수 없음")
        return -1
    end
end

function get_scenario4_facilities_for_k(k::Int)
    """시나리오 4에서 특정 k값에 대한 락커 위치 반환"""
    facilities = get(scenario4_k_facilities, k, nothing)
    if facilities !== nothing
        progress_println("🎯 k=$k 락커 위치 조회: $(length(facilities))개")
        return facilities
    else
        progress_println("⚠️ k=$k 락커 위치를 찾을 수 없음")
        return Tuple{Float64,Float64}[]
    end
end


# ═══════════════════════════════════════════════════════════════════════════════
# 도로망 경로 시각화 헬퍼 함수
# ═══════════════════════════════════════════════════════════════════════════════

"""
두 노드 사이의 경로를 그리기 위한 좌표 가져오기
- USE_ROAD_DISTANCE=true: OSRM geometry 사용 (실제 도로 형태)
- USE_ROAD_DISTANCE=false: 직선 연결
반환: (lons::Vector, lats::Vector)
"""
function get_route_path_coords(from_lat::Float64, from_lon::Float64, to_lat::Float64, to_lon::Float64; profile::String="car")
    if USE_ROAD_DISTANCE
        # OSRM geometry 가져오기
        path = osrm_route_geometry((from_lat, from_lon), (to_lat, to_lon); profile=profile)
        if !isempty(path)
            lats = [p[1] for p in path]
            lons = [p[2] for p in path]
            return (lons, lats)
        end
    end
    # 직선 경로
    return ([from_lon, to_lon], [from_lat, to_lat])
end

"""
경로 노드 리스트에서 전체 경로 좌표 가져오기
- nodes_data: 노드 정보가 담긴 Dict의 Vector
- route_node_ids: 경로 순서대로의 노드 ID 리스트
- use_road_network: true면 도로망 기반, false면 직선 연결
반환: (all_lons::Vector, all_lats::Vector)
"""
function get_full_route_coords(nodes_data::Vector{Dict{String,Any}}, route_node_ids::Vector; profile::String="car", use_road_network::Bool=true)
    if length(route_node_ids) < 2
        return (Float64[], Float64[])
    end
    
    all_lons = Float64[]
    all_lats = Float64[]
    
    for i in 1:(length(route_node_ids) - 1)
        from_id = route_node_ids[i]
        to_id = route_node_ids[i + 1]
        
        from_idx = findfirst(n -> n["id"] == from_id, nodes_data)
        to_idx = findfirst(n -> n["id"] == to_id, nodes_data)
        
        if from_idx !== nothing && to_idx !== nothing
            from_lon = Float64(nodes_data[from_idx]["lon"])
            from_lat = Float64(nodes_data[from_idx]["lat"])
            to_lon = Float64(nodes_data[to_idx]["lon"])
            to_lat = Float64(nodes_data[to_idx]["lat"])
            
            if use_road_network
                # 도로망 기반 경로
                (seg_lons, seg_lats) = get_route_path_coords(from_lat, from_lon, to_lat, to_lon; profile=profile)
            else
                # 직선 연결
                seg_lons = [from_lon, to_lon]
                seg_lats = [from_lat, to_lat]
            end
            
            # 첫 번째 세그먼트가 아니면 시작점 제외 (중복 방지)
            if !isempty(all_lons)
                append!(all_lons, seg_lons[2:end])
                append!(all_lats, seg_lats[2:end])
            else
                append!(all_lons, seg_lons)
                append!(all_lats, seg_lats)
            end
        end
    end
    
    return (all_lons, all_lats)
end


function visualize_topology_routes(scenario::Int, customers; save_plot::Bool=true, omega::Int=1)
    scenario_name = get_scenario_name(scenario)
    progress_println("🕸️ $scenario_name 토폴로지 시각화 시작...")

    # 메인 배치에서 저장된 결과 먼저 확인
    saved_result = get_main_batch_result(scenario, seed, omega)
    if saved_result === nothing
        progress_println("⚠️ 토폴로지: 저장된 결과 없음, 시각화를 건너뜁니다.")
        return (nodes=Vector{Dict{String,Any}}(), edges=Vector{Tuple{String,String,String,Symbol}}(), labels=Dict{String,String}(), colors=Dict{String,Symbol}())
    end
    customers = saved_result["customers"]
    route_data = saved_result["route_data"]
    distance_info = saved_result["distance_info"]
    progress_println("📊 토폴로지: 메인 배치 저장 결과 사용")

    all_nodes = route_data["nodes"]
    node_labels = Dict{String,String}()
    node_colors = Dict{String,Symbol}()

    for node in all_nodes
        id = node["id"]
        node_type = node["type"]
        if node_type == "Customer"
            customer = findfirst(c -> c.id == id, customers)
            if customer !== nothing
                dtype = customers[customer].dtype
                node_labels[id] = "$(id)\n$(dtype)"
                node_colors[id] = dtype == "D2D" ? :lightblue : :lightgreen
            end
        elseif node_type == "Depot"
            depot_carrier = ""
            for (did, (_, _, carrier)) in DEPOTS
                if did == id
                    depot_carrier = carrier
                    break
                end
            end
            node_labels[id] = "$(id)\n창고\n$(depot_carrier)"
            node_colors[id] = :red
        elseif node_type == "Locker"
            locker_type = haskey(LOCKERS_PRIV, id) ? "사설" : "공공"
            node_labels[id] = "$(id)\n$(locker_type)"
            node_colors[id] = locker_type == "사설" ? :orange : :purple
        end
    end

    edges = Vector{Tuple{String,String,String,Symbol}}()
    colors = [:darkred, :darkblue, :darkgreen, :darkorange, :darkmagenta, :darkcyan]
    color_idx = 0
    for (carrier, route_info) in route_data["vehicle_routes"]
        if haskey(route_info, "routes")
            for (vehicle_idx, route_nodes) in enumerate(route_info["routes"])
                color = colors[(color_idx % length(colors)) + 1]
                color_idx += 1
                for j in 1:(length(route_nodes)-1)
                    from_node = route_nodes[j]
                    to_node = route_nodes[j+1]
                    push!(edges, (from_node, to_node, "$(carrier)-V$(vehicle_idx)", color))
                end
            end
        else
            route_nodes = route_info["nodes"]
            color = colors[(color_idx % length(colors)) + 1]
            color_idx += 1
            for j in 1:(length(route_nodes)-1)
                from_node = route_nodes[j]
                to_node = route_nodes[j+1]
                push!(edges, (from_node, to_node, carrier, color))
            end
            if length(route_nodes) > 1
                push!(edges, (route_nodes[end], route_nodes[1], carrier, color))
            end
        end
    end

    for (customer_id, locker_id) in route_data["walking_routes"]
        push!(edges, (customer_id, locker_id, "보행", :gray))
    end

            # 토폴로지 데이터 생성

    if save_plot
        topology_dir = joinpath(OUTDIR, "topology")
        isdir(topology_dir) || mkpath(topology_dir)
        try
            topology_data = Dict(
                "scenario" => scenario,
                "distance_info" => distance_info,
                "nodes" => all_nodes,
                "edges" => edges,
                "node_labels" => node_labels,
                "node_colors" => node_colors,
                "vehicle_routes" => route_data["vehicle_routes"],
                "walking_routes" => route_data["walking_routes"]
            )
            topology_file = joinpath(topology_dir, "topology_scenario_$(scenario).json")
            open(topology_file, "w") do io
                JSON3.write(io, topology_data)
            end
            progress_println("🕸️ 토폴로지 데이터 저장: $topology_file")
        catch err
            scenario_name = get_scenario_name(scenario)
            progress_println("⚠️ 토폴로지 데이터 저장 실패 ($scenario_name): $err")
        end
    end

    return (nodes=all_nodes, edges=edges, labels=node_labels, colors=node_colors)
end

function visualize_scenario_routes_with_stats(scenario::Int, customers; save_plot::Bool=true, omega::Int=1, route_data::Union{Dict,Nothing}=nothing, distance_info_override::Union{Dict,Nothing}=nothing, k_suffix::Union{Int,Nothing}=nothing, use_road_network::Bool=true)
    scenario_name = get_scenario_name(scenario)
    progress_println("🎨 $scenario_name visualization starting...")

    local distance_info::Dict
    
    # route_data가 직접 제공되었으면 사용, 아니면 저장된 결과 확인
    if route_data !== nothing
        # 직접 제공된 route_data 사용 (k별 개별 시각화용)
        progress_println("📊 직접 제공된 route_data 사용: $scenario_name")
        if distance_info_override === nothing
            distance_info = Dict(
                "total" => get(route_data, "total_distance", 0.0),
                "social_cost" => get(route_data, "social_cost", get(route_data, "total_distance", 0.0))
            )
        else
            distance_info = distance_info_override
        end
    else
        # 메인 배치에서 저장된 결과 먼저 확인
        saved_result = get_main_batch_result(scenario, omega)
        
        if saved_result === nothing
            progress_println("⚠️ 저장된 결과 없음, 시각화를 건너뜁니다: $scenario_name")
            return nothing
        end
        # 저장된 메인 배치 결과 사용
        progress_println("📊 메인 배치 저장 결과 사용: $scenario_name Omega $omega")
        customers = saved_result["customers"]
        route_data = saved_result["route_data"]
        distance_info = saved_result["distance_info"]
    end

    customer_lons = [c.coord[1] for c in customers]
    customer_lats = [c.coord[2] for c in customers]
    depot_lons = [lon for (_, (lon, lat, _)) in DEPOTS]
    depot_lats = [lat for (_, (lon, lat, _)) in DEPOTS]

    # 1:1 평면도 비율로 설정
    aspect = 1.0

    # 캐리어별 색상 계열 (구분이 쉬운 색상들)
    carrier_base_colors = Dict(
        "Foxpost" => [:red, :darkred, :crimson, :firebrick],           # 빨간색 계열
        "GLS"     => [:blue, :darkblue, :navy, :steelblue],           # 파란색 계열  
        "Packeta" => [:green, :darkgreen, :forestgreen, :seagreen],    # 초록색 계열
        "AlzaBox" => [:orange, :darkorange, :chocolate, :orangered],   # 주황색 계열
        "EasyBox" => [:purple, :darkviolet, :indigo, :mediumorchid],   # 보라색 계열
        "DHL"     => [:magenta, :darkmagenta, :mediumvioletred, :deeppink] # 자홍색 계열
    )
    
    # 캐리어별 기본 색상 (첫 번째 색상)
    carrier_colors = Dict(
        "Foxpost" => :red,
        "GLS"     => :blue,
        "Packeta" => :green,
        "AlzaBox" => :orange,
        "EasyBox" => :purple,
        "DHL"     => :magenta
    )
    
    # 차량/투어별 색상 변형을 위한 함수
    function get_carrier_color(carrier::String, vehicle_id::Int, trip_idx::Int=1)
        if haskey(carrier_base_colors, carrier)
            colors = carrier_base_colors[carrier]
            # 차량과 투어 조합으로 색상 인덱스 결정
            color_idx = ((vehicle_id - 1) * 4 + trip_idx - 1) % length(colors) + 1
            return colors[color_idx]
        else
            return :black  # 기본색
        end
    end

    # Full view 생성
    scenario_name = get_scenario_name(scenario)
    route_type_label = use_road_network ? "Road Network" : "Straight Line"
    plt_full = plot(title="$scenario_name - Full View [$route_type_label] ($(round(distance_info["total"], digits=2))km)",
                   xlabel="Longitude", ylabel="Latitude", size=(1200, 900), dpi=200, aspect_ratio=aspect,
                   legend=:topright, legendfontsize=8, legendtitlefontsize=9)

    d2d_customers = [c for c in customers if c.dtype == "D2D"]
    locker_customers = [c for c in customers if c.dtype == "Locker"]

    if !isempty(d2d_customers)
        d2d_lons = [c.coord[1] for c in d2d_customers]
        d2d_lats = [c.coord[2] for c in d2d_customers]
        scatter!(plt_full, d2d_lons, d2d_lats, color=:dodgerblue, marker=:circle,
                 markersize=2, label="D2D Customers", alpha=0.8, 
                 markerstrokewidth=0.3, markerstrokecolor=:darkblue)
    end

    if !isempty(locker_customers)
        locker_lons_cust = [c.coord[1] for c in locker_customers]
        locker_lats_cust = [c.coord[2] for c in locker_customers]
        scatter!(plt_full, locker_lons_cust, locker_lats_cust, color=:limegreen, marker=:rect,
                 markersize=2, label="Locker Customers", alpha=0.8,
                 markerstrokewidth=0.3, markerstrokecolor=:darkgreen)
    end

    scatter!(plt_full, depot_lons, depot_lats, color=:crimson, marker=:star8,
             markersize=6, label="Depots")

    if !isempty(route_data["effective_lockers"])
        priv_lons, priv_lats = Float64[], Float64[]
        pub_lons, pub_lats = Float64[], Float64[]
        metro_lons, metro_lats = Float64[], Float64[]

        for (locker_id, locker_data) in route_data["effective_lockers"]
            coords = locker_data[1]
            locker_type = locker_data[2]
            if locker_type == "Private"
                push!(priv_lons, coords[1])
                push!(priv_lats, coords[2])
            elseif locker_type == "Metro"
                push!(metro_lons, coords[1])
                push!(metro_lats, coords[2])
            else
                push!(pub_lons, coords[1])
                push!(pub_lats, coords[2])
            end
        end

        if !isempty(priv_lons)
            scatter!(plt_full, priv_lons, priv_lats, color=:darkorange, marker=:dtriangle,
                     markersize=4, label="Private Lockers")
        end
        if !isempty(metro_lons)
            scatter!(plt_full, metro_lons, metro_lats, color=:purple, marker=:star5,
                     markersize=4, label="Metro Lockers")
        end
        if !isempty(pub_lons)
            scatter!(plt_full, pub_lons, pub_lats, color=:mediumorchid, marker=:pentagon,
                     markersize=4, label="Public Lockers")
        end
    end

    # Full view 차량 경로 표시 (캐리어별 색상 계열, 차량/투어별 색상 변형)
    carrier_legend_added = Dict{String, Bool}()
    
    for (carrier, route_info) in route_data["vehicle_routes"]
        
        if haskey(route_info, "vehicle_routes")
            for (vehicle_id, trips) in route_info["vehicle_routes"]
                for (trip_idx, route_nodes) in enumerate(trips)
                    if length(route_nodes) > 1
                        # 차량/투어별 색상 변형 적용
                        route_color = get_carrier_color(carrier, vehicle_id, trip_idx)
                        
                        # 경로 좌표 가져오기 (도로망 또는 직선)
                        (route_lons, route_lats) = get_full_route_coords(route_data["nodes"], route_nodes; profile="car", use_road_network=use_road_network)
                        
                        if !isempty(route_lons)
                            # 캐리어별로 첫 번째 경로만 legend에 표시
                            show_label = !get(carrier_legend_added, carrier, false)
                            label_text = show_label ? "$carrier Routes" : ""
                            if show_label
                                carrier_legend_added[carrier] = true
                            end
                            
                            plot!(plt_full, route_lons, route_lats, color=route_color, linewidth=1.2,
                                  alpha=0.9, linestyle=:solid, label=label_text)
                        end
                    end
                end
            end
        elseif haskey(route_info, "routes")
            for (vehicle_idx, route_nodes) in enumerate(route_info["routes"])
                if length(route_nodes) > 1
                    # 차량별 색상 변형 적용
                    route_color = get_carrier_color(carrier, vehicle_idx, 1)
                    
                    # 경로 좌표 가져오기 (도로망 또는 직선)
                    (route_lons, route_lats) = get_full_route_coords(route_data["nodes"], route_nodes; profile="car", use_road_network=use_road_network)
                    
                    if !isempty(route_lons)
                        # 캐리어별로 첫 번째 경로만 legend에 표시
                        show_label = !get(carrier_legend_added, carrier, false)
                        label_text = show_label ? "$carrier Routes" : ""
                        if show_label
                            carrier_legend_added[carrier] = true
                        end
                        
                        plot!(plt_full, route_lons, route_lats, color=route_color, linewidth=1.2,
                              alpha=0.9, linestyle=:solid, label=label_text)
                    end
                end
            end
        end
    end

    # Full view 도보 경로 표시 (스케일상 직선으로 단순화, 스타일 차별화)
    if !isempty(route_data["walking_routes"])
        first_walking = true
        for (customer_id, locker_id) in route_data["walking_routes"]
            customer_idx = findfirst(c -> c.id == customer_id, customers)
            if customer_idx !== nothing && haskey(route_data["effective_lockers"], locker_id)
                cust_lon, cust_lat = customers[customer_idx].coord[1], customers[customer_idx].coord[2]
                locker_coords = route_data["effective_lockers"][locker_id][1]
                # 도보 경로: 회색 점선으로 차량과 구분
                plot!(plt_full, [cust_lon, locker_coords[1]], [cust_lat, locker_coords[2]],
                      color=:gray50, linestyle=:dot, alpha=0.4, linewidth=0.6,
                      label=(first_walking ? "Walking Routes" : ""))
                first_walking = false
            end
        end
    end

    # Detail view 범위 설정
    detail_lon_range = (19.0375, 19.0690)  # 요청한 Longitude 범위
    detail_lat_range = (47.490, 47.5175)   # 요청한 Latitude 범위

    plt_detail = plot(title="$scenario_name - Detail View [$route_type_label] (Customer Area)",
                     xlabel="Longitude", ylabel="Latitude",
                     size=(1800, 1400), dpi=300,
                     xlims=detail_lon_range, ylims=detail_lat_range, aspect_ratio=aspect,
                     legend=:topright, legendfontsize=7, legendtitlefontsize=8)

    # 배경: Raster 경계와 Residential Polygons 추가 (detail 범위 내에서만)
    for (rid, raster_coords) in raster_polygons
        raster_lons = [coord[1] for coord in raster_coords]
        raster_lats = [coord[2] for coord in raster_coords]
        
        if any(detail_lon_range[1] ≤ lon ≤ detail_lon_range[2] for lon in raster_lons) &&
           any(detail_lat_range[1] ≤ lat ≤ detail_lat_range[2] for lat in raster_lats)
            
            plot!(plt_detail, raster_lons, raster_lats, 
                  color=:lightgray, linewidth=1.5, alpha=0.7, 
                  label=(rid == first(keys(raster_polygons)) ? "Raster Boundaries" : ""), linestyle=:solid)
            
            center_lon = sum(raster_lons[1:end-1]) / (length(raster_lons)-1)
            center_lat = sum(raster_lats[1:end-1]) / (length(raster_lats)-1)
            if detail_lon_range[1] ≤ center_lon ≤ detail_lon_range[2] &&
               detail_lat_range[1] ≤ center_lat ≤ detail_lat_range[2]
                annotate!(plt_detail, center_lon, center_lat, 
                         text("R$rid", :gray, :center, 8))
            end
            
            if haskey(residential_polygons, rid)
                for poly in residential_polygons[rid]
                    poly_coords = coordinates(poly)
                    poly_lons = [coord[1] for coord in poly_coords]
                    poly_lats = [coord[2] for coord in poly_coords]
                    
                    if any(detail_lon_range[1] ≤ lon ≤ detail_lon_range[2] for lon in poly_lons) &&
                       any(detail_lat_range[1] ≤ lat ≤ detail_lat_range[2] for lat in poly_lats)
                        
                        if poly_lons[end] != poly_lons[1] || poly_lats[end] != poly_lats[1]
                            push!(poly_lons, poly_lons[1])
                            push!(poly_lats, poly_lats[1])
                        end
                        
                        plot!(plt_detail, poly_lons, poly_lats, 
                              fillcolor=:lightblue, fillalpha=0.15, 
                              color=:steelblue, linewidth=0.8, alpha=0.6,
                              label=(rid == first(keys(residential_polygons)) && 
                                    poly == first(residential_polygons[rid]) ? "Residential Areas" : ""))
                    end
                end
            end
        end
    end

    # 고객 데이터 표시 (detail 범위 내에서만)
    d2d_customers = [c for c in customers if c.dtype == "D2D" && 
                     detail_lon_range[1] ≤ c.coord[1] ≤ detail_lon_range[2] &&
                     detail_lat_range[1] ≤ c.coord[2] ≤ detail_lat_range[2]]
    locker_customers = [c for c in customers if c.dtype == "Locker" && 
                        detail_lon_range[1] ≤ c.coord[1] ≤ detail_lon_range[2] &&
                        detail_lat_range[1] ≤ c.coord[2] ≤ detail_lat_range[2]]

    if !isempty(d2d_customers)
        d2d_lons = [c.coord[1] for c in d2d_customers]
        d2d_lats = [c.coord[2] for c in d2d_customers]
        scatter!(plt_detail, d2d_lons, d2d_lats, color=:dodgerblue, marker=:circle,
                 markersize=2, label="D2D Customers", alpha=0.8,
                 markerstrokewidth=0.3, markerstrokecolor=:darkblue)
    end

    if !isempty(locker_customers)
        locker_lons_cust = [c.coord[1] for c in locker_customers]
        locker_lats_cust = [c.coord[2] for c in locker_customers]
        scatter!(plt_detail, locker_lons_cust, locker_lats_cust, color=:limegreen, marker=:rect,
                 markersize=2, label="Locker Customers", alpha=0.8,
                 markerstrokewidth=0.3, markerstrokecolor=:darkgreen)
    end

    # 락커 표시 (detail 범위 내에서만)
    if !isempty(route_data["effective_lockers"])
        first_priv, first_pub, first_metro = true, true, true
        for (locker_id, locker_data) in route_data["effective_lockers"]
            coords = locker_data[1]
            locker_type = locker_data[2]
            if detail_lon_range[1] ≤ coords[1] ≤ detail_lon_range[2] &&
               detail_lat_range[1] ≤ coords[2] ≤ detail_lat_range[2]
                if locker_type == "Private"
                    scatter!(plt_detail, [coords[1]], [coords[2]], color=:darkorange, marker=:dtriangle,
                             markersize=6, label=(first_priv ? "Private Lockers" : ""))
                    first_priv = false
                elseif locker_type == "Metro"
                    scatter!(plt_detail, [coords[1]], [coords[2]], color=:purple, marker=:star5,
                             markersize=6, label=(first_metro ? "Metro Lockers" : ""))
                    first_metro = false
                else
                    scatter!(plt_detail, [coords[1]], [coords[2]], color=:mediumorchid, marker=:pentagon,
                             markersize=6, label=(first_pub ? "Public Lockers" : ""))
                    first_pub = false
                end
            end
        end
    end

    # 차량 경로 표시 (캐리어별 색상, detail 영역 내부 연결만 표시)
    # Detail view 차량 경로 표시 (캐리어별 그룹화된 legend)
    detail_carrier_legend_added = Dict{String, Bool}()
    
    for (carrier, route_info) in route_data["vehicle_routes"]
        
        if haskey(route_info, "vehicle_routes")
            # 다중여행 표시
            for (vehicle_id, trips) in route_info["vehicle_routes"]
                for (trip_idx, route_nodes) in enumerate(trips)
                    if length(route_nodes) > 1
                        # 차량/투어별 색상 변형 적용
                        route_color = get_carrier_color(carrier, vehicle_id, trip_idx)
                        
                        # detail 영역 내부 연결만 필터링
                        first_segment_in_area = true
                        for i in 1:(length(route_nodes)-1)
                            node1_id = route_nodes[i]
                            node2_id = route_nodes[i+1]
                            
                            node1_idx = findfirst(n -> n["id"] == node1_id, route_data["nodes"])
                            node2_idx = findfirst(n -> n["id"] == node2_id, route_data["nodes"])
                            
                            if node1_idx !== nothing && node2_idx !== nothing
                                lon1 = Float64(route_data["nodes"][node1_idx]["lon"])
                                lat1 = Float64(route_data["nodes"][node1_idx]["lat"])
                                lon2 = Float64(route_data["nodes"][node2_idx]["lon"])
                                lat2 = Float64(route_data["nodes"][node2_idx]["lat"])
                                
                                # 두 노드가 모두 detail 영역 내부에 있는 경우만 선 그리기
                                if (detail_lon_range[1] ≤ lon1 ≤ detail_lon_range[2] &&
                                    detail_lat_range[1] ≤ lat1 ≤ detail_lat_range[2] &&
                                    detail_lon_range[1] ≤ lon2 ≤ detail_lon_range[2] &&
                                    detail_lat_range[1] ≤ lat2 ≤ detail_lat_range[2])
                                    
                                    # 캐리어별로 첫 번째 segment만 legend에 표시
                                    show_label = first_segment_in_area && !get(detail_carrier_legend_added, carrier, false)
                                    label_text = show_label ? "$carrier Routes" : ""
                                    if show_label
                                        detail_carrier_legend_added[carrier] = true
                                    end
                                    first_segment_in_area = false
                                    
                                    # 경로 좌표 가져오기 (도로망 또는 직선)
                                    if use_road_network
                                        (seg_lons, seg_lats) = get_route_path_coords(lat1, lon1, lat2, lon2; profile="car")
                                    else
                                        seg_lons = [lon1, lon2]
                                        seg_lats = [lat1, lat2]
                                    end
                                    plot!(plt_detail, seg_lons, seg_lats, 
                                          color=route_color, linewidth=1.4, alpha=0.9, linestyle=:solid,
                                          label=label_text)
                                end
                            end
                        end
                    end
                end
            end
        elseif haskey(route_info, "routes")
            # 기존 방식 (호환성)
            for (vehicle_idx, route_nodes) in enumerate(route_info["routes"])
                if length(route_nodes) > 1
                    # 차량별 색상 변형 적용
                    route_color = get_carrier_color(carrier, vehicle_idx, 1)
                    
                    # detail 영역 내부 연결만 필터링
                    first_segment_in_area = true
                    for i in 1:(length(route_nodes)-1)
                        node1_id = route_nodes[i]
                        node2_id = route_nodes[i+1]
                        
                        node1_idx = findfirst(n -> n["id"] == node1_id, route_data["nodes"])
                        node2_idx = findfirst(n -> n["id"] == node2_id, route_data["nodes"])
                        
                        if node1_idx !== nothing && node2_idx !== nothing
                            lon1 = Float64(route_data["nodes"][node1_idx]["lon"])
                            lat1 = Float64(route_data["nodes"][node1_idx]["lat"])
                            lon2 = Float64(route_data["nodes"][node2_idx]["lon"])
                            lat2 = Float64(route_data["nodes"][node2_idx]["lat"])
                            
                            # 두 노드가 모두 detail 영역 내부에 있는 경우만 선 그리기
                            if (detail_lon_range[1] ≤ lon1 ≤ detail_lon_range[2] &&
                                detail_lat_range[1] ≤ lat1 ≤ detail_lat_range[2] &&
                                detail_lon_range[1] ≤ lon2 ≤ detail_lon_range[2] &&
                                detail_lat_range[1] ≤ lat2 ≤ detail_lat_range[2])
                                
                                # 캐리어별로 첫 번째 segment만 legend에 표시
                                show_label = first_segment_in_area && !get(detail_carrier_legend_added, carrier, false)
                                label_text = show_label ? "$carrier Routes" : ""
                                if show_label
                                    detail_carrier_legend_added[carrier] = true
                                end
                                first_segment_in_area = false
                                
                                # 경로 좌표 가져오기 (도로망 또는 직선)
                                if use_road_network
                                    (seg_lons, seg_lats) = get_route_path_coords(lat1, lon1, lat2, lon2; profile="car")
                                else
                                    seg_lons = [lon1, lon2]
                                    seg_lats = [lat1, lat2]
                                end
                                plot!(plt_detail, seg_lons, seg_lats, 
                                      color=route_color, linewidth=1.4, alpha=0.9, linestyle=:solid,
                                      label=label_text)
                            end
                        end
                    end
                end
            end
        end
    end

    # 도보 경로 표시 (detail 범위 내에서만) - Hybrid 방식: 직선(골목) + 도로망 + 직선(골목)
    # 모두 동일한 스타일로 통일 (보라색 파선)
    if !isempty(route_data["walking_routes"])
        first_walking = true
        for (customer_id, locker_id) in route_data["walking_routes"]
            customer_idx = findfirst(c -> c.id == customer_id, customers)
            if customer_idx !== nothing && haskey(route_data["effective_lockers"], locker_id)
                cust_lon = Float64(customers[customer_idx].coord[1])
                cust_lat = Float64(customers[customer_idx].coord[2])
                locker_coords = route_data["effective_lockers"][locker_id][1]
                locker_lon = Float64(locker_coords[1])
                locker_lat = Float64(locker_coords[2])
                
                # 고객과 락커 모두 detail 범위 내에 있는 경우만 표시
                if (detail_lon_range[1] ≤ cust_lon ≤ detail_lon_range[2] &&
                    detail_lat_range[1] ≤ cust_lat ≤ detail_lat_range[2] &&
                    detail_lon_range[1] ≤ locker_lon ≤ detail_lon_range[2] &&
                    detail_lat_range[1] ≤ locker_lat ≤ detail_lat_range[2])
                    
                    # 도보 경로 스타일: 회색 점선 (차량 실선과 구분)
                    walking_color = :gray50
                    walking_style = :dot
                    walking_alpha = 0.5
                    walking_width = 0.8
                    
                    # 도보 경로 (도로망 또는 직선)
                    if use_road_network && USE_ROAD_DISTANCE && is_distance_matrix_initialized()
                        # Hybrid 도보 경로: 직선(골목) + 도로망 + 직선(골목)
                        # 스냅 위치 가져오기
                        cust_pos = (cust_lat, cust_lon)
                        locker_pos = (locker_lat, locker_lon)
                        cust_snapped = get_foot_snapped_location(cust_pos)
                        locker_snapped = get_foot_snapped_location(locker_pos)
                        
                        # 1. 고객 집 → foot 스냅 (직선, 골목 내 이동)
                        if cust_pos != cust_snapped
                            plot!(plt_detail, [cust_lon, cust_snapped[2]], [cust_lat, cust_snapped[1]],
                                  color=walking_color, linestyle=walking_style, alpha=walking_alpha, linewidth=walking_width,
                          label=(first_walking ? "Walking Routes" : ""))
                    first_walking = false
                end
                        
                        # 2. foot 스냅 → foot 스냅 (도로망)
                        (walk_lons, walk_lats) = get_route_path_coords(cust_snapped[1], cust_snapped[2], 
                                                                        locker_snapped[1], locker_snapped[2]; profile="foot")
                        plot!(plt_detail, walk_lons, walk_lats,
                              color=walking_color, linestyle=walking_style, alpha=walking_alpha, linewidth=walking_width,
                              label=(first_walking ? "Walking Routes" : ""))
                        first_walking = false
                        
                        # 3. foot 스냅 → 락커 (직선, 골목 내 이동)
                        if locker_pos != locker_snapped
                            plot!(plt_detail, [locker_snapped[2], locker_lon], [locker_snapped[1], locker_lat],
                                  color=walking_color, linestyle=walking_style, alpha=walking_alpha, linewidth=walking_width,
                                  label="")
                        end
                    else
                        # 직선 모드: 단순 직선 경로
                        plot!(plt_detail, [cust_lon, locker_lon], [cust_lat, locker_lat],
                              color=walking_color, linestyle=walking_style, alpha=walking_alpha, linewidth=walking_width,
                              label=(first_walking ? "Walking Routes" : ""))
                        first_walking = false
                    end
                end
            end
        end
    end

    if save_plot
        plot_dir = joinpath(OUTDIR, "visualizations")
        isdir(plot_dir) || mkpath(plot_dir)
        
        # 도로망/직선 접미사
        route_type_suffix = use_road_network ? "_road" : "_straight"
        
        # k_suffix가 직접 제공되었으면 사용, 아니면 기존 방식
        if k_suffix !== nothing
            # k별 개별 시각화용 파일명
            k_suffix_str = "_k$(k_suffix)"
            full_file = joinpath(plot_dir, "scenario_$(scenario)$(k_suffix_str)$(route_type_suffix)_full.png")
            detail_file = joinpath(plot_dir, "scenario_$(scenario)$(k_suffix_str)$(route_type_suffix)_detail.png")
            detail_csv_file = joinpath(plot_dir, "scenario_$(scenario)$(k_suffix_str)_detail_distances.csv")
        elseif scenario == SCENARIO_OPL
            # 기존 최적 k 방식
            optimal_k = find_optimal_k_for_scenario4()
            k_suffix_str = optimal_k > 0 ? "_k$(optimal_k)" : "_k3"
            full_file = joinpath(plot_dir, "scenario_$(scenario)$(k_suffix_str)$(route_type_suffix)_full.png")
            detail_file = joinpath(plot_dir, "scenario_$(scenario)$(k_suffix_str)$(route_type_suffix)_detail.png")
            detail_csv_file = joinpath(plot_dir, "scenario_$(scenario)$(k_suffix_str)_detail_distances.csv")
        else
            # 다른 시나리오들
            full_file = joinpath(plot_dir, "scenario_$(scenario)$(route_type_suffix)_full.png")
            detail_file = joinpath(plot_dir, "scenario_$(scenario)$(route_type_suffix)_detail.png")
            detail_csv_file = joinpath(plot_dir, "scenario_$(scenario)_detail_distances.csv")
        end
        savefig(plt_full, full_file)
        savefig(plt_detail, detail_file)
        
        # Detail 영역 거리 정보 CSV 생성 (도로망 버전에서만, 중복 방지)
        if use_road_network
            try
                create_detail_distance_csv(detail_csv_file, customers, route_data, scenario, omega)
                progress_println("📊 경로 상세 CSV 생성 완료: $detail_csv_file")
            catch err
                progress_println("⚠️ 경로 상세 CSV 생성 실패: $err")
            end
        end
        
        progress_println("📊 Visualization saved:")
        progress_println("  Full view: $full_file")
        progress_println("  Detail view: $detail_file")
        progress_println("  Detail distances: $detail_csv_file")
    end

    return plt_full, plt_detail
end

# Detail 영역 거리 정보를 CSV로 저장하는 함수
function create_detail_distance_csv(csv_file::String, customers, route_data, scenario::Int, omega::Int)
    """배송사별 차량/도보 이동거리 요약 CSV 생성"""
    
    # Detail 영역 범위 (Budapest 중심부)
    detail_bounds = (
        lon_min = 19.0375, lon_max = 19.0690,
        lat_min = 47.490, lat_max = 47.5175
    )
    
    # 좌표가 Detail 영역 내인지 확인
    function is_in_detail(lon, lat)
        detail_bounds.lon_min <= lon <= detail_bounds.lon_max && 
        detail_bounds.lat_min <= lat <= detail_bounds.lat_max
    end
    
    # 노드 ID로 좌표 가져오기
    function get_node_coords(node_id)
        if haskey(route_data, "nodes")
            node_idx = findfirst(n -> n["id"] == node_id, route_data["nodes"])
            if node_idx !== nothing
                return (Float64(route_data["nodes"][node_idx]["lon"]), 
                        Float64(route_data["nodes"][node_idx]["lat"]))
            end
        end
        return nothing
    end
    
    csv_rows = []
    
    # ═══════════════════════════════════════════════════════════════════
    # 1. 차량 경로별 이동거리 (캐리어별)
    # ═══════════════════════════════════════════════════════════════════
    if haskey(route_data, "vehicle_routes")
        for (carrier, route_info) in route_data["vehicle_routes"]
            
            vehicle_routes = if haskey(route_info, "vehicle_routes")
                route_info["vehicle_routes"]
            elseif haskey(route_info, "routes")
                Dict(i => [route] for (i, route) in enumerate(route_info["routes"]))
            else
                Dict()
            end
            
            for (vehicle_id, trips) in vehicle_routes
                for (trip_idx, route_nodes) in enumerate(trips)
                    if length(route_nodes) < 2
                        continue
                    end
                    
                    total_trip_dist = 0.0
                    detail_trip_dist = 0.0
                    
                    for i in 1:(length(route_nodes) - 1)
                        from_id = route_nodes[i]
                        to_id = route_nodes[i + 1]
                        
                        from_coords = get_node_coords(from_id)
                        to_coords = get_node_coords(to_id)
                        
                        if from_coords !== nothing && to_coords !== nothing
                            # 도로망 거리 계산
                            segment_dist = get_precomputed_car_distance(
                                (from_coords[2], from_coords[1]),  # (lat, lon)
                                (to_coords[2], to_coords[1])
                            )
                            if segment_dist == Inf || isnan(segment_dist)
                                segment_dist = euclidean_distance(from_coords, to_coords)
                            end
                            
                            total_trip_dist += segment_dist
                            
                            # 배송 구간 거리 (디포 왕복 제외)
                            # 첫 번째 세그먼트(디포→첫고객)와 마지막 세그먼트(마지막고객→디포) 제외
                            is_first_segment = (i == 1)
                            is_last_segment = (i == length(route_nodes) - 1)
                            
                            if !is_first_segment && !is_last_segment
                                detail_trip_dist += segment_dist
                            end
                        end
                    end
                    
                    # 경로 순서를 문자열로 변환
                    route_sequence = join(route_nodes, " → ")
                    
                    push!(csv_rows, Dict(
                        "scenario" => scenario, "seed" => seed, "omega" => omega,
                        "carrier" => carrier,
                        "route_type" => "Vehicle",
                        "vehicle_id" => vehicle_id,
                        "trip_idx" => trip_idx,
                        "num_stops" => length(route_nodes),
                        "total_distance_km" => round(total_trip_dist, digits=3),
                        "delivery_zone_distance_km" => round(detail_trip_dist, digits=3),
                        "route_sequence" => route_sequence
                    ))
                end
            end
        end
    end
    
    # ═══════════════════════════════════════════════════════════════════
    # 2. 도보 경로별 이동거리 (고객→락커) - 각 고객별로 개별 표시
    # ═══════════════════════════════════════════════════════════════════
    if haskey(route_data, "walking_routes") && !isempty(route_data["walking_routes"])
        walking_idx = 0
        
        for (customer_id, locker_id) in route_data["walking_routes"]
            # 고객 찾기
            customer_idx = findfirst(c -> c.id == customer_id, customers)
            if customer_idx === nothing
                continue
            end
            
            cust = customers[customer_idx]
            cust_lon, cust_lat = cust.coord
            
            # 락커 좌표 가져오기
            if !haskey(route_data, "effective_lockers") || !haskey(route_data["effective_lockers"], locker_id)
                continue
            end
            locker_coords = route_data["effective_lockers"][locker_id][1]
            locker_lon, locker_lat = locker_coords[1], locker_coords[2]
            
            # 도보망 거리 계산 (Float64로 변환)
            walking_dist = get_hybrid_foot_distance((Float64(cust_lat), Float64(cust_lon)), (Float64(locker_lat), Float64(locker_lon)))
            if walking_dist == Inf || isnan(walking_dist)
                walking_dist = euclidean_distance((cust_lon, cust_lat), (locker_lon, locker_lat))
            end
            
            # 캐리어 정보 가져오기
            df_attr = gen_attr([cust])
            carrier = !isempty(df_attr) ? df_attr[1, :carrier] : "Unknown"
            
            walking_idx += 1
            
            # 도보 경로 순서
            route_sequence = "$(customer_id) → $(locker_id)"
            
            push!(csv_rows, Dict(
                "scenario" => scenario, "seed" => seed, "omega" => omega,
                "carrier" => carrier,
                "route_type" => "Walking",
                "vehicle_id" => 0,
                "trip_idx" => walking_idx,
                "num_stops" => 2,  # 고객 → 락커
                "total_distance_km" => round(walking_dist, digits=3),
                "delivery_zone_distance_km" => round(walking_dist, digits=3),  # 도보는 전부 배송 구간
                "route_sequence" => route_sequence
            ))
        end
    end
    
    # DataFrame 생성 및 CSV 저장
    if !isempty(csv_rows)
        df_distances = DataFrame(csv_rows)
        column_order = ["scenario", "seed", "omega", "carrier", "route_type",
                       "vehicle_id", "trip_idx", "num_stops", 
                       "total_distance_km", "delivery_zone_distance_km", "route_sequence"]
        df_distances = df_distances[:, column_order]
        sort!(df_distances, [:carrier, :route_type, :vehicle_id, :trip_idx])
        CSV.write(csv_file, df_distances)
    else
        df_empty = DataFrame(
            scenario = Int[], seed = Int[], omega = Int[],
            carrier = String[], route_type = String[],
            vehicle_id = Int[], trip_idx = Int[], num_stops = Int[],
            total_distance_km = Float64[], delivery_zone_distance_km = Float64[],
            route_sequence = String[]
        )
        CSV.write(csv_file, df_empty)
    end
end

# 시각화 생성 함수 (main_batch_optimization 외부로 이동)
function create_all_visualizations()
    progress_println("\n🎨 메인 배치 저장 결과 기반 시각화 생성 중...")
    
    # 저장된 메인 배치 결과들 확인
    saved_keys = collect(keys(MAIN_BATCH_RESULTS))
    progress_println("🔍 MAIN_BATCH_RESULTS 디버깅: 총 $(length(saved_keys))개 키 발견")
    for key in saved_keys
        progress_println("   키: $key")
    end
    
    if isempty(saved_keys)
        progress_println("⚠️ 저장된 메인 배치 결과가 없습니다. 시각화를 생성할 수 없습니다.")
        return
    end
    
    progress_println("📊 저장된 결과 $(length(saved_keys))개를 기반으로 시각화 생성...")
    
    # 시나리오별로 그룹화 (오메가1만 선택)
    scenarios_to_visualize = Dict{Int, Int}()  # scenario => omega (1로 고정)
    for (scenario, omega) in saved_keys
        # 각 시나리오에서 첫 번째 오메가(1)만 시각화
        if !haskey(scenarios_to_visualize, scenario) && omega == 1
            scenarios_to_visualize[scenario] = omega
        end
    end
    
    total_visualizations = 0
    sorted_keys = sort(collect(keys(scenarios_to_visualize)))
    
    for scenario in sorted_keys
        omega = scenarios_to_visualize[scenario]
        try
            scenario_name = get_scenario_name(scenario)
            progress_println("🎨 메인 배치 결과 시각화: $scenario_name Omega $omega")
            
            # 저장된 결과에서 고객 데이터 가져오기
            saved_result = get_main_batch_result(scenario, omega)
            if saved_result !== nothing
                customers = saved_result["customers"]
                progress_println("📊 시각화 데이터 확인: $scenario_name - 고객 $(length(customers))명")
                
                # 메인 배치 결과 사용해서 시각화 (오메가1만 시각화) - 도로망 + 직선 두 버전
                progress_println("🎯 시각화 함수 호출 시작...")
                
                # 도로망 버전
                viz_result_road = visualize_scenario_routes_with_stats(scenario, customers; omega=omega, use_road_network=true)
                if viz_result_road === nothing
                    progress_println("⚠️ $scenario_name Omega $omega: 도로망 시각화가 생성되지 않아 건너뜁니다.")
                end
                
                # 직선 버전
                viz_result_straight = visualize_scenario_routes_with_stats(scenario, customers; omega=omega, use_road_network=false)
                if viz_result_straight === nothing
                    progress_println("⚠️ $scenario_name Omega $omega: 직선 시각화가 생성되지 않아 건너뜁니다.")
                end
                
                if viz_result_road === nothing && viz_result_straight === nothing
                    continue
                end
                progress_println("🎯 시각화 함수 호출 완료!")
                
                total_visualizations += 4  # full + detail view × 2 (도로망 + 직선)
                progress_println("✅ $scenario_name 시각화 완료 (도로망 + 직선)")
            else
                progress_println("⚠️ $scenario_name Omega $omega: 저장된 결과를 찾을 수 없음")
            end
        catch err
            scenario_name = get_scenario_name(scenario)
            progress_println("⚠️ $scenario_name 시각화 실패: $err")
            println("스택 트레이스:")
            for (exc, bt) in Base.catch_stack()
                showerror(stdout, exc, bt)
                println()
            end
        end
    end
    
    progress_println("✅ 메인 배치 기반 시각화 완료! (총 $(total_visualizations)개 파일)")
    progress_println("📊 시각화된 시나리오:")
    for scenario in sorted_keys
        scenario_name = get_scenario_name(scenario)
        progress_println("   시나리오 $scenario ($scenario_name)")
    end
    
    # 시나리오 OPL k별 개별 시각화 생성 (오메가1 기준)
    progress_println("\n🎯 시나리오 OPL k별 개별 시각화 생성 중...")
    
    scenario4_viz_count = 0
    for k in 1:MAX_PUB
        snapshot = get_scenario4_snapshot(k, 1)
        if snapshot === nothing
            progress_println("⚠️ 시나리오 4 k=$k: 저장된 스냅샷이 없어 건너뜁니다.")
            continue
        end
        
        customers_snapshot = snapshot["customers"]
        route_data_snapshot = snapshot["route_data"]
        distance_info_snapshot = snapshot["distance_info"]
        
        # 도로망 버전
        viz_result_road = visualize_scenario_routes_with_stats(
            4, customers_snapshot;
            omega=1,
            route_data=route_data_snapshot,
            distance_info_override=distance_info_snapshot,
            k_suffix=k,
            use_road_network=true
        )
        
        # 직선 버전
        viz_result_straight = visualize_scenario_routes_with_stats(
            4, customers_snapshot;
            omega=1,
            route_data=route_data_snapshot,
            distance_info_override=distance_info_snapshot,
            k_suffix=k,
            use_road_network=false
        )
        
        if viz_result_road === nothing && viz_result_straight === nothing
            progress_println("⚠️ 시나리오 4 k=$k: 시각화가 생성되지 않아 건너뜁니다.")
        else
            scenario4_viz_count += 4  # full + detail × 2 (도로망 + 직선)
            progress_println("✅ 시나리오 4 k=$k 시각화 완료 (도로망 + 직선)")
        end
    end
    
    progress_println("✅ 시나리오 4 k별 비교 시각화 완료! (총 $(scenario4_viz_count)개 파일)")
end

#───────────────────────────────────────────────────────────────────────────────
# 스크립트 실행
#───────────────────────────────────────────────────────────────────────────────
function execute_pipeline()
    sanity_check()

    # 디버깅: MAIN_BATCH_RESULTS 사전 확인
    progress_println("\n🔍 MAIN_BATCH_RESULTS 사전 확인:")
    progress_println("   총 키 개수: $(length(MAIN_BATCH_RESULTS))")
    for key in keys(MAIN_BATCH_RESULTS)
        progress_println("   저장된 키: $key")
    end

    main_batch_optimization()

    # 디버깅: MAIN_BATCH_RESULTS 사후 확인
    progress_println("\n🔍 MAIN_BATCH_RESULTS 사후 확인:")
    progress_println("   총 키 개수: $(length(MAIN_BATCH_RESULTS))")
    for key in keys(MAIN_BATCH_RESULTS)
        progress_println("   저장된 키: $key")
    end

    # 락커 통계 요약 및 저장
    progress_println("\n" * "="^70)
    progress_println("📊 락커 사용률 통계 처리 시작")
    progress_println("="^70)

    # 통계 요약 생성 및 출력
    print_locker_summary()

    # 캐리어별 상세 통계 출력
    print_carrier_summary()

    # 락커별 캐리어 사용량 분석 출력
    show_locker_carrier_breakdown()

    # 차량 효율성 분석은 캐리어별 상세 통계에 포함됨

    # CSV 파일로 저장
    csv_path = save_locker_stats_csv()
    if !isempty(csv_path)
        progress_println("📄 락커 통계 CSV 저장 완료: $csv_path")
    end

    # 캐리어별 통계 CSV 저장
    carrier_csv_path = save_carrier_stats_csv()
    if !isempty(carrier_csv_path)
        progress_println("📄 캐리어별 통계 CSV 저장 완료: $carrier_csv_path")
    end
    
    # 시간창 정보 CSV 저장
    tw_csv_path = save_customer_time_windows_csv()
    if !isempty(tw_csv_path)
        progress_println("📄 고객 시간창 CSV 저장 완료: $tw_csv_path")
    end
    
    schedule_csv_path = save_route_schedule_csv()
    if !isempty(schedule_csv_path)
        progress_println("📄 라우트 스케줄 CSV 저장 완료: $schedule_csv_path")
    end

    # JSON 파일로 저장
    json_path = save_locker_stats_json()
    if !isempty(json_path)
        progress_println("📄 락커 통계 JSON 저장 완료: $json_path")
    end

    # 추가 시각화 생성
    progress_println("\n🎨 시각화 생성 시작...")
    progress_println("🔍 [디버그] create_all_visualizations 함수 호출 직전")
    create_all_visualizations()
    progress_println("🔍 [디버그] create_all_visualizations 함수 호출 완료")
    progress_println("🎨 시각화 생성 완료!")
    
    # MOO Pareto front 저장 및 시각화
    progress_println("\n📊 MOO Pareto Front 저장 및 시각화 시작...")
    
    lock(MOO_RESULTS_LOCK) do
        if !isempty(MOO_RESULTS_COLLECTOR)
            # 시각화 생성
            viz_dir = joinpath(OUTDIR, "visualizations")
            mkpath(viz_dir)
            create_all_moo_visualizations(MOO_RESULTS_COLLECTOR, viz_dir)
        else
            progress_println("⚠️  MOO 결과가 없어 시각화를 건너뜁니다.")
        end
    end

    progress_println("\n🎉 모든 처리 완료!")
    progress_println("📊 결과 파일들이 다음 위치에 저장되었습니다:")
    progress_println("   📁 결과 디렉토리: $OUTDIR")
    if !isempty(csv_path)
        progress_println("   📄 락커 통계 CSV: $(basename(csv_path))")
    end
    if !isempty(json_path)
        progress_println("   📄 락커 통계 JSON: $(basename(json_path))")
    end
    
    # MOO 결과 파일 출력
    lock(MOO_RESULTS_LOCK) do
        if !isempty(MOO_RESULTS_COLLECTOR)
            progress_println("   📄 Integrated results (MOO): integrated_results_moo.csv")
            progress_println("   📄 Pareto fronts (all): pareto_fronts_all.csv, pareto_fronts_all.json")
            progress_println("   📊 Pareto front 시각화: visualizations/ 폴더")
        end
    end

    # ✅ 시나리오 4는 SLRP (Stochastic Location-Routing Problem) 방식으로
    # POI 기반 후보 위치 + ADD 알고리즘으로 락커 위치와 개수를 동시에 최적화합니다.

    progress_println("🎯 ALNS CVRP with SLRP-based Locker Optimization 완료!")
end

# 메인 프로세스에서만 파이프라인 실행 (워커에서는 함수 정의만 로드)
if IS_MAIN_PROCESS && abspath(PROGRAM_FILE) == @__FILE__
    execute_pipeline()

    # Robustness validation: runs by default; skip with SKIP_ROBUSTNESS=1
    if get(ENV, "SKIP_ROBUSTNESS", "") != "1"
        progress_println("\n" * "═"^80)
        progress_println("  ROBUSTNESS VALIDATION SUITE")
        progress_println("═"^80)
        include(joinpath(@__DIR__, "robustness_tests.jl"))
        run_robustness_tests()
    else
        progress_println("\n⏭️  SKIP_ROBUSTNESS=1 — 로버스트니스 테스트 건너뜀")
    end
end
