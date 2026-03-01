#!/usr/bin/env julia
# ═══════════════════════════════════════════════════════════════════════════
# nsga2_moo_backend.jl
#
# NSGA-II-HI 기반 Multi-Objective Optimization (MOO) VRP 백엔드
# - Python pymoo 기반 NSGA-II-HI (Hybrid Insertion) 사용
# - VRP 특화 인코딩: Permutation (Giant Tour) 
# - 교차: OX/PMX, 돌연변이: Swap/Inversion/Or-opt
# - 수리: Hybrid Insertion (Nearest/Cheapest/Random Path)
# - 목적함수: [f1(기업), f2(고객), f3(사회)] 벡터 (가중합 아님!)
#   • f1(기업): 이동거리×유류비 + 차량수×고정비 (EUR)
#   • f2(고객): 락커고객 이동불편 + 배송지연 불만족도 (정규화)
#   • f3(사회): CO2 배출량 (kg CO2)
# - 시간창 (Time Windows) 완전 지원
# - Pareto Front 생성 및 관리
# 
# Chen et al. (IEEE CEC 2024) 논문 기반 NSGA-II-HI 구현
# ═══════════════════════════════════════════════════════════════════════════

using PyCall
using Random
# Metaheuristics.jl fallback 제거 — pymoo 실패 시 에러를 그대로 노출하여 원인 분석
using JSON3

const MOO_LOCK = ReentrantLock()  # 동시성 제어용 락

# ═══════════════════════════════════════════════════════════════════════════
# MOO 비용 파라미터 (EUR 단위 통일)
# ═══════════════════════════════════════════════════════════════════════════

# f1: 기업 비용 파라미터
const MOO_FUEL_COST_PER_KM = 0.20              # 유류비 (EUR/km) - 디젤 밴 기준
const MOO_VEHICLE_DAILY_COST = 7.0             # 차량 1대당 일일 운행비용 (EUR/대/일)
                                               # - 감가상각, 보험, 유지보수 포함
                                               # - 유럽 LCV TCO 기준 (~€0.07/km × 100km/day)
const MOO_DRIVER_WAGE_PER_HOUR = 12.0          # 운전기사 시급 (EUR/h)

# f2 정규화: Upper-Lower-Bound Min-Max (Marler & Arora 2005, Ishibuchi et al. 2017)
# z_L=ideal point, z_U=Pareto maximum. z' = (z - z_L) / (z_U - z_L + ε), range≈0이면 z'=0
mutable struct F2NormalizationParams
    mobility_L::Float64
    mobility_U::Float64
    dissatisfaction_L::Float64
    dissatisfaction_U::Float64
    initialized::Bool
end

const F2_NORM_PARAMS = F2NormalizationParams(0.0, 1.0, 0.0, 1.0, false)
const F2_NORM_EPSILON = 1e-6
const F2_NORM_RANGE_MIN = 1e-10  # range < 이면 z'=0 (변별력 없음)

"""시나리오 전환 시 f2 정규화 파라미터 리셋 (시나리오별 재계산 필수)"""
function reset_f2_normalization!()
    F2_NORM_PARAMS.mobility_L = 0.0
    F2_NORM_PARAMS.mobility_U = 1.0
    F2_NORM_PARAMS.dissatisfaction_L = 0.0
    F2_NORM_PARAMS.dissatisfaction_U = 1.0
    F2_NORM_PARAMS.initialized = false
end

"""
시나리오 4 전용: mobility range를 사전 계산된 값으로 설정
- 같은 k에서 모든 후보 위치의 mobility를 미리 계산하여 range 설정
- dissatisfaction은 아직 미설정 (NSGA-II 첫 호출 시 Python에서 계산)
- initialized = false 유지 → Python에서 dissatisfaction만 추가 계산
"""
function set_f2_mobility_range!(mob_L::Float64, mob_U::Float64)
    F2_NORM_PARAMS.mobility_L = mob_L
    F2_NORM_PARAMS.mobility_U = mob_U
    # dissatisfaction은 미설정 상태 유지, initialized=false → Python에서 dissatisfaction ideal point만 탐색
    F2_NORM_PARAMS.dissatisfaction_L = 0.0
    F2_NORM_PARAMS.dissatisfaction_U = 1.0
    F2_NORM_PARAMS.initialized = false
    println("  📐 f2 mobility range 사전설정: [$(round(mob_L, digits=4)), $(round(mob_U, digits=4))] km")
end

# f3: 사회 비용 파라미터 (CO2)
const MOO_VEHICLE_CO2_PER_KM = 0.12            # 배송차량 CO2 (kg/km)
const MOO_CUSTOMER_VEHICLE_CO2_PER_KM = 0.12   # 승용차 CO2 (kg/km) - 배송 밴과 동일
const MOO_LOCKER_CO2_PER_UNIT_PER_DAY = 0.5    # 락커 CO2 (kg/일)

# ═══════════════════════════════════════════════════════════════════════════
# CO2 가중 비율 계산
# - 메인 파일(moo_multitrip_cvrp_budapest.jl)의 MODE_SHARE_TABLE, get_mode_share 사용
# - 도보/자전거: CO2 = 0
# - 전용차량: CO2 100% 책임
# - 연계차량: CO2 50% 책임 (다른 목적 이동에 연계)
# ═══════════════════════════════════════════════════════════════════════════

"""
거리(km) 기반 CO2 발생 이동거리 비율 계산
- 메인 파일(moo_multitrip_cvrp_budapest.jl)의 get_mode_share 함수 사용
- 도보/자전거: CO2 = 0
- 전용차량: CO2 100% 책임
- 연계차량: CO2 50% 책임 (다른 목적 이동에 연계되므로 절반만 할당)

주의: 이 함수는 실행 시점에 get_mode_share가 정의되어 있어야 함
      (메인 파일에서 include 후 정의됨)
"""
function get_co2_weighted_vehicle_share(distance_km::Float64)
    # 메인 파일의 get_mode_share 사용 (walk, bicycle, dedicated, linked)
    walk, bicycle, dedicated, linked = get_mode_share(distance_km)
    # 도보/자전거는 CO2 없음
    # 전용차량은 100%, 연계차량은 50% CO2 책임
    return dedicated * 1.0 + linked * 0.5
end

# MOO 결과 저장용 전역 변수
# MOO_PARETO_RESULTS 제거 — fallback 없음, pymoo 결과는 MOO_LAST_PF_MATRIX에 직접 저장
const MOO_ALL_PARETO_FRONTS = Vector{Tuple{Matrix{Float64}, String}}()  # (pareto_front, label)
const MOO_PARETO_LOCK = ReentrantLock()
const MOO_LAST_DETAIL = Ref{Any}(nothing)  # 마지막 호출의 MOO 세부 정보
const MOO_LAST_PF_MATRIX = Ref{Any}(nothing)  # pymoo Pareto front matrix 직접 저장

# Python pymoo 스크립트 경로
const PYMOO_SCRIPT_PATH = joinpath(@__DIR__, "pymoo_vrp_nsga2_hi.py")


# ═══════════════════════════════════════════════════════════════════════════
# 백엔드 버전 확인
# ═══════════════════════════════════════════════════════════════════════════
function pyvrp_version()
    return "NSGA-II-HI MOO v2.0 (pymoo + Hybrid Insertion)"
end

# ═══════════════════════════════════════════════════════════════════════════
# MOO 목적함수 계산 헬퍼 함수들
# ═══════════════════════════════════════════════════════════════════════════

"""순열을 경로로 디코딩 (용량 제약 기반 분할)"""
function decode_routes_from_perm(perm::Vector{Int}, demands::Vector{Int}, capacity::Int;
    time_matrix::Union{Matrix{Float64}, Nothing}=nothing,
    time_windows::Union{Vector{Tuple{Int,Int}}, Nothing}=nothing,
    service_times::Union{Vector{Int}, Nothing}=nothing,
    depot_tw::Union{Tuple{Int,Int}, Nothing}=nothing
)
    routes = Vector{Vector{Int}}()
    current_route = Int[]
    current_load = 0
    
    # 시간창 인식 모드
    tw_aware = (time_matrix !== nothing && time_windows !== nothing && 
                service_times !== nothing && depot_tw !== nothing)
    # Julia Tuple: depot_tw = (earliest_sec, latest_sec)
    # depot_tw[1] = earliest (보통 0), depot_tw[2] = latest (보통 86400)
    current_time = tw_aware ? Float64(depot_tw[1]) / 3600.0 : 0.0  # 시간(h)
    
    for idx in perm
        if idx < 1 || idx > length(demands)
            continue
        end
        demand = demands[idx]
        
        # 1) 용량 초과 → 새 경로
        if current_load + demand > capacity
            if !isempty(current_route)
                push!(routes, current_route)
            end
            current_route = [idx]
            current_load = demand
            if tw_aware
                current_time = Float64(depot_tw[1]) / 3600.0
                travel_h = time_matrix[1, idx + 1] / 3600.0  # depot(row 1) → customer(col idx+1)
                arrival_h = current_time + travel_h
                tw_early_h = Float64(time_windows[idx][1]) / 3600.0  # [1]=earliest
                current_time = max(arrival_h, tw_early_h) + Float64(service_times[idx]) / 3600.0
            end
            continue
        end
        
        # 2) 시간창 확인 (tw_aware 모드)
        if tw_aware
            prev_matrix_idx = isempty(current_route) ? 1 : (current_route[end] + 1)
            travel_h = time_matrix[prev_matrix_idx, idx + 1] / 3600.0
            arrival_h = current_time + travel_h
            tw_late_h = Float64(time_windows[idx][2]) / 3600.0  # [2]=latest
            
            if arrival_h > tw_late_h
                if !isempty(current_route)
                    push!(routes, current_route)
                end
                current_route = [idx]
                current_load = demand
                current_time = Float64(depot_tw[1]) / 3600.0
                travel_from_depot_h = time_matrix[1, idx + 1] / 3600.0
                arrival_from_depot_h = current_time + travel_from_depot_h
                tw_early_h = Float64(time_windows[idx][1]) / 3600.0
                current_time = max(arrival_from_depot_h, tw_early_h) + Float64(service_times[idx]) / 3600.0
                continue
            end
            
            tw_early_h = Float64(time_windows[idx][1]) / 3600.0
            current_time = max(arrival_h, tw_early_h) + Float64(service_times[idx]) / 3600.0
        end
        
        push!(current_route, idx)
        current_load += demand
    end
    
    if !isempty(current_route)
        push!(routes, current_route)
    end
    
    return routes
end

"""
    initialize_f2_normalization!(mobility_values, dissatisfaction_values)

f2 Min-Max 파라미터 설정 (Marler & Arora 2005, Eq.7).
단일 목적 최적화로 구한 ideal 해들의 mobility/dissatisfaction에서 z_L, z_U 결정.
range≈0(D2D 전용 등)이면 z'=0 (변별력 없음).
"""
function initialize_f2_normalization!(mobility_samples::Vector{Float64}, dissatisfaction_samples::Vector{Float64})
    if !F2_NORM_PARAMS.initialized
        F2_NORM_PARAMS.mobility_L = minimum(mobility_samples)
        F2_NORM_PARAMS.mobility_U = maximum(mobility_samples)
        F2_NORM_PARAMS.dissatisfaction_L = minimum(dissatisfaction_samples)
        F2_NORM_PARAMS.dissatisfaction_U = maximum(dissatisfaction_samples)
        F2_NORM_PARAMS.initialized = true
        println("✅ f2 정규화 (Min-Max): 이동불편[$(round(F2_NORM_PARAMS.mobility_L, digits=3)), $(round(F2_NORM_PARAMS.mobility_U, digits=3))], 불만족도[$(round(F2_NORM_PARAMS.dissatisfaction_L, digits=3)), $(round(F2_NORM_PARAMS.dissatisfaction_U, digits=3))]")
    end
end

"""
고객 만족도 계산 (논문: Chen et al., IEEE CEC 2024 기반)

만족도 공식: S = 1 / (1 + waiting_time)²
- waiting_time = max(0, arrival_time - desired_time)
- 일찍 도착 → waiting_time = 0 → S = 1 (최대 만족)
- 늦게 도착 → waiting_time ↑ → S ↓ (만족도 하락)

D2D 고객: 시간창 내 빠른 도착 → 만족도 높음
락커 고객: 24시간 운영, 희망시간보다 늦으면 만족도 하락
"""
function calculate_customer_satisfaction(arrival_time_h::Float64, tw_early_h::Float64)
    # 대기시간(지연) = max(0, 도착시간 - 희망시간)
    # 희망시간 = 시간창 시작 (일찍 도착할수록 좋음)
    waiting_time = max(0.0, arrival_time_h - tw_early_h)
    
    # 만족도: S = 1 / (1 + w)²  (논문 공식)
    satisfaction = 1.0 / (1.0 + waiting_time)^2
    
    return satisfaction, waiting_time
end

"""경로 평가: 거리, 시간, D2D 고객 만족도, 시간창 위반 계산

D2D 고객만 만족도/불만족도 측정 (락커 고객은 이동거리로 측정)

node_individual_desired_times: 노드별 개별 고객 희망시간(초) 리스트
  - D2D 노드: [tw_early] (1명) → 만족도 계산에 사용
  - 락커 가상 노드: 만족도 계산에서 제외 (이동거리로 대체)
  - 비어있으면 기존 방식 (time_windows 기반 + demands 가중)
"""
function evaluate_routes(
    routes::Vector{Vector{Int}},
    dist_matrix::Matrix{Float64},
    time_matrix::Matrix{Float64},
    time_windows::Vector{Tuple{Int,Int}},
    service_times::Vector{Int},
    depot_tw::Tuple{Int,Int};
    customer_types::Vector{Symbol}=Symbol[],  # :d2d 또는 :locker
    demands::Vector{Int}=Int[],  # 노드별 수요 (가상 노드 가중 만족도 계산용)
    node_individual_desired_times::Vector{Vector{Int}}=Vector{Vector{Int}}()  # 노드별 개별 고객 희망시간
)
    total_distance = 0.0
    total_time_hours = 0.0
    total_wait_time_hours = 0.0  # 차량 대기시간 (일찍 도착 시)
    tw_violations = 0.0
    
    # D2D 고객 만족도 관련 (불만족도 계산용)
    d2d_total_satisfaction = 0.0
    d2d_num_customers = 0
    d2d_total_delay_hours = 0.0
    
    # 개별 희망시간 사용 가능 여부
    use_individual_times = !isempty(node_individual_desired_times)
    
    for route in routes
        if isempty(route)
            continue
        end
        
        route_dist = 0.0
        current_time = Float64(depot_tw[1]) / 3600.0  # hours
        
        # depot(1) → first customer
        first_idx = route[1] + 1  # 행렬 인덱스 (depot=1)
        route_dist += dist_matrix[1, first_idx]
        current_time += time_matrix[1, first_idx] / 3600.0
        
        for i in 1:length(route)
            cust_idx = route[i]
            matrix_idx = cust_idx + 1
            
            # 시간창 정보
            tw_early_h = time_windows[cust_idx][1] / 3600.0
            tw_late_h = time_windows[cust_idx][2] / 3600.0
            
            # 고객 유형 확인 (락커 = 24시간, D2D = 시간창 제약)
            is_locker = !isempty(customer_types) && 
                        cust_idx <= length(customer_types) && 
                        customer_types[cust_idx] == :locker
            
            arrival_time = current_time
            
            # 차량 대기 (일찍 도착 시)
            if current_time < tw_early_h
                wait_time = tw_early_h - current_time
                total_wait_time_hours += wait_time
                current_time = tw_early_h
            end
            
            # 고객 만족도 계산 (D2D 고객만 - 불만족도 목적함수용)
            # 락커 고객은 이동거리로 측정하므로 여기서 제외
            if !is_locker
                if use_individual_times && cust_idx <= length(node_individual_desired_times) && !isempty(node_individual_desired_times[cust_idx])
                    for desired_sec in node_individual_desired_times[cust_idx]
                        desired_h = desired_sec / 3600.0
                        sat, delay = calculate_customer_satisfaction(arrival_time, desired_h)
                        d2d_total_satisfaction += sat
                        d2d_total_delay_hours += delay
                        d2d_num_customers += 1
                    end
                else
                    node_weight = (!isempty(demands) && cust_idx <= length(demands)) ? 
                        max(1, demands[cust_idx]) : 1
                    satisfaction, customer_delay = calculate_customer_satisfaction(arrival_time, tw_early_h)
                    d2d_total_satisfaction += satisfaction * node_weight
                    d2d_total_delay_hours += customer_delay * node_weight
                    d2d_num_customers += node_weight
                end
            end
            
            # 시간창 위반 체크 (D2D만 엄격 적용, 락커는 24시간이므로 위반 없음)
            if !is_locker && current_time > tw_late_h
                tw_violations += (current_time - tw_late_h)
            end
            
            # 서비스 시간
            current_time += service_times[cust_idx] / 3600.0
            
            # 다음 노드로 이동
            if i < length(route)
                next_idx = route[i+1] + 1
                route_dist += dist_matrix[matrix_idx, next_idx]
                current_time += time_matrix[matrix_idx, next_idx] / 3600.0
            end
        end
        
        # last customer → depot
        last_idx = route[end] + 1
        route_dist += dist_matrix[last_idx, 1]
        current_time += time_matrix[last_idx, 1] / 3600.0
        
        total_distance += route_dist
        total_time_hours += (current_time - depot_tw[1] / 3600.0)
    end
    
    # D2D 고객 평균 만족도 (0~1)
    avg_d2d_satisfaction = d2d_num_customers > 0 ? 
        d2d_total_satisfaction / d2d_num_customers : 1.0
    
    # D2D 고객 평균 지연시간 (시간)
    avg_d2d_delay = d2d_num_customers > 0 ?
        d2d_total_delay_hours / d2d_num_customers : 0.0
    
    return (
        total_distance = total_distance,
        total_time_hours = total_time_hours,
        total_wait_time_hours = total_wait_time_hours,
        tw_violations = tw_violations,
        avg_satisfaction = avg_d2d_satisfaction,      # D2D 고객만
        avg_customer_delay = avg_d2d_delay,           # D2D 고객만
        num_d2d_customers = d2d_num_customers         # D2D 고객 수
    )
end

"""
MOO 목적함수 평가 (f1, f2, f3)

f2 고객 만족도: Chen et al. (IEEE CEC 2024) 논문 기반
- 만족도 S = 1 / (1 + waiting_time)²
- 일찍 도착 → waiting_time = 0 → S = 1 (최대 만족)
- 늦게 도착 → waiting_time ↑ → S ↓ (만족도 하락)

매개변수:
- customer_locker_distances: 고객별 락커까지 거리 (없으면 빈 배열)
- num_active_lockers: 활성화된 락커 수
- customer_types: 고객별 유형 (:d2d 또는 :locker)
"""
function evaluate_moo_objectives(
    routes::Vector{Vector{Int}},
    dist_matrix::Matrix{Float64},
    time_matrix::Matrix{Float64},
    time_windows::Vector{Tuple{Int,Int}},
    service_times::Vector{Int},
    depot_tw::Tuple{Int,Int};
    customer_locker_distances::Vector{Float64}=Float64[],
    num_active_lockers::Int=0,
    customer_types::Vector{Symbol}=Symbol[],
    demands::Vector{Int}=Int[],
    node_individual_desired_times::Vector{Vector{Int}}=Vector{Vector{Int}}()
)
    n_customers = size(dist_matrix, 1) - 1
    num_vehicles = length(routes)
    
    # 경로 평가 (개별 고객 희망시간 기반 만족도 계산)
    eval_result = evaluate_routes(
        routes, dist_matrix, time_matrix, time_windows, service_times, depot_tw;
        customer_types=customer_types,
        demands=demands,
        node_individual_desired_times=node_individual_desired_times
    )
    
    total_distance = eval_result.total_distance
    total_time_hours = eval_result.total_time_hours
    total_wait_time_hours = eval_result.total_wait_time_hours
    tw_violations = eval_result.tw_violations
    avg_d2d_satisfaction = eval_result.avg_satisfaction        # D2D 고객만
    avg_d2d_delay = eval_result.avg_customer_delay            # D2D 고객만
    num_d2d_customers = eval_result.num_d2d_customers         # D2D 고객 수
    
    # ─────────────────────────────────────────────────────────────────────────
    # f1: 기업 비용 (EUR)
    # (이동거리 × 유류비) + (차량수 × 일일운행비)
    # 주의: 인건비 제외
    # ─────────────────────────────────────────────────────────────────────────
    f1_fuel = total_distance * MOO_FUEL_COST_PER_KM
    f1_vehicle = num_vehicles * MOO_VEHICLE_DAILY_COST
    f1 = f1_fuel + f1_vehicle  # 인건비 제외
    
    # ─────────────────────────────────────────────────────────────────────────
    # f2: 고객 비용 (Min-Max 정규화, range=0 fallback)
    # z' = (z - z_L) / max(z_U - z_L, range_min) + ε, range < 1e-10이면 분모 1
    # ─────────────────────────────────────────────────────────────────────────
    
    # 1. 락커 고객 실제 이동거리 계산 (MODE_SHARE 기반)
    total_customer_actual_dist = 0.0    # 락커 고객 실제 총 이동거리 (km)
    num_locker_customers = 0            # 락커 사용 고객 수
    total_customer_vehicle_co2 = 0.0    # 차량 이용분만 CO2 (도보/자전거 = 0)
    
    if !isempty(customer_locker_distances)
        for dist in customer_locker_distances
            if dist > 0
                num_locker_customers += 1
                
                # MODE_SHARE 기반 실제 이동거리 계산
                walk, bicycle, dedicated, linked = get_mode_share(dist)
                
                # 도보/자전거/전용차량: 왕복 (편도 × 2)
                # 연계차량: 편도만 (다른 목적 이동 중 들르므로)
                round_trip_dist = dist * 2.0
                one_way_dist = dist
                
                actual_dist = (walk + bicycle + dedicated) * round_trip_dist + linked * one_way_dist
                total_customer_actual_dist += actual_dist
                
                # f3용: CO2 계산 (전용차량 100%, 연계차량 50%)
                # 도보/자전거는 CO2 = 0
                dedicated_co2 = dedicated * round_trip_dist * MOO_CUSTOMER_VEHICLE_CO2_PER_KM
                linked_co2 = linked * one_way_dist * MOO_CUSTOMER_VEHICLE_CO2_PER_KM  # 연계는 편도
                total_customer_vehicle_co2 += dedicated_co2 + linked_co2
            end
        end
    end
    
    # 평균 실제 이동거리 (km)
    avg_customer_actual_dist = num_locker_customers > 0 ? 
        total_customer_actual_dist / num_locker_customers : 0.0
    
    # 2. D2D 고객 불만족도 (논문: Chen et al., IEEE CEC 2024 기반)
    # 논문: Max Σ S_j = Max Σ 1/(1+w_j)²
    # 최소화 형태: Min Σ (1 - S_j)
    # D2D 고객만 측정 (락커 고객은 이동거리로 측정)
    d2d_total_dissatisfaction = num_d2d_customers * (1.0 - avg_d2d_satisfaction)
    
    # Min-Max: z' = (z - z_L)/(z_U - z_L + ε), range=0이면 z'=0 (변별력 없음)
    range_mob = F2_NORM_PARAMS.mobility_U - F2_NORM_PARAMS.mobility_L
    range_dis = F2_NORM_PARAMS.dissatisfaction_U - F2_NORM_PARAMS.dissatisfaction_L
    mobility_normalized = range_mob >= F2_NORM_RANGE_MIN ? 
        (avg_customer_actual_dist - F2_NORM_PARAMS.mobility_L) / (range_mob + F2_NORM_EPSILON) : 0.0
    dissatisfaction_normalized = range_dis >= F2_NORM_RANGE_MIN ?
        (d2d_total_dissatisfaction - F2_NORM_PARAMS.dissatisfaction_L) / (range_dis + F2_NORM_EPSILON) : 0.0
    f2 = mobility_normalized + dissatisfaction_normalized
    
    # ─────────────────────────────────────────────────────────────────────────
    # f3: 사회 비용 (kg CO2) - 환경 영향
    # 배송차량 CO2 + 고객 차량이용분 CO2 + 락커 CO2
    # 
    # 고객 CO2 (MODE_SHARE 기반):
    # - 도보/자전거: CO2 = 0
    # - 전용차량: 왕복거리 × CO2/km
    # - 연계차량: 편도거리 × CO2/km (다른 목적 이동에 연계)
    # 
    # 주의: EUR 환산 제거 (kg CO2 단위 직접 사용)
    # ─────────────────────────────────────────────────────────────────────────
    vehicle_co2 = total_distance * effective_vehicle_co2()
    locker_co2 = num_active_lockers * MOO_LOCKER_CO2_PER_UNIT_PER_DAY
    
    f3 = vehicle_co2 + total_customer_vehicle_co2 + locker_co2  # kg CO2
    
    # 세부 정보를 NamedTuple로 반환
    detail = (
        # 목적함수
        f1 = f1,
        f2 = f2,
        f3 = f3,
        
        # f1 구성요소
        f1_fuel_cost = f1_fuel,
        f1_vehicle_cost = f1_vehicle,
        f1_driver_cost = 0.0,  # 인건비 제외
        
        # f2 구성요소 (정규화된 값)
        f2_mobility_inconvenience = mobility_normalized,       # 락커 고객 이동거리 (정규화)
        f2_dissatisfaction = dissatisfaction_normalized,       # D2D 고객 불만족도 (정규화)
        avg_customer_satisfaction = avg_d2d_satisfaction,      # D2D 고객 평균 만족도
        avg_customer_delay = avg_d2d_delay,                   # D2D 고객 평균 지연시간
        avg_customer_actual_dist = avg_customer_actual_dist,  # 락커 고객 평균 이동거리 (km)
        mobility_raw_km = avg_customer_actual_dist,           # 원시값 (시나리오 비교용)
        dissatisfaction_raw = d2d_total_dissatisfaction,      # 원시값 (시나리오 비교용)
        
        # f3 구성요소 (kg CO2)
        f3_vehicle_co2 = vehicle_co2,
        f3_customer_co2 = total_customer_vehicle_co2,
        f3_locker_co2 = locker_co2,
        
        # 기타 정보
        total_distance = total_distance,
        num_vehicles = num_vehicles,
        tw_violations = tw_violations
    )
    
    return [f1, f2, f3], [tw_violations], total_distance, avg_d2d_satisfaction, detail
end

# ═══════════════════════════════════════════════════════════════════════════
# NSGA-II MOO CVRPTW 솔버 (Single Depot)
# ═══════════════════════════════════════════════════════════════════════════
"""
    pyvrp_solve_cvrptw(...)

NSGA-II 기반 다목적 최적화 CVRPTW 솔버
- f1(기업): 유류비 + 차량 고정비 (EUR)
- f2(고객): 이동불편 + 배송지연 불만족도 (EUR + 무차원)
- f3(사회): CO2 배출량 (kg CO2)

기존 인터페이스 호환: Pareto front 중 f1 최소 해 반환
"""
function pyvrp_solve_cvrptw(
    dist_matrix::Matrix{Float64},
    time_matrix::Matrix{Float64},
    demands::Vector{Int},
    time_windows::Vector{Tuple{Int,Int}},
    service_times::Vector{Int},
    capacity::Int;
    num_vehicles::Int=1000,
    max_iterations::Int=1000,
    depot_tw::Tuple{Int,Int}=(0, 86400),
    customer_locker_distances::Vector{Float64}=Float64[],
    num_active_lockers::Int=0,
    moo_pop_size::Int=50,
    moo_generations::Int=100,
    node_individual_desired_times::Vector{Vector{Int}}=Vector{Vector{Int}}(),
    moo_seed::Int=42
)
    n = size(dist_matrix, 1)
    n_customers = n - 1
    
    @assert length(demands) == n_customers "demands length must be N-1"
    @assert length(time_windows) == n_customers "time_windows length must be N-1"
    @assert length(service_times) == n_customers "service_times length must be N-1"
    
    # 고객이 없으면 빈 결과 반환
    if n_customers == 0
        return Vector{Vector{Int}}(), 0.0
    end
    
    # Inf/NaN 값 처리
    MAX_VAL = 1.0e8
    clean_dist = copy(dist_matrix)
    clean_time = copy(time_matrix)
    for i in 1:n, j in 1:n
        if !isfinite(clean_dist[i,j]) || clean_dist[i,j] > MAX_VAL
            clean_dist[i,j] = MAX_VAL
        end
        if !isfinite(clean_time[i,j]) || clean_time[i,j] > MAX_VAL
            clean_time[i,j] = MAX_VAL
        end
    end
    
    # ─────────────────────────────────────────────────────────────────────────
    # Python pymoo NSGA-II-HI 호출
    # ─────────────────────────────────────────────────────────────────────────
    
    # f2 정규화 파라미터 (Julia 측에서 계산하여 Python에 전달)
    f2_initialized = F2_NORM_PARAMS.initialized
    
    # mobility range가 사전설정되었는지 확인 (시나리오 4: set_f2_mobility_range! 호출됨)
    mob_range = F2_NORM_PARAMS.mobility_U - F2_NORM_PARAMS.mobility_L
    has_preset_mobility = !f2_initialized && mob_range > F2_NORM_RANGE_MIN
    
    # 고객 유형 리스트 생성
    customer_types_list = String[]
    if !isempty(customer_locker_distances)
        customer_types_list = [d > 0 ? "locker" : "d2d" for d in customer_locker_distances]
    end
    
    # f2_norm_params 구성
    f2_params_to_send = if f2_initialized
        # 완전히 초기화됨 (mobility + dissatisfaction 모두 설정)
        Dict(
            "mobility_L" => F2_NORM_PARAMS.mobility_L,
            "mobility_U" => F2_NORM_PARAMS.mobility_U,
            "dissatisfaction_L" => F2_NORM_PARAMS.dissatisfaction_L,
            "dissatisfaction_U" => F2_NORM_PARAMS.dissatisfaction_U
        )
    elseif has_preset_mobility
        # mobility만 사전설정됨 → Python에서 dissatisfaction만 추가 계산
        Dict(
            "mobility_L" => F2_NORM_PARAMS.mobility_L,
            "mobility_U" => F2_NORM_PARAMS.mobility_U,
            "preset_mobility_only" => true
        )
    else
        nothing  # 전혀 미초기화 → Python에서 전체 계산
    end
    
    # 문제 데이터를 JSON으로 직렬화
    problem_data = Dict{String,Any}(
        "dist_matrix" => [collect(clean_dist[i, :]) for i in 1:n],
        "time_matrix" => [collect(clean_time[i, :]) for i in 1:n],
        "demands" => collect(demands),
        "time_windows" => [[tw[1], tw[2]] for tw in time_windows],
        "service_times" => collect(service_times),
        "depot_tw" => [depot_tw[1], depot_tw[2]],
        "capacity" => capacity,
        "customer_locker_distances" => isempty(customer_locker_distances) ? nothing : collect(customer_locker_distances),
        "num_active_lockers" => num_active_lockers,
        "customer_types" => isempty(customer_types_list) ? nothing : customer_types_list,
        "node_individual_desired_times" => isempty(node_individual_desired_times) ? nothing : 
            [isempty(times) ? Int[] : collect(times) for times in node_individual_desired_times],
        "pop_size" => moo_pop_size,
        "n_gen" => min(moo_generations, max(50, max_iterations ÷ 10)),
        "seed" => moo_seed,
        "crossover_type" => "ox",
        "f2_norm_params" => f2_params_to_send,
        "cost_overrides" => Dict{String,Float64}(
            "fuel_cost_per_km"     => effective_fuel_cost(),
            "vehicle_daily_cost"   => effective_vehicle_daily_cost(),
            "vehicle_co2_per_km"   => effective_vehicle_co2(),
        ),
    )
    
    # 임시 파일 경로
    input_path = tempname() * "_pymoo_input.json"
    output_path = tempname() * "_pymoo_output.json"
    
    try
        # JSON 파일 작성
        open(input_path, "w") do f
            JSON3.write(f, problem_data)
        end
        
        # Python pymoo 호출
        pymoo_cmd = `python3 $(PYMOO_SCRIPT_PATH) --input $(input_path) --output $(output_path) --quiet`
        
        run(pymoo_cmd; wait=true)
        
        # 출력 파일 확인
        if !isfile(output_path)
            error("pymoo 출력 파일 없음: $output_path")
        end
        
        # 결과 JSON 읽기
        result_json = JSON3.read(read(output_path, String))
        
        # Pareto front 파싱
        pf_data = result_json[:pareto_front]
        if isempty(pf_data)
            error("pymoo Pareto front가 비어있음 — NSGA-II-HI 결과 없음")
        end
        
        n_solutions = length(pf_data)
        pf_matrix = zeros(Float64, n_solutions, 3)
        for i in 1:n_solutions
            pf_matrix[i, 1] = Float64(pf_data[i][1])
            pf_matrix[i, 2] = Float64(pf_data[i][2])
            pf_matrix[i, 3] = Float64(pf_data[i][3])
        end
        
        # Pareto front 저장 (전역)
        MOO_LAST_PF_MATRIX[] = copy(pf_matrix)
        # (pymoo 전용 — fallback 제거됨)
        
        lock(MOO_PARETO_LOCK) do
            push!(MOO_ALL_PARETO_FRONTS, (copy(pf_matrix), "NSGA2-HI_pymoo"))
        end
        
        # f2 정규화 파라미터 업데이트 (Python에서 계산한 값)
        if haskey(result_json, :f2_norm_params) && result_json[:f2_norm_params] !== nothing
            p = result_json[:f2_norm_params]
            if !F2_NORM_PARAMS.initialized
                # mobility가 사전설정된 경우 유지, 아니면 Python 결과 사용
                if has_preset_mobility
                    # mobility range는 사전설정값 유지, dissatisfaction만 Python에서 가져옴
                    F2_NORM_PARAMS.dissatisfaction_L = Float64(p[:dissatisfaction_L])
                    F2_NORM_PARAMS.dissatisfaction_U = Float64(p[:dissatisfaction_U])
                else
                    F2_NORM_PARAMS.mobility_L = Float64(p[:mobility_L])
                    F2_NORM_PARAMS.mobility_U = Float64(p[:mobility_U])
                    F2_NORM_PARAMS.dissatisfaction_L = Float64(p[:dissatisfaction_L])
                    F2_NORM_PARAMS.dissatisfaction_U = Float64(p[:dissatisfaction_U])
                end
                F2_NORM_PARAMS.initialized = true
                println("  ✅ f2 정규화 (pymoo): 이동불편[$(round(F2_NORM_PARAMS.mobility_L, digits=3)), $(round(F2_NORM_PARAMS.mobility_U, digits=3))], 불만족도[$(round(F2_NORM_PARAMS.dissatisfaction_L, digits=3)), $(round(F2_NORM_PARAMS.dissatisfaction_U, digits=3))]")
            end
        end
        
        # MOO detail 파싱
        detail_data = result_json[:moo_detail]
        moo_detail = (
            f1 = Float64(detail_data[:f1]),
            f2 = Float64(detail_data[:f2]),
            f3 = Float64(detail_data[:f3]),
            f1_fuel_cost = Float64(detail_data[:f1_fuel_cost]),
            f1_vehicle_cost = Float64(detail_data[:f1_vehicle_cost]),
            f1_driver_cost = Float64(get(detail_data, :f1_driver_cost, 0.0)),
            f2_mobility_inconvenience = Float64(detail_data[:f2_mobility_inconvenience]),
            f2_dissatisfaction = Float64(detail_data[:f2_dissatisfaction]),
            avg_customer_satisfaction = Float64(detail_data[:avg_customer_satisfaction]),
            avg_customer_delay = Float64(detail_data[:avg_customer_delay]),
            avg_customer_actual_dist = Float64(detail_data[:avg_customer_actual_dist]),
            mobility_raw_km = Float64(get(detail_data, :mobility_raw_km, detail_data[:avg_customer_actual_dist])),
            dissatisfaction_raw = Float64(get(detail_data, :dissatisfaction_raw, 0.0)),
            f3_vehicle_co2 = Float64(detail_data[:f3_vehicle_co2]),
            f3_customer_co2 = Float64(detail_data[:f3_customer_co2]),
            f3_locker_co2 = Float64(detail_data[:f3_locker_co2]),
            total_distance = Float64(detail_data[:total_distance]),
            num_vehicles = Int(detail_data[:num_vehicles]),
            tw_violations = Float64(detail_data[:tw_violations]),
        )
        
        MOO_LAST_DETAIL[] = moo_detail
        
        # 경로 파싱 (Python에서 반환한 routes: [[0, c1, c2, ..., 0], ...])
        routes_data = result_json[:selected_routes]
        total_cost = Float64(result_json[:total_cost])
        
        routes_output = Vector{Vector{Int}}()
        for r in routes_data
            push!(routes_output, [Int(v) for v in r])
        end
        
        # 경로가 비어있으면 detail에서 경로 재구성
        if isempty(routes_output)
            return Vector{Vector{Int}}(), 0.0
        end
        
        return routes_output, total_cost
        
    finally
        # 임시 파일 정리
        isfile(input_path) && rm(input_path; force=true)
        isfile(output_path) && rm(output_path; force=true)
    end
end

"""
MOO Pareto Front 조회 함수
"""
function get_moo_pareto_front()
    if MOO_LAST_PF_MATRIX[] !== nothing
        return MOO_LAST_PF_MATRIX[]
    end
    return nothing
end

"""
MOO 결과에서 특정 목적함수 기준 최적 해 선택
- objective_idx: 1=f1(기업), 2=f2(고객), 3=f3(사회)
"""
function select_moo_solution(objective_idx::Int=1)
    pf = get_moo_pareto_front()
    if pf === nothing || size(pf, 1) == 0
        return nothing, nothing
    end
    
    best_idx = argmin(pf[:, objective_idx])
    best_f = pf[best_idx, :]
    
    return best_idx, best_f
end

"""
마지막 VRP 호출의 MOO 세부 정보 가져오기
"""
function get_last_moo_detail()
    return MOO_LAST_DETAIL[]
end

"""
마지막 Pareto front 가져오기
"""
function get_last_pareto_front()
    if MOO_LAST_PF_MATRIX[] !== nothing
        return MOO_LAST_PF_MATRIX[]
    end
    return nothing
end


# ═══════════════════════════════════════════════════════════════════════════
# Multi-Depot CVRPTW 솔버 (NSGA-II MOO 기반)
# ═══════════════════════════════════════════════════════════════════════════
"""
    pyvrp_solve_mdcvrptw(...)

Multi-Depot CVRPTW 문제를 NSGA-II MOO로 해결합니다.
각 디포별로 고객을 분할하여 단일 디포 NSGA-II 최적화를 수행합니다.

# Arguments
- `vehicle_dist_dict`: (from_id, to_id) => 거리 (미터)
- `vehicle_time_dict`: (from_id, to_id) => 이동 시간 (초)
- `depot_ids`: 디포 ID 리스트
- `customer_ids`: 고객 ID 리스트
- `demand_by_customer`: 고객 ID => 수요
- `time_windows`: 노드 ID => (earliest, latest) 초 단위
- `service_times`: 노드 ID => 서비스 시간 (초)
- `vehicles_by_depot`: 디포 ID => 가용 차량 수
- `capacity_by_depot`: 디포 ID => 차량 용량

# Returns
- `(cost, routes_by_depot)`: 총 비용, 디포별 라우트 딕셔너리
"""
function pyvrp_solve_mdcvrptw(
    vehicle_dist_dict::Dict{Tuple{String,String},Float64},
    vehicle_time_dict::Dict{Tuple{String,String},Float64},
    depot_ids::Vector{String},
    customer_ids::Vector{String},
    demand_by_customer::Dict{String,Int},
    time_windows::Dict{String,Tuple{Int,Int}},
    service_times::Dict{String,Int},
    vehicles_by_depot::Dict{String,Int},
    capacity_by_depot::Dict{String,Int};
    max_iterations::Int=1000
)
    # 고객이 없으면 빈 결과 반환
    if isempty(customer_ids)
        routes_by_depot = Dict{String, Vector{Vector{String}}}()
        for did in depot_ids
            routes_by_depot[did] = Vector{Vector{String}}()
        end
        return 0.0, routes_by_depot
    end
    
    # 단일 디포인 경우 직접 NSGA-II 호출
    if length(depot_ids) == 1
        depot_id = depot_ids[1]
        capacity = get(capacity_by_depot, depot_id, 100)
        
        demands = [get(demand_by_customer, cid, 1) for cid in customer_ids]
        tw_list = [get(time_windows, cid, (0, 86400)) for cid in customer_ids]
        svc_list = [get(service_times, cid, 0) for cid in customer_ids]
        depot_tw = get(time_windows, depot_id, (0, 86400))
        
        cost, routes = solve_vrp_ids_with_tw(
            vehicle_dist_dict, vehicle_time_dict, depot_id, customer_ids,
            demands, tw_list, svc_list, capacity;
            max_iterations=max_iterations, depot_tw=depot_tw
        )
        
        routes_by_depot = Dict{String, Vector{Vector{String}}}()
        routes_by_depot[depot_id] = routes
        
        return cost, routes_by_depot
    end
    
    # Multi-Depot: 각 디포별로 가장 가까운 고객 할당 후 개별 최적화
    big_dist = 1.0e9
    
    # 고객-디포 할당 (가장 가까운 디포)
    customer_to_depot = Dict{String, String}()
    for cid in customer_ids
        min_dist = big_dist
        nearest_depot = depot_ids[1]
        for did in depot_ids
            d = get(vehicle_dist_dict, (did, cid), big_dist)
            if d < min_dist
                min_dist = d
                nearest_depot = did
            end
        end
        customer_to_depot[cid] = nearest_depot
    end
    
    # 디포별 고객 그룹화
    customers_by_depot = Dict{String, Vector{String}}()
    for did in depot_ids
        customers_by_depot[did] = String[]
    end
    for (cid, did) in customer_to_depot
        push!(customers_by_depot[did], cid)
    end
    
    # 디포별 NSGA-II MOO 최적화
    total_cost = 0.0
    routes_by_depot = Dict{String, Vector{Vector{String}}}()
    
    for did in depot_ids
        depot_customers = customers_by_depot[did]
        
        if isempty(depot_customers)
            routes_by_depot[did] = Vector{Vector{String}}()
            continue
        end
        
        capacity = get(capacity_by_depot, did, 100)
        demands = [get(demand_by_customer, cid, 1) for cid in depot_customers]
        tw_list = [get(time_windows, cid, (0, 86400)) for cid in depot_customers]
        svc_list = [get(service_times, cid, 0) for cid in depot_customers]
        depot_tw = get(time_windows, did, (0, 86400))
        
        cost, routes = solve_vrp_ids_with_tw(
            vehicle_dist_dict, vehicle_time_dict, did, depot_customers,
            demands, tw_list, svc_list, capacity;
            max_iterations=max_iterations, depot_tw=depot_tw
        )
        
        total_cost += cost
        routes_by_depot[did] = routes
    end
    
    return total_cost, routes_by_depot
end

# ═══════════════════════════════════════════════════════════════════════════
# ID 기반 MDCVRPTW 솔버 (NSGA-II MOO 기반)
# ═══════════════════════════════════════════════════════════════════════════
"""
    solve_vrp_ids_with_tw(...)

NSGA-II MOO 기반 단일 디포 CVRPTW 솔버
- f1(기업 EUR), f2(고객 EUR+무차원), f3(사회 kg CO2) 3개 목적함수 동시 최적화
- 기존 인터페이스 호환 (단일 해 반환)

# Returns
- `(cost, routes)`: 총 비용, 라우트 리스트 (각 라우트는 노드 ID 벡터)
- Pareto front는 get_moo_pareto_front()로 조회 가능
"""
function solve_vrp_ids_with_tw(
    vehicle_dist_dict::Dict{Tuple{String,String},Float64},
    vehicle_time_dict::Dict{Tuple{String,String},Float64},
    depot_id::String,
    node_ids::Vector{String},
    demands::Vector{Int},
    time_windows::Vector{Tuple{Int,Int}},
    service_times::Vector{Int},
    capacity::Int;
    num_vehicles::Int=1000,
    max_iterations::Int=1000,
    depot_tw::Tuple{Int,Int}=(0, 86400),
    customer_locker_distances::Vector{Float64}=Float64[],
    num_active_lockers::Int=0,
    node_individual_desired_times::Vector{Vector{Int}}=Vector{Vector{Int}}(),
    moo_seed::Int=42
)
    # 거리/시간 행렬 구축 (1=depot, 2..=customers)
    num_customers = length(node_ids)
    N = num_customers + 1
    big = 1.0e8
    
    dist_matrix = zeros(Float64, N, N)
    time_matrix = zeros(Float64, N, N)
    ids = vcat([depot_id], node_ids)
    
    for i in 1:N, j in 1:N
        if i == j
            dist_matrix[i, j] = 0.0
            time_matrix[i, j] = 0.0
        else
            a = ids[i]; b = ids[j]
            d = get(vehicle_dist_dict, (a, b), big)
            t = get(vehicle_time_dict, (a, b), big)
            dist_matrix[i, j] = (isinf(d) || isnan(d)) ? big : d
            time_matrix[i, j] = (isinf(t) || isnan(t)) ? big : t
        end
    end

    routes_idx, cost = pyvrp_solve_cvrptw(
        dist_matrix, time_matrix, demands, time_windows, service_times, capacity;
        num_vehicles=num_vehicles, max_iterations=max_iterations, depot_tw=depot_tw,
        customer_locker_distances=customer_locker_distances,
        num_active_lockers=num_active_lockers,
        node_individual_desired_times=node_individual_desired_times,
        moo_seed=moo_seed
    )

    # 인덱스를 ID로 변환
    routes = Vector{Vector{String}}()
    for r in routes_idx
        route_ids = String[]
        for v in r
            if v == 0
                push!(route_ids, depot_id)
            else
                push!(route_ids, node_ids[v])
            end
        end
        push!(routes, route_ids)
    end

    return cost, routes
end

"""
MOO 목적함수 값을 함께 반환하는 버전
Returns: (cost, routes, moo_objectives)
- moo_objectives: [f1, f2, f3] (기업 EUR, 고객 EUR+무차원, 사회 kg CO2)
"""
function solve_vrp_ids_with_tw_moo(
    vehicle_dist_dict::Dict{Tuple{String,String},Float64},
    vehicle_time_dict::Dict{Tuple{String,String},Float64},
    depot_id::String,
    node_ids::Vector{String},
    demands::Vector{Int},
    time_windows::Vector{Tuple{Int,Int}},
    service_times::Vector{Int},
    capacity::Int;
    num_vehicles::Int=1000,
    max_iterations::Int=1000,
    depot_tw::Tuple{Int,Int}=(0, 86400),
    customer_locker_distances::Vector{Float64}=Float64[],
    num_active_lockers::Int=0,
    node_individual_desired_times::Vector{Vector{Int}}=Vector{Vector{Int}}(),
    moo_seed::Int=42
)
    cost, routes = solve_vrp_ids_with_tw(
        vehicle_dist_dict, vehicle_time_dict, depot_id, node_ids,
        demands, time_windows, service_times, capacity;
        num_vehicles=num_vehicles, max_iterations=max_iterations, depot_tw=depot_tw,
        customer_locker_distances=customer_locker_distances,
        num_active_lockers=num_active_lockers,
        node_individual_desired_times=node_individual_desired_times,
        moo_seed=moo_seed
    )
    
    # Pareto front에서 선택된 해의 목적함수 값 조회
    _, best_f = select_moo_solution(1)  # f1 기준 최적
    
    moo_objectives = best_f !== nothing ? best_f : [0.0, 0.0, 0.0]
    
    return cost, routes, moo_objectives
end

# ═══════════════════════════════════════════════════════════════════════════
# 시간창 없는 버전 (하위 호환성)
# ═══════════════════════════════════════════════════════════════════════════
"""
    solve_vrp_ids(vehicle_dist_dict, depot_id, node_ids, demands, capacity)

시간창 없는 CVRP 솔버 (하위 호환성).
"""
function solve_vrp_ids(
    vehicle_dist_dict::Dict{Tuple{String,String},Float64},
    depot_id::String,
    node_ids::Vector{String},
    demands::Vector{Int},
    capacity::Int;
    max_iterations::Int=1000
)
    # 시간창 없음 = 모든 노드에 [0, 하루] 시간창 할당 (충분히 넓게)
    n = length(node_ids)
    big_tw = 86400  # 24시간 (1일)
    time_windows = [(0, big_tw) for _ in 1:n]
    service_times = zeros(Int, n)
    
    # 거리(km)를 시간(초)으로 변환: 30km/h 가정
    # 시간(초) = 거리(km) / 30 * 3600 = 거리(km) * 120
    vehicle_time_dict = Dict{Tuple{String,String},Float64}()
    for (key, dist_km) in vehicle_dist_dict
        vehicle_time_dict[key] = dist_km * 120.0  # 30km/h 기준 초 단위
    end
    
    return solve_vrp_ids_with_tw(
        vehicle_dist_dict, vehicle_time_dict, depot_id, node_ids,
        demands, time_windows, service_times, capacity;
        max_iterations=max_iterations,
        depot_tw=(0, big_tw)
    )
end

# ═══════════════════════════════════════════════════════════════════════════
# 테스트
# ═══════════════════════════════════════════════════════════════════════════
if abspath(PROGRAM_FILE) == @__FILE__
    println("NSGA-II MOO Backend Test")
    println("="^50)
    
    try
        println("Backend version: ", pyvrp_version())
        
        # 간단한 테스트
        dist = Dict{Tuple{String,String},Float64}(
            ("D", "C1") => 10.0, ("C1", "D") => 10.0,
            ("D", "C2") => 20.0, ("C2", "D") => 20.0,
            ("D", "C3") => 15.0, ("C3", "D") => 15.0,
            ("C1", "C2") => 15.0, ("C2", "C1") => 15.0,
            ("C1", "C3") => 25.0, ("C3", "C1") => 25.0,
            ("C2", "C3") => 10.0, ("C3", "C2") => 10.0,
        )
        
        # 시간 딕셔너리 (거리 ≈ 시간 가정)
        time_dict = Dict{Tuple{String,String},Float64}()
        for (k, v) in dist
            time_dict[k] = v * 120.0  # 30km/h
        end
        
        # 시간창과 함께 테스트
        time_windows = [(0, 86400), (0, 86400), (0, 86400)]
        service_times = [60, 60, 60]
        
        cost, routes = solve_vrp_ids_with_tw(
            dist, time_dict, "D", ["C1", "C2", "C3"],
            [1, 1, 1], time_windows, service_times, 10;
            max_iterations=500
        )
        
        println("Cost (distance): ", cost)
        println("Routes: ", routes)
        
        # MOO Pareto Front 확인
        pf = get_moo_pareto_front()
        if pf !== nothing && size(pf, 1) > 0
            println("\n📊 MOO Pareto Front:")
            println("   f1 (기업): ", round(minimum(pf[:, 1]), digits=2), " ~ ", round(maximum(pf[:, 1]), digits=2), " EUR")
            println("   f2 (고객): ", round(minimum(pf[:, 2]), digits=2), " ~ ", round(maximum(pf[:, 2]), digits=2), " EUR")
            println("   f3 (사회): ", round(minimum(pf[:, 3]), digits=2), " ~ ", round(maximum(pf[:, 3]), digits=2), " kg CO2")
            println("   해 개수: ", size(pf, 1))
        end
        
        println("\n✅ Test passed!")
    catch e
        @error "Test failed" exception=(e, catch_backtrace())
        rethrow()
    end
end
