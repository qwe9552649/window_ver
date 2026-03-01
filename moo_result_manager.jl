#!/usr/bin/env julia
# ═══════════════════════════════════════════════════════════════════════════
# moo_result_manager.jl
#
# Multi-Objective Optimization (MOO) 결과 관리 및 저장
# - Pareto Front 수집 및 저장 (CSV, JSON)
# - 시나리오별, omega별 MOO 목적함수 값 관리
# - integrated_results.csv 새로운 MOO 형식으로 저장
# ═══════════════════════════════════════════════════════════════════════════

using DataFrames, CSV, JSON3, Statistics, HypothesisTests

# MOO 결과를 담는 구조체
mutable struct MOOScenarioResult
    scenario::Int
    scenario_name::String
    k::Int  # 락커 개수
    omega::Int  # Monte Carlo 반복 번호
    
    # Pareto Front 정보
    pareto_front::Matrix{Float64}  # [n_solutions × 3] matrix (f1, f2, f3)
    selected_solution_idx::Int     # 선택된 해의 인덱스 (compromise solution 등)
    
    # 선택된 해의 MOO 목적함수 값
    f1_enterprise::Float64    # 기업 비용 (EUR)
    f2_customer::Float64      # 고객 비용 (EUR)
    f3_society::Float64       # 사회 비용 (EUR)
    
    # 선택된 해의 세부 정보 (f1 구성요소)
    f1_fuel_cost::Float64
    f1_vehicle_cost::Float64
    f1_driver_cost::Float64
    
    # 선택된 해의 세부 정보 (f2 구성요소)
    f2_mobility_inconvenience::Float64
    f2_dissatisfaction::Float64
    avg_customer_satisfaction::Float64
    avg_customer_delay::Float64
    avg_customer_actual_dist::Float64  # 실제 평균 이동거리 (km)
    
    # 시나리오 비교용 원시값 (정규화 전)
    mobility_raw_km::Float64       # 락커 고객 평균 이동거리 (km)
    dissatisfaction_raw::Float64  # D2D 불만족도 = N_D2D×(1-avg_satisfaction)
    
    # 선택된 해의 세부 정보 (f3 구성요소)
    f3_vehicle_co2::Float64
    f3_customer_co2::Float64
    f3_locker_co2::Float64
    
    # 경로 정보
    total_distance_km::Float64
    vehicles_used::Int
    total_trips::Int
    
    # 고객 정보
    total_customers::Int
    locker_customers::Int
    actual_locker_customers::Int
    d2d_conversions::Int
    
    # 락커 정보
    num_active_lockers::Int
    locker_ids::String  # comma-separated
    
    # 고객 이동 수단 선택 상세
    num_walk_choice::Float64
    num_bicycle_choice::Float64
    num_vehicle_dedicated::Float64
    num_vehicle_linked::Float64
    
    customer_walking_dist::Float64
    customer_bicycle_dist::Float64
    customer_vehicle_ded_dist::Float64
    customer_vehicle_link_dist::Float64
end

# Pareto Front 통계 정보
struct ParetoFrontStats
    n_solutions::Int
    
    # f1 범위
    f1_min::Float64
    f1_max::Float64
    f1_mean::Float64
    f1_std::Float64
    
    # f2 범위
    f2_min::Float64
    f2_max::Float64
    f2_mean::Float64
    f2_std::Float64
    
    # f3 범위
    f3_min::Float64
    f3_max::Float64
    f3_mean::Float64
    f3_std::Float64
    
    # 극단해 인덱스
    idx_min_f1::Int  # f1 최소 해
    idx_min_f2::Int  # f2 최소 해
    idx_min_f3::Int  # f3 최소 해
    idx_compromise::Int  # compromise solution (가장 균형잡힌 해)
end

"""
Pareto front의 통계 정보 계산
"""
function compute_pareto_stats(pareto_front::Matrix{Float64})::ParetoFrontStats
    n = size(pareto_front, 1)
    
    f1_vals = pareto_front[:, 1]
    f2_vals = pareto_front[:, 2]
    f3_vals = pareto_front[:, 3]
    
    # Compromise solution: 정규화된 거리 기반
    # 각 목적함수를 [0,1]로 정규화 후, Euclidean distance가 최소인 점
    f1_norm = (f1_vals .- minimum(f1_vals)) ./ (maximum(f1_vals) - minimum(f1_vals) + 1e-9)
    f2_norm = (f2_vals .- minimum(f2_vals)) ./ (maximum(f2_vals) - minimum(f2_vals) + 1e-9)
    f3_norm = (f3_vals .- minimum(f3_vals)) ./ (maximum(f3_vals) - minimum(f3_vals) + 1e-9)
    
    distances = sqrt.(f1_norm.^2 .+ f2_norm.^2 .+ f3_norm.^2)
    idx_compromise = argmin(distances)
    
    return ParetoFrontStats(
        n,
        minimum(f1_vals), maximum(f1_vals), mean(f1_vals), std(f1_vals),
        minimum(f2_vals), maximum(f2_vals), mean(f2_vals), std(f2_vals),
        minimum(f3_vals), maximum(f3_vals), mean(f3_vals), std(f3_vals),
        argmin(f1_vals), argmin(f2_vals), argmin(f3_vals),
        idx_compromise
    )
end

"""
모든 Pareto front를 하나의 CSV로 저장
"""
function save_all_pareto_fronts_csv(results::Vector{MOOScenarioResult}, output_path::String)
    rows = []
    
    for result in results
        for sol_idx in 1:size(result.pareto_front, 1)
            push!(rows, (
                scenario = result.scenario,
                scenario_name = result.scenario_name,
                k = result.k,
                omega = result.omega,
                solution_id = sol_idx,
                is_selected = (sol_idx == result.selected_solution_idx),
                f1_enterprise = result.pareto_front[sol_idx, 1],
                f2_customer = result.pareto_front[sol_idx, 2],
                f3_society = result.pareto_front[sol_idx, 3]
            ))
        end
    end
    
    df = DataFrame(rows)
    CSV.write(output_path, df)
    println("✅ Pareto fronts 저장: $output_path ($(length(rows))개 해)")
end

"""
모든 Pareto front를 JSON으로 저장
"""
function save_all_pareto_fronts_json(results::Vector{MOOScenarioResult}, output_path::String)
    json_data = []
    
    for result in results
        pareto_list = []
        for sol_idx in 1:size(result.pareto_front, 1)
            push!(pareto_list, Dict(
                "solution_id" => sol_idx,
                "is_selected" => (sol_idx == result.selected_solution_idx),
                "f1_enterprise" => result.pareto_front[sol_idx, 1],
                "f2_customer" => result.pareto_front[sol_idx, 2],
                "f3_society" => result.pareto_front[sol_idx, 3]
            ))
        end
        
        push!(json_data, Dict(
            "scenario" => result.scenario,
            "scenario_name" => result.scenario_name,
            "k" => result.k,
            "omega" => result.omega,
            "pareto_front" => pareto_list
        ))
    end
    
    open(output_path, "w") do io
        JSON3.pretty(io, json_data)
    end
    println("✅ Pareto fronts JSON 저장: $output_path")
end

"""
Integrated results를 Pareto 최적 결과 형식으로 저장
- 시나리오별, omega별 Pareto front의 모든 비지배해를 출력
- 선택된 해(compromise, min_f1 등) 표시
- 시나리오 메타 정보 (고객수, 락커수 등) 포함
"""
function save_integrated_results_moo(results::Vector{MOOScenarioResult}, output_path::String)
    # 중복 제거: (scenario, omega) 조합으로 unique한 결과만 선택
    unique_results = Dict{Tuple{Int,Int}, MOOScenarioResult}()
    for result in results
        key = (result.scenario, result.omega)
        if !haskey(unique_results, key) || unique_results[key].total_trips < result.total_trips
            unique_results[key] = result
        end
    end
    
    rows = []
    
    for result in values(unique_results)
        stats = compute_pareto_stats(result.pareto_front)
        n_solutions = size(result.pareto_front, 1)
        
        for sol_idx in 1:n_solutions
            # 해의 유형 결정
            solution_type = if sol_idx == stats.idx_compromise
                "compromise"
            elseif sol_idx == stats.idx_min_f1
                "min_f1"
            elseif sol_idx == stats.idx_min_f2
                "min_f2"
            elseif sol_idx == stats.idx_min_f3
                "min_f3"
            else
                ""
            end
            
            push!(rows, (
                scenario = result.scenario,
                scenario_name = result.scenario_name,
                omega = result.omega,
                
                # Pareto front 해 정보
                solution_id = sol_idx,
                n_pareto_solutions = n_solutions,
                is_selected = (sol_idx == result.selected_solution_idx),
                solution_type = solution_type,
                
                # 목적함수 값
                f1_enterprise_EUR = result.pareto_front[sol_idx, 1],
                f2_customer_normalized = result.pareto_front[sol_idx, 2],
                f3_society_kgCO2 = result.pareto_front[sol_idx, 3],
                
                # 시나리오 메타 정보 (모든 해에 동일)
                k_lockers = result.k,
                total_customers = result.total_customers,
                d2d_customers = result.total_customers - result.actual_locker_customers,
                locker_customers = result.actual_locker_customers,
                d2d_conversions = result.d2d_conversions,
                total_distance_km = result.total_distance_km,
                vehicles_used = result.vehicles_used,
                
                # 선택된 해의 세부 정보 (참고용)
                selected_f1 = result.f1_enterprise,
                selected_f2 = result.f2_customer,
                selected_f3 = result.f3_society,
                selected_avg_d2d_satisfaction = result.avg_customer_satisfaction,
                selected_avg_d2d_delay_h = result.avg_customer_delay,
                selected_avg_locker_dist_km = result.avg_customer_actual_dist,
                # 시나리오 비교용 원시값 (정규화 전)
                mobility_raw_km = result.mobility_raw_km,
                dissatisfaction_raw = result.dissatisfaction_raw,
            ))
        end
    end
    
    df = DataFrame(rows)
    
    # 시나리오 → omega → solution_id 순서로 정렬
    sort!(df, [:scenario, :omega, :solution_id])
    
    CSV.write(output_path, df)
    
    # 요약 통계 출력
    n_scenarios = length(unique(df.scenario))
    n_omegas = nrow(unique(df[!, [:scenario, :omega]]))
    println("✅ Pareto 최적 결과 저장: $output_path")
    println("   $(n_scenarios)개 시나리오, $(n_omegas)개 omega-scenario, 총 $(nrow(df))개 비지배해")
end

"""
Pareto Front 통계 요약 출력 (논문용 상세 통계)
"""
function print_pareto_summary(results::Vector{MOOScenarioResult})
    println("\n" * "="^80)
    println("📊 MOO 결과 통계 요약 (Monte Carlo Simulation)")
    println("="^80)
    
    # 시나리오별 그룹화
    scenario_groups = Dict{Int, Vector{MOOScenarioResult}}()
    for result in results
        if !haskey(scenario_groups, result.scenario)
            scenario_groups[result.scenario] = []
        end
        push!(scenario_groups[result.scenario], result)
    end
    
    for scn in sort(collect(keys(scenario_groups)))
        scn_results = scenario_groups[scn]
        scn_name = scn_results[1].scenario_name
        n_omega = length(scn_results)
        
        # 선택된 해의 목적함수 값들
        f1_vals = [r.f1_enterprise for r in scn_results]
        f2_vals = [r.f2_customer for r in scn_results]
        f3_vals = [r.f3_society for r in scn_results]
        
        # 통계 계산
        avg_f1, std_f1 = mean(f1_vals), std(f1_vals)
        avg_f2, std_f2 = mean(f2_vals), std(f2_vals)
        avg_f3, std_f3 = mean(f3_vals), std(f3_vals)
        
        min_f1, max_f1 = minimum(f1_vals), maximum(f1_vals)
        min_f2, max_f2 = minimum(f2_vals), maximum(f2_vals)
        min_f3, max_f3 = minimum(f3_vals), maximum(f3_vals)
        
        # CV (변동계수)
        cv_f1 = (std_f1 / avg_f1) * 100
        cv_f2 = (std_f2 / avg_f2) * 100
        cv_f3 = (std_f3 / avg_f3) * 100
        
        # Pareto 크기 통계
        pareto_sizes = [size(r.pareto_front, 1) for r in scn_results]
        avg_pareto_size = mean(pareto_sizes)
        std_pareto_size = std(pareto_sizes)
        
        println("\n" * "─"^80)
        println("🎯 시나리오 $scn: $scn_name (N=$n_omega)")
        println("─"^80)
        
        println("\n📈 선택된 해 (Compromise Solution) 통계:")
        println("   f1 (Enterprise Cost):")
        println("      Mean ± Std: $(round(avg_f1, digits=2)) ± $(round(std_f1, digits=2)) EUR")
        println("      Range: [$(round(min_f1, digits=2)), $(round(max_f1, digits=2))] EUR")
        println("      CV: $(round(cv_f1, digits=2))%")
        
        println("\n   f2 (Customer Cost):")
        println("      Mean ± Std: $(round(avg_f2, digits=2)) ± $(round(std_f2, digits=2)) EUR")
        println("      Range: [$(round(min_f2, digits=2)), $(round(max_f2, digits=2))] EUR")
        println("      CV: $(round(cv_f2, digits=2))%")
        
        println("\n   f3 (Social Cost):")
        println("      Mean ± Std: $(round(avg_f3, digits=2)) ± $(round(std_f3, digits=2)) EUR")
        println("      Range: [$(round(min_f3, digits=2)), $(round(max_f3, digits=2))] EUR")
        println("      CV: $(round(cv_f3, digits=2))%")
        
        println("\n   Total Cost:")
        println("      Mean: $(round(avg_f1 + avg_f2 + avg_f3, digits=2)) EUR")
        
        println("\n📊 Pareto Front 크기:")
        println("      Mean ± Std: $(round(avg_pareto_size, digits=1)) ± $(round(std_pareto_size, digits=1)) 개 해")
        println("      Range: [$(minimum(pareto_sizes)), $(maximum(pareto_sizes))] 개")
        
        # 시나리오 비교용 원시값
        mobility_raw_vals = [r.mobility_raw_km for r in scn_results]
        dissatisfaction_raw_vals = [r.dissatisfaction_raw for r in scn_results]
        println("\n📋 시나리오 비교용 원시값 (동일 척도 비교):")
        println("   mobility_raw (km):   Mean ± Std: $(round(mean(mobility_raw_vals), digits=4)) ± $(round(length(mobility_raw_vals) > 1 ? std(mobility_raw_vals) : 0.0, digits=4))")
        println("   dissatisfaction_raw: Mean ± Std: $(round(mean(dissatisfaction_raw_vals), digits=4)) ± $(round(length(dissatisfaction_raw_vals) > 1 ? std(dissatisfaction_raw_vals) : 0.0, digits=4))")
    end
    
    # 시나리오별 원시값 통합 비교 테이블
    print_raw_values_comparison(results)
    
    println("\n" * "="^80)
end

"""
시나리오별 원시값 통합 비교 테이블 출력 (최종 비교용)
- mobility_raw_km, dissatisfaction_raw를 동일 척도로 시나리오 간 비교
"""
function print_raw_values_comparison(results::Vector{MOOScenarioResult})
    scenario_groups = Dict{Int, Vector{MOOScenarioResult}}()
    for result in results
        if !haskey(scenario_groups, result.scenario)
            scenario_groups[result.scenario] = []
        end
        push!(scenario_groups[result.scenario], result)
    end
    
    println("\n" * "─"^80)
    println("📊 최종 원시값 비교 (시나리오 간 동일 척도 비교)")
    println("   mobility_raw_km: 락커 고객 평균 이동거리 [km]")
    println("   dissatisfaction_raw: D2D 불만족도 = N_D2D × (1 - avg_satisfaction)")
    println("─"^80)
    
    # 테이블 헤더
    header = "시나리오 │ mobility_raw_km (Mean±Std)      │ dissatisfaction_raw (Mean±Std)"
    println(header)
    println("─"^80)
    
    for scn in sort(collect(keys(scenario_groups)))
        scn_results = scenario_groups[scn]
        scn_name = scn_results[1].scenario_name
        mobility_vals = [r.mobility_raw_km for r in scn_results]
        dissat_vals = [r.dissatisfaction_raw for r in scn_results]
        mob_mean = round(mean(mobility_vals), digits=4)
        mob_std = round(length(mobility_vals) > 1 ? std(mobility_vals) : 0.0, digits=4)
        diss_mean = round(mean(dissat_vals), digits=4)
        diss_std = round(length(dissat_vals) > 1 ? std(dissat_vals) : 0.0, digits=4)
        row = "  $scn ($scn_name) │ $(mob_mean) ± $(mob_std) km │ $(diss_mean) ± $(diss_std)"
        println(row)
    end
    println("="^80)
end

"""
원시값 비교 전용 CSV 저장 ( omega별 선택해 기준 )
- 시나리오 간 최종 비교 시 mobility_raw_km, dissatisfaction_raw 원시값 사용
"""
function save_raw_values_comparison(results::Vector{MOOScenarioResult}, output_path::String)
    # (scenario, omega)당 선택된 해 하나만 사용
    unique_results = Dict{Tuple{Int,Int}, MOOScenarioResult}()
    for result in results
        key = (result.scenario, result.omega)
        if !haskey(unique_results, key) || unique_results[key].total_trips < result.total_trips
            unique_results[key] = result
        end
    end
    
    rows = []
    for ((scn, omega), result) in unique_results
        push!(rows, (
            scenario = scn,
            scenario_name = result.scenario_name,
            omega = omega,
            mobility_raw_km = result.mobility_raw_km,
            dissatisfaction_raw = result.dissatisfaction_raw,
            f1_enterprise_EUR = result.f1_enterprise,
            f2_customer_EUR = result.f2_customer,
            f3_society_kgCO2 = result.f3_society
        ))
    end
    
    df = DataFrame(rows)
    sort!(df, [:scenario, :omega])
    CSV.write(output_path, df)
    println("✅ 원시값 비교 CSV 저장: $output_path ($(nrow(df))개 omega-scenario)")
end

"""
시나리오 간 비교 테이블 생성 및 저장 (논문 Table용)
"""
function save_scenario_comparison_table(results::Vector{MOOScenarioResult}, output_path::String)
    # 시나리오별 그룹화
    scenario_groups = Dict{Int, Vector{MOOScenarioResult}}()
    for result in results
        if !haskey(scenario_groups, result.scenario)
            scenario_groups[result.scenario] = []
        end
        push!(scenario_groups[result.scenario], result)
    end
    
    rows = []
    
    for scn in sort(collect(keys(scenario_groups)))
        scn_results = scenario_groups[scn]
        scn_name = scn_results[1].scenario_name
        n_omega = length(scn_results)
        
        # 통계 계산
        f1_vals = [r.f1_enterprise for r in scn_results]
        f2_vals = [r.f2_customer for r in scn_results]
        f3_vals = [r.f3_society for r in scn_results]
        
        # 시나리오 비교용 원시값
        mobility_raw_vals = [r.mobility_raw_km for r in scn_results]
        dissatisfaction_raw_vals = [r.dissatisfaction_raw for r in scn_results]
        
        push!(rows, (
            scenario = scn,
            scenario_name = scn_name,
            n_omega = n_omega,
            
            # f1 통계
            f1_mean = mean(f1_vals),
            f1_std = std(f1_vals),
            f1_min = minimum(f1_vals),
            f1_max = maximum(f1_vals),
            f1_cv = (std(f1_vals) / mean(f1_vals)) * 100,
            
            # f2 통계
            f2_mean = mean(f2_vals),
            f2_std = std(f2_vals),
            f2_min = minimum(f2_vals),
            f2_max = maximum(f2_vals),
            f2_cv = (std(f2_vals) / mean(f2_vals)) * 100,
            
            # f3 통계
            f3_mean = mean(f3_vals),
            f3_std = std(f3_vals),
            f3_min = minimum(f3_vals),
            f3_max = maximum(f3_vals),
            f3_cv = (std(f3_vals) / mean(f3_vals)) * 100,
            
            # 시나리오 비교용 원시값 (정규화 전, 동일 척도로 비교 가능)
            mobility_raw_km_mean = mean(mobility_raw_vals),
            mobility_raw_km_std = length(mobility_raw_vals) > 1 ? std(mobility_raw_vals) : 0.0,
            dissatisfaction_raw_mean = mean(dissatisfaction_raw_vals),
            dissatisfaction_raw_std = length(dissatisfaction_raw_vals) > 1 ? std(dissatisfaction_raw_vals) : 0.0,
            
            # 총 비용
            total_mean = mean(f1_vals) + mean(f2_vals) + mean(f3_vals),
            
            # Pareto front 크기
            pareto_size_mean = mean([size(r.pareto_front, 1) for r in scn_results]),
            pareto_size_std = std([size(r.pareto_front, 1) for r in scn_results])
        ))
    end
    
    df = DataFrame(rows)
    CSV.write(output_path, df)
    println("✅ 시나리오 비교 테이블 저장: $output_path")
end

"""
통계적 검정: 시나리오 간 유의성 검정 (Wilcoxon signed-rank test)
"""
function perform_statistical_tests(results::Vector{MOOScenarioResult})
    println("\n" * "="^80)
    println("📊 통계적 유의성 검정 (Wilcoxon Signed-Rank Test)")
    println("="^80)
    
    # 시나리오별 그룹화
    scenario_groups = Dict{Int, Vector{MOOScenarioResult}}()
    for result in results
        if !haskey(scenario_groups, result.scenario)
            scenario_groups[result.scenario] = []
        end
        push!(scenario_groups[result.scenario], result)
    end
    
    scenarios = sort(collect(keys(scenario_groups)))
    
    # 쌍별 비교
    for i in 1:length(scenarios)-1
        for j in i+1:length(scenarios)
            scn1 = scenarios[i]
            scn2 = scenarios[j]
            
            results1 = scenario_groups[scn1]
            results2 = scenario_groups[scn2]
            
            # 같은 omega 수일 때만 검정
            if length(results1) != length(results2)
                continue
            end
            
            scn1_name = results1[1].scenario_name
            scn2_name = results2[1].scenario_name
            
            # f1, f2, f3에 대해 검정
            f1_vals1 = [r.f1_enterprise for r in results1]
            f1_vals2 = [r.f1_enterprise for r in results2]
            
            f2_vals1 = [r.f2_customer for r in results1]
            f2_vals2 = [r.f2_customer for r in results2]
            
            f3_vals1 = [r.f3_society for r in results1]
            f3_vals2 = [r.f3_society for r in results2]
            
            try
                test_f1 = SignedRankTest(f1_vals1, f1_vals2)
                test_f2 = SignedRankTest(f2_vals1, f2_vals2)
                test_f3 = SignedRankTest(f3_vals1, f3_vals2)
                
                p_f1 = pvalue(test_f1)
                p_f2 = pvalue(test_f2)
                p_f3 = pvalue(test_f3)
                
                println("\n$scn1_name vs $scn2_name:")
                println("   f1: p = $(round(p_f1, digits=4)) $(p_f1 < 0.001 ? "***" : p_f1 < 0.01 ? "**" : p_f1 < 0.05 ? "*" : "n.s.")")
                println("   f2: p = $(round(p_f2, digits=4)) $(p_f2 < 0.001 ? "***" : p_f2 < 0.01 ? "**" : p_f2 < 0.05 ? "*" : "n.s.")")
                println("   f3: p = $(round(p_f3, digits=4)) $(p_f3 < 0.001 ? "***" : p_f3 < 0.01 ? "**" : p_f3 < 0.05 ? "*" : "n.s.")")
            catch e
                println("\n$scn1_name vs $scn2_name: 검정 실패 ($e)")
            end
        end
    end
    
    println("\n(*** p<0.001, ** p<0.01, * p<0.05, n.s. not significant)")
    println("="^80)
end

