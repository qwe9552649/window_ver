#!/usr/bin/env julia
# ═══════════════════════════════════════════════════════════════════════════
# moo_pareto_visualization.jl
#
# Multi-Objective Optimization (MOO) Pareto Front 시각화
# - 3D 시각화 (f1, f2, f3)
# - 2D 시각화 (f1 vs f2, f1 vs f3, f2 vs f3)
# - 시나리오별 Pareto Front 시각화
# ═══════════════════════════════════════════════════════════════════════════

using Plots, Statistics, Printf, Dates

"""
MOO 결과로부터 시나리오별 Pareto front 3D 시각화 생성
"""
function create_scenario_pareto_3d_plots(moo_results::Vector{MOOScenarioResult}, output_dir::String)
    # 시나리오별 그룹화
    scenario_groups = Dict{Int, Vector{MOOScenarioResult}}()
    for result in moo_results
        if !haskey(scenario_groups, result.scenario)
            scenario_groups[result.scenario] = []
        end
        push!(scenario_groups[result.scenario], result)
    end
    
    for scn in sort(collect(keys(scenario_groups)))
        scn_results = scenario_groups[scn]
        scn_name = scn_results[1].scenario_name
        
        # 모든 omega의 Pareto fronts 수집
        all_f1 = Float64[]
        all_f2 = Float64[]
        all_f3 = Float64[]
        selected_f1 = Float64[]
        selected_f2 = Float64[]
        selected_f3 = Float64[]
        
        for result in scn_results
            pf = result.pareto_front
            append!(all_f1, pf[:, 1])
            append!(all_f2, pf[:, 2])
            append!(all_f3, pf[:, 3])
            
            # 선택된 해
            push!(selected_f1, result.f1_enterprise)
            push!(selected_f2, result.f2_customer)
            push!(selected_f3, result.f3_society)
        end
        
        # 3D scatter plot
        p = scatter(all_f1, all_f2, all_f3,
            label="Pareto Front",
            marker=(:circle, 3, 0.5),
            color=:lightblue,
            xlabel="f1: Enterprise Cost (EUR)",
            ylabel="f2: Customer Cost (EUR)",
            zlabel="f3: Society Cost (EUR)",
            title="Scenario $scn ($scn_name) - Pareto Front (n=$(length(scn_results)) ω)",
            size=(800, 600),
            camera=(45, 30)
        )
        
        # 선택된 해 표시 (빨간색, 큰 마커)
        scatter!(p, selected_f1, selected_f2, selected_f3,
            label="Selected Solutions",
            marker=(:star, 8, 1.0),
            color=:red
        )
        
        output_path = joinpath(output_dir, "pareto_3d_scenario_$(scn)_$(scn_name).png")
        savefig(p, output_path)
        println("✅ Scenario $scn 3D Pareto front 저장: $output_path")
    end
end

"""
MOO 결과로부터 시나리오별 Pareto front 2D 시각화 생성 (f1 vs f2, f1 vs f3, f2 vs f3)
"""
function create_scenario_pareto_2d_plots(moo_results::Vector{MOOScenarioResult}, output_dir::String)
    # 시나리오별 그룹화
    scenario_groups = Dict{Int, Vector{MOOScenarioResult}}()
    for result in moo_results
        if !haskey(scenario_groups, result.scenario)
            scenario_groups[result.scenario] = []
        end
        push!(scenario_groups[result.scenario], result)
    end
    
    for scn in sort(collect(keys(scenario_groups)))
        scn_results = scenario_groups[scn]
        scn_name = scn_results[1].scenario_name
        
        # 모든 omega의 Pareto fronts 수집
        all_f1 = Float64[]
        all_f2 = Float64[]
        all_f3 = Float64[]
        selected_f1 = Float64[]
        selected_f2 = Float64[]
        selected_f3 = Float64[]
        
        for result in scn_results
            pf = result.pareto_front
            append!(all_f1, pf[:, 1])
            append!(all_f2, pf[:, 2])
            append!(all_f3, pf[:, 3])
            
            push!(selected_f1, result.f1_enterprise)
            push!(selected_f2, result.f2_customer)
            push!(selected_f3, result.f3_society)
        end
        
        # f1 vs f2
        p1 = scatter(all_f1, all_f2,
            label="Pareto Front",
            marker=(:circle, 3, 0.5),
            color=:lightblue,
            xlabel="f1: Enterprise Cost (EUR)",
            ylabel="f2: Customer Cost (EUR)",
            title="Scenario $scn ($scn_name) - f1 vs f2",
            size=(600, 500)
        )
        scatter!(p1, selected_f1, selected_f2,
            label="Selected",
            marker=(:star, 8, 1.0),
            color=:red
        )
        savefig(p1, joinpath(output_dir, "pareto_2d_f1_f2_scenario_$(scn)_$(scn_name).png"))
        
        # f1 vs f3
        p2 = scatter(all_f1, all_f3,
            label="Pareto Front",
            marker=(:circle, 3, 0.5),
            color=:lightgreen,
            xlabel="f1: Enterprise Cost (EUR)",
            ylabel="f3: Society Cost (EUR)",
            title="Scenario $scn ($scn_name) - f1 vs f3",
            size=(600, 500)
        )
        scatter!(p2, selected_f1, selected_f3,
            label="Selected",
            marker=(:star, 8, 1.0),
            color=:red
        )
        savefig(p2, joinpath(output_dir, "pareto_2d_f1_f3_scenario_$(scn)_$(scn_name).png"))
        
        # f2 vs f3
        p3 = scatter(all_f2, all_f3,
            label="Pareto Front",
            marker=(:circle, 3, 0.5),
            color=:lightcoral,
            xlabel="f2: Customer Cost (EUR)",
            ylabel="f3: Society Cost (EUR)",
            title="Scenario $scn ($scn_name) - f2 vs f3",
            size=(600, 500)
        )
        scatter!(p3, selected_f2, selected_f3,
            label="Selected",
            marker=(:star, 8, 1.0),
            color=:red
        )
        savefig(p3, joinpath(output_dir, "pareto_2d_f2_f3_scenario_$(scn)_$(scn_name).png"))
        
        println("✅ Scenario $scn 2D Pareto fronts 저장 (3개 플롯)")
    end
end

"""
전체 시나리오 통합 Pareto front 3D 시각화
"""
function create_all_scenarios_pareto_3d(moo_results::Vector{MOOScenarioResult}, output_dir::String)
    # 시나리오별 그룹화 및 색상 지정
    scenario_groups = Dict{Int, Vector{MOOScenarioResult}}()
    for result in moo_results
        if !haskey(scenario_groups, result.scenario)
            scenario_groups[result.scenario] = []
        end
        push!(scenario_groups[result.scenario], result)
    end
    
    colors = [:blue, :red, :green, :purple, :orange]
    p = plot(xlabel="f1: Enterprise Cost (EUR)",
             ylabel="f2: Customer Cost (EUR)",
             zlabel="f3: Society Cost (EUR)",
             title="All Scenarios - Pareto Fronts Comparison",
             size=(900, 700),
             camera=(45, 30),
             legend=:outertopright)
    
    for (idx, scn) in enumerate(sort(collect(keys(scenario_groups))))
        scn_results = scenario_groups[scn]
        scn_name = scn_results[1].scenario_name
        color = colors[mod1(idx, length(colors))]
        
        # 모든 omega의 Pareto fronts 수집
        all_f1 = Float64[]
        all_f2 = Float64[]
        all_f3 = Float64[]
        
        for result in scn_results
            pf = result.pareto_front
            append!(all_f1, pf[:, 1])
            append!(all_f2, pf[:, 2])
            append!(all_f3, pf[:, 3])
        end
        
        scatter!(p, all_f1, all_f2, all_f3,
            label="$scn_name",
            marker=(:circle, 3, 0.5),
            color=color
        )
    end
    
    output_path = joinpath(output_dir, "pareto_3d_all_scenarios.png")
    savefig(p, output_path)
    println("✅ 전체 시나리오 통합 3D Pareto front 저장: $output_path")
end

"""
MOO 결과로부터 모든 Pareto front 시각화 생성
"""
function create_all_moo_visualizations(moo_results::Vector{MOOScenarioResult}, output_dir::String)
    if isempty(moo_results)
        println("⚠️  시각화할 MOO 결과가 없습니다.")
        return
    end
    
    println("\n" * "="^80)
    println("📊 MOO Pareto Front 시각화 생성 중...")
    println("="^80)
    
    # 시나리오별 3D 플롯
    create_scenario_pareto_3d_plots(moo_results, output_dir)
    
    # 시나리오별 2D 플롯
    create_scenario_pareto_2d_plots(moo_results, output_dir)
    
    # 전체 시나리오 통합 3D 플롯
    create_all_scenarios_pareto_3d(moo_results, output_dir)

    # 시나리오 비교 종합 그래프
    create_scenario_comparison_figures(moo_results, output_dir)

    # 종합 결과 보고서
    generate_main_results_report(moo_results, output_dir)

    println("="^80)
    println("✅ 모든 MOO 시각화 및 보고서 생성 완료")
    println("="^80)
end

# ═══════════════════════════════════════════════════════════════════════════
# Publication-Quality Scenario Comparison Figures
# ═══════════════════════════════════════════════════════════════════════════

function create_scenario_comparison_figures(moo_results::Vector{MOOScenarioResult}, output_dir::String)
    if isempty(moo_results)
        return
    end

    # Group by scenario, compute per-omega compromise averages
    scenario_groups = Dict{Int, Vector{MOOScenarioResult}}()
    for r in moo_results
        push!(get!(scenario_groups, r.scenario, MOOScenarioResult[]), r)
    end

    scn_ids = sort(collect(keys(scenario_groups)))
    scn_names = String[]
    f1_means = Float64[]; f1_stds = Float64[]
    f2_means = Float64[]; f2_stds = Float64[]
    f3_means = Float64[]; f3_stds = Float64[]
    mob_means = Float64[]; dissat_means = Float64[]
    mob_stds = Float64[]; dissat_stds = Float64[]
    cust_counts = Float64[]; locker_counts = Float64[]
    dist_means = Float64[]; dist_stds = Float64[]
    veh_means = Float64[]; veh_stds = Float64[]
    conv_means = Float64[]; pareto_means = Float64[]
    n_omegas = Int[]

    for scn in scn_ids
        results = scenario_groups[scn]
        push!(scn_names, results[1].scenario_name)
        push!(n_omegas, length(results))

        f1v = [r.f1_enterprise for r in results]
        f2v = [r.f2_customer for r in results]
        f3v = [r.f3_society for r in results]

        push!(f1_means, mean(f1v)); push!(f1_stds, std(f1v))
        push!(f2_means, mean(f2v)); push!(f2_stds, std(f2v))
        push!(f3_means, mean(f3v)); push!(f3_stds, std(f3v))

        mobv = [r.mobility_raw_km for r in results]
        dissv = [r.dissatisfaction_raw for r in results]
        push!(mob_means, mean(mobv)); push!(mob_stds, std(mobv))
        push!(dissat_means, mean(dissv)); push!(dissat_stds, std(dissv))
        push!(cust_counts, mean([r.total_customers for r in results]))
        push!(locker_counts, mean([r.actual_locker_customers for r in results]))
        dv = [r.total_distance_km for r in results]
        vv = [Float64(r.vehicles_used) for r in results]
        push!(dist_means, mean(dv)); push!(dist_stds, std(dv))
        push!(veh_means, mean(vv)); push!(veh_stds, std(vv))
        push!(conv_means, mean([r.d2d_conversions for r in results]))
        push!(pareto_means, mean([size(r.pareto_front, 1) for r in results]))
    end

    # ──── Save visualization data CSVs ────
    viz_summary = DataFrame(
        scenario = scn_ids,
        scenario_name = scn_names,
        n_omega = n_omegas,
        f1_mean = f1_means, f1_std = f1_stds,
        f2_mean = f2_means, f2_std = f2_stds,
        f3_mean = f3_means, f3_std = f3_stds,
        mobility_raw_mean = mob_means, mobility_raw_std = mob_stds,
        dissatisfaction_raw_mean = dissat_means, dissatisfaction_raw_std = dissat_stds,
        total_customers = cust_counts, locker_customers = locker_counts,
        total_distance_mean = dist_means, total_distance_std = dist_stds,
        vehicles_mean = veh_means, vehicles_std = veh_stds,
        d2d_conversions_mean = conv_means,
        pareto_solutions_mean = pareto_means
    )
    CSV.write(joinpath(output_dir, "viz_scenario_summary.csv"), viz_summary)
    println("✅ 시각화 데이터 저장: viz_scenario_summary.csv")

    pareto_rows = NamedTuple[]
    for scn in scn_ids
        results = scenario_groups[scn]
        scn_name = results[1].scenario_name
        for r in results
            pf = r.pareto_front
            for sol_idx in 1:size(pf, 1)
                push!(pareto_rows, (
                    scenario = scn, scenario_name = scn_name, omega = r.omega,
                    f1 = pf[sol_idx, 1], f2 = pf[sol_idx, 2], f3 = pf[sol_idx, 3],
                    is_selected = (sol_idx == r.selected_solution_idx),
                    compromise_f1 = r.f1_enterprise,
                    compromise_f3 = r.f3_society
                ))
            end
        end
    end
    CSV.write(joinpath(output_dir, "viz_pareto_points.csv"), DataFrame(pareto_rows))
    println("✅ Pareto front 데이터 저장: viz_pareto_points.csv")

    n = length(scn_ids)
    x = 1:n
    colors_bar = [:royalblue, :forestgreen, :crimson, :darkorange, :purple]

    # ──── Figure 1: Compromise Solution Comparison (f1, f2, f3) ────
    p1 = plot(layout=(1, 3), size=(1200, 420), dpi=300,
              margin=6Plots.mm, bottom_margin=14Plots.mm, top_margin=8Plots.mm,
              titlefontsize=11, guidefontsize=10, tickfontsize=9, legendfontsize=8)

    bar!(p1[1], x, f1_means, yerr=f1_stds, bar_width=0.6,
         fillcolor=permutedims([colors_bar[mod1(i,5)] for i in x]),
         linecolor=permutedims([colors_bar[mod1(i,5)] for i in x]),
         alpha=0.85, label=false)
    for i in 1:n
        annotate!(p1[1], [(i, f1_means[i] + f1_stds[i] + maximum(f1_means)*0.03,
                  text(@sprintf("%.1f", f1_means[i]), 8, :center))])
    end
    xticks!(p1[1], x, scn_names)
    ylabel!(p1[1], "EUR")
    title!(p1[1], "(a) f₁: Enterprise Cost")

    bar!(p1[2], x, f2_means, yerr=f2_stds, bar_width=0.6,
         fillcolor=permutedims([colors_bar[mod1(i,5)] for i in x]),
         linecolor=permutedims([colors_bar[mod1(i,5)] for i in x]),
         alpha=0.85, label=false)
    for i in 1:n
        annotate!(p1[2], [(i, f2_means[i] + f2_stds[i] + maximum(f2_means)*0.03,
                  text(@sprintf("%.3f", f2_means[i]), 8, :center))])
    end
    xticks!(p1[2], x, scn_names)
    ylabel!(p1[2], "Normalized")
    title!(p1[2], "(b) f₂: Customer Disutility")

    bar!(p1[3], x, f3_means, yerr=f3_stds, bar_width=0.6,
         fillcolor=permutedims([colors_bar[mod1(i,5)] for i in x]),
         linecolor=permutedims([colors_bar[mod1(i,5)] for i in x]),
         alpha=0.85, label=false)
    for i in 1:n
        annotate!(p1[3], [(i, f3_means[i] + f3_stds[i] + maximum(f3_means)*0.03,
                  text(@sprintf("%.1f", f3_means[i]), 8, :center))])
    end
    xticks!(p1[3], x, scn_names)
    ylabel!(p1[3], "kg CO₂")
    title!(p1[3], "(c) f₃: Social Cost")

    plot!(p1, plot_title="Scenario Comparison — Compromise Solutions (μ ± σ over Nω)", plot_titlefontsize=13)

    savefig(p1, joinpath(output_dir, "fig_scenario_comparison.png"))
    println("✅ 시나리오 비교 그래프 저장: fig_scenario_comparison.png")

    # ──── Figure 2: Raw Values (mobility, dissatisfaction) ────
    p2 = plot(layout=(1, 2), size=(900, 420), dpi=300,
              margin=6Plots.mm, bottom_margin=14Plots.mm, top_margin=8Plots.mm,
              titlefontsize=11, guidefontsize=10, tickfontsize=9, legendfontsize=8)

    bar!(p2[1], x, mob_means, bar_width=0.6,
         fillcolor=permutedims([colors_bar[mod1(i,5)] for i in x]),
         linecolor=permutedims([colors_bar[mod1(i,5)] for i in x]),
         alpha=0.85, label=false)
    for i in 1:n
        annotate!(p2[1], [(i, mob_means[i] + maximum(mob_means)*0.03,
                  text(@sprintf("%.2f", mob_means[i]), 8, :center))])
    end
    xticks!(p2[1], x, scn_names)
    ylabel!(p2[1], "km")
    title!(p2[1], "(a) Mobility Raw (avg locker→customer)")

    bar!(p2[2], x, dissat_means, bar_width=0.6,
         fillcolor=permutedims([colors_bar[mod1(i,5)] for i in x]),
         linecolor=permutedims([colors_bar[mod1(i,5)] for i in x]),
         alpha=0.85, label=false)
    for i in 1:n
        annotate!(p2[2], [(i, dissat_means[i] + maximum(dissat_means)*0.03,
                  text(@sprintf("%.3f", dissat_means[i]), 8, :center))])
    end
    xticks!(p2[2], x, scn_names)
    ylabel!(p2[2], "Dissatisfaction")
    title!(p2[2], "(b) D2D Dissatisfaction Raw")

    plot!(p2, plot_title="Raw Objective Components (Pre-Normalization)", plot_titlefontsize=13)

    savefig(p2, joinpath(output_dir, "fig_raw_values.png"))
    println("✅ 원시값 비교 그래프 저장: fig_raw_values.png")

    # ──── Figure 3: All-Scenario 2D Pareto (f1 vs f3) ────
    p3 = plot(size=(700, 500), dpi=300,
              margin=6Plots.mm, bottom_margin=10Plots.mm,
              titlefontsize=12, guidefontsize=10, tickfontsize=9, legendfontsize=9)

    for (idx, scn) in enumerate(scn_ids)
        results = scenario_groups[scn]
        scn_name = results[1].scenario_name
        all_f1 = Float64[]; all_f3 = Float64[]
        sel_f1 = Float64[]; sel_f3 = Float64[]
        for r in results
            append!(all_f1, r.pareto_front[:, 1])
            append!(all_f3, r.pareto_front[:, 3])
            push!(sel_f1, r.f1_enterprise)
            push!(sel_f3, r.f3_society)
        end

        scatter!(p3, all_f1, all_f3,
                 label=scn_name, color=colors_bar[mod1(idx, 5)],
                 markersize=3, alpha=0.4, markerstrokewidth=0)
        scatter!(p3, [mean(sel_f1)], [mean(sel_f3)],
                 label=false, color=colors_bar[mod1(idx, 5)],
                 markersize=10, markershape=:star5,
                 markerstrokewidth=1.5, markerstrokecolor=:black)
        annotate!(p3, [(mean(sel_f1)+0.5, mean(sel_f3),
                  text(scn_name, 8, colors_bar[mod1(idx, 5)], :left))])
    end

    xlabel!(p3, "f₁: Enterprise Cost (EUR)")
    ylabel!(p3, "f₃: Social Cost (kg CO₂)")
    title!(p3, "Pareto Front Comparison — All Scenarios (★ = compromise mean)")

    savefig(p3, joinpath(output_dir, "fig_pareto_f1_f3_comparison.png"))
    println("✅ Pareto front 비교 그래프 저장: fig_pareto_f1_f3_comparison.png")
end

# ═══════════════════════════════════════════════════════════════════════════
# Comprehensive Main Results Report
# ═══════════════════════════════════════════════════════════════════════════

function generate_main_results_report(moo_results::Vector{MOOScenarioResult}, output_dir::String)
    if isempty(moo_results)
        return
    end

    report_path = joinpath(output_dir, "main_results_report.md")

    scenario_groups = Dict{Int, Vector{MOOScenarioResult}}()
    for r in moo_results
        push!(get!(scenario_groups, r.scenario, MOOScenarioResult[]), r)
    end
    scn_ids = sort(collect(keys(scenario_groups)))

    io = IOBuffer()
    println(io, "# Main Simulation Results Report")
    println(io, "")
    println(io, "**Generated**: $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))")
    println(io, "**Model**: Multi-Trip CVRP Budapest — NSGA-II-HI Multi-Objective Optimization")
    println(io, "**Scenarios**: $(length(scn_ids)) evaluated")
    println(io, "")
    println(io, "---")
    println(io, "")

    # ──── 1. Scenario Overview ────
    println(io, "## 1. Scenario Overview")
    println(io, "")
    println(io, "| ID | Scenario | Description |")
    println(io, "|----|----------|-------------|")
    scn_descriptions = Dict(
        1 => "Door-to-Door only (no lockers)",
        2 => "Dedicated Private Lockers (carrier-exclusive)",
        3 => "Shared Private Lockers (all carriers share)",
        4 => "Optimized Public Lockers (SLRP-based placement)",
        5 => "Partially Shared Private Lockers (brand-group sharing)"
    )
    for scn in scn_ids
        name = scenario_groups[scn][1].scenario_name
        desc = get(scn_descriptions, scn, "—")
        println(io, "| $scn | $name | $desc |")
    end
    println(io, "")

    # ──── 2. Compromise Solution Comparison ────
    println(io, "## 2. Compromise Solution Comparison")
    println(io, "")
    println(io, "Average compromise solution values across Nω Monte Carlo replications (μ ± σ):")
    println(io, "")
    println(io, "| Scenario | f₁ Enterprise (EUR) | f₂ Customer Disutility | f₃ Social (kg CO₂) | Nω |")
    println(io, "|----------|--------------------|-----------------------|--------------------|----|")

    best_f1 = (Inf, ""); best_f2 = (Inf, ""); best_f3 = (Inf, "")
    scn_stats = Dict{Int, NamedTuple}()

    for scn in scn_ids
        results = scenario_groups[scn]
        name = results[1].scenario_name
        n_omega = length(results)

        f1v = [r.f1_enterprise for r in results]
        f2v = [r.f2_customer for r in results]
        f3v = [r.f3_society for r in results]

        f1m, f1s = mean(f1v), std(f1v)
        f2m, f2s = mean(f2v), std(f2v)
        f3m, f3s = mean(f3v), std(f3v)

        scn_stats[scn] = (f1m=f1m, f1s=f1s, f2m=f2m, f2s=f2s, f3m=f3m, f3s=f3s, name=name, n_omega=n_omega)

        @printf(io, "| %s | %.2f ± %.2f | %.4f ± %.4f | %.2f ± %.2f | %d |\n",
                name, f1m, f1s, f2m, f2s, f3m, f3s, n_omega)

        if f1m < best_f1[1] best_f1 = (f1m, name) end
        if f2m < best_f2[1] best_f2 = (f2m, name) end
        if f3m < best_f3[1] best_f3 = (f3m, name) end
    end
    println(io, "")

    println(io, "**Best per objective**:")
    println(io, "- f₁ (lowest enterprise cost): **$(best_f1[2])** ($(@sprintf("%.2f", best_f1[1])) EUR)")
    println(io, "- f₂ (lowest customer disutility): **$(best_f2[2])** ($(@sprintf("%.4f", best_f2[1])))")
    println(io, "- f₃ (lowest social cost): **$(best_f3[2])** ($(@sprintf("%.2f", best_f3[1])) kg CO₂)")
    println(io, "")

    # Compromise winner (lowest Euclidean distance from ideal)
    if length(scn_stats) > 1
        ideal_f1 = minimum(s.f1m for s in values(scn_stats))
        ideal_f2 = minimum(s.f2m for s in values(scn_stats))
        ideal_f3 = minimum(s.f3m for s in values(scn_stats))
        range_f1 = maximum(s.f1m for s in values(scn_stats)) - ideal_f1
        range_f2 = maximum(s.f2m for s in values(scn_stats)) - ideal_f2
        range_f3 = maximum(s.f3m for s in values(scn_stats)) - ideal_f3

        best_dist = Inf; best_name = ""
        println(io, "**Normalized Euclidean distance to ideal point**:")
        println(io, "")
        println(io, "| Scenario | d(ideal) |")
        println(io, "|----------|----------|")
        for scn in scn_ids
            s = scn_stats[scn]
            d1 = range_f1 > 0 ? (s.f1m - ideal_f1) / range_f1 : 0.0
            d2 = range_f2 > 0 ? (s.f2m - ideal_f2) / range_f2 : 0.0
            d3 = range_f3 > 0 ? (s.f3m - ideal_f3) / range_f3 : 0.0
            d = sqrt(d1^2 + d2^2 + d3^2)
            @printf(io, "| %s | %.4f |\n", s.name, d)
            if d < best_dist best_dist = d; best_name = s.name end
        end
        println(io, "")
        println(io, "**Overall compromise winner**: **$(best_name)** (d = $(@sprintf("%.4f", best_dist)))")
    end
    println(io, "")

    println(io, "![Scenario Comparison](fig_scenario_comparison.png)")
    println(io, "")
    println(io, "---")
    println(io, "")

    # ──── 3. Raw Values ────
    println(io, "## 3. Raw Objective Components (Pre-Normalization)")
    println(io, "")
    println(io, "| Scenario | Mobility Raw (km) | Dissatisfaction Raw | Locker Customers | D2D Customers | Total |")
    println(io, "|----------|-------------------|--------------------|-----------------:|--------------:|------:|")

    for scn in scn_ids
        results = scenario_groups[scn]
        name = results[1].scenario_name
        mob = mean([r.mobility_raw_km for r in results])
        dis = mean([r.dissatisfaction_raw for r in results])
        lck = mean([r.actual_locker_customers for r in results])
        tot = mean([r.total_customers for r in results])
        d2d = tot - lck
        @printf(io, "| %s | %.3f | %.4f | %.0f | %.0f | %.0f |\n",
                name, mob, dis, lck, d2d, tot)
    end
    println(io, "")
    println(io, "![Raw Values](fig_raw_values.png)")
    println(io, "")
    println(io, "---")
    println(io, "")

    # ──── 4. Operational Metrics ────
    println(io, "## 4. Operational Metrics")
    println(io, "")
    println(io, "| Scenario | Total Distance (km) | Vehicles Used | D2D Conversions | Pareto Solutions |")
    println(io, "|----------|--------------------:|--------------:|----------------:|-----------------:|")

    for scn in scn_ids
        results = scenario_groups[scn]
        name = results[1].scenario_name
        dist = mean([r.total_distance_km for r in results])
        veh = mean([r.vehicles_used for r in results])
        conv = mean([r.d2d_conversions for r in results])
        pf = mean([size(r.pareto_front, 1) for r in results])
        @printf(io, "| %s | %.1f | %.1f | %.1f | %.1f |\n", name, dist, veh, conv, pf)
    end
    println(io, "")
    println(io, "---")
    println(io, "")

    # ──── 5. Pareto Front Analysis ────
    println(io, "## 5. Pareto Front Analysis")
    println(io, "")
    println(io, "![Pareto Comparison](fig_pareto_f1_f3_comparison.png)")
    println(io, "")

    # ──── 6. Interpretation ────
    println(io, "## 6. Key Findings")
    println(io, "")

    if length(scn_stats) > 1
        f1_sorted = sort(collect(scn_stats), by=x->x.second.f1m)
        f3_sorted = sort(collect(scn_stats), by=x->x.second.f3m)

        println(io, "1. **Enterprise cost ranking** (f₁, low → high): ",
                join([s.second.name for s in f1_sorted], " < "))
        println(io, "2. **Social cost ranking** (f₃, low → high): ",
                join([s.second.name for s in f3_sorted], " < "))

        d2d_scn = get(scn_stats, 1, nothing)
        if d2d_scn !== nothing
            for scn in scn_ids
                if scn == 1 continue end
                s = scn_stats[scn]
                f1_change = (s.f1m - d2d_scn.f1m) / abs(d2d_scn.f1m) * 100
                f3_change = (s.f3m - d2d_scn.f3m) / abs(d2d_scn.f3m) * 100
                @printf(io, "3. **%s vs D2D**: f₁ %+.1f%%, f₃ %+.1f%%\n", s.name, f1_change, f3_change)
            end
        end
    end

    println(io, "")
    println(io, "---")
    println(io, "*Report generated by moo_pareto_visualization.jl*")

    open(report_path, "w") do f
        write(f, String(take!(io)))
    end
    println("✅ 종합 결과 보고서 저장: main_results_report.md")
end

# ═══════════════════════════════════════════════════════════════════════════
# Standalone Regeneration from Saved CSVs (no re-run needed)
# ═══════════════════════════════════════════════════════════════════════════

"""
    regenerate_main_figures(output_dir::String)

Re-generate all main result figures and report from previously saved CSVs.
No simulation re-run needed — reads viz_scenario_summary.csv and viz_pareto_points.csv.

Usage:
    regenerate_main_figures("/Users/.../Desktop/runs/visualizations")
"""
function regenerate_main_figures(output_dir::String)
    summary_csv = joinpath(output_dir, "viz_scenario_summary.csv")
    pareto_csv = joinpath(output_dir, "viz_pareto_points.csv")

    if !isfile(summary_csv)
        println("❌ $summary_csv 파일이 없습니다. 먼저 전체 파이프라인을 실행하세요.")
        return
    end

    println("🔄 저장된 CSV로부터 시각화 재생성 중...")

    df_summary = CSV.read(summary_csv, DataFrame)

    scn_names = df_summary.scenario_name
    n = nrow(df_summary)
    x = 1:n
    colors_bar = [:royalblue, :forestgreen, :crimson, :darkorange, :purple]

    # ──── Figure 1: Scenario Comparison (f1, f2, f3) ────
    p1 = plot(layout=(1, 3), size=(1200, 420), dpi=300,
              margin=6Plots.mm, bottom_margin=14Plots.mm, top_margin=8Plots.mm,
              titlefontsize=11, guidefontsize=10, tickfontsize=9, legendfontsize=8)

    bar!(p1[1], x, df_summary.f1_mean, yerr=df_summary.f1_std, bar_width=0.6,
         fillcolor=permutedims([colors_bar[mod1(i,5)] for i in x]),
         linecolor=permutedims([colors_bar[mod1(i,5)] for i in x]),
         alpha=0.85, label=false)
    for i in 1:n
        annotate!(p1[1], [(i, df_summary.f1_mean[i] + df_summary.f1_std[i] + maximum(df_summary.f1_mean)*0.03,
                  text(@sprintf("%.1f", df_summary.f1_mean[i]), 8, :center))])
    end
    xticks!(p1[1], x, scn_names); ylabel!(p1[1], "EUR"); title!(p1[1], "(a) f₁: Enterprise Cost")

    bar!(p1[2], x, df_summary.f2_mean, yerr=df_summary.f2_std, bar_width=0.6,
         fillcolor=permutedims([colors_bar[mod1(i,5)] for i in x]),
         linecolor=permutedims([colors_bar[mod1(i,5)] for i in x]),
         alpha=0.85, label=false)
    for i in 1:n
        annotate!(p1[2], [(i, df_summary.f2_mean[i] + df_summary.f2_std[i] + maximum(df_summary.f2_mean)*0.03,
                  text(@sprintf("%.3f", df_summary.f2_mean[i]), 8, :center))])
    end
    xticks!(p1[2], x, scn_names); ylabel!(p1[2], "Normalized"); title!(p1[2], "(b) f₂: Customer Disutility")

    bar!(p1[3], x, df_summary.f3_mean, yerr=df_summary.f3_std, bar_width=0.6,
         fillcolor=permutedims([colors_bar[mod1(i,5)] for i in x]),
         linecolor=permutedims([colors_bar[mod1(i,5)] for i in x]),
         alpha=0.85, label=false)
    for i in 1:n
        annotate!(p1[3], [(i, df_summary.f3_mean[i] + df_summary.f3_std[i] + maximum(df_summary.f3_mean)*0.03,
                  text(@sprintf("%.1f", df_summary.f3_mean[i]), 8, :center))])
    end
    xticks!(p1[3], x, scn_names); ylabel!(p1[3], "kg CO₂"); title!(p1[3], "(c) f₃: Social Cost")

    plot!(p1, plot_title="Scenario Comparison — Compromise Solutions (μ ± σ over Nω)", plot_titlefontsize=13)
    savefig(p1, joinpath(output_dir, "fig_scenario_comparison.png"))
    println("  ✅ fig_scenario_comparison.png")

    # ──── Figure 2: Raw Values ────
    p2 = plot(layout=(1, 2), size=(900, 420), dpi=300,
              margin=6Plots.mm, bottom_margin=14Plots.mm, top_margin=8Plots.mm,
              titlefontsize=11, guidefontsize=10, tickfontsize=9, legendfontsize=8)

    bar!(p2[1], x, df_summary.mobility_raw_mean, bar_width=0.6,
         fillcolor=permutedims([colors_bar[mod1(i,5)] for i in x]),
         linecolor=permutedims([colors_bar[mod1(i,5)] for i in x]),
         alpha=0.85, label=false)
    for i in 1:n
        annotate!(p2[1], [(i, df_summary.mobility_raw_mean[i] + maximum(df_summary.mobility_raw_mean)*0.03,
                  text(@sprintf("%.2f", df_summary.mobility_raw_mean[i]), 8, :center))])
    end
    xticks!(p2[1], x, scn_names); ylabel!(p2[1], "km"); title!(p2[1], "(a) Mobility Raw (avg locker→customer)")

    bar!(p2[2], x, df_summary.dissatisfaction_raw_mean, bar_width=0.6,
         fillcolor=permutedims([colors_bar[mod1(i,5)] for i in x]),
         linecolor=permutedims([colors_bar[mod1(i,5)] for i in x]),
         alpha=0.85, label=false)
    for i in 1:n
        annotate!(p2[2], [(i, df_summary.dissatisfaction_raw_mean[i] + maximum(df_summary.dissatisfaction_raw_mean)*0.03,
                  text(@sprintf("%.3f", df_summary.dissatisfaction_raw_mean[i]), 8, :center))])
    end
    xticks!(p2[2], x, scn_names); ylabel!(p2[2], "Dissatisfaction"); title!(p2[2], "(b) D2D Dissatisfaction Raw")
    plot!(p2, plot_title="Raw Objective Components (Pre-Normalization)", plot_titlefontsize=13)
    savefig(p2, joinpath(output_dir, "fig_raw_values.png"))
    println("  ✅ fig_raw_values.png")

    # ──── Figure 3: Pareto front comparison ────
    if isfile(pareto_csv)
        df_pf = CSV.read(pareto_csv, DataFrame)
        p3 = plot(size=(700, 500), dpi=300,
                  margin=6Plots.mm, bottom_margin=10Plots.mm,
                  titlefontsize=12, guidefontsize=10, tickfontsize=9, legendfontsize=9)

        scenarios_pf = sort(unique(df_pf.scenario))
        for (idx, scn) in enumerate(scenarios_pf)
            sub = df_pf[df_pf.scenario .== scn, :]
            scn_name = first(sub.scenario_name)
            c = colors_bar[mod1(idx, 5)]

            scatter!(p3, sub.f1, sub.f3,
                     label=scn_name, color=c, markersize=3, alpha=0.4, markerstrokewidth=0)
            sel = sub[sub.is_selected .== true, :]
            if nrow(sel) > 0
                scatter!(p3, [mean(sel.compromise_f1)], [mean(sel.compromise_f3)],
                         label=false, color=c, markersize=10, markershape=:star5,
                         markerstrokewidth=1.5, markerstrokecolor=:black)
                annotate!(p3, [(mean(sel.compromise_f1)+0.5, mean(sel.compromise_f3),
                          text(scn_name, 8, c, :left))])
            end
        end
        xlabel!(p3, "f₁: Enterprise Cost (EUR)"); ylabel!(p3, "f₃: Social Cost (kg CO₂)")
        title!(p3, "Pareto Front Comparison — All Scenarios (★ = compromise mean)")
        savefig(p3, joinpath(output_dir, "fig_pareto_f1_f3_comparison.png"))
        println("  ✅ fig_pareto_f1_f3_comparison.png")
    end

    # ──── Report ────
    regenerate_main_report(output_dir, df_summary)

    println("🔄 메인 시각화 재생성 완료!")
end

function regenerate_main_report(output_dir::String, df::DataFrame)
    report_path = joinpath(output_dir, "main_results_report.md")

    scn_descriptions = Dict(
        1 => "Door-to-Door only (no lockers)",
        2 => "Dedicated Private Lockers (carrier-exclusive)",
        3 => "Shared Private Lockers (all carriers share)",
        4 => "Optimized Public Lockers (SLRP-based placement)",
        5 => "Partially Shared Private Lockers (brand-group sharing)"
    )

    io = IOBuffer()
    println(io, "# Main Simulation Results Report")
    println(io, "")
    println(io, "**Generated**: $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))")
    println(io, "**Model**: Multi-Trip CVRP Budapest — NSGA-II-HI Multi-Objective Optimization")
    println(io, "**Scenarios**: $(nrow(df)) evaluated")
    println(io, "")
    println(io, "---")
    println(io, "")

    println(io, "## 1. Scenario Overview")
    println(io, "")
    println(io, "| ID | Scenario | Description |")
    println(io, "|----|----------|-------------|")
    for r in eachrow(df)
        desc = get(scn_descriptions, r.scenario, "—")
        println(io, "| $(r.scenario) | $(r.scenario_name) | $desc |")
    end
    println(io, "")

    println(io, "## 2. Compromise Solution Comparison")
    println(io, "")
    println(io, "Average compromise solution values across Nω Monte Carlo replications (μ ± σ):")
    println(io, "")
    println(io, "| Scenario | f₁ Enterprise (EUR) | f₂ Customer Disutility | f₃ Social (kg CO₂) | Nω |")
    println(io, "|----------|--------------------|-----------------------|--------------------|----|")

    best_f1 = (Inf, ""); best_f2 = (Inf, ""); best_f3 = (Inf, "")
    for r in eachrow(df)
        @printf(io, "| %s | %.2f ± %.2f | %.4f ± %.4f | %.2f ± %.2f | %d |\n",
                r.scenario_name, r.f1_mean, r.f1_std, r.f2_mean, r.f2_std,
                r.f3_mean, r.f3_std, r.n_omega)
        if r.f1_mean < best_f1[1] best_f1 = (r.f1_mean, r.scenario_name) end
        if r.f2_mean < best_f2[1] best_f2 = (r.f2_mean, r.scenario_name) end
        if r.f3_mean < best_f3[1] best_f3 = (r.f3_mean, r.scenario_name) end
    end
    println(io, "")
    println(io, "**Best per objective**:")
    println(io, "- f₁ (lowest enterprise cost): **$(best_f1[2])** ($(@sprintf("%.2f", best_f1[1])) EUR)")
    println(io, "- f₂ (lowest customer disutility): **$(best_f2[2])** ($(@sprintf("%.4f", best_f2[1])))")
    println(io, "- f₃ (lowest social cost): **$(best_f3[2])** ($(@sprintf("%.2f", best_f3[1])) kg CO₂)")
    println(io, "")

    if nrow(df) > 1
        ideal_f1 = minimum(df.f1_mean); ideal_f2 = minimum(df.f2_mean); ideal_f3 = minimum(df.f3_mean)
        range_f1 = maximum(df.f1_mean) - ideal_f1
        range_f2 = maximum(df.f2_mean) - ideal_f2
        range_f3 = maximum(df.f3_mean) - ideal_f3

        best_dist = Inf; best_name = ""
        println(io, "**Normalized Euclidean distance to ideal point**:")
        println(io, "")
        println(io, "| Scenario | d(ideal) |")
        println(io, "|----------|----------|")
        for r in eachrow(df)
            d1 = range_f1 > 0 ? (r.f1_mean - ideal_f1) / range_f1 : 0.0
            d2 = range_f2 > 0 ? (r.f2_mean - ideal_f2) / range_f2 : 0.0
            d3 = range_f3 > 0 ? (r.f3_mean - ideal_f3) / range_f3 : 0.0
            d = sqrt(d1^2 + d2^2 + d3^2)
            @printf(io, "| %s | %.4f |\n", r.scenario_name, d)
            if d < best_dist best_dist = d; best_name = r.scenario_name end
        end
        println(io, "")
        println(io, "**Overall compromise winner**: **$(best_name)** (d = $(@sprintf("%.4f", best_dist)))")
    end
    println(io, "")
    println(io, "![Scenario Comparison](fig_scenario_comparison.png)")
    println(io, "")
    println(io, "---")
    println(io, "")

    println(io, "## 3. Raw Objective Components (Pre-Normalization)")
    println(io, "")
    println(io, "| Scenario | Mobility Raw (km) | Dissatisfaction Raw | Locker Customers | D2D Customers | Total |")
    println(io, "|----------|-------------------|--------------------|-----------------:|--------------:|------:|")
    for r in eachrow(df)
        d2d = r.total_customers - r.locker_customers
        @printf(io, "| %s | %.3f | %.4f | %.0f | %.0f | %.0f |\n",
                r.scenario_name, r.mobility_raw_mean, r.dissatisfaction_raw_mean,
                r.locker_customers, d2d, r.total_customers)
    end
    println(io, "")
    println(io, "![Raw Values](fig_raw_values.png)")
    println(io, "")
    println(io, "---")
    println(io, "")

    println(io, "## 4. Operational Metrics")
    println(io, "")
    println(io, "| Scenario | Total Distance (km) | Vehicles Used | D2D Conversions | Pareto Solutions |")
    println(io, "|----------|--------------------:|--------------:|----------------:|-----------------:|")
    for r in eachrow(df)
        @printf(io, "| %s | %.1f | %.1f | %.1f | %.1f |\n",
                r.scenario_name, r.total_distance_mean, r.vehicles_mean,
                r.d2d_conversions_mean, r.pareto_solutions_mean)
    end
    println(io, "")
    println(io, "---")
    println(io, "")

    println(io, "## 5. Pareto Front Analysis")
    println(io, "")
    println(io, "![Pareto Comparison](fig_pareto_f1_f3_comparison.png)")
    println(io, "")

    println(io, "## 6. Key Findings")
    println(io, "")
    if nrow(df) > 1
        f1_sorted = sort(collect(eachrow(df)), by=r->r.f1_mean)
        f3_sorted = sort(collect(eachrow(df)), by=r->r.f3_mean)
        println(io, "1. **Enterprise cost ranking** (f₁, low → high): ",
                join([r.scenario_name for r in f1_sorted], " < "))
        println(io, "2. **Social cost ranking** (f₃, low → high): ",
                join([r.scenario_name for r in f3_sorted], " < "))

        d2d_row = df[df.scenario .== 1, :]
        if nrow(d2d_row) > 0
            d2d = first(d2d_row)
            for r in eachrow(df)
                if r.scenario == 1 continue end
                f1c = (r.f1_mean - d2d.f1_mean) / abs(d2d.f1_mean) * 100
                f3c = (r.f3_mean - d2d.f3_mean) / abs(d2d.f3_mean) * 100
                @printf(io, "3. **%s vs D2D**: f₁ %+.1f%%, f₃ %+.1f%%\n", r.scenario_name, f1c, f3c)
            end
        end
    end
    println(io, "")
    println(io, "---")
    println(io, "*Report generated by moo_pareto_visualization.jl*")

    open(report_path, "w") do f
        write(f, String(take!(io)))
    end
    println("  ✅ main_results_report.md")
end
