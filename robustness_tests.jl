#!/usr/bin/env julia
# ═══════════════════════════════════════════════════════════════════════════
# robustness_tests.jl
#
# Robustness Testing Suite for Multi-Trip CVRP Budapest
# - Test 1: SAA Convergence Analysis
# - Test 2: Out-of-Sample Validation
# - Test 3: Algorithmic Stability (Hypervolume-based)
# - Test 4: Parameter Sensitivity Analysis
#
# Usage:
#   (Integrated) julia moo_multitrip_cvrp_budapest.jl  # Pipeline + robustness
#   (Standalone)  julia robustness_tests.jl             # Robustness only
#   ROBUSTNESS_TEST=saa julia robustness_tests.jl       # Specific test only
#   SKIP_ROBUSTNESS=1 julia moo_multitrip_cvrp_budapest.jl  # Pipeline only
# ═══════════════════════════════════════════════════════════════════════════

using Distributed
using Random
using Statistics
using DataFrames
using CSV
using Printf
using Dates
using HypothesisTests
using StatsPlots

# Guard: skip include if already loaded (when called from main pipeline)
if !@isdefined(SCENARIO_D2D)
    include(joinpath(@__DIR__, "moo_multitrip_cvrp_budapest.jl"))
end

const ROBUSTNESS_OUTDIR = joinpath(expanduser("~/Desktop/runs"), "robustness_results")

# ═══════════════════════════════════════════════════════════════════════════
# Hypervolume Computation (3D, WFG-based)
# ═══════════════════════════════════════════════════════════════════════════

"""
    compute_hypervolume_3d(pareto_front::Matrix{Float64}, ref_point::Vector{Float64})

Compute the exact hypervolume indicator for a 3D Pareto front.
Uses the slice-based algorithm (HSO) for 3 objectives.

All objectives are assumed to be minimized. Points dominating the reference
point contribute to the hypervolume; others are excluded.
"""
function compute_hypervolume_3d(pareto_front::Matrix{Float64}, ref_point::Vector{Float64})
    @assert length(ref_point) == 3 "Reference point must be 3-dimensional"
    n = size(pareto_front, 1)
    if n == 0
        return 0.0
    end
    
    # Filter points that are dominated by the reference point (valid contributors)
    valid_mask = [all(pareto_front[i, :] .< ref_point) for i in 1:n]
    pf = pareto_front[valid_mask, :]
    n = size(pf, 1)
    if n == 0
        return 0.0
    end
    
    # Sort by first objective (ascending)
    order = sortperm(pf[:, 1])
    pf = pf[order, :]
    
    hv = 0.0
    
    for i in 1:n
        # For each point, compute the volume of its exclusive "slab" in f1 direction
        x_lo = pf[i, 1]
        x_hi = (i < n) ? pf[i+1, 1] : ref_point[1]
        
        if x_hi <= x_lo
            continue
        end
        
        # In this slab, compute the 2D hypervolume of all points with f1 <= x_lo
        # (i.e., points 1..i projected onto f2-f3 plane)
        points_2d = pf[1:i, 2:3]
        hv2d = compute_hypervolume_2d(points_2d, ref_point[2:3])
        
        hv += (x_hi - x_lo) * hv2d
    end
    
    return hv
end

"""
    compute_hypervolume_2d(points::Matrix{Float64}, ref::Vector{Float64})

Exact 2D hypervolume computation using the sweep-line algorithm.
"""
function compute_hypervolume_2d(points::Matrix{Float64}, ref::Vector{Float64})
    n = size(points, 1)
    if n == 0
        return 0.0
    end
    
    # Filter dominated points (only keep non-dominated in 2D)
    valid = [all(points[i, :] .< ref) for i in 1:n]
    pts = points[valid, :]
    n = size(pts, 1)
    if n == 0
        return 0.0
    end
    
    # Sort by f2 ascending
    order = sortperm(pts[:, 1])
    pts = pts[order, :]
    
    # Remove dominated points in 2D
    nd_pts = Matrix{Float64}(undef, 0, 2)
    min_f3 = ref[2]
    for i in 1:n
        if pts[i, 2] < min_f3
            nd_pts = vcat(nd_pts, pts[i, :]')
            min_f3 = pts[i, 2]
        end
    end
    
    n = size(nd_pts, 1)
    if n == 0
        return 0.0
    end
    
    # Sweep-line: area under the staircase
    hv = 0.0
    for i in 1:n
        y_hi = (i < n) ? nd_pts[i+1, 1] : ref[1]
        z_depth = ref[2] - nd_pts[i, 2]
        hv += (y_hi - nd_pts[i, 1]) * z_depth
    end
    
    return hv
end

"""
    compute_reference_point(all_fronts::Vector{Matrix{Float64}}; margin::Float64=1.1)

Compute a common reference point from all Pareto fronts (max * margin per objective).
"""
function compute_reference_point(all_fronts::Vector{Matrix{Float64}}; margin::Float64=1.1)
    ref = fill(-Inf, 3)
    for pf in all_fronts
        for j in 1:3
            ref[j] = max(ref[j], maximum(pf[:, j]))
        end
    end
    return ref .* margin
end

"""
    compute_igd_plus(pareto_front::Matrix{Float64}, reference_front::Matrix{Float64})

Compute the Inverted Generational Distance Plus (IGD+) indicator.
IGD+ uses the modified distance that only penalizes points worse than the reference.
"""
function compute_igd_plus(pareto_front::Matrix{Float64}, reference_front::Matrix{Float64})
    n_ref = size(reference_front, 1)
    if n_ref == 0 || size(pareto_front, 1) == 0
        return Inf
    end
    
    total_dist = 0.0
    for i in 1:n_ref
        min_dist = Inf
        for j in 1:size(pareto_front, 1)
            d_plus = 0.0
            for k in 1:size(reference_front, 2)
                d_plus += max(0.0, pareto_front[j, k] - reference_front[i, k])^2
            end
            min_dist = min(min_dist, sqrt(d_plus))
        end
        total_dist += min_dist
    end
    
    return total_dist / n_ref
end

# ═══════════════════════════════════════════════════════════════════════════
# Utility: Generate omega scenarios with deterministic seeding
# ═══════════════════════════════════════════════════════════════════════════

"""
    generate_omega_scenarios(n_omega; master_seed=12345, customer_scale=1.0)

Generate `n_omega` customer scenarios with deterministic seeding.
`customer_scale` multiplies region-level customer counts (e.g., 0.7 = −30%, 1.3 = +30%).
Nested sampling property: the first k scenarios from n_omega=N are identical
to the first k scenarios from n_omega=M (M > N) when using the same master_seed and scale.
"""
function generate_omega_scenarios(n_omega::Int; master_seed::Int=12345, customer_scale::Float64=1.0)
    omega_customers = Vector{Vector{NamedTuple}}()
    omega_attrs = Vector{DataFrame}()
    
    # Temporarily scale customer_counts if needed
    original_counts = Dict{Int, Dict{String,Int}}()
    if customer_scale != 1.0
        for (rid, delivs) in customer_counts
            original_counts[rid] = copy(delivs)
            for dtype in keys(delivs)
                delivs[dtype] = max(1, round(Int, delivs[dtype] * customer_scale))
            end
        end
    end
    
    try
        for omega in 1:n_omega
            rng = MersenneTwister(master_seed + omega)
            customers = gen_customers(; rng=rng)
            
            if CUSTOMER_LIMIT > 0
                customers = customers[1:min(end, CUSTOMER_LIMIT)]
            end
            
            attrs = gen_attr(customers; rng=rng)
            push!(omega_customers, customers)
            push!(omega_attrs, attrs)
        end
    finally
        if customer_scale != 1.0
            for (rid, delivs) in customer_counts
                for (dtype, cnt) in original_counts[rid]
                    delivs[dtype] = cnt
                end
            end
        end
    end
    
    return omega_customers, omega_attrs
end

"""
    run_scenario_and_collect(scn::Int, omega_customers::Vector, omega_attrs::Vector;
                             moo_seed::Int=42)

Run a single scenario with given customer data and return MOO results.
Returns a vector of MOOScenarioResult (one per omega).
"""
function run_scenario_and_collect(scn::Int, omega_customers::Vector, omega_attrs::Vector;
                                  moo_seed::Int=42)
    # Reset global state
    reset_f2_normalization!()
    empty!(MOO_RESULTS_COLLECTOR)
    
    result = solve_scenario_omega_average(scn, omega_customers, omega_attrs; moo_seed=moo_seed)
    
    if haskey(result, :omega_results) && result.omega_results !== nothing
        lock_pub = Dict{String,Any}()
        if scn != SCENARIO_D2D
            for (id, (lon, lat, carrier_name)) in LOCKERS_PRIV
                lock_pub[id] = ((lon, lat), "Private")
            end
        end
        
        moo_results = collect_moo_results_from_omega(
            scn, get_scenario_name(scn), result.k,
            result.omega_results, lock_pub
        )
        return moo_results
    end
    
    return MOOScenarioResult[]
end

"""
    extract_compromise_solution(moo_results::Vector{MOOScenarioResult})

Extract the average compromise solution (f1, f2, f3) across all omega results.
"""
function extract_compromise_solution(moo_results::Vector{MOOScenarioResult})
    if isempty(moo_results)
        return (f1=NaN, f2=NaN, f3=NaN)
    end
    
    f1_vals = [r.f1_enterprise for r in moo_results]
    f2_vals = [r.f2_customer for r in moo_results]
    f3_vals = [r.f3_society for r in moo_results]
    
    return (f1=mean(f1_vals), f2=mean(f2_vals), f3=mean(f3_vals))
end

"""
    extract_pareto_fronts(moo_results::Vector{MOOScenarioResult})

Collect all individual Pareto front matrices from omega-level results.
"""
function extract_pareto_fronts(moo_results::Vector{MOOScenarioResult})
    fronts = Matrix{Float64}[]
    for r in moo_results
        if size(r.pareto_front, 1) > 0
            push!(fronts, r.pareto_front)
        end
    end
    return fronts
end

"""
    merge_pareto_fronts(fronts::Vector{Matrix{Float64}})

Merge multiple Pareto fronts into a single non-dominated set.
"""
function merge_pareto_fronts(fronts::Vector{Matrix{Float64}})
    if isempty(fronts)
        return Matrix{Float64}(undef, 0, 3)
    end
    
    all_points = vcat(fronts...)
    n = size(all_points, 1)
    if n == 0
        return Matrix{Float64}(undef, 0, 3)
    end
    
    # Non-dominated sorting
    is_dominated = falses(n)
    for i in 1:n
        for j in 1:n
            if i != j && !is_dominated[j]
                if all(all_points[j, :] .<= all_points[i, :]) && any(all_points[j, :] .< all_points[i, :])
                    is_dominated[i] = true
                    break
                end
            end
        end
    end
    
    return all_points[.!is_dominated, :]
end

# ═══════════════════════════════════════════════════════════════════════════
# Test 1: SAA Convergence Analysis
# ═══════════════════════════════════════════════════════════════════════════

"""
    test_saa_convergence(; scenarios_to_test::Vector{Int}=ALL_SCENARIOS,
                          n_omega_values::Vector{Int}=[5, 10, 15, 20, 30, 50],
                          master_seed::Int=12345)

Validate that the Sample Average Approximation converges as N_omega increases.
Uses nested sampling to ensure consistency across sample sizes.

Reports: CV (coefficient of variation) and 95% CI width / mean for each N_omega.
Convergence criterion: CI ratio < 10% (95% CI width / mean).
"""
function test_saa_convergence(; scenarios_to_test::Vector{Int}=[SCENARIO_D2D, SCENARIO_DPL, SCENARIO_SPL],
                                n_omega_values::Vector{Int}=[5, 10, 15, 20, 30, 50],
                                master_seed::Int=12345)
    println("\n" * "="^80)
    println("  TEST 1: SAA CONVERGENCE ANALYSIS")
    println("="^80)
    
    max_n = maximum(n_omega_values)
    println("  Generating $max_n omega scenarios (master_seed=$master_seed)...")
    all_omega_customers, all_omega_attrs = generate_omega_scenarios(max_n; master_seed=master_seed)
    
    results_df = DataFrame(
        scenario=Int[], scenario_name=String[], n_omega=Int[],
        f1_mean=Float64[], f1_std=Float64[], f1_cv=Float64[], f1_ci_ratio=Float64[],
        f2_mean=Float64[], f2_std=Float64[], f2_cv=Float64[], f2_ci_ratio=Float64[],
        f3_mean=Float64[], f3_std=Float64[], f3_cv=Float64[], f3_ci_ratio=Float64[],
        converged=Bool[]
    )
    
    for scn in scenarios_to_test
        scn_name = get_scenario_name(scn)
        println("\n  ─── Scenario $scn: $scn_name ───")
        
        for n_omega in n_omega_values
            println("    N_ω = $n_omega ...")
            
            # Use first n_omega scenarios (nested sampling)
            sub_customers = all_omega_customers[1:n_omega]
            sub_attrs = all_omega_attrs[1:n_omega]
            
            moo_results = run_scenario_and_collect(scn, sub_customers, sub_attrs)
            
            if isempty(moo_results)
                println("      ⚠ No results — skipping")
                continue
            end
            
            f1_vals = [r.f1_enterprise for r in moo_results]
            f2_vals = [r.f2_customer for r in moo_results]
            f3_vals = [r.f3_society for r in moo_results]
            
            function stats_for(vals)
                m = mean(vals)
                s = std(vals)
                cv = m > 0 ? s / m * 100.0 : NaN
                n = length(vals)
                ci_width = n > 1 ? 2 * 1.96 * s / sqrt(n) : NaN
                ci_ratio = m > 0 ? ci_width / m * 100.0 : NaN
                return (m, s, cv, ci_ratio)
            end
            
            f1_m, f1_s, f1_cv, f1_ci = stats_for(f1_vals)
            f2_m, f2_s, f2_cv, f2_ci = stats_for(f2_vals)
            f3_m, f3_s, f3_cv, f3_ci = stats_for(f3_vals)
            
            converged = f1_ci < 10.0 && f3_ci < 10.0
            
            push!(results_df, (scn, scn_name, n_omega,
                              f1_m, f1_s, f1_cv, f1_ci,
                              f2_m, f2_s, f2_cv, f2_ci,
                              f3_m, f3_s, f3_cv, f3_ci,
                              converged))
            
            status = converged ? "✓ CONVERGED" : "  not yet"
            @printf("      f1: %.2f ± %.2f (CV=%.1f%%)  f3: %.2f ± %.2f (CV=%.1f%%)  %s\n",
                    f1_m, f1_s, f1_cv, f3_m, f3_s, f3_cv, status)
        end
    end
    
    # Save results
    mkpath(ROBUSTNESS_OUTDIR)
    csv_path = joinpath(ROBUSTNESS_OUTDIR, "saa_convergence.csv")
    CSV.write(csv_path, results_df)
    println("\n  ✅ Results saved to $csv_path")
    
    return results_df
end

# ═══════════════════════════════════════════════════════════════════════════
# Test 2: Out-of-Sample Validation
# ═══════════════════════════════════════════════════════════════════════════

"""
    test_out_of_sample(; scenarios_to_test::Vector{Int}=ALL_SCENARIOS,
                        n_omega_in::Int=30, n_omega_out::Int=200,
                        seed_in::Int=12345, seed_out::Int=99999)

Validate that in-sample optimization generalizes to out-of-sample scenarios.
Following Verweij et al. (2003): independently optimize in-sample and out-of-sample,
then compare average objective values.

Reports: Optimality gap = (Z_out - Z_in) / Z_in × 100%.
"""
function test_out_of_sample(; scenarios_to_test::Vector{Int}=[SCENARIO_D2D, SCENARIO_DPL, SCENARIO_SPL],
                              n_omega_in::Int=30, n_omega_out::Int=200,
                              seed_in::Int=12345, seed_out::Int=99999)
    println("\n" * "="^80)
    println("  TEST 2: OUT-OF-SAMPLE VALIDATION")
    println("="^80)
    
    println("  In-sample:  N_ω=$n_omega_in  (seed=$seed_in)")
    println("  Out-sample: N_ω=$n_omega_out (seed=$seed_out)")
    
    # Generate in-sample scenarios
    println("  Generating in-sample scenarios...")
    in_customers, in_attrs = generate_omega_scenarios(n_omega_in; master_seed=seed_in)
    
    # Generate out-of-sample scenarios (independent seed)
    println("  Generating out-of-sample scenarios...")
    out_customers, out_attrs = generate_omega_scenarios(n_omega_out; master_seed=seed_out)
    
    results_df = DataFrame(
        scenario=Int[], scenario_name=String[],
        f1_in=Float64[], f2_in=Float64[], f3_in=Float64[],
        f1_out=Float64[], f2_out=Float64[], f3_out=Float64[],
        gap_f1_pct=Float64[], gap_f2_pct=Float64[], gap_f3_pct=Float64[]
    )
    
    for scn in scenarios_to_test
        scn_name = get_scenario_name(scn)
        println("\n  ─── Scenario $scn: $scn_name ───")
        
        # In-sample optimization
        println("    In-sample optimization (N_ω=$n_omega_in)...")
        in_results = run_scenario_and_collect(scn, in_customers, in_attrs)
        in_sol = extract_compromise_solution(in_results)
        
        # Out-of-sample optimization
        println("    Out-of-sample optimization (N_ω=$n_omega_out)...")
        out_results = run_scenario_and_collect(scn, out_customers, out_attrs)
        out_sol = extract_compromise_solution(out_results)
        
        if isnan(in_sol.f1) || isnan(out_sol.f1)
            println("    ⚠ Insufficient results — skipping")
            continue
        end
        
        gap_f1 = (out_sol.f1 - in_sol.f1) / abs(in_sol.f1) * 100.0
        gap_f2 = in_sol.f2 != 0 ? (out_sol.f2 - in_sol.f2) / abs(in_sol.f2) * 100.0 : NaN
        gap_f3 = (out_sol.f3 - in_sol.f3) / abs(in_sol.f3) * 100.0
        
        push!(results_df, (scn, scn_name,
                          in_sol.f1, in_sol.f2, in_sol.f3,
                          out_sol.f1, out_sol.f2, out_sol.f3,
                          gap_f1, gap_f2, gap_f3))
        
        @printf("    f1: in=%.2f  out=%.2f  gap=%+.1f%%\n", in_sol.f1, out_sol.f1, gap_f1)
        @printf("    f3: in=%.2f  out=%.2f  gap=%+.1f%%\n", in_sol.f3, out_sol.f3, gap_f3)
    end
    
    mkpath(ROBUSTNESS_OUTDIR)
    csv_path = joinpath(ROBUSTNESS_OUTDIR, "out_of_sample_validation.csv")
    CSV.write(csv_path, results_df)
    println("\n  ✅ Results saved to $csv_path")
    
    return results_df
end

# ═══════════════════════════════════════════════════════════════════════════
# Test 3: Algorithmic Stability (Hypervolume-based)
# ═══════════════════════════════════════════════════════════════════════════

"""
    test_algorithmic_stability(; scenarios_to_test::Vector{Int}=ALL_SCENARIOS,
                                n_runs::Int=20, n_omega::Int=30,
                                master_seed::Int=12345)

Test NSGA-II stability across independent runs with different random seeds.
Uses the same customer scenarios but varies the NSGA-II seed.

Reports: Hypervolume mean ± std, IGD+ mean ± std, Wilcoxon signed-rank test.
"""
function test_algorithmic_stability(; scenarios_to_test::Vector{Int}=[SCENARIO_D2D, SCENARIO_DPL, SCENARIO_SPL],
                                     n_runs::Int=20, n_omega::Int=30,
                                     master_seed::Int=12345)
    println("\n" * "="^80)
    println("  TEST 3: ALGORITHMIC STABILITY (HYPERVOLUME)")
    println("="^80)
    
    println("  Independent runs: $n_runs")
    println("  Fixed N_ω: $n_omega (master_seed=$master_seed)")
    
    # Generate fixed customer scenarios
    println("  Generating fixed customer scenarios...")
    omega_customers, omega_attrs = generate_omega_scenarios(n_omega; master_seed=master_seed)
    
    # NSGA-II seeds for independent runs
    nsga2_seeds = [42 + 100*i for i in 0:(n_runs-1)]
    
    results_df = DataFrame(
        scenario=Int[], scenario_name=String[], run=Int[], nsga2_seed=Int[],
        hv=Float64[], igd_plus=Float64[],
        f1_compromise=Float64[], f2_compromise=Float64[], f3_compromise=Float64[],
        n_pareto_solutions=Int[]
    )
    
    summary_df = DataFrame(
        scenario=Int[], scenario_name=String[],
        hv_mean=Float64[], hv_std=Float64[], hv_cv=Float64[],
        igd_mean=Float64[], igd_std=Float64[],
        signedrank_p=Float64[]
    )
    
    for scn in scenarios_to_test
        scn_name = get_scenario_name(scn)
        println("\n  ─── Scenario $scn: $scn_name ───")
        
        all_fronts = Matrix{Float64}[]
        front_run_indices = Int[]  # track which run_idx produced each front
        hv_values = Float64[]
        igd_values = Float64[]
        
        for (run_idx, seed) in enumerate(nsga2_seeds)
            println("    Run $run_idx/$n_runs (seed=$seed)...")
            
            moo_results = run_scenario_and_collect(scn, omega_customers, omega_attrs; moo_seed=seed)
            
            if isempty(moo_results)
                println("      ⚠ No results")
                continue
            end
            
            # Merge omega-level Pareto fronts into system-level front
            fronts = extract_pareto_fronts(moo_results)
            merged_pf = merge_pareto_fronts(fronts)
            
            if size(merged_pf, 1) == 0
                println("      ⚠ Empty Pareto front")
                continue
            end
            
            push!(all_fronts, merged_pf)
            push!(front_run_indices, run_idx)
            
            sol = extract_compromise_solution(moo_results)
            
            push!(results_df, (scn, scn_name, run_idx, seed,
                              NaN, NaN,  # HV/IGD computed after all runs
                              sol.f1, sol.f2, sol.f3,
                              size(merged_pf, 1)))
        end
        
        if length(all_fronts) < 2
            println("    ⚠ Too few valid runs for stability analysis")
            continue
        end
        
        # Compute common reference point
        ref_point = compute_reference_point(all_fronts; margin=1.1)
        println("    Reference point: [$(join([@sprintf("%.2f", r) for r in ref_point], ", "))]")
        
        # Compute reference front (union of all non-dominated solutions)
        reference_front = merge_pareto_fronts(all_fronts)
        
        # Compute HV and IGD+ for each run (using tracked run indices)
        for (i, pf) in enumerate(all_fronts)
            hv = compute_hypervolume_3d(pf, ref_point)
            igd = compute_igd_plus(pf, reference_front)
            push!(hv_values, hv)
            push!(igd_values, igd)
            
            actual_run_idx = front_run_indices[i]
            matching_rows = findall(r -> results_df.scenario[r] == scn && results_df.run[r] == actual_run_idx, 1:nrow(results_df))
            if !isempty(matching_rows)
                results_df.hv[matching_rows[1]] = hv
                results_df.igd_plus[matching_rows[1]] = igd
            end
        end
        
        hv_m = mean(hv_values)
        hv_s = std(hv_values)
        hv_cv = hv_m > 0 ? hv_s / hv_m * 100.0 : NaN
        igd_m = mean(igd_values)
        igd_s = std(igd_values)
        
        # Wilcoxon signed-rank test: H0 = HV values are symmetric around their median
        # Low p-value would indicate systematic bias; high p-value indicates consistency
        signedrank_p = NaN
        if length(hv_values) >= 3
            try
                signedrank_p = pvalue(SignedRankTest(hv_values .- median(hv_values)))
            catch
                signedrank_p = NaN
            end
        end
        
        push!(summary_df, (scn, scn_name,
                          hv_m, hv_s, hv_cv,
                          igd_m, igd_s, signedrank_p))
        
        @printf("    HV:   %.4f ± %.4f (CV=%.1f%%)\n", hv_m, hv_s, hv_cv)
        @printf("    IGD+: %.4f ± %.4f\n", igd_m, igd_s)
    end
    
    mkpath(ROBUSTNESS_OUTDIR)
    csv_path_detail = joinpath(ROBUSTNESS_OUTDIR, "algorithm_stability_hv.csv")
    csv_path_summary = joinpath(ROBUSTNESS_OUTDIR, "algorithm_stability_summary.csv")
    CSV.write(csv_path_detail, results_df)
    CSV.write(csv_path_summary, summary_df)
    println("\n  ✅ Results saved to $csv_path_detail")
    println("  ✅ Summary saved to $csv_path_summary")
    
    return results_df, summary_df
end

# ═══════════════════════════════════════════════════════════════════════════
# Test 4: Parameter Sensitivity Analysis
# ═══════════════════════════════════════════════════════════════════════════

"""
    test_parameter_sensitivity(; n_omega=30, master_seed=12345)

One-at-a-time (OAT) sensitivity analysis for 3 key model parameters:

| Parameter        | Symbol              | Values         | Default | Rationale         | Scenarios     |
|------------------|---------------------|----------------|---------|-------------------|---------------|
| Customer count C | CUSTOMER_SCALE      | ×0.7, 1.0, 1.3| 1.0     | Demand uncertainty| D2D, DPL, SPL |
| Vehicle capacity | CAPACITY            | 80, 100, 120   | 100     | Fleet variation   | D2D, DPL, SPL |
| Public locker cap| PUBLIC_LOCKER_CAP   | 50, 69, 90     | 69      | Policy alternative| OPL           |
"""
function test_parameter_sensitivity(; n_omega::Int=30, master_seed::Int=12345)
    println("\n" * "="^80)
    println("  TEST 4: PARAMETER SENSITIVITY ANALYSIS (3 parameters, OAT)")
    println("="^80)

    scn_general = [SCENARIO_D2D, SCENARIO_DPL, SCENARIO_SPL]
    scn_opl     = [SCENARIO_OPL]

    # (display_name, param_symbol, test_values, default, affected_scenarios, is_data_param)
    param_configs = [
        ("C_customers",  :CUSTOMER_SCALE,          [0.7, 1.0, 1.3],    1.0,  scn_general, true),
        ("Q_vehicle",    :CAPACITY,                 [80, 100, 120],     100,  scn_general, false),
        ("Q_pub",        :PUBLIC_LOCKER_CAPACITY,   [50, 69, 90],        69,  scn_opl,     false),
    ]

    # Pre-generate base scenarios (used by all non-data params)
    println("  Generating base customer scenarios (N_ω=$n_omega, seed=$master_seed)...")
    base_customers, base_attrs = generate_omega_scenarios(n_omega; master_seed=master_seed)

    results_df = DataFrame(
        parameter=String[], param_value=Float64[],
        scenario=Int[], scenario_name=String[],
        f1_compromise=Float64[], f2_compromise=Float64[], f3_compromise=Float64[],
        relative_rank=Int[]
    )

    function set_override!(sym::Symbol, val)
        PARAM_OVERRIDES[sym] = val
        if nprocs() > 1
            sym_node = QuoteNode(sym)
            @everywhere workers() PARAM_OVERRIDES[$sym_node] = $val
        end
    end

    function clear_override!(sym::Symbol)
        delete!(PARAM_OVERRIDES, sym)
        if nprocs() > 1
            sym_node = QuoteNode(sym)
            @everywhere workers() delete!(PARAM_OVERRIDES, $sym_node)
        end
    end

    for (param_name, param_sym, values, default_val, affected_scns, is_data_param) in param_configs
        println("\n  ─── Parameter: $param_name (default=$default_val) ───")
        println("      Scenarios: $(join([get_scenario_name(s) for s in affected_scns], ", "))")

        for val in values
            println("    $param_name = $val")

            if is_data_param
                # Data parameter (C): regenerate scenarios with scaled customer count
                scale = Float64(val)
                oc, oa = generate_omega_scenarios(n_omega; master_seed=master_seed, customer_scale=scale)
                total_c = isempty(oc) ? 0 : length(first(oc))
                println("      Generated $total_c customers per ω (scale=$scale)")
            else
                # Runtime parameter: set override and use base scenarios
                set_override!(param_sym, val)
                oc, oa = base_customers, base_attrs
            end

            for scn in affected_scns
                scn_name = get_scenario_name(scn)

                moo_results = run_scenario_and_collect(scn, oc, oa)
                sol = extract_compromise_solution(moo_results)

                if !isnan(sol.f1)
                    push!(results_df, (param_name, Float64(val),
                                      scn, scn_name,
                                      sol.f1, sol.f2, sol.f3, 0))
                    @printf("      Scn %d (%s): f1=%.2f  f2=%.4f  f3=%.2f\n",
                            scn, scn_name, sol.f1, sol.f2, sol.f3)
                end
            end

            if !is_data_param
                clear_override!(param_sym)
            end
        end
    end

    # Compute relative ranks within each parameter-value group
    for param_name in unique(results_df.parameter)
        for val in unique(results_df[results_df.parameter .== param_name, :param_value])
            mask = (results_df.parameter .== param_name) .& (results_df.param_value .== val)
            subset = results_df[mask, :]
            if nrow(subset) > 0
                f3_order = sortperm(subset.f3_compromise)
                ranks = invperm(f3_order)
                results_df.relative_rank[findall(mask)] .= ranks
            end
        end
    end

    mkpath(ROBUSTNESS_OUTDIR)
    csv_path = joinpath(ROBUSTNESS_OUTDIR, "parameter_sensitivity.csv")
    CSV.write(csv_path, results_df)
    println("\n  ✅ Results saved to $csv_path")

    # Rank consistency check
    println("\n  📊 Rank Consistency Check:")
    all_tested_scns = unique(results_df.scenario)
    for scn in all_tested_scns
        scn_name = get_scenario_name(scn)
        scn_rows = results_df[results_df.scenario .== scn, :]
        if nrow(scn_rows) > 0
            ranks = scn_rows.relative_rank
            println("    Scn $scn ($scn_name): ranks = $ranks")
        end
    end

    return results_df
end

# ═══════════════════════════════════════════════════════════════════════════
# Visualization: Publication-Quality Figures
# ═══════════════════════════════════════════════════════════════════════════

const FIG_DPI = 300

"""
    plot_saa_convergence(csv_path::String)

Generate SAA convergence plot: N_ω vs 95% CI ratio (%) with 10% threshold line.
Produces one subplot per objective (f1, f3).
"""
function plot_saa_convergence(csv_path::String=joinpath(ROBUSTNESS_OUTDIR, "saa_convergence.csv");
                              output_dir::String=dirname(csv_path))
    df = CSV.read(csv_path, DataFrame)
    if nrow(df) == 0
        println("  ⚠ No SAA data to plot")
        return
    end

    scenarios = unique(df.scenario_name)

    p = plot(layout=(1, 2), size=(1000, 420), dpi=FIG_DPI,
             margin=6Plots.mm, bottom_margin=10Plots.mm, top_margin=8Plots.mm,
             titlefontsize=11, guidefontsize=10, tickfontsize=9, legendfontsize=8)

    markers = [:circle, :square, :diamond, :utriangle, :star5]
    colors = [:royalblue, :crimson, :forestgreen, :darkorange, :purple]

    for (i, obj) in enumerate(["f1", "f3"])
        ci_col = Symbol("$(obj)_ci_ratio")
        obj_title = obj == "f1" ? "(a) Enterprise Cost f₁" : "(b) Social Cost f₃"

        for (j, scn) in enumerate(scenarios)
            sub = df[df.scenario_name .== scn, :]
            plot!(p[i], sub.n_omega, sub[!, ci_col],
                  label=scn, marker=markers[mod1(j, 5)], color=colors[mod1(j, 5)],
                  linewidth=2, markersize=5)

            if nrow(sub) > 0
                last_row = sub[end, :]
                final_ci = last_row[ci_col]
                annotate!(p[i], [(last_row.n_omega + 1, final_ci,
                          text(@sprintf("%.1f%%", final_ci), 7, colors[mod1(j, 5)], :left))])
            end
        end

        hline!(p[i], [10.0], color=:red, linestyle=:dash, linewidth=1.5,
               label="Threshold (10%)", alpha=0.7)

        xlabel!(p[i], "Sample Size (Nω)")
        ylabel!(p[i], "95% CI / Mean (%)")
        title!(p[i], obj_title)
    end

    plot!(p, plot_title="SAA Convergence Analysis", plot_titlefontsize=13)

    fig_path = joinpath(output_dir, "fig_saa_convergence.png")
    savefig(p, fig_path)
    println("  📊 Saved: $fig_path")
    return p
end

"""
    plot_out_of_sample(csv_path::String)

Generate out-of-sample gap bar chart per scenario.
"""
function plot_out_of_sample(csv_path::String=joinpath(ROBUSTNESS_OUTDIR, "out_of_sample_validation.csv");
                            output_dir::String=dirname(csv_path))
    df = CSV.read(csv_path, DataFrame)
    if nrow(df) == 0
        println("  ⚠ No OOS data to plot")
        return
    end

    n = nrow(df)
    x = 1:n

    p = plot(size=(max(600, 150*n), 450), dpi=FIG_DPI,
             margin=6Plots.mm, bottom_margin=14Plots.mm, top_margin=8Plots.mm,
             titlefontsize=11, guidefontsize=10, tickfontsize=9, legendfontsize=9)

    width = 0.25
    bar!(p, x .- width, df.gap_f1_pct, bar_width=width, label="f₁ (Enterprise Cost)",
         color=:royalblue, alpha=0.85)
    bar!(p, x, df.gap_f2_pct, bar_width=width, label="f₂ (Customer Disutility)",
         color=:forestgreen, alpha=0.85)
    bar!(p, x .+ width, df.gap_f3_pct, bar_width=width, label="f₃ (Social Cost)",
         color=:crimson, alpha=0.85)

    hline!(p, [0.0], color=:black, linewidth=0.8, label=false)
    hline!(p, [15.0], color=:orange, linestyle=:dash, linewidth=1.2,
           label="Acceptable threshold (15%)", alpha=0.7)
    hline!(p, [-15.0], color=:orange, linestyle=:dash, linewidth=1.2, label=false, alpha=0.7)

    for i in 1:n
        for (dx, col, c) in [(-width, :gap_f1_pct, :royalblue),
                              (0.0, :gap_f2_pct, :forestgreen),
                              (width, :gap_f3_pct, :crimson)]
            v = df[i, col]
            if !isnan(v)
                annotate!(p, [(i + dx, v + sign(v)*1.5,
                          text(@sprintf("%+.1f%%", v), 7, c, :center))])
            end
        end
    end

    xticks!(p, x, df.scenario_name)
    xlabel!(p, "Scenario")
    ylabel!(p, "Optimality Gap: (Z_out − Z_in) / Z_in × 100%")
    title!(p, "Out-of-Sample Validation (Nω_in=30, Nω_out=200)")

    fig_path = joinpath(output_dir, "fig_out_of_sample.png")
    savefig(p, fig_path)
    println("  📊 Saved: $fig_path")
    return p
end

"""
    plot_algorithmic_stability(csv_path::String)

Generate HV box plot across independent runs per scenario.
"""
function plot_algorithmic_stability(csv_path::String=joinpath(ROBUSTNESS_OUTDIR, "algorithm_stability_hv.csv");
                                    output_dir::String=dirname(csv_path))
    df = CSV.read(csv_path, DataFrame)
    if nrow(df) == 0
        println("  ⚠ No stability data to plot")
        return
    end

    df = df[.!isnan.(df.hv), :]
    scenarios = unique(df.scenario_name)
    colors = [:royalblue, :crimson, :forestgreen, :darkorange, :purple]

    summary_path = joinpath(output_dir, "algorithm_stability_summary.csv")
    summary_df = isfile(summary_path) ? CSV.read(summary_path, DataFrame) : DataFrame()

    p = plot(layout=(1, 2), size=(1050, 450), dpi=FIG_DPI,
             margin=6Plots.mm, bottom_margin=12Plots.mm, top_margin=8Plots.mm,
             titlefontsize=11, guidefontsize=10, tickfontsize=9, legendfontsize=8)

    for (i, scn) in enumerate(scenarios)
        hv_vals = df[df.scenario_name .== scn, :hv]
        boxplot!(p[1], fill(scn, length(hv_vals)), hv_vals,
                 label=false, color=colors[mod1(i, 5)], alpha=0.7,
                 whisker_width=0.5, outliers=true)

        if nrow(summary_df) > 0
            row = summary_df[summary_df.scenario_name .== scn, :]
            if nrow(row) > 0
                cv_val = row.hv_cv[1]
                p_val = row.signedrank_p[1]
                lbl = @sprintf("CV=%.1f%%\np=%.3f", cv_val, p_val)
                annotate!(p[1], [(i, maximum(hv_vals) * 1.02,
                          text(lbl, 7, :center))])
            end
        end
    end
    ylabel!(p[1], "Hypervolume (HV)")
    title!(p[1], "(a) HV Distribution (n=20 runs)")

    for (i, scn) in enumerate(scenarios)
        sub = df[df.scenario_name .== scn, :]
        scatter!(p[2], sub.f1_compromise, sub.f3_compromise,
                 label=scn, color=colors[mod1(i, 5)], markersize=5,
                 alpha=0.6, markerstrokewidth=0.5)
        f1_m = mean(sub.f1_compromise)
        f3_m = mean(sub.f3_compromise)
        scatter!(p[2], [f1_m], [f3_m], label=false,
                 color=colors[mod1(i, 5)], markersize=9,
                 markershape=:star5, markerstrokewidth=1.5, markerstrokecolor=:black)
        annotate!(p[2], [(f1_m, f3_m - 0.5,
                  text(@sprintf("μ=(%.1f, %.1f)", f1_m, f3_m), 7, colors[mod1(i, 5)], :center))])
    end
    xlabel!(p[2], "f₁ (Enterprise Cost, EUR)")
    ylabel!(p[2], "f₃ (Social Cost, kg CO₂)")
    title!(p[2], "(b) Compromise Solutions (★ = mean)")

    plot!(p, plot_title="Algorithmic Stability — NSGA-II-HI", plot_titlefontsize=13)

    fig_path = joinpath(output_dir, "fig_algorithmic_stability.png")
    savefig(p, fig_path)
    println("  📊 Saved: $fig_path")
    return p
end

"""
    plot_parameter_sensitivity(csv_path::String)

Generate tornado/grouped bar chart for parameter sensitivity.
"""
function plot_parameter_sensitivity(csv_path::String=joinpath(ROBUSTNESS_OUTDIR, "parameter_sensitivity.csv");
                                     output_dir::String=dirname(csv_path))
    df = CSV.read(csv_path, DataFrame)
    if nrow(df) == 0
        println("  ⚠ No sensitivity data to plot")
        return
    end

    params = unique(df.parameter)
    n_params = length(params)

    n_cols = min(n_params, 3)
    n_rows = ceil(Int, n_params / n_cols)
    p = plot(layout=(n_rows, n_cols),
             size=(400*n_cols, 380*n_rows), dpi=FIG_DPI,
             margin=6Plots.mm, bottom_margin=12Plots.mm, top_margin=8Plots.mm,
             titlefontsize=10, guidefontsize=9, tickfontsize=8, legendfontsize=7)

    colors = [:royalblue, :crimson, :forestgreen, :darkorange, :purple]
    markers = [:circle, :square, :diamond, :utriangle, :star5]

    param_obj_map = Dict(
        "C_customers" => (:f1_compromise, "Δf₁ (%)"),
        "Q_vehicle"   => (:f1_compromise, "Δf₁ (%)"),
        "Q_pub"       => (:f1_compromise, "Δf₁ (%)"),
    )
    param_labels = Dict(
        "C_customers" => "(a) Customer Count (×C)",
        "Q_vehicle"   => "(b) Vehicle Capacity (Q)",
        "Q_pub"       => "(c) Public Locker Cap. (Qᵖᵘᵇ)",
    )
    param_defaults = Dict("C_customers" => 1.0, "Q_vehicle" => 100.0, "Q_pub" => 69.0)

    for (pi, param) in enumerate(params)
        sub_param = df[df.parameter .== param, :]
        obj_col, y_label = get(param_obj_map, param, (:f1_compromise, "Δf₁ (%)"))
        scenarios = unique(sub_param.scenario_name)
        default_val = get(param_defaults, param, NaN)

        for (si, scn) in enumerate(scenarios)
            sub = sub_param[sub_param.scenario_name .== scn, :]
            if nrow(sub) == 0 continue end

            vals = sub.param_value
            obj_vals = sub[!, obj_col]

            default_idx = findfirst(v -> isapprox(v, default_val; atol=0.01), vals)
            if default_idx === nothing default_idx = ceil(Int, length(vals)/2) end
            obj_default = obj_vals[default_idx]
            obj_pct = abs(obj_default) > 1e-12 ?
                (obj_vals .- obj_default) ./ abs(obj_default) .* 100.0 :
                zeros(length(obj_vals))

            plot!(p[pi], vals, obj_pct,
                  label=scn, marker=markers[mod1(si, 5)], color=colors[mod1(si, 5)],
                  linewidth=2, markersize=6)

            for idx in [1, length(obj_pct)]
                if abs(obj_pct[idx]) > 0.1
                    annotate!(p[pi], [(vals[idx], obj_pct[idx] + sign(obj_pct[idx]) * 2.0,
                              text(@sprintf("%+.1f%%", obj_pct[idx]), 7,
                                   colors[mod1(si, 5)], :center))])
                end
            end
        end

        hline!(p[pi], [0.0], color=:gray, linestyle=:dash, linewidth=0.8, label=false)
        vline!(p[pi], [default_val], color=:gray, linestyle=:dot, linewidth=0.8, label=false)

        xlabel!(p[pi], get(param_labels, param, param))
        ylabel!(p[pi], y_label)
        title!(p[pi], get(param_labels, param, param))
    end

    plot!(p, plot_title="Parameter Sensitivity Analysis (OAT)", plot_titlefontsize=13)

    fig_path = joinpath(output_dir, "fig_parameter_sensitivity.png")
    savefig(p, fig_path)
    println("  📊 Saved: $fig_path")
    return p
end

"""
    generate_robustness_report()

Generate a comprehensive Markdown report summarizing all robustness test results
with full interpretation for each test. Suitable for direct inclusion in academic papers.
"""
function generate_robustness_report(output_dir::String=ROBUSTNESS_OUTDIR)
    mkpath(output_dir)
    report_path = joinpath(output_dir, "robustness_report.md")

    io = IOBuffer()
    println(io, "# Robustness Validation Report")
    println(io, "")
    println(io, "**Generated**: $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))")
    println(io, "**Model**: Multi-Trip CVRP Budapest — NSGA-II-HI Multi-Objective Optimization")
    println(io, "")
    println(io, "---")
    println(io, "")

    # ──── Test 1: SAA Convergence ────
    saa_csv = joinpath(output_dir, "saa_convergence.csv")
    println(io, "## 1. SAA Convergence Analysis")
    println(io, "")
    println(io, "**Purpose**: Validate that the Sample Average Approximation (SAA) estimator")
    println(io, "converges as the Monte Carlo sample size Nω increases.")
    println(io, "")
    println(io, "**Criterion**: 95% CI width / Mean < 10% for both f₁ and f₃.")
    println(io, "")
    if isfile(saa_csv)
        df = CSV.read(saa_csv, DataFrame)
        println(io, "| Scenario | Nω | f₁ Mean ± SD | f₁ CI Ratio | f₃ Mean ± SD | f₃ CI Ratio | Converged |")
        println(io, "|----------|-----|-------------|-------------|-------------|-------------|-----------|")
        for r in eachrow(df)
            conv = r.converged ? "✓" : "✗"
            @printf(io, "| %s | %d | %.2f ± %.2f | %.1f%% | %.2f ± %.2f | %.1f%% | %s |\n",
                    r.scenario_name, r.n_omega,
                    r.f1_mean, r.f1_std, r.f1_ci_ratio,
                    r.f3_mean, r.f3_std, r.f3_ci_ratio, conv)
        end
        println(io, "")

        # Interpretation
        final_rows = combine(groupby(df, :scenario_name), last)
        all_converged = all(final_rows.converged)
        max_n = maximum(df.n_omega)
        println(io, "**Interpretation**: ")
        if all_converged
            println(io, "All scenarios achieved convergence (CI ratio < 10%) at Nω = $max_n.")
            println(io, "The chosen sample size is sufficient for stable SAA estimates.")
        else
            not_conv = final_rows[.!final_rows.converged, :scenario_name]
            println(io, "Scenarios **$(join(not_conv, ", "))** did not fully converge at Nω = $max_n.")
            println(io, "Consider increasing the sample size for these scenarios.")
        end
    else
        println(io, "*No results available — test not yet executed.*")
    end
    println(io, "")
    println(io, "![SAA Convergence](fig_saa_convergence.png)")
    println(io, "")
    println(io, "---")
    println(io, "")

    # ──── Test 2: Out-of-Sample ────
    oos_csv = joinpath(output_dir, "out_of_sample_validation.csv")
    println(io, "## 2. Out-of-Sample Validation")
    println(io, "")
    println(io, "**Purpose**: Verify that optimization with Nω in-sample scenarios")
    println(io, "generalizes to unseen demand realizations.")
    println(io, "")
    println(io, "**Metric**: Optimality gap = (Z_out − Z_in) / Z_in × 100%.")
    println(io, "Gaps below ±15% indicate acceptable generalization.")
    println(io, "")
    if isfile(oos_csv)
        df = CSV.read(oos_csv, DataFrame)
        println(io, "| Scenario | f₁ In-Sample | f₁ Out-of-Sample | Gap f₁ | f₃ In | f₃ Out | Gap f₃ |")
        println(io, "|----------|-------------|-----------------|--------|-------|--------|--------|")
        for r in eachrow(df)
            @printf(io, "| %s | %.2f | %.2f | %+.1f%% | %.2f | %.2f | %+.1f%% |\n",
                    r.scenario_name, r.f1_in, r.f1_out, r.gap_f1_pct,
                    r.f3_in, r.f3_out, r.gap_f3_pct)
        end
        println(io, "")

        max_gap_f1 = maximum(abs.(df.gap_f1_pct))
        max_gap_f3 = maximum(abs.(df.gap_f3_pct))
        println(io, "**Interpretation**: ")
        if max_gap_f1 < 15.0 && max_gap_f3 < 15.0
            @printf(io, "Maximum absolute gaps are %.1f%% (f₁) and %.1f%% (f₃), ", max_gap_f1, max_gap_f3)
            println(io, "both within the 15% threshold.")
            println(io, "The SAA solutions generalize well to unseen scenarios.")
        else
            @printf(io, "Maximum absolute gaps of %.1f%% (f₁) and %.1f%% (f₃) ", max_gap_f1, max_gap_f3)
            println(io, "suggest potential overfitting to in-sample scenarios.")
            println(io, "Increasing Nω or regularization may be warranted.")
        end
    else
        println(io, "*No results available — test not yet executed.*")
    end
    println(io, "")
    println(io, "![Out-of-Sample Validation](fig_out_of_sample.png)")
    println(io, "")
    println(io, "---")
    println(io, "")

    # ──── Test 3: Algorithmic Stability ────
    stab_csv = joinpath(output_dir, "algorithm_stability_summary.csv")
    stab_detail = joinpath(output_dir, "algorithm_stability_hv.csv")
    println(io, "## 3. Algorithmic Stability (NSGA-II-HI)")
    println(io, "")
    println(io, "**Purpose**: Assess whether NSGA-II-HI produces consistent Pareto fronts")
    println(io, "across independent runs with different random seeds.")
    println(io, "")
    println(io, "**Metrics**: Hypervolume (HV) and IGD+ across 20 runs.")
    println(io, "CV(HV) < 5% indicates high stability.")
    println(io, "Wilcoxon signed-rank test: p > 0.05 confirms no systematic bias.")
    println(io, "")
    if isfile(stab_csv)
        df = CSV.read(stab_csv, DataFrame)
        println(io, "| Scenario | HV Mean ± SD | HV CV (%) | IGD+ Mean ± SD | Wilcoxon p |")
        println(io, "|----------|-------------|-----------|---------------|------------|")
        for r in eachrow(df)
            p_str = isnan(r.signedrank_p) ? "N/A" : @sprintf("%.4f", r.signedrank_p)
            @printf(io, "| %s | %.4f ± %.4f | %.1f | %.4f ± %.4f | %s |\n",
                    r.scenario_name, r.hv_mean, r.hv_std, r.hv_cv,
                    r.igd_mean, r.igd_std, p_str)
        end
        println(io, "")

        max_cv = maximum(df.hv_cv)
        all_stable = max_cv < 5.0
        println(io, "**Interpretation**: ")
        if all_stable
            @printf(io, "All scenarios exhibit HV coefficient of variation below 5%% (max: %.1f%%), ", max_cv)
            println(io, "confirming that NSGA-II-HI produces highly reproducible Pareto fronts.")
        else
            high_cv = df[df.hv_cv .>= 5.0, :scenario_name]
            @printf(io, "Scenarios **%s** show HV CV ≥ 5%% (max: %.1f%%), ", join(high_cv, ", "), max_cv)
            println(io, "indicating moderate variability in Pareto front quality.")
        end

        valid_p = filter(!isnan, df.signedrank_p)
        if !isempty(valid_p)
            min_p = minimum(valid_p)
            if min_p > 0.05
                @printf(io, "\nAll Wilcoxon p-values exceed 0.05 (min: %.4f), ", min_p)
                println(io, "confirming no systematic bias in the optimization.")
            else
                @printf(io, "\nSome Wilcoxon p-values are below 0.05 (min: %.4f), ", min_p)
                println(io, "suggesting potential systematic variation in those scenarios.")
            end
        end
    else
        println(io, "*No results available — test not yet executed.*")
    end

    if isfile(stab_detail)
        df_d = CSV.read(stab_detail, DataFrame)
        df_d = df_d[.!isnan.(df_d.hv), :]
        if nrow(df_d) > 0
            println(io, "")
            println(io, "**Per-run Compromise Solutions** (f₁, f₂, f₃):")
            println(io, "")
            println(io, "| Scenario | Run | NSGA-II Seed | f₁ | f₂ | f₃ | HV | IGD+ | Pareto |")
            println(io, "|----------|-----|-------------|-----|------|------|------|------|--------|")
            for r in eachrow(df_d)
                @printf(io, "| %s | %d | %d | %.2f | %.4f | %.2f | %.4f | %.4f | %d |\n",
                        r.scenario_name, r.run, r.nsga2_seed,
                        r.f1_compromise, r.f2_compromise, r.f3_compromise,
                        r.hv, r.igd_plus, r.n_pareto_solutions)
            end
        end
    end
    println(io, "")
    println(io, "![Algorithmic Stability](fig_algorithmic_stability.png)")
    println(io, "")
    println(io, "---")
    println(io, "")

    # ──── Test 4: Parameter Sensitivity ────
    sens_csv = joinpath(output_dir, "parameter_sensitivity.csv")
    println(io, "## 4. Parameter Sensitivity Analysis (OAT)")
    println(io, "")
    println(io, "**Purpose**: Evaluate how robust the results are to variations in key model parameters.")
    println(io, "One parameter is varied at a time while others remain at default values.")
    println(io, "")
    println(io, "| Parameter | Symbol | Values Tested | Default | Rationale |")
    println(io, "|-----------|--------|---------------|---------|-----------|")
    println(io, "| Customer count | C | ×0.7, ×1.0, ×1.3 | ×1.0 (2,033) | Demand uncertainty |")
    println(io, "| Vehicle capacity | Q | 80, 100, 120 | 100 | Fleet type variation |")
    println(io, "| Public locker capacity | Qᵖᵘᵇ | 50, 69, 90 | 69 | Policy alternatives |")
    println(io, "")
    if isfile(sens_csv)
        df = CSV.read(sens_csv, DataFrame)
        println(io, "### Raw Results")
        println(io, "")
        println(io, "| Parameter | Value | Scenario | f₁ (EUR) | f₂ (Disutility) | f₃ (kg CO₂) |")
        println(io, "|-----------|-------|----------|---------|-----------------|-------------|")
        for r in eachrow(df)
            @printf(io, "| %s | %.2f | %s | %.2f | %.4f | %.2f |\n",
                    r.parameter, r.param_value, r.scenario_name,
                    r.f1_compromise, r.f2_compromise, r.f3_compromise)
        end
        println(io, "")

        # Compute and report % changes per parameter
        println(io, "### Percentage Change from Default")
        println(io, "")
        param_defaults_map = Dict("C_customers" => 1.0, "Q_vehicle" => 100.0, "Q_pub" => 69.0)
        for param in unique(df.parameter)
            sub = df[df.parameter .== param, :]
            default_v = get(param_defaults_map, param, NaN)
            println(io, "**$param** (default = $default_v):")
            println(io, "")
            println(io, "| Scenario | Value | f₁ Δ(%) | f₃ Δ(%) |")
            println(io, "|----------|-------|---------|---------|")
            for scn in unique(sub.scenario_name)
                scn_sub = sub[sub.scenario_name .== scn, :]
                default_row = scn_sub[isapprox.(scn_sub.param_value, default_v; atol=0.01), :]
                if nrow(default_row) == 0 continue end
                f1_def = default_row.f1_compromise[1]
                f3_def = default_row.f3_compromise[1]
                for r in eachrow(scn_sub)
                    f1_pct = abs(f1_def) > 1e-12 ? (r.f1_compromise - f1_def) / abs(f1_def) * 100 : 0.0
                    f3_pct = abs(f3_def) > 1e-12 ? (r.f3_compromise - f3_def) / abs(f3_def) * 100 : 0.0
                    @printf(io, "| %s | %.2f | %+.1f%% | %+.1f%% |\n",
                            r.scenario_name, r.param_value, f1_pct, f3_pct)
                end
            end
            println(io, "")
        end

        println(io, "**Interpretation**: ")
        max_change = 0.0
        most_sensitive = ""
        for param in unique(df.parameter)
            sub = df[df.parameter .== param, :]
            default_v = get(param_defaults_map, param, NaN)
            if isnan(default_v) continue end
            for scn in unique(sub.scenario_name)
                scn_sub = sub[sub.scenario_name .== scn, :]
                default_row = scn_sub[isapprox.(scn_sub.param_value, default_v; atol=0.01), :]
                if nrow(default_row) == 0 continue end
                f1_def = default_row.f1_compromise[1]
                for r in eachrow(scn_sub)
                    pct = abs(f1_def) > 1e-12 ? abs((r.f1_compromise - f1_def) / f1_def * 100) : 0.0
                    if pct > max_change
                        max_change = pct
                        most_sensitive = param
                    end
                end
            end
        end
        if !isempty(most_sensitive)
            @printf(io, "The most sensitive parameter is **%s** (max |Δf₁| = %.1f%%). ", most_sensitive, max_change)
        end
        if max_change < 20.0
            println(io, "All parameters show moderate sensitivity (< 20%), confirming model robustness.")
        else
            @printf(io, "Sensitivity of **%s** exceeds 20%%, indicating that this parameter ", most_sensitive)
            println(io, "choice significantly affects results and warrants careful calibration.")
        end

        # Rank consistency
        println(io, "")
        println(io, "**Rank Consistency**: ")
        all_consistent = true
        for scn in unique(df.scenario)
            scn_rows = df[df.scenario .== scn, :]
            ranks = scn_rows.relative_rank
            if length(unique(ranks)) > 1
                all_consistent = false
            end
        end
        if all_consistent
            println(io, "Scenario rankings remain consistent across all parameter perturbations,")
            println(io, "confirming that policy recommendations are robust to parameter uncertainty.")
        else
            println(io, "Some scenario rankings shift under parameter perturbation,")
            println(io, "suggesting cautious interpretation of near-equivalent scenarios.")
        end
    else
        println(io, "*No results available — test not yet executed.*")
    end
    println(io, "")
    println(io, "![Parameter Sensitivity](fig_parameter_sensitivity.png)")
    println(io, "")
    println(io, "---")
    println(io, "")
    println(io, "## Summary")
    println(io, "")
    println(io, "| Test | Criterion | Status |")
    println(io, "|------|-----------|--------|")

    # SAA status
    saa_status = "—"
    if isfile(saa_csv)
        df_saa = CSV.read(saa_csv, DataFrame)
        final = combine(groupby(df_saa, :scenario_name), last)
        saa_status = all(final.converged) ? "✓ Passed" : "△ Partial"
    end
    println(io, "| SAA Convergence | CI ratio < 10% | $saa_status |")

    # OOS status
    oos_status = "—"
    if isfile(oos_csv)
        df_oos = CSV.read(oos_csv, DataFrame)
        mg = max(maximum(abs.(df_oos.gap_f1_pct)), maximum(abs.(df_oos.gap_f3_pct)))
        oos_status = mg < 15.0 ? "✓ Passed (max $(round(mg; digits=1))%)" : "✗ Failed (max $(round(mg; digits=1))%)"
    end
    println(io, "| Out-of-Sample | Gap < 15% | $oos_status |")

    # Stability status
    stab_status = "—"
    if isfile(stab_csv)
        df_stab = CSV.read(stab_csv, DataFrame)
        mc = maximum(df_stab.hv_cv)
        stab_status = mc < 5.0 ? "✓ Passed (CV max $(round(mc; digits=1))%)" : "△ Moderate (CV max $(round(mc; digits=1))%)"
    end
    println(io, "| Algorithmic Stability | HV CV < 5% | $stab_status |")

    # Sensitivity status
    sens_status = "—"
    if isfile(sens_csv)
        sens_status = "✓ Completed"
    end
    println(io, "| Parameter Sensitivity | OAT analysis | $sens_status |")

    println(io, "")
    println(io, "---")
    println(io, "*Report generated by robustness_tests.jl*")

    # Write to file
    open(report_path, "w") do f
        write(f, String(take!(io)))
    end
    println("  📄 Saved: $report_path")
end

"""
    generate_all_figures()

Generate all publication-quality figures from saved CSV results.
Can be called independently after tests are complete.
"""
function generate_all_figures()
    println("\n" * "="^60)
    println("  GENERATING PUBLICATION FIGURES")
    println("="^60)
    
    mkpath(ROBUSTNESS_OUTDIR)
    
    for (name, func, file) in [
        ("SAA Convergence", plot_saa_convergence, "saa_convergence.csv"),
        ("Out-of-Sample", plot_out_of_sample, "out_of_sample_validation.csv"),
        ("Algorithmic Stability", plot_algorithmic_stability, "algorithm_stability_hv.csv"),
        ("Parameter Sensitivity", plot_parameter_sensitivity, "parameter_sensitivity.csv"),
    ]
        csv = joinpath(ROBUSTNESS_OUTDIR, file)
        if isfile(csv)
            println("\n  ─── $name ───")
            try
                func(csv)
            catch e
                println("  ⚠ Failed: $e")
            end
        else
            println("  ⚠ $file not found — run test first")
        end
    end
    
    # Generate comprehensive report from all CSV results
    println("\n  ─── Robustness Report ───")
    try
        generate_robustness_report()
    catch e
        println("  ⚠ Report generation failed: $e")
    end

    println("\n" * "="^60)
    println("  FIGURES & REPORT COMPLETE")
    println("="^60)
end

# ═══════════════════════════════════════════════════════════════════════════
# Standalone Figure/Report Regeneration (no re-run needed)
# ═══════════════════════════════════════════════════════════════════════════

"""
    regenerate_robustness_figures(result_dir::String)

Re-generate all robustness figures and report from previously saved CSVs.
No test re-run needed — reads existing CSV files in result_dir.

Usage:
    regenerate_robustness_figures(joinpath(homedir(), "Desktop", "runs", "robustness_results"))
"""
function regenerate_robustness_figures(result_dir::String=ROBUSTNESS_OUTDIR)
    if !isdir(result_dir)
        println("❌ $result_dir 디렉토리가 없습니다. 먼저 robustness 테스트를 실행하세요.")
        return
    end

    println("\n🔄 저장된 CSV로부터 robustness 시각화 재생성 중...")
    println("   결과 디렉토리: $result_dir")

    for (name, func, file) in [
        ("SAA Convergence", plot_saa_convergence, "saa_convergence.csv"),
        ("Out-of-Sample", plot_out_of_sample, "out_of_sample_validation.csv"),
        ("Algorithmic Stability", plot_algorithmic_stability, "algorithm_stability_hv.csv"),
        ("Parameter Sensitivity", plot_parameter_sensitivity, "parameter_sensitivity.csv"),
    ]
        csv = joinpath(result_dir, file)
        if isfile(csv)
            println("\n  ─── $name ───")
            try
                func(csv; output_dir=result_dir)
            catch e
                println("  ⚠ Failed: $e")
            end
        else
            println("  ⚠ $file not found — skipping")
        end
    end

    println("\n  ─── Robustness Report ───")
    try
        generate_robustness_report(result_dir)
    catch e
        println("  ⚠ Report generation failed: $e")
    end

    println("\n🔄 robustness 시각화 재생성 완료!")
end

# ═══════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════

function run_robustness_tests()
    test_name = get(ENV, "ROBUSTNESS_TEST", "all")
    
    println("\n" * "═"^80)
    println("  ROBUSTNESS TEST SUITE — Multi-Trip CVRP Budapest")
    println("  Test: $(test_name)")
    println("  Time: $(Dates.now())")
    println("═"^80)
    
    total_start = time()
    
    if test_name == "all" || test_name == "saa"
        test_saa_convergence()
    end
    
    if test_name == "all" || test_name == "oos"
        test_out_of_sample()
    end
    
    if test_name == "all" || test_name == "stability"
        test_algorithmic_stability()
    end
    
    if test_name == "all" || test_name == "sensitivity"
        test_parameter_sensitivity()
    end
    
    # Generate publication figures from results
    if test_name == "all" || test_name == "figures"
        generate_all_figures()
    end
    
    total_elapsed = round(time() - total_start; digits=1)
    println("\n" * "═"^80)
    println("  ALL TESTS COMPLETE — Total time: $(total_elapsed)s")
    println("═"^80)
end

# Auto-run when executed as main script
if abspath(PROGRAM_FILE) == @__FILE__
    run_robustness_tests()
end
