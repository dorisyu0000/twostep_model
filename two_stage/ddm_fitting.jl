
using Pkg
using Distributed


@time @everywhere begin

    include("ddm.jl")
    include("box.jl")
    include("ibs.jl")


    using Sobol
    using ProgressMeter
    using Serialization

    struct Trial
        value1::Vector{Float64}  # Rewards for each path: [R, L]
        value2::Vector{Float64}  # Rewards for each path: [R_LL, R_LR, R_RL, R_RR]
        choice::Int  # Decision at stage 2: 0 for no decision, 12 for LL, 22 for LR, 21 for RL,22 for RR
        rt1::Float64  # Reaction time for the first decision
        rt2::Float64 # Reaction time for the second decision
    end


    function log_likelihood(model, trials::Vector{Trial}; kws...)
        mapreduce(+, trials) do trial
            log_likelihood(model, trial; kws...)
        end
    end

    struct LapseModel
        N::Int  # number of paths
        max_rt::Int
        min_rt1::Int
        min_rt2::Int
    end
    
    function LapseModel(trials::Vector{Trial})
        N = maximum(length(trial.value2) for trial in trials)  # Maximum number of paths
        max_rt = maximum((trial.rt1) + maximum(trial.rt2) for trial in trials)
        min_rt1 = minimum(trial.rt1 for trial in trials)
        min_rt2 = minimum(trial.rt2 for trial in trials)
        return LapseModel(N, max_rt, min_rt1, min_rt2)
    end
    
    function simulate_two_stage_lapse(model::LapseModel)
        (;N, max_rt, min_rt1, min_rt2) = model
        choice = rand(1:N)
        rt1 = rand(min_rt1:max_rt-min_rt2-1)  # Random reaction time for the first stage
        rt2 = rand(min_rt2:max_rt-rt1)  # Random reaction time for the second stage
        return (choice, rt1, rt2)
    end
    

    function log_likelihood(model::LapseModel, trial::Trial; rt_tol=5)
        p_choice = 1 / model.N  # Probability of choosing any of the N paths ui
        n_rt1_hits = length(max(model.min_rt1, trial.rt1 + rt_tol):min(model.max_rt - model.min_rt2, trial.rt1 - rt_tol))
        n_rt2_hits = length(max(model.min_rt2, trial.rt2 + rt_tol):min(model.max_rt - trial.rt1, trial.rt2 - rt_tol))

        p_rt1 = n_rt1_hits / (model.max_rt- model.min_rt2 - 1)
        p_rt2 = n_rt2_hits / (model.max_rt - trial.rt1 )
        logp = log(p_choice)+ log(p_rt1) + log(p_rt2)
        return logp
    end
    
    function log_likelihood(model::DDM, trials::Vector{Trial}; parallel=false, ε=.01, rt_tol=5, kws...)
        lapse = LapseModel(trials)
        min_logp = log_likelihood(lapse, trials; rt_tol)
        (logp, std) = ibs(trials; parallel, min_logp, kws...) do t
            if rand() < ε
                choice, rt1, rt2 = simulate_two_stage_lapse(lapse)
            else
                choice, rt1, rt2 = simulate_two_stage(model, t.value1, t.value2; maxt = t.rt1 +t.rt2 +rt_tol  +rt_tol + 1) 
            end
            choice == t.choice && abs(rt1 - t.rt1) ≤ rt_tol && abs(rt2 - t.rt2) ≤ rt_tol
        end
        if ismissing(std)
            std = 1.
        end
        logp, std
    end
end


# %% --------
# import trials
using JSON
function load_trials(filename::String)
    trials = Trial[]  # Initialize an empty array to store the trials
    open(filename, "r") do file
        for line in eachline(file)
            data = JSON.parse(line)
            trial = Trial(
                Float64.(data["value1"]),
                Float64.(data["value2"]),
                data["choice2"],
                Int(round(data["rt1"]/10)),
                Int(round(data["rt2"]/10))

            )
            push!(trials, trial)
        end
    end
    return trials
end

trials = load_trials("AllTrial.json")

# trials = trials[1:min(200, length(trials))] 
# %% --------m.

# model fitting 

# box = Box(
#     :d1 => (.001, .1),
#     :d2 => (.01, .1),
#     :threshold1 => (0.4, 0.8),
#     :threshold2 => (0.8, 1.2),
#     :t1_error => (1, 3),
#     :t2_error => (1, 2),
# )

# params = grid(5, box)

# like_grid = @showprogress pmap(params) do prm
#     log_likelihood(DDM(;prm...), trials; rt_tol=10, repeats=3)
# end

# Bayesian Optimization
# include("bads.jl")
# box = Box(
#     :d1 => (.0001, .005),
#     :d2 => (.0001, .01),
#     :threshold1 => (0.8, 1.2),
#     :threshold2 => (1.2, 1.8),
# )

# lower_bounds = [0.0000001, 0.0000001, 1, 1.8,30,15]  # Corresponding to the lower bounds of d1, d2, threshold1, threshold2, t1_error, t2_error
# upper_bounds = [0.00001, 0.00001, 1.8, 2.5,50,30]    # Corresponding to the upper bounds

# bads = optimize_bads(lower_bounds=lower_bounds, upper_bounds=upper_bounds, specify_target_noise=true, tol_fun=5, max_fun_evals=1000) do params
#     # Extract parameters from the vector
#     d1, d2, threshold1, threshold2, t1_error, t2_error= params
#     model = DDM(d1=d1, d2=d2, threshold1=threshold1, threshold2=threshold2, t1_error=t1_error, t2_error=t2_error)
#     logp, std = log_likelihood(model, trials; repeats=1)
#     if ismissing(std)
#         std = 1.0  # Set a default std value if missing
#     end
#     (-logp, std)
# end

# d1, d2, threshold1, threshold2, t1_error, t2_error= get_result(bads)[:x]
d1, d2, threshold1, threshold2, t1_error, t2_error= [9.991000000000001e-5, 1.6347947822679198e-6, 0.46989218749999995, 1.0005, 10.0, 3.0]
true_model = DDM(;d1, d2, threshold1, threshold2, t1_error, t2_error)
true_logp = log_likelihood(true_model, trials)

# println(true_logp)
# println(true_model)

# %% --------
# Generate some random trials by using given parmeter
# Random.seed!(1)

# function generate_random_floats(n, min_val, max_val)
#     return [float(rand(min_val:max_val)) for _ in 1:n]
# end

# parameters = [0.0001, 0.001, 0.7, 0.9, 30,5]
# model = DDM(parameters...)

# v1 = generate_random_floats(2, -4, 4)
# v2 = generate_random_floats(4, -4, 4)
# random_trials = map(1:10) do i
#     choice, rt1, rt2 = simulate_two_stage(model, v1,v2)
#     trials = Trial(v1,v2, choice, rt1, rt2)
# end

# prm = (d1=.0001,d2=.001,threshold1=0.8, threshold2=1.0)

# log_likelihood(DDM(;prm...),trials; rt_tol=1, repeats=5)
