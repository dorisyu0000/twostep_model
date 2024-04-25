
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
        choice1::Int  # Decision at stage 1: 0 for no decision, 1 for left, 2 for right
        choice2::Int  # Decision at stage 2: 0 for no decision, 12 for LL, 22 for LR, 21 for RL,22 for RR
        rt1::Float64  # Reaction time for the first decision
        rt2::Float64 # Reaction time for the second decision
    end


    function log_likelihood(model, trials::Vector{Trial}; kws...)
        mapreduce(+, trials) do trial
            log_likelihood(model, trial; kws...)
        end
    end

    struct LapseModel
        N::Int  # Number of options at stage 2 (assuming max number of paths possible)
        max_rt1::Int
        max_rt2::Int
    end
    
    function LapseModel(trials::Vector{Trial})
        N = maximum([length(trial.value2) for trial in trials])  # Maximum number of paths in stage 2
        max_rt1 = maximum(trial.rt1 for trial in trials)
        max_rt2 = maximum(trial.rt2 for trial in trials)
        max_rt1 = round(Int, max_rt1)
        max_rt2 = round(Int, max_rt2)
        return LapseModel(N, max_rt1, max_rt2)
    end
    
    function simulate_two_stage_lapse(model::LapseModel)
        choice2 = rand(1:model.N)  
        rt1 = rand(1:model.max_rt1)  # Random reaction time for the first stage
        rt2 = rand(1:model.max_rt2)  # Random reaction time for the second stage
        choice1 = choice2 <= 2 ? 1 : 2
        return (choice1, rt1, choice2, rt2)
    end
    

    function log_likelihood(model::LapseModel, trial::Trial; rt_tol=20)
        p_choice = 1 / model.N  # Uniform probability over N choices
        p_rt1 = 1 / model.max_rt1  # Uniform distribution of rt1
        p_rt2 = 1 / model.max_rt2  # Uniform distribution of rt2
        
        n_rt1_hits = length(max(1, trial.rt1 - rt_tol):min(model.max_rt1, trial.rt1 + rt_tol))
        n_rt2_hits = length(max(1, trial.rt2 - rt_tol):min(model.max_rt2, trial.rt2 + rt_tol))
        p_rt1 = n_rt1_hits / model.max_rt1
        p_rt2 = n_rt2_hits / model.max_rt2

        log_prob_choice = log(p_choice)
        log_prob_rt1 = n_rt1_hits > 0 ? log(p_rt1) : Float64(-Inf) 
        log_prob_rt2 = n_rt2_hits > 0 ? log(p_rt2) : Float64(-Inf)

        return log_prob_choice + log_prob_rt1 + log_prob_rt2
    end
    
    function log_likelihood(model::DDM, trials::Vector{Trial}; parallel=false, ε=.1, rt_tol=20, kws...)
        lapse = LapseModel(trials)
        min_logp = log_likelihood(lapse, trials; rt_tol)
        (logp, std) = ibs(trials; parallel, min_logp, kws...) do t
            if rand() < ε
                choice1, choice2, rt1, rt2 = simulate_two_stage_lapse(lapse)
            else
                choice1, choice2, rt1, rt2 = simulate_two_stage(model, t.value1, t.value2; maxt = t.rt1 +t.rt2 + rt_tol + 1) 
            end
            choice2 == t.choice2 && abs(rt1 - t.rt1) ≤ rt_tol && abs(rt2 - t.rt2) ≤ rt_tol
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
                data["choice1"],
                data["choice2"],
                Int(round(data["rt1"] / 100)),
                Int(round(data["rt2"] / 100))

            )
            push!(trials, trial)
        end
    end
    return trials
end

trials = load_trials("/Users/dorisyu/Documents/GitHub/fitting_example/trials1.json")


# %% --------
# model fitting 

box = Box(
    :d1 => (.001, .1),
    :d2 => (.01, .1),
    :threshold1 => (0.4, 0.8),
    :threshold2 => (0.8, 1.2),
    :t1_error => (1, 3),
    :t2_error => (1, 2),
)

params = grid(5, box)

like_grid = @showprogress pmap(params) do prm
    log_likelihood(DDM(;prm...), trials; rt_tol=10, repeats=3)
end



# %% --------
# Generate some random trials
Random.seed!(1)

function generate_random_floats(n, min_val, max_val)
    return [float(rand(min_val:max_val)) for _ in 1:n]
end

parameters = [0.001, 0.01, 0.8, 1.0, 3.36,0.88]
model = DDM(parameters...)

v1 = generate_random_floats(2, -4, 4)
v2 = generate_random_floats(4, -4, 4)
random_trials = map(1:10) do i
    choice1, choice2, rt1, rt2 = simulate_two_stage(model, v1,v2)
    trials = Trial(v1,v2, choice1, choice2, rt1, rt2)
end

prm = (d1=.01,d2=.0005,threshold1=1.0, threshold2=1.2)

log_likelihood(DDM(;prm...), trials; rt_tol=10, repeats=3)


# Bayesian Optimization
include("bads.jl")
box = Box(
    :d1 => (.0001, .01),
    :d2 => (.01, .1),
    :threshold1 => (0.5, 1.0),
    :threshold2 => (1.0, 1.5),
    :t1_error => (2, 8),
    :t2_error => (2, 8),
)
lower_bounds = [0.0001, 0.01, 0.5, 1.0, 2, 2]  # Corresponding to the lower bounds of d1, d2, threshold1, threshold2, t1_error, t2_error
upper_bounds = [0.01, 0.1, 1.0, 1.5, 8, 8]    # Corresponding to the upper bounds

bads = optimize_bads(lower_bounds=lower_bounds, upper_bounds=upper_bounds, specify_target_noise=true, tol_fun=10, max_fun_evals=1000) do params
    # Extract parameters from the vector
    d1, d2, threshold1, threshold2, t1_error, t2_error = params
    model = DDM(d1=d1, d2=d2, threshold1=threshold1, threshold2=threshold2, t1_error=t1_error, t2_error=t2_error)
    logp, std = log_likelihood(model, trials; repeats=1)
    if ismissing(std)
        std = 1.0  # Set a default std value if missing
    end
    (-logp, std)
end

d1, d2, threshold1, threshold2, t1_error, t2_error = get_result(bads)[:x]
DDM(;d1, d2, threshold1, threshold2, t1_error, t2_error)


# initialize with values drawn from sobol sequence (covers space better than grid)
include("gp_min.jl")


initX = sobol(20, box)

init = @showprogress map(initX) do x
    model = DDM(;box(x)...)  # Assuming this constructs a DDM model correctly
    log_likelihood(model,trials; repeats=5)
end

# IBS gives a variance estimate, so we tell this to the Gaussian Process
lognoise = log(maximum(getfield.(init, :std)))
noisebounds = [lognoise, lognoise]

# use the seed values as the GP initialization points
y = getfield.(init, :logp)
init_Xy = (reduce(hcat, initX), -y)

result_gp = gp_minimize(length(box); iterations=180, verbose=true, init_Xy, noisebounds) do x
    model = DDM(;box(x)...)
    -log_likelihood(model, trials; repeats=5).logp
end


mle = DDM(;box(result_gp.model_optimizer)...)
log_likelihood(mle, trials; repeats=50)
log_likelihood(model, trials; repeats=50)

