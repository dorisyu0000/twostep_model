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
end 


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
                    Int(round(data["rt1"])),
                    Int(round(data["rt2"]))

                )
                push!(trials, trial)
            end
        end
        return trials
    end
trials1 = load_trials("/Users/dorisyu/Documents/GitHub/fitting_example/trials1.json")
trials2 = load_trials("/Users/dorisyu/Documents/GitHub/fitting_example/trials2.json")

# Define the DDM model parameters
parameters1 = [0.005047848112247565, 0.008870085305771531, 0.5, 1.0998, 14.0, 4.007]
parameters2 = [0.0046764985540873605, 0.009999999999658402, 0.5478767192200233, 0.9000000000005152, 10.0, 8.39574789114749]
# parameters2 = [0.04335455108562024, 0.0885242554918941, 0.3969524656211612, 0.3358071042959825, 0.9960389831987981, 1.0047869628386545, 0.21312658886546265]
model1 = DDM(parameters1...)
model2 = DDM(parameters2...)

# Assuming the existence of a function `simulate_two_stage` which does the actual simulation
# for the two-stage decision process. You will need to define this function based on your model.
# Here is a placeholder for the function:

function simulate_two_stage(model, v1, v2)
    choice, rt1, rt2 = simulate_two_stage(model, v1,v2)
    return choice, rt1, rt2
end


random_trial1 = map(trials1) do trial
    v1 = trial.value1
    v2 = trial.value2
    choice, rt1, rt2 = simulate_two_stage(model1, v1, v2)
    Trial(v1, v2, choice, rt1, rt2)
end

average_rt1 = mean(trial.rt1 for trial in trials1)
average_rt2 = mean(trial.rt2 for trial in trials1)

average_rt1 = mean(trial.rt1 for trial in random_trial1)
average_rt2 = mean(trial.rt2 for trial in random_trial1)

random_trial2 = map(trials2) do trial
    v1 = trial.value1
    v2 = trial.value2
    choice, rt1, rt2 = simulate_two_stage(model2, v1, v2)
    Trial(v1, v2, choice, rt1, rt2)
end

average_rt1 = mean(trial.rt1 for trial in trials2)
average_rt2 = mean(trial.rt2 for trial in trials2)
average_rt2 = mean([trial.rt2 for trial in trials2 if trial.choice == 1 || trial.choice == 2])

average_rt1 = mean(trial.rt1 for trial in random_trial2)
average_rt2 = mean(trial.rt2 for trial in random_trial2)
average_rt2 = mean([trial.rt2 for trial in random_trial2 if trial.choice == 1 || trial.choice == 2])
