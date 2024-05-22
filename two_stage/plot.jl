using Pkg
using Distributed
using JSON

@time @everywhere begin

    include("ddm.jl")
    include("box.jl")
    include("ibs.jl")

    using Sobol
    using ProgressMeter
    using Serialization

    struct Trial
        rewards::Vector{Float64} # Rewards for each path: [R, L]
        value1::Vector{Float64}  # Rewards for each path: [R, L]
        value2::Vector{Float64}  # Rewards for each path: [R_LL, R_LR, R_RL, R_RR]
        choice::Int  # Decision at stage 2: 0 for no decision, 12 for LL, 22 for LR, 21 for RL,22 for RR
        rt1::Float64  # Reaction time for the first decision
        rt2::Float64 # Reaction time for the second decision
        diff::Int # Difference between the two values
    end
end 

function load_trials(filename::String)
    trials = Trial[]  # Initialize an empty array to store the trials
    open(filename, "r") do file
        for line in eachline(file)
            data = JSON.parse(line)
            rewards = Float64.(data["rewards"])

            trial = Trial(
                rewards,
                Float64.(data["value1"]),
                Float64.(data["value2"]),
                data["choice2"],
                Float64(round(data["rt1"]/10)),
                Float64(round(data["rt2"]/10)),
                calculate_difficulty(rewards)
            )
            push!(trials, trial)
        end
    end
    return trials
end



function calculate_difficulty(rewards::Vector{Float64})
    max_r = maximum(rewards)
    filtered_rewards = filter(r -> r != max_r, rewards)
    avg_r = mean(filtered_rewards)
    return round(Int, max_r - avg_r)  # Ensure you're returning an Int.
end

file_trials = load_trials("/Users/dorisyu/Documents/GitHub/fitting_example/AllTrial.json")



# Define the DDM model parameters


parameters1 = [0.00002659494247718415, 0.000000030975827291885197, 0.988125, 1.8001999999999999, 300, 100]
model1 = DDM(parameters1...)

# model2 = DDM(parameters2...)

random_trials = map(file_trials) do trial
    r = trial.rewards
    v1 = trial.value1
    v2 = trial.value2
    diff = trial.diff
    choice, rt1, rt2 = simulate_two_stage(model1, v1, v2)
    Trial(r,v1, v2, choice, rt1, rt2,diff)
end

average_rt1 = mean(trial.rt1 for trial in file_trials)
average_rt2 = mean(trial.rt2 for trial in file_trials)

average_rt1 = mean(trial.rt1 for trial in random_trials)
average_rt2 = mean(trial.rt2 for trial in random_trials)

# Function to calculate average reaction times by difficulty
function calculate_metrics(trials)
    rt1_by_difficulty = Dict()
    rt2_by_difficulty = Dict()
    acc_by_difficulty = Dict()
    
    for trial in trials
        # Calculate accuracy dynamically
        accuracy = (argmax(trial.rewards) == trial.choice) ? 1 : 0

        push!(get!(rt1_by_difficulty, trial.diff, Float64[]), trial.rt1)
        push!(get!(rt2_by_difficulty, trial.diff, Float64[]), trial.rt2)
        push!(get!(acc_by_difficulty, trial.diff, Int[]), accuracy)
    end

    avg_rt1 = [(diff, mean(rts)) for (diff, rts) in rt1_by_difficulty]
    avg_rt2 = [(diff, mean(rts)) for (diff, rts) in rt2_by_difficulty]
    avg_acc = [(diff, mean(accs)) for (diff, accs) in acc_by_difficulty]

    sort!(avg_rt1, by = x -> x[1])
    sort!(avg_rt2, by = x -> x[1])
    sort!(avg_acc, by = x -> x[1])

    return avg_rt1, avg_rt2, avg_acc
end


avg_rt1_file, avg_rt2_file, avg_acc_file = calculate_metrics(file_trials)
avg_rt1_random, avg_rt2_random, avg_acc_random = calculate_metrics(random_trials)

# Plotting
plot(layout = 3, size = (900, 400))
plot!(subplot = 1, [x[1] for x in avg_rt1_file], [x[2] for x in avg_rt1_file], label = "RT1 - File Trials", color = :blue, marker = :circle)
plot!(subplot = 1, [x[1] for x in avg_rt1_random], [x[2] for x in avg_rt1_random], label = "RT1 - Random Trials", color = :red, marker = :square)
plot!(subplot = 2, [x[1] for x in avg_rt2_file], [x[2] for x in avg_rt2_file], label = "RT2 - File Trials", color = :blue, marker = :circle)
plot!(subplot = 2, [x[1] for x in avg_rt2_random], [x[2] for x in avg_rt2_random], label = "RT2 - Random Trials", color = :red, marker = :square)
plot!(subplot = 3, [x[1] for x in avg_acc_file], [x[2] for x in avg_acc_file], label = "Accuracy - File Trials", color = :blue, marker = :circle)
plot!(subplot = 3, [x[1] for x in avg_acc_random], [x[2] for x in avg_acc_random], label = "Accuracy - Random Trials", color = :red, marker = :circle)






