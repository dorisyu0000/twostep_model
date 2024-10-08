using Pkg
using Distributed
using JSON
using Plots
using RCall
using DataFrames



    include("ddm.jl")
    include("box.jl")
    include("ibs.jl")


    using Sobol
    using ProgressMeter
    using Serialization

    mutable struct plot_trial
        rewards::Vector{Float64}
        value1::Vector{Float64}
        value2::Vector{Float64}
        choice::Int
        rt1::Float64
        rt2::Float64
        diff::Int
        diff_2::Int
    end


function load_trials(filename::String)
    trials = Vector{plot_trial}()
    open(filename, "r") do file
        for line in eachline(file)
            data = JSON.parse(line)
            # Initialize with default value for diff_2, will be calculated later
            trial = plot_trial(
                Float64.(data["rewards"]),
                Float64.(data["value1"]),
                Float64.(data["value2"]),
                data["choice2"],
                Float64(round(data["rt1"]/100)),
                Float64(round(data["rt2"]/100)),
                calculate_difficulty(Float64.(data["rewards"])), # Compute initial difficulty
                0  # Default value for diff_2, will be computed post-simulation
            )
            push!(trials, trial)
        end
    end
    return trials
end

function update_diff_2(trials::Vector{plot_trial})
    for trial in trials
        # Update diff_2 based on current information
        trial.diff_2 = calculate_diff_2(trial.rewards, trial.choice)
    end
    return trials  # Return updated trials
end




function calculate_diff_2(rewards::Vector{Float64}, choice::Int)
    if length(rewards) == 3 && choice == 3
        return -1  # Special case for choice 3 with 3 rewards
    elseif choice == 0 || choice == 1
        return abs(rewards[2] - rewards[1])  # Assuming indices need correction
    elseif choice == 2 || choice == 3
        return abs(rewards[3] - rewards[2])  # Assuming indices need correction
    end
    return 0  # Default case if no conditions are met
end



function calculate_difficulty(rewards::Vector{Float64})
    max_r = maximum(rewards)
    filtered_rewards = filter(r -> r != max_r, rewards)
    avg_r = mean(filtered_rewards)
    return round(Int, max_r - avg_r)  # Ensure you're returning an Int.
end

file_trials = load_trials("AllTrial.json")
trial_1 = load_trials("trials2.json")
trial_2 = load_trials("trials1.json")


# Define the DDM model parameters
parameters1 = [0.05859494247718415, 0.02975827291885197, 0.53688125, 0.9901999999999999, 13.0, 6.0]
# model = DDM_inhb(0.0093458974576076, 0.01099803975014784, 0.529015027867913, 0.887650683005397, 14.045589015309448, 1.59411932510365, 0.03204594505578, 0.03001855203555304)
# model = DDM(0.008783272201699212, 0.02000252503317679, 0.54688125, 0.9929024645912078, 15.568269433353877, 7.640827345191999)
# model = DDM(0.004901306944711048, 0.01827669192844268, 0.4917352094165164, 0.9285735791587721, 9.999999999999993, 4.403870491950274)
model = DDM(0.0090840931549732664, 0.015486032258723427, 0.5571922706540302, 1.0000000000008584, 12.005657553271893, 6.391783543961144)
# model = DDM_inhb(0.009726529863138292, 0.010008232021078822, 0.6812176275686801, 1.99944663320936, 10.033504640268763, 1.509387776355169, 0.08457085792362388, 0.02944467305109707)
# model = DDM(0.0044862351481092634, 0.01410375304853078, 0.5111433272885392, 1.0223317948525654, 9.989053798147943, 4.674152437064331)
# Function to simulate trials

function simulate_trial(trial, model)
    r = trial.rewards
    v1 = trial.value1
    v2 = trial.value2
    diff = trial.diff
    choice, rt1, rt2 = simulate_two_stage(model, v1, v2)
    return (choice, rt1, rt2, r, v1, v2, diff)
end



# Function to simulate multiple times for each trial and compute average and most frequent choice
function simulate_trials(trials, model, num_simulations=100)
    all_results = Vector{plot_trial}()  # Ensure this is correctly initialized
    for trial in trials
        all_rt1 = []
        all_rt2 = []
        all_choices = []
        
        for _ in 1:num_simulations
            result = simulate_trial(trial, model)
            choice, rt1, rt2 = result[1], result[2], result[3]
            push!(all_choices, choice)
            push!(all_rt1, rt1)
            push!(all_rt2, rt2)
        end
        
        average_rt1 = mean(all_rt1)
        average_rt2 = mean(all_rt2)
        highest_freq_choice = rand(all_choices)
        
        r = trial.rewards
        v1 = trial.value1
        v2 = trial.value2
        diff = trial.diff
        
        # Calculate diff_2 based on the most frequent choice
        diff_2 = calculate_diff_2(r, highest_freq_choice)
        
        # Ensure new trials are added to the results
        push!(all_results, plot_trial(r, v1, v2, highest_freq_choice, average_rt1, average_rt2, diff, diff_2))
    end
    return all_results
end

# Assuming simulate_trials is defined correctly as shown above
random_trials_file = simulate_trials(file_trials, model)
random_trials_1 = simulate_trials(trial_1, model)
random_trials_2 = simulate_trials(trial_2, model)

# If filtering is required, make sure it returns the correct type
filtered_trial_1 = filter(t -> t.choice != 3, random_trials_1)

# Ensure these are vectors of plot_trial before calling update_diff_2
random_trials_1 = update_diff_2(random_trials_1) 
random_trials_2 = update_diff_2(random_trials_2)


function calculate_metrics(trials)
    rt1_by_difficulty = Dict()
    rt2_by_difficulty = Dict()
    acc_by_difficulty = Dict()

    for trial in trials
        if trial.diff >= 2  # Ensuring this uses the correct difficulty level if needed
            accuracy = (argmax(trial.rewards) == trial.choice) ? 1 : 0
            # Ensure diff_2 is calculated post-simulation and integrated correctly
            trial.diff_2 = calculate_diff_2(trial.rewards, trial.choice)
            push!(get!(rt1_by_difficulty, trial.diff, Float64[]), trial.rt1 * 100)  # Scaling RT1 by 100
            push!(get!(rt2_by_difficulty, trial.diff, Float64[]), trial.rt2 * 100)  # Scaling RT2 by 100
            push!(get!(acc_by_difficulty, trial.diff, Int[]), accuracy)
        end
    end

    function calc_stats(data::Vector)
        n = length(data)
        mu = mean(data)
        sigma = std(data)
        se = sigma / sqrt(n)
        (mu, sigma, se)
    end

    avg_rt1 = [(diff, calc_stats(rts)...) for (diff, rts) in rt1_by_difficulty]
    avg_rt2 = [(diff, calc_stats(rts)...) for (diff, rts) in rt2_by_difficulty]
    avg_acc = [(diff, mean(accs), std(accs) / sqrt(length(accs))) for (diff, accs) in acc_by_difficulty]

    sort!(avg_rt1, by = x -> x[1])
    sort!(avg_rt2, by = x -> x[1])
    sort!(avg_acc, by = x -> x[1])

    (avg_rt1, avg_rt2, avg_acc)
end


function prepare_data(avg_data::Array{Tuple{Float64, Float64, Float64}}, group_label::String)
    df = DataFrame(
        Difficulty = [x[1] for x in avg_data],
        Value = [x[2] for x in avg_data],
        SE = [x[3] for x in avg_data],
        Group = group_label
    )

    # Standardize the x and y values
    df.Difficulty = (df.Difficulty .- mean(df.Difficulty)) ./ std(df.Difficulty)
    df.Value = (df.Value .- mean(df.Value)) ./ std(df.Value)
    return df
end

# Helper function to prepare RT2 by Diff2 data
function prepare_rt2_diff2_data(trials, group_label::String)
    data = [(trial.diff_2, trial.rt2 * 100) for trial in trials]
    df = DataFrame(
        Diff2 = [x[1] for x in data],
        RT2 = [x[2] for x in data],
        Group = group_label
    )

    # Standardize Diff2 and RT2
    df.Diff2 = (df.Diff2 .- mean(df.Diff2)) ./ std(df.Diff2)
    df.RT2 = (df.RT2 .- mean(df.RT2)) ./ std(df.RT2)
    return df
end

# Modified plot_and_save function
function plot_and_save(trials, random_trials, filename)
    # Calculate metrics
    avg_rt1, avg_rt2, avg_acc = calculate_metrics(trials)
    avg_rt1_random, avg_rt2_random, avg_acc_random = calculate_metrics(random_trials)

    # Prepare and standardize data
    df_rt1_exp = prepare_data(avg_rt1, "Experiment")
    df_rt1_model = prepare_data(avg_rt1_random, "Model")
    df_rt2_exp = prepare_data(avg_rt2, "Experiment")
    df_rt2_model = prepare_data(avg_rt2_random, "Model")
    df_acc_exp = prepare_data(avg_acc, "Experiment")
    df_acc_model = prepare_data(avg_acc_random, "Model")

    # Combine data frames
    df_rt1 = vcat(df_rt1_exp, df_rt1_model)
    df_rt2 = vcat(df_rt2_exp, df_rt2_model)
    df_acc = vcat(df_acc_exp, df_acc_model)

    # Prepare data for RT2 by Diff2
    df_rt2_diff2_exp = prepare_rt2_diff2_data(trials, "Experiment")
    df_rt2_diff2_model = prepare_rt2_diff2_data(random_trials, "Model")
    df_rt2_diff2 = vcat(df_rt2_diff2_exp, df_rt2_diff2_model)

    # Convert Julia DataFrames to R data frames
    @rput df_rt1 df_rt2 df_acc df_rt2_diff2

    # Load required R libraries
    R"""
    library(ggplot2)
    library(gridExtra)

    # Plot RT1
    p1 <- ggplot(df_rt1, aes(x = Difficulty, y = Value, color = Group)) +
        geom_point() +
        geom_errorbar(aes(ymin = Value - SE, ymax = Value + SE), width = 0.2) +
        labs(title = "RT1", x = "Standardized Difficulty", y = "Standardized Reaction Time") +
        theme_minimal()

    # Plot RT2
    p2 <- ggplot(df_rt2, aes(x = Difficulty, y = Value, color = Group)) +
        geom_point() +
        geom_errorbar(aes(ymin = Value - SE, ymax = Value + SE), width = 0.2) +
        labs(title = "RT2", x = "Standardized Difficulty", y = "Standardized Reaction Time") +
        theme_minimal()

    # Plot Accuracy
    p3 <- ggplot(df_acc, aes(x = Difficulty, y = Value, color = Group)) +
        geom_point() +
        geom_errorbar(aes(ymin = Value - SE, ymax = Value + SE), width = 0.2) +
        labs(title = "Accuracy", x = "Standardized Difficulty", y = "Standardized Accuracy") +
        theme_minimal()

    # Plot RT2 by Diff2
    p4 <- ggplot(df_rt2_diff2, aes(x = Diff2, y = RT2, color = Group)) +
        geom_point() +
        labs(title = "RT2 by Diff2", x = "Standardized Diff2", y = "Standardized RT2 (scaled)") +
        theme_minimal()

    # Arrange the plots in a grid
    g <- grid.arrange(p1, p2, p3, p4, ncol = 2)

    # Save the combined plot
    ggsave(filename = $filename, plot = g, width = 10, height = 8)
    """
end

# function plot_RT2(trials, random_trials, filename)
#     # Helper function to calculate stats
#     function calc_stats(data::Vector{Float64})
#         mu = mean(data)
#         se = std(data) / sqrt(length(data))
#         return (mu, se)
#     end

#     # Extracting RT2 data based on diff_2
#     function extract_rt2_data(trials)
#         rt2_by_diff_2 = Dict()
#         for trial in trials
#             if trial.diff_2 in 1:8  # Assuming diff_2 ranges from -8 to 8
#                 push!(get!(rt2_by_diff_2, trial.diff_2, Float64[]), trial.rt2 * 100)  # Scaling RT2 by 100 for visibility
#             end
#         end
#         return rt2_by_diff_2
#     end

#     rt2_data = extract_rt2_data(trials)
#     rt2_random_data = extract_rt2_data(random_trials)

#     # Calculate average RT2 and standard errors
#     avg_rt2 = [(diff_2, calc_stats(data)...) for (diff_2, data) in rt2_data]
#     avg_rt2_random = [(diff_2, calc_stats(data)...) for (diff_2, data) in rt2_random_data]

#     # Sorting data for consistent plotting
#     sort!(avg_rt2, by = x -> x[1])
#     sort!(avg_rt2_random, by = x -> x[1])

#     # Plotting
#     plt = plot(size=(600, 400), legend=:outertopright, title="RT2 by Diff2")
#     plot!(plt, [x[1] for x in avg_rt2], [x[2] for x in avg_rt2], ribbon=[x[3] for x in avg_rt2], label="RT2 - Experiment", color=:blue, marker=:circle)
#     plot!(plt, [x[1] for x in avg_rt2_random], [x[2] for x in avg_rt2_random], ribbon=[x[3] for x in avg_rt2_random], label="RT2 - Model", color=:red, marker=:square)

#     xlabel!(plt, "Diff2")
#     ylabel!(plt, "Reaction Time (scaled)")
#     savefig(plt, filename)
# end




# Plot and save each set of trials with simulated data
plot_and_save(file_trials, random_trials_file, "file_trialsDDM.png")
plot_and_save(trial_1, random_trials_1, "trial_1_DDM.png")
plot_and_save(trial_2, random_trials_2, "trial_2_DDM.png")

plot_RT2(trial_1, random_trials_1, "trial_1_RT2.png")
plot_RT2(trial_2, random_trials_2, "trial_2_RT2.png")

# Define the number of parameters
k = 6

# Define the number of observations (length of trial data)
n = length(file_trials)  # replace file_trials with the actual trial data

# Define the log-likelihood value at the estimated parameters
# log_likelihood = -12978.20419604053 