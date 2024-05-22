include("ddm.jl")
include("plot.jl")

using Plots

# Define a structure to hold the log data
struct DVLog
    stage1::Dict{Int, Vector{Float64}}
    stage2::Dict{Int, Vector{Float64}}
end


function dv_logger(dvlog::DVLog, dv::Vector{Float64}, stage::Int, t::Int)
    if stage == 1
        dvlog.stage1[t] = dv
    else
        dvlog.stage2[t] = dv
    end
end

function simulate_two_stage(model::DDM, v1::Vector{Float64}, v2::Vector{Float64}; maxt=10000, dvlog::DVLog)
    N = length(v2)  # Assume v2 determines the number of options in the decision model

    noise1 = Normal(0, 0.1)
    noise2 = Normal(0, 0.1)
    t1_error = model.t1_error
    t2_error = model.t2_error
    v, v1, v2 = value_function(v1, v2)  # Ensure this function correctly manipulates v1 and v2
    stage1_drifts = model.d1 .* v
    stage2_drifts = model.d2 .* v2 

    rt1, rt2 = 0, 0
    choice1, choice2 = 0, 0

    dv = zeros(N)  # Decision variables for stage 1

    for t in 1:maxt
        dv .= dv .+ stage1_drifts .+ rand.(noise1, N)  # Ensure vector sizes match
        dv_logger(dvlog, dv, 1, t)  # Log dv at each timestep for stage 1

        if any(dv .>= model.threshold1)
            choice1 = findfirst(dv .>= model.threshold1)
            rt1 = t + t1_error
            break
        end
    end

    if N == 3
        indices = choice1 < 3 ? [1, 2] : [3]
    elseif N == 4
        indices = choice1 < 3 ? [1, 2] : [3, 4]
    end

    if length(indices) == 1
        rt2 = t2_error
        return (choice1, rt1, rt2)
    end

    for t in 1:(maxt-rt1)
        dv_logger(dvlog, dv[indices], 2, t)  # Log dv at each timestep for stage 2
        dv[indices] .= dv[indices] .+ stage2_drifts[indices] .+ rand.(noise2, length(indices))

        if any(dv[indices] .>= model.threshold2)
            choice2 = indices[findfirst(dv[indices] .>= model.threshold2)]
            rt2 = t + t2_error
            return (choice1, choice2, rt1, rt2)
        end
    end

    return (0, -1, -1)
end






# Plotting function check
function plot_dv_changes(dvlog::DVLog, model::DDM)
    plt = plot(xlabel="Time (t)", ylabel="DV Value", title="DV Changes Over Time", legend=:outertopright)

    # Initialize DV values for each option
    dv_values = Dict(i => Float64[] for i in 1:4)

    # Plot stage 1
    for (t, dv_values_t) in dvlog.stage1
        for (i, dv) in enumerate(dv_values_t)
            push!(dv_values[i], dv)
        end
    end

    # Plot stage 2
    for (t, dv_values_t) in dvlog.stage2
        for (i, dv) in enumerate(dv_values_t)
            push!(dv_values[i], dv)
        end
    end

    # Plot DV values for each option
    for i in 1:4
        plot!(plt, 1:length(dv_values[i]), dv_values[i], label="Option $i", seriestype=:path)
    end

    # Plot threshold lines
    plot!(plt, [0, maximum(values(map(length, dv_values)))], [model.threshold1, model.threshold1], label="Threshold Stage 1", linestyle=:dash)
    plot!(plt, [0, maximum(values(map(length, dv_values)))], [model.threshold2, model.threshold2], label="Threshold Stage 2", linestyle=:dash)

    display(plt)
end



# Initialize the log
dvlog = init_log()

parameters1 = [0.004659494247718415, 0.0030975827291885197, 0.49688125, 0.9001999999999999, 12.5, 8.0]
model = DDM(parameters1...)

trials = trials[1:min(1, length(trials))] 

# Run simulation
dvlog = init_log()  # Initialize log before simulations
for trial in trials
    simulate_two_stage(model, v1, v2, maxt=5000, dvlog=dvlog)  # Pass dvlog correctly
    plot_dv_changes(dvlog) 
end


