using Random
using Distributions
using Base: @kwdef
# using Revise

@kwdef struct DDM
    # d::Float64 = .0002
    d1::Float64 = .0001
    d2::Float64 = .0001
    threshold1::Float64 = 0.8
    threshold2::Float64 = 1.0
    t1_error::Float64  = 15
    t2_error::Float64  = 10
end


function final_termination(dv, threshold)
    if length(dv) == 2
        a, b = dv
        a - b > threshold && return 1
        b - a > threshold && return 2
        return 0
    elseif length(dv) == 3
        a, b, c = dv
        a - max(b, c) > threshold && return 1
        b - max(a, c) > threshold && return 2
        c - max(a, b) > threshold && return 3
        return 0
    elseif length(dv) == 4
        a, b, c, d = dv
        a - max(b, c, d) > threshold && return 1
        b - max(a, c, d) > threshold && return 2
        c - max(a, b, d) > threshold && return 3
        d - max(a, b, c) > threshold && return 4
        return 0
    else
        best, next = partialsortperm(dv, 1:2, rev=true)
        dv[best] - dv[next] > model.threshold && return best
        return 0
    end
end

function simulate(model::DDM, v::Vector{Float64}; maxt=100000, logger=(dv, t) -> nothing)
    N = length(v)
    noise = Normal(0, model.σ)
    drift = model.d * v
    dv = zeros(N)  # total accumulated evidence
    choice = 0

    for t in 1:maxt
        logger(dv, t)
        for i in 1:N
            dv[i] += drift[i] + rand(noise)
        end
        choice = final_termination(dv, model.threshold)
        if choice != 0
            return (choice, t)
        end
    end
    (0, -1)
end



function value_function(v1::Vector{Float64}, v2::Vector{Float64}, weight=1.0)
    v = Vector{Float64}(undef, length(v2))
    weight1 = weight
    weight2 = 1.0 
    expanded_v1 = Vector{Float64}(undef, length(v2))

    if length(v2) == 4
        expanded_v1[1:2] .= v1[1]  
        expanded_v1[3:4] .= v1[2]
        v[1:2] .= weight1 * v1[1] .+ weight2 * (v2[1] + v2[2])/2
        v[3:4] .= weight1 * v1[2] .+ weight2 * (v2[3] + v2[4])/2
    elseif length(v2) == 3
        expanded_v1[1] = v1[1]  # Direct assignment for single elements
        expanded_v1[2:3] .= v1[2]  # Broadcasting for filling more than one element

        # Direct assignment for single elements and broadcasting for slices
        v[1] = weight1 * v1[1] + weight2 * v2[1]
        v[2:3] .= weight1 * v1[2] .+ weight2 * (v2[2] + v2[3])/2
    else
        error("Unsupported configuration of v2 with length $(length(v2))")
    end

    return v, expanded_v1, v2
end


function simulate_two_stage(model::DDM, v1::Vector{Float64}, v2::Vector{Float64}; maxt=5000, logger=(dv, stage, t) -> nothing)
    N = length(v2)  # There are always 3/4 options in the two-stage decision model



    noise1 = Normal(0,0.1)
    noise2 = Normal(0,0.1)
    t1_error = model.t1_error
    t2_error = model.t2_error
    v,v1,v2 = value_function(v1, v2)
    stage1_drifts = model.d1 .* v
    stage2_drifts = model.d2 .* v2 


    rt1, rt2 = 0, 0
    choice1, choice2 = 0, 0

    # Stage 1: Decide between L and R
    # dv_stage1 = zeros(N)
    dv = zeros(N)
    dv_alt = zeros(N)
    for t in 1:maxt
        logger(copy(dv), 1, t)

        # ε1 = rand(noise1)
        # ε2 = rand(noise1)
        # dv[i] += stage1_drifts[i] + stage1_noise + rand(noise1)
        
        for i in 1:N
            dv[i] += stage1_drifts[i] + rand(noise1)
        end
        # apply_inhibition!(model, dv, dv_alt)
        choice1 = final_termination(dv, model.threshold1)
        if choice1 != 0
            rt1 = t + t1_error
            break
        end
    end

    # if model.restart_stage2
    #     dv .= 0
    # end

    # Stage 2
    if N == 3
        indices = choice1 < 3 ? [1,2] : [3]  # Adjusted for a 3-option scenario
    elseif N == 4
        indices = choice1 < 3 ? [1, 2] : [3, 4]  # Original setup for 4 options
    end

    # Assuming decision can be made directly if only one option
    if length(indices) == 1
        rt2 = t2_error # Assuming a fixed time for the second decision
        return (choice1, rt1, rt2)
    end

    # Otherwise, possibly change decision
    for t in 1:(maxt-rt1)
        logger(copy(dv), 2, t)
        # Stage 1: Decide between L and R
        for i in indices
            dv[i] += stage2_drifts[i] + rand(noise2)  # Continue accumulating evidence for the second decision
        end
        # apply_inhibition!(model, @view(dv[indices]), dv_alt)
        choice = final_termination(@view(dv[indices]), model.threshold2)  # Pass the relevant evidence to final_termination

        if choice != 0
            choice2 = indices[choice]
            rt2 = t + t2_error # Calculate the reaction time for the second decision
            return (choice2, rt1, rt2)
        end
    end
    return (0, -1, -1)
end


function simulate_one_stage(model::DDM, v1::Vector{Float64}, v2::Vector{Float64}; maxt=5000, logger=(dv, stage, t) -> nothing)
    N = length(v2)  # There are always 3/4 options in the two-stage decision model

    noise1 = Normal(0,0.1)
    noise2 = Normal(0,0.1)
    t1_error = model.t1_error
    t2_error = model.t2_error
    v,v1,v2 = value_function(v1, v2)
    stage1_drifts = model.d1 .* v
    stage2_drifts = model.d2 .* v2 
    rt1, rt2 = 0, 0
    choice1, choice2 = 0, 0

    # Stage 1: Decide between L and R
    dv = zeros(N)
    # dv_alt = zeros(N)
    for t in 1:maxt
        logger(copy(dv), 1, t)

        # ε1 = rand(noise1)
        # ε2 = rand(noise1)
        # dv[i] += stage1_drifts[i] + stage1_noise + rand(noise1)
        
        for i in 1:N
            dv[i] += stage1_drifts[i] + rand(noise1)
        end
        # apply_inhibition!(model, dv, dv_alt)
        choice1 = final_termination(dv, model.threshold1)
        if choice1 != 0
            rt1 = t + t1_error
            break
        end
    end

    # if model.restart_stage2
    #     dv .= 0
    # end

    # Stage 2
    if N == 3
        indices = choice1 < 3 ? [1,2] : [3]  # Adjusted for a 3-option scenario
    elseif N == 4
        indices = choice1 < 3 ? [1, 2] : [3, 4]  # Original setup for 4 options
    end

    # Assuming decision can be made directly if only one option
    if length(indices) == 1
        rt2 = t2_error # Assuming a fixed time for the second decision
        return (choice1, rt1, rt2)
    end

    # Otherwise, possibly change decision
    for t in 1:(maxt-rt1)
        # reset dv 
        dv .= 0
        # Stage 1: Decide between L and R
        for i in indices
            dv[i] += stage2_drifts[i] + rand(noise2)  # Continue accumulating evidence for the second decision
        end
        # apply_inhibition!(model, @view(dv[indices]), dv_alt)
        choice = final_termination(@view(dv_stage2[indices]), model.threshold2)  # Pass the relevant evidence to final_termination

        if choice != 0
            choice2 = indices[choice]
            rt2 = t + t2_error # Calculate the reaction time for the second decision
            return (choice2, rt1, rt2)
        end
    end
    return (0, -1, -1)
end


