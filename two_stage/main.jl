using Pkg
using Distributed


@time @everywhere begin
    include("ddm.jl")
    include("box.jl")
    include("ibs.jl")
    include("bads.jl")

    using Sobol
    using ProgressMeter
    using Serialization


    
end


box = Box(
    :d1 => (.0001, .005),
    :d2 => (.0001, .01),
    :threshold1 => (0.8, 1.2),
    :threshold2 => (1.2, 1.8),
)
lower_bounds = [0.0001, 0.0001, 0.2, 0.8,8,4]  # Corresponding to the lower bounds of d1, d2, threshold1, threshold2, t1_error, t2_error
upper_bounds = [0.01, 0.01, 0.8, 1.2,20,11]    # Corresponding to the upper bounds

bads = optimize_bads(lower_bounds=lower_bounds, upper_bounds=upper_bounds, specify_target_noise=true, tol_fun=5, max_fun_evals=1000) do params
    # Extract parameters from the vector
    d1, d2, threshold1, threshold2, t1_error, t2_error= params
    model = DDM(d1=d1, d2=d2, threshold1=threshold1, threshold2=threshold2, t1_error=t1_error, t2_error=t2_error)
    logp, std = log_likelihood(model, trials; repeats=1)
    if ismissing(std)
        std = 1.0  # Set a default std value if missing
    end
    (-logp, std)
end

d1, d2, threshold1, threshold2, t1_error, t2_error= get_result(bads)[:x]
true_model = DDM(;d1, d2, threshold1, threshold2, t1_error, t2_error)
true_logp = log_likelihood(true_model, trials)