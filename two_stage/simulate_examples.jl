include("ddm.jl")

model = DDM()
v = [1., 2.]

function logged_simulate(model, v1, v2; kws...)
    dvs = (Vector{Float64}[], Vector{Float64}[])
    choice, rt1, rt2 = simulate_two_stage(model, v1, v2; kws..., logger=(dv, stage, t) -> push!(dvs[stage], dv))
    (;choice, rt1, rt2, dvs)
end

logged_simulate(model, [1., 1.], [1., 1., 2., 1.]).dvs