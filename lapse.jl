
struct Trial
    v::Vector{Float64}
    choice::Int
    rt::Int
end

function log_likelihood(model, trials::Vector{Trial}; kws...)
    mapreduce(+, trials) do trial
        log_likelihood(model, trial; kws...)
    end
end

struct LapseModel
    N::Int
    max_rt::Int
end

function LapseModel(trials)
    N = only(unique(length(t.v for t in trials)))
    max_rt = maximum(t.rt for t in trials)
    LapseModel(N, max_rt)
end

simulate(model::LapseModel) = (rand(1:model.N), rand(1:model.max_rt))


