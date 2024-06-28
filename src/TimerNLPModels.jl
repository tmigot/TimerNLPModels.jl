module TimerNLPModels

using NLPModels
using TimerOutputs

export TimerNLPModel, get_timer

mutable struct TimerNLPModel{T, S, Model} <: AbstractNLPModel{T, S}
  nlp::Model
  meta::AbstractNLPModelMeta{T, S}
  timer
  function TimerNLPModel(nlp::AbstractNLPModel{T, S}) where {T, S}
    return new{T, S, typeof(nlp)}(nlp, nlp.meta, TimerOutput())
  end
end

@default_counters TimerNLPModel nlp

get_nlp(model::TimerNLPModel) = model.nlp
get_timer(model::TimerNLPModel) = model.timer

function Base.show(io::IO, nlp::TimerNLPModel)
  show(io, get_nlp(nlp))
end

include("api.jl")

end
