module TimerNLPModels

using NLPModels
using TimerOutputs

export TimerNLPModel, TimerNLSModel, get_timer

mutable struct TimerNLPModel{T, S, Model} <: AbstractNLPModel{T, S}
  nlp::Model
  meta::AbstractNLPModelMeta{T, S}
  timer
  function TimerNLPModel(nlp::AbstractNLPModel{T, S}) where {T, S}
    return new{T, S, typeof(nlp)}(nlp, nlp.meta, TimerOutput())
  end
end

mutable struct TimerNLSModel{T, S, Model} <: AbstractNLSModel{T, S}
  nlp::Model
  meta::AbstractNLPModelMeta{T, S}
  nls_meta::NLSMeta{T, S}
  timer
  function TimerNLSModel(nlp::AbstractNLSModel{T, S}) where {T, S}
    return new{T, S, typeof(nlp)}(nlp, nlp.meta, nlp.nls_meta, TimerOutput())
  end
end

const TimerModel{T, S} = Union{TimerNLPModel{T, S}, TimerNLSModel{T, S}}

@default_counters TimerModel nlp
@default_nlscounters TimerNLSModel nlp

get_nlp(model::TimerModel) = model.nlp
get_timer(model::TimerModel) = model.timer

function Base.show(io::IO, nlp::TimerModel)
  show(io, get_nlp(nlp))
end

include("api.jl")
include("nls_api.jl")

end
