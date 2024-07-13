using TimerNLPModels
using NLPModels, NLPModelsTest
using Test

@testset "TimerNLPModels.jl for NLP" begin
  @testset "NLP: $pb" for pb in NLPModelsTest.nlp_problems
    nlp = TimerNLPModel(eval(Meta.parse(pb))())
    consistent_nlps([nlp, nlp]; exclude = [], linear_api = true)
  end
end

@testset "TimerNLPModels.jl for NLS" begin
  @testset "NLS: $pb" for pb in NLPModelsTest.nls_problems
    exclude = pb == "LLS" ? [hess_coord, hess] : []
    nls = TimerNLSModel(eval(Meta.parse(pb))())
    consistent_nlss([nls, nls]; exclude = exclude, linear_api = true)
  end
end
