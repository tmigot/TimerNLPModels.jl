using Distributed

np = Sys.CPU_THREADS
addprocs(np - 1)

@everywhere using TimerNLPModels
@everywhere using CUDA, NLPModels, NLPModelsTest, Test

@everywhere function nlp_tests(p)
  @testset "TimerNLPModels.jl for NLP: $p" begin
    nlp = eval(Meta.parse(p))()
    timed_nlp = TimerNLPModel(nlp)
    consistent_nlps([nlp, timed_nlp]; exclude = [], linear_api = true)
  end
end
@everywhere function nls_tests(p)
  @testset "TimerNLPModels.jl for NLS: $p" begin
    nls = eval(Meta.parse(p))()
    timed_nls = TimerNLSModel(nls)
    exclude = p == "LLS" ? [hess_coord, hess] : []
    consistent_nlss([nls, timed_nls]; exclude = exclude, linear_api = true)
  end
end
@testset "TimerNLPModels.jl for NLP" begin
  pmap(nlp_tests, NLPModelsTest.nlp_problems)
end
@testset "TimerNLPModels.jl for NLS" begin
  pmap(nls_tests, NLPModelsTest.nls_problems)
end
