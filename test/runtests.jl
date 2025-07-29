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

io = IOBuffer();
map(
  nlp -> print_nlp_allocations(io, nlp; linear_api = true),
  map(x -> eval(Symbol(x))(), NLPModelsTest.nlp_problems),
)
print_nlp_allocations(io, LLS(); linear_api = true, exclude = [hess])
map(
  nlp -> print_nlp_allocations(io, nlp; linear_api = true),
  map(x -> eval(Symbol(x))(), setdiff(NLPModelsTest.nls_problems, ["LLS"])),
)

#=
map(
  nlp -> test_zero_allocations(nlp; linear_api = true),
  map(x -> TimerNLPModel(eval(Symbol(x))()), NLPModelsTest.nlp_problems),
)
test_zero_allocations(LLS(); linear_api = true, exclude = [hess])
map(
  nlp -> test_zero_allocations(nlp; linear_api = true),
  map(x -> TimerNLSModel(eval(Symbol(x))()), setdiff(NLPModelsTest.nls_problems, ["LLS"])),
)
=#

if CUDA.functional()
  @everywhere function nlp_gpu_tests(p)
    @testset "NLP tests of problem $p" begin
      nlp_from_T = T -> TimerNLPModel(eval(Symbol(p))(T))
      @testset "GPU multiple precision support of problem $p" begin
        CUDA.allowscalar() do
          multiple_precision_nlp_array(nlp_from_T, CuArray, linear_api = true, exclude = [])
        end
      end
    end
  end

  @everywhere function nls_gpu_tests(p)
    @testset "NLS tests of problem $p" begin
      nls_from_T = T -> TimerNLSModel(eval(Symbol(p))(T))
      exclude = p == "LLS" ? [hess_coord, hess] : []
      @testset "GPU multiple precision support of problem $p" begin
        CUDA.allowscalar() do
          multiple_precision_nls_array(nls_from_T, CuArray, linear_api = true, exclude = exclude)
        end
      end
    end
  end

  pmap(nlp_gpu_tests, NLPModelsTest.nlp_problems)
  pmap(nls_gpu_tests, NLPModelsTest.nls_problems)
end
