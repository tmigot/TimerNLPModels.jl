using BenchmarkTools, ADNLPModels, NLPModels
using OptimizationProblems
using TimerNLPModels

# Run locally with `tune!(SUITE)` and then `run(SUITE)`
const SUITE = BenchmarkGroup()

for n in [100, 1000]
  g = zeros(n)
  SUITE["grad! ref"]["$n"] = @benchmarkable grad!(nlp, get_x0(nlp), $g) setup =
    (nlp = OptimizationProblems.ADNLPProblems.arglina(n = $n))
  SUITE["grad! tim"]["$n"] = @benchmarkable grad!(timed_nlp, get_x0(timed_nlp), $g) setup =
    (timed_nlp = TimerNLPModel(OptimizationProblems.ADNLPProblems.arglina(n = $n)))
end
for n in [100, 1000]
  Hv = zeros(n)
  SUITE["hprod! ref"]["$n"] = @benchmarkable hprod!(nlp, get_x0(nlp), get_x0(nlp), $Hv) setup =
    (nlp = OptimizationProblems.ADNLPProblems.arglina(n = $n))
  SUITE["hprod! tim"]["$n"] =
    @benchmarkable hprod!(timed_nlp, get_x0(timed_nlp), get_x0(timed_nlp), $Hv) setup =
      (timed_nlp = TimerNLPModel(OptimizationProblems.ADNLPProblems.arglina(n = $n)))
end
