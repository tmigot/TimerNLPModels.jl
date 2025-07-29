using BenchmarkTools, ADNLPModels
using OptimizationProblems
using TimerNLPModels

# Run locally with `tune!(SUITE)` and then `run(SUITE)`
const SUITE = BenchmarkGroup()

for n in [100, 1000]
    g = zeros(T, n)
    SUITE["grad! ref"]["$n"] = @benchmarkable grad!(nlp, get_x0(nlp), $g) setup = (nlp = OptimizationProblems.ADNLPProblems.arglina(n = $n), timed_nlp = TimerNLPModel(nlp))
    SUITE["grad! tim"]["$n"] = @benchmarkable grad!(timed_nlp, get_x0(nlp), $g) setup = (nlp = OptimizationProblems.ADNLPProblems.arglina(n = $n), timed_nlp = TimerNLPModel(nlp))
end
for n in [100, 1000]
    Hv = zeros(T, n)
    SUITE["hprod!"]["$n"] = @benchmarkable hprod!(nlp, get_x0(nlp), get_x0(nlp), $Hv) setup = (nlp = OptimizationProblems.ADNLPProblems.arglina(n = $n))
end
