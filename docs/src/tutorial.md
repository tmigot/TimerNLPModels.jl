```@example ex1
using NLPModelsIpopt, ADNLPModels, TimerNLPModels
nlp = TimerNLPModel(ADNLPModel(x -> sum(x.^2), ones(3)))
stats = ipopt(nlp, print_level = 0)
```

```@example ex1
get_timer(nlp)
```