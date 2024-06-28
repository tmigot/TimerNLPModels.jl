# Nonlinear optimization

```@example ex1
using NLPModelsIpopt, ADNLPModels, TimerNLPModels
nlp = TimerNLPModel(ADNLPModel(x -> sum(x.^2), ones(3)))
stats = ipopt(nlp, print_level = 0)
```

```@example ex1
get_timer(nlp)
```

# Nonlinear least squares

```@example ex2
using NLPModelsIpopt, ADNLPModels, TimerNLPModels
nls = TimerNLPModel(ADNLSModel(x -> x, ones(3), 3))
stats = ipopt(nls, print_level = 0)
```

```@example ex2
get_timer(nls)
```
