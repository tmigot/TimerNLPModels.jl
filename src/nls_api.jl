for meth in (:jac_structure_residual, :hess_structure_residual)
  @eval begin
    function NLPModels.$meth(nlp::TimerNLSModel)
      @timeit nlp.timer "$($meth)" $meth(get_nlp(nlp))
    end
  end
end

for meth in (:residual,)
  @eval begin
    function NLPModels.$meth(nlp::TimerNLSModel, x::AbstractVector)
      @timeit nlp.timer "$($meth)" $meth(get_nlp(nlp), x)
    end
  end
end

for meth in (:residual!, :jac_coord_residual!, :jprod_residual, :jtprod_residual)
  @eval begin
    function NLPModels.$meth(nlp::TimerNLSModel, x::AbstractVector, y::AbstractVector)
      @timeit nlp.timer "$($meth)" $meth(get_nlp(nlp), x, y)
    end
  end
end

for meth in (:hess_coord_residual!, :jprod_residual!, :jtprod_residual!)
  @eval begin
    function NLPModels.$meth(
      nlp::TimerNLSModel,
      x::AbstractVector,
      y::AbstractVector,
      z::AbstractVector,
    )
      @timeit nlp.timer "$($meth)" $meth(get_nlp(nlp), x, y, z)
    end
  end
end

function NLPModels.hprod_residual!(
  nlp::TimerNLSModel,
  x::AbstractVector,
  i::Int,
  v::AbstractVector,
  Hiv::AbstractVector,
)
  @timeit nlp.timer "hprod_residual!" hprod_residual!(get_nlp(nlp), x, i, v, Hiv)
end

function NLPModels.hprod_residual(nlp::TimerNLSModel, x::AbstractVector, i::Int, v::AbstractVector)
  @timeit nlp.timer "hprod_residual" hprod_residual(get_nlp(nlp), x, i, v)
end
