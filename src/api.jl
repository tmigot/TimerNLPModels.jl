for meth in (
  :obj,
  :grad,
  :cons,
  :cons_lin,
  :cons_nln,
  :jac_coord,
  :jac_lin_coord,
  :jac_nln_coord,
  :jac,
  :jac_lin,
  :jac_nln,
)
  @eval begin
    function NLPModels.$meth(nlp::TimerModel, x::AbstractVector)
      @timeit nlp.timer "$($meth)" $meth(get_nlp(nlp), x)
    end
  end
end
for meth in (
  :grad!,
  :cons!,
  :cons_lin!,
  :cons_nln!,
  :jprod,
  :jprod_lin,
  :jprod_nln,
  :jtprod,
  :jtprod_lin,
  :jtprod_nln,
  :objgrad,
  :objgrad!,
  :jac_coord!,
  :jac_lin_coord!,
  :jac_nln_coord!,
  :ghjvprod,
)
  @eval begin
    function NLPModels.$meth(nlp::TimerModel, x::AbstractVector, y::AbstractVector)
      @timeit nlp.timer "$($meth)" $meth(get_nlp(nlp), x, y)
    end
  end
end
for meth in (:ghjvprod!,)
  @eval begin
    function NLPModels.$meth(
      nlp::TimerModel,
      x::AbstractVector,
      y::AbstractVector,
      z::AbstractVector,
      Hv::AbstractVector,
    )
      @timeit nlp.timer "$($meth)" $meth(get_nlp(nlp), x, y, z, Hv)
    end
  end
end
for meth in (:hess, :hess_op, :hess_coord)
  @eval begin
    function NLPModels.$meth(
      nlp::TimerModel,
      x::AbstractVector{T};
      obj_weight::T = one(T),
    ) where {T}
      @timeit nlp.timer "$($meth)" $meth(get_nlp(nlp), x; obj_weight = obj_weight)
    end
  end
end
for meth in (:hess, :hprod, :hess_op, :hess_op!, :hess_coord, :hess_coord!)
  @eval begin
    function NLPModels.$meth(
      nlp::TimerModel,
      x::AbstractVector{T},
      y::AbstractVector;
      obj_weight::T = one(T),
    ) where {T}
      @timeit nlp.timer "$($meth)" $meth(get_nlp(nlp), x, y; obj_weight = obj_weight)
    end
  end
end
for meth in (:hprod!, :hess_op!, :hess_coord!)
  @eval begin
    function NLPModels.$meth(
      nlp::TimerModel,
      x::AbstractVector{T},
      y::AbstractVector,
      z::AbstractVector;
      obj_weight::T = one(T),
    ) where {T}
      @timeit nlp.timer "$($meth)" $meth(get_nlp(nlp), x, y, z; obj_weight = obj_weight)
    end
  end
end
for meth in (:hprod!,)
  @eval begin
    function NLPModels.$meth(
      nlp::TimerModel,
      x::AbstractVector{T},
      y::AbstractVector,
      z::AbstractVector,
      Hv::AbstractVector;
      obj_weight::T = one(T),
    ) where {T}
      @timeit nlp.timer "$($meth)" $meth(get_nlp(nlp), x, y, z, Hv; obj_weight = obj_weight)
    end
  end
end
for meth in (:jprod!, :jprod_lin!, :jprod_nln!, :jtprod!, :jtprod_lin!, :jtprod_nln!)
  @eval begin
    function NLPModels.$meth(
      nlp::TimerModel,
      x::AbstractVector,
      y::AbstractVector,
      z::AbstractVector,
    )
      @timeit nlp.timer "$($meth)" $meth(get_nlp(nlp), x, y, z)
    end
  end
end
for meth in (
  :jac_structure!,
  :jac_structure_residual!,
  :jac_lin_structure!,
  :jac_nln_structure!,
  :hess_structure!,
  :hess_structure_residual!,
)
  @eval begin
    function NLPModels.$meth(
      nlp::TimerModel,
      rows::AbstractVector{<:Integer},
      cols::AbstractVector{<:Integer},
    )
      @timeit nlp.timer "$($meth)" $meth(get_nlp(nlp), rows, cols)
    end
  end
end
for meth in (:jth_hess_coord,)
  @eval begin
    function NLPModels.$meth(nlp::TimerModel, x::AbstractVector, j::Int)
      @timeit nlp.timer "$($meth)" $meth(get_nlp(nlp), x, j)
    end
  end
end
for meth in (:jth_hess_coord!,)
  @eval begin
    function NLPModels.$meth(nlp::TimerModel, x::AbstractVector, j::Int, y::AbstractVector)
      @timeit nlp.timer "$($meth)" $meth(get_nlp(nlp), x, j, y)
    end
  end
end
for meth in (:jth_hprod,)
  @eval begin
    function NLPModels.$meth(nlp::TimerModel, x::AbstractVector, y::AbstractVector, j::Int)
      @timeit nlp.timer "$($meth)" $meth(get_nlp(nlp), x, y, j)
    end
  end
end
for meth in (:jth_hprod!,)
  @eval begin
    function NLPModels.$meth(
      nlp::TimerModel,
      x::AbstractVector,
      y::AbstractVector,
      j::Int,
      z::AbstractVector,
    )
      @timeit nlp.timer "$($meth)" $meth(get_nlp(nlp), x, y, j, z)
    end
  end
end
for meth in (:jac_structure, :jac_lin_structure, :jac_nln_structure, :hess_structure)
  @eval begin
    function NLPModels.$meth(nlp::TimerModel)
      @timeit nlp.timer "$($meth)" $meth(get_nlp(nlp))
    end
  end
end
