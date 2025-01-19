@non_differentiable TensorKit.TensorMap(f::Function, storagetype, cod, dom)
@non_differentiable TensorKit.id(args...)
@non_differentiable TensorKit.isomorphism(args...)
@non_differentiable TensorKit.isometry(args...)
@non_differentiable TensorKit.unitary(args...)

function ChainRulesCore.rrule(::Type{<:TensorMap}, d::DenseArray, args...; kwargs...)
    function TensorMap_pullback(Δt)
        ∂d = convert(Array, unthunk(Δt))
        return NoTangent(), ∂d, ntuple(_ -> NoTangent(), length(args))...
    end
    return TensorMap(d, args...; kwargs...), TensorMap_pullback
end

function ChainRulesCore.rrule(::Type{<:DiagonalTensorMap}, d::DenseVector, args...; kwargs...)
    D=TensorMap(d, args...; kwargs...)
    project_D=ProjectTo(D)
    function DiagonalTensorMap_pullback(Δt)
        ∂d = project_D(unthunk(Δt)).data
        return NoTangent(), ∂d, ntuple(_ -> NoTangent(), length(args))...
    end
    return D, DiagonalTensorMap_pullback
end

function ChainRulesCore.rrule(::typeof(Base.copy), t::AbstractTensorMap)
    copy_pullback(Δt) = NoTangent(), Δt
    return copy(t), copy_pullback
end

function ChainRulesCore.rrule(::typeof(TensorKit.copy_oftype), t::AbstractTensorMap,
                              T::Type{<:Number})
    project = ProjectTo(t)
    copy_oftype_pullback(Δt) = NoTangent(), project(unthunk(Δt)), NoTangent()
    return TensorKit.copy_oftype(t, T), copy_oftype_pullback
end

function ChainRulesCore.rrule(::typeof(TensorKit.permutedcopy_oftype), t::AbstractTensorMap,
                              T::Type{<:Number}, p::Index2Tuple)
    project = ProjectTo(t)
    function permutedcopy_oftype_pullback(Δt)
        invp = TensorKit._canonicalize(TupleTools.invperm(linearize(p)), t)
        return NoTangent(), project(TensorKit.permute(unthunk(Δt), invp)), NoTangent(),
               NoTangent()
    end
    return TensorKit.permutedcopy_oftype(t, T, p), permutedcopy_oftype_pullback
end

function ChainRulesCore.rrule(::typeof(Base.convert), T::Type{<:Array},
                              t::AbstractTensorMap)
    A = convert(T, t)
    function convert_pullback(ΔA)
        # use constructor to (unconditionally) project back onto symmetric subspace
        ∂t = TensorMap(unthunk(ΔA), codomain(t), domain(t); tol=Inf)
        return NoTangent(), NoTangent(), ∂t
    end
    return A, convert_pullback
end

function ChainRulesCore.rrule(::typeof(Base.convert), ::Type{Dict}, t::AbstractTensorMap)
    out = convert(Dict, t)
    function convert_pullback(c′)
        c = unthunk(c′)
        if haskey(c, :data) # :data is the only thing for which this dual makes sense
            dual = copy(out)
            dual[:data] = c[:data]
            return (NoTangent(), NoTangent(), convert(TensorMap, dual))
        else
            # instead of zero(t) you can also return ZeroTangent(), which is type unstable
            return (NoTangent(), NoTangent(), zero(t))
        end
    end
    return out, convert_pullback
end
function ChainRulesCore.rrule(::typeof(Base.convert), ::Type{TensorMap},
                              t::Dict{Symbol,Any})
    return convert(TensorMap, t), v -> (NoTangent(), NoTangent(), convert(Dict, v))
end

function ChainRulesCore.rrule(T::Type{<:TensorKit.AdjointTensorMap}, t::AbstractTensorMap)
    return T(t), Δt -> (NoTangent(), adjoint(unthunk(Δt)))
end
