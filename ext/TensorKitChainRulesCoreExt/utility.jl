# Utility
# -------
trivtuple(N) = ntuple(identity, N)

Base.@constprop :aggressive function _repartition(p::IndexTuple, N₁::Int)
    length(p) >= N₁ ||
        throw(ArgumentError("cannot repartition $(typeof(p)) to $N₁, $(length(p) - N₁)"))
    return TupleTools.getindices(p, trivtuple(N₁)),
           TupleTools.getindices(p, trivtuple(length(p) - N₁) .+ N₁)
end
Base.@constprop :aggressive function _repartition(p::Index2Tuple, N₁::Int)
    return _repartition(linearize(p), N₁)
end
function _repartition(p::Union{IndexTuple,Index2Tuple}, ::Index2Tuple{N₁}) where {N₁}
    return _repartition(p, N₁)
end
function _repartition(p::Union{IndexTuple,Index2Tuple},
                      t::AbstractTensorMap)
    return _repartition(p, TensorKit.numout(t))
end

TensorKit.block(t::ZeroTangent, c::Sector) = t

ChainRulesCore.ProjectTo(::T) where {T<:AbstractTensorMap} = ProjectTo{T}()
function (::ProjectTo{T1})(x::T2) where {S,N1,N2,T1<:AbstractTensorMap{<:Any,S,N1,N2},
                                         T2<:AbstractTensorMap{<:Any,S,N1,N2}}
    T1 === T2 && return x
    y = similar(x, scalartype(T1))
    for (c, b) in blocks(y)
        p = ProjectTo(b)
        b .= p(block(x, c))
    end
    return y
end
