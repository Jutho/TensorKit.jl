# Utility
# -------
function _repartition(p::IndexTuple, N₁::Int)
    length(p) >= N₁ ||
        throw(ArgumentError("cannot repartition $(typeof(p)) to $N₁, $(length(p) - N₁)"))
    return p[1:N₁], p[(N₁ + 1):end]
end
_repartition(p::Index2Tuple, N₁::Int) = _repartition(linearize(p), N₁)
function _repartition(p::Union{IndexTuple,Index2Tuple}, ::Index2Tuple{N₁}) where {N₁}
    return _repartition(p, N₁)
end
function _repartition(p::Union{IndexTuple,Index2Tuple},
                      ::AbstractTensorMap{<:Any,N₁}) where {N₁}
    return _repartition(p, N₁)
end

TensorKit.block(t::ZeroTangent, c::Sector) = t

ChainRulesCore.ProjectTo(::T) where {T<:AbstractTensorMap} = ProjectTo{T}()
function (::ProjectTo{T1})(x::T2) where {S,N1,N2,T1<:AbstractTensorMap{S,N1,N2},
                                         T2<:AbstractTensorMap{S,N1,N2}}
    T1 === T2 && return x
    y = similar(x, scalartype(T1))
    for (c, b) in blocks(y)
        p = ProjectTo(b)
        b .= p(block(x, c))
    end
    return y
end
