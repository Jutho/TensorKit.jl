# scalartype
#------------
VectorInterface.scalartype(T::Type{<:AbstractTensorMap}) = scalartype(storagetype(T))

# zerovector & zerovector!!
#---------------------------
function VectorInterface.zerovector(t::AbstractTensorMap, ::Type{S}) where {S<:Number}
    return zerovector!(similar(t, S))
end
function VectorInterface.zerovector!(t::AbstractTensorMap)
    for (c, b) in blocks(t)
        zerovector!(b)
    end
    return t
end
VectorInterface.zerovector!!(t::AbstractTensorMap) = zerovector!(t)

# scale, scale! & scale!!
#-------------------------
function VectorInterface.scale(t::AbstractTensorMap, α::Number)
    T = Base.promote_op(scale, scalartype(t), scalartype(α))
    return scale!(similar(t, T), t, α)
end
function VectorInterface.scale!(t::AbstractTensorMap, α::Number)
    for (c, b) in blocks(t)
        scale!(b, α)
    end
    return t
end
function VectorInterface.scale!!(t::AbstractTensorMap, α::Number)
    α === _one && return t
    α === _zero && return zerovector!!(t)
    T = Base.promote_op(scale, scalartype(t), scalartype(α))
    return T <: scalartype(t) ? scale!(t, α) : scale(t, α)
end

function VectorInterface.scale!(ty::AbstractTensorMap, tx::AbstractTensorMap, α::Number)
    space(ty) == space(tx) || throw(SpaceMismatch("$(space(ty)) ≠ $(space(tx))"))
    for c in blocksectors(tx)
        scale!(block(ty, c), block(tx, c), α)
    end
    return ty
end
function VectorInterface.scale!!(ty::AbstractTensorMap, tx::AbstractTensorMap, α::Number)
    T = Base.promote_op(scale, scalartype(tx), scalartype(α))
    if T <: scalartype(ty)
        return scale!(ty, tx, α)
    else
        return scale(tx, α)
    end
end

# add, add! & add!!
#-------------------
# TODO: remove VectorInterface from calls to `add!` when `TensorKit.add!` is renamed
function VectorInterface.add(ty::AbstractTensorMap, tx::AbstractTensorMap,
                             α::Number=_one, β::Number=_one)
    space(ty) == space(tx) || throw(SpaceMismatch("$(space(ty)) ≠ $(space(tx))"))
    T = Base.promote_op(VectorInterface.add, scalartype(ty), scalartype(tx), scalartype(α), scalartype(β))
    return VectorInterface.add!(scale!(similar(ty, T), ty, β), tx, α)
end
function VectorInterface.add!(ty::AbstractTensorMap, tx::AbstractTensorMap,
                              α::Number=_one, β::Number=_one)
    space(ty) == space(tx) || throw(SpaceMismatch("$(space(ty)) ≠ $(space(tx))"))
    for c in blocksectors(tx)
        VectorInterface.add!(block(ty, c), block(tx, c), α, β)
    end
    return ty
end
function VectorInterface.add!!(ty::AbstractTensorMap, tx::AbstractTensorMap,
                               α::Number=_one, β::Number=_one)
    T = Base.promote_op(VectorInterface.add, scalartype(ty), scalartype(tx), scalartype(α),
                        scalartype(β))
    if T <: scalartype(ty)
        return VectorInterface.add!(ty, tx, α, β)
    else
        return VectorInterface.add(ty, tx, α, β)
    end
end

# inner
#-------
function VectorInterface.inner(tx::AbstractTensorMap, ty::AbstractTensorMap)
    space(tx) == space(ty) || throw(SpaceMismatch("$(space(tx)) ≠ $(space(ty))"))
    InnerProductStyle(tx) === EuclideanProduct() ||
        throw(ArgumentError("dot requires Euclidean inner product"))
    T = Base.promote_op(VectorInterface.inner, scalartype(tx), scalartype(ty))
    s = zero(T)
    for c in blocksectors(tx)
        s += convert(T, dim(c)) * dot(block(tx, c), block(ty, c))
    end
    return s
end
