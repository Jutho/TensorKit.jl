using VectorInterface: ONumber, _one
_isone(α::ONumber) = α == _one || isone(α)

# scalartype
#------------
VectorInterface.scalartype(T::Type{<:AbstractTensorMap}) = scalartype(storagetype(T))

# zerovector & zerovector!!
#---------------------------
function VectorInterface.zerovector(t::AbstractTensorMap, ::Type{S}) where {S<:Number}
    return zero!(similar(t, S))
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
VectorInterface.scale(t::TensorMap, α::ONumber) = _isone(α) ? t : t * α
function VectorInterface.scale!(t::AbstractTensorMap, α::ONumber)
    for (c, b) in blocks(t)
        scale!(b, α)
    end
    return t
end
function VectorInterface.scale!!(t::AbstractTensorMap, α::ONumber)
    _isone(α) && return t
    if promote_type(scalartype(t), typeof(α)) <: scalartype(t)
        return scale!(t, α)
    else
        return scale(t, α)
    end
end

function VectorInterface.scale!(ty::AbstractTensorMap, tx::AbstractTensorMap, α::ONumber)
    space(ty) == space(tx) || throw(SpaceMismatch())
    for c in blocksectors(tx)
        scale!(block(ty, c), block(tx, c), α)
    end
    return ty
end
function VectorInterface.scale!!(ty::AbstractTensorMap, tx::AbstractTensorMap, α::ONumber)
    space(ty) == space(tx) || throw(SpaceMismatch())
    T = scalartype(ty)
    if promote_type(T, typeof(α), scalartype(tx)) <: T
        return scale!(ty, tx, α)
    else
        return scale(tx, α)
    end
end

# add, add! & add!!
#-------------------
# TODO: remove VectorInterface from calls to `add!` when `TensorKit.add!` is renamed
function VectorInterface.add(ty::AbstractTensorMap, tx::AbstractTensorMap, α::ONumber=_one,
                             β::ONumber=_one)
    space(ty) == space(tx) || throw(SpaceMismatch())
    T = promote_type(scalartype(ty), scalartype(tx), typeof(α), typeof(β))
    return VectorInterface.add!(scale!(similar(ty, T), ty, β), tx, α)
end
function VectorInterface.add!(ty::AbstractTensorMap, tx::AbstractTensorMap, α::ONumber=_one,
                              β::ONumber=_one)
    space(ty) == space(tx) || throw(SpaceMismatch())
    for c in blocksectors(tx)
        VectorInterface.add!(block(ty, c), block(tx, c), α, β)
    end
    return ty
end
function VectorInterface.add!!(ty::AbstractTensorMap, tx::AbstractTensorMap,
                               α::ONumber=_one, β::ONumber=_one)
    T = scalartype(ty)
    if promote_type(T, typeof(α), typeof(β), scalartype(tx)) <: T
        return VectorInterface.add!(ty, tx, α, β)
    else
        return VectorInterface.add(ty, tx, α, β)
    end
end

# inner
#-------
function VectorInterface.inner(tx::AbstractTensorMap, ty::AbstractTensorMap)
    space(tx) == space(ty) || throw(SpaceMismatch())
    InnerProductStyle(tx) === EuclideanProduct() ||
        throw(ArgumentError("dot requires Euclidean inner product"))
    T = promote_type(scalartype(tx), scalartype(ty))
    s = zero(T)
    for c in blocksectors(tx)
        s += convert(T, dim(c)) * dot(block(tx, c), block(ty, c))
    end
    return s
end
