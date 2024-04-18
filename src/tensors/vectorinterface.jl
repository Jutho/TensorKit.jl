# scalartype
#------------
VectorInterface.scalartype(T::Type{<:AbstractTensorMap}) = scalartype(storagetype(T))

# zerovector & zerovector!!
#---------------------------
function VectorInterface.zerovector(t::AbstractTensorMap, ::Type{S};
                                    scheduler::Scheduler=SerialScheduler()) where {S<:Number}
    return zerovector!(similar(t, S); scheduler)
end
function VectorInterface.zerovector!(t::AbstractTensorMap;
                                     scheduler::Scheduler=SerialScheduler())
    tforeach(zerovector!, values(blocks(t)); scheduler)
    return t
end
function VectorInterface.zerovector!!(t::AbstractTensorMap;
                                      scheduler::Scheduler=SerialScheduler())
    return zerovector!(t)
end

# scale, scale! & scale!!
#-------------------------
function VectorInterface.scale(t::AbstractTensorMap, α::Number;
                               scheduler::Scheduler=SerialScheduler())
    T = VectorInterface.promote_scale(t, α)
    return scale!(similar(t, T), t, α; scheduler)
end
function VectorInterface.scale!(t::AbstractTensorMap, α::Number;
                                scheduler::Scheduler=SerialScheduler())
    _scale!(c) = scale!(block(t, c), α)
    tforeach(_scale!, blocksectors(t); scheduler)
    return t
end
function VectorInterface.scale!!(t::AbstractTensorMap, α::Number;
                                 scheduler::Scheduler=SerialScheduler())
    α === One() && return t
    T = VectorInterface.promote_scale(t, α)
    return T <: scalartype(t) ? scale!(t, α; scheduler) : scale(t, α; scheduler)
end
function VectorInterface.scale!(ty::AbstractTensorMap, tx::AbstractTensorMap, α::Number;
                                scheduler::Scheduler=SerialScheduler())
    space(ty) == space(tx) || throw(SpaceMismatch("$(space(ty)) ≠ $(space(tx))"))
    for c in blocksectors(tx)
        scale!(block(ty, c), block(tx, c), α)
    end
    return ty
end
function VectorInterface.scale!!(ty::AbstractTensorMap, tx::AbstractTensorMap, α::Number;
                                 scheduler::Scheduler=SerialScheduler())
    T = VectorInterface.promote_scale(tx, α)
    if T <: scalartype(ty)
        return scale!(ty, tx, α; scheduler)
    else
        return scale(tx, α; scheduler)
    end
end

# add, add! & add!!
#-------------------
# TODO: remove VectorInterface from calls to `add!` when `TensorKit.add!` is renamed
function VectorInterface.add(ty::AbstractTensorMap, tx::AbstractTensorMap,
                             α::Number, β::Number;
                             scheduler::Scheduler=SerialScheduler())
    space(ty) == space(tx) || throw(SpaceMismatch("$(space(ty)) ≠ $(space(tx))"))
    T = VectorInterface.promote_add(ty, tx, α, β)
    return VectorInterface.add!(scale!(similar(ty, T), ty, β; scheduler), tx, α, One();
                                scheduler)
end
function VectorInterface.add!(ty::AbstractTensorMap, tx::AbstractTensorMap,
                              α::Number, β::Number;
                              scheduler::Scheduler=SerialScheduler())
    space(ty) == space(tx) || throw(SpaceMismatch("$(space(ty)) ≠ $(space(tx))"))
    _add!(c) = VectorInterface.add!(block(ty, c), block(tx, c), α, β)
    tforeach(_add!, blocksectors(tx); scheduler)
    return ty
end
function VectorInterface.add!!(ty::AbstractTensorMap, tx::AbstractTensorMap,
                               α::Number, β::Number;
                               scheduler::Scheduler=SerialScheduler())
    # spacecheck is done in add(!)
    T = VectorInterface.promote_add(ty, tx, α, β)
    if T <: scalartype(ty)
        return VectorInterface.add!(ty, tx, α, β; scheduler)
    else
        return VectorInterface.add(ty, tx, α, β; scheduler)
    end
end

# inner
#-------
function VectorInterface.inner(tx::AbstractTensorMap, ty::AbstractTensorMap)
    space(tx) == space(ty) || throw(SpaceMismatch("$(space(tx)) ≠ $(space(ty))"))
    InnerProductStyle(tx) === EuclideanProduct() || throw_invalid_innerproduct(:inner)
    T = VectorInterface.promote_inner(tx, ty)
    s = zero(T)
    for c in blocksectors(tx)
        s += convert(T, dim(c)) * dot(block(tx, c), block(ty, c))
    end
    return s
end
