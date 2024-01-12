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
    T = VectorInterface.promote_scale(t, α)
    return scale!(similar(t, T), t, α)
end
function VectorInterface.scale!(t::AbstractTensorMap, α::Number)
    for (c, b) in blocks(t)
        scale!(b, α)
    end
    return t
end
function VectorInterface.scale!!(t::AbstractTensorMap, α::Number)
    α === One() && return t
    T = VectorInterface.promote_scale(t, α)
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
    T = VectorInterface.promote_scale(tx, α)
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
                             α::Number, β::Number)
    space(ty) == space(tx) || throw(SpaceMismatch("$(space(ty)) ≠ $(space(tx))"))
    T = VectorInterface.promote_add(ty, tx, α, β)
    return VectorInterface.add!(scale!(similar(ty, T), ty, β), tx, α)
end
function VectorInterface.add!(ty::AbstractTensorMap, tx::AbstractTensorMap,
                              α::Number, β::Number;
                              numthreads::Int64=1)
    space(ty) == space(tx) || throw(SpaceMismatch("$(space(ty)) ≠ $(space(tx))"))
    if numthreads == 1 || length(blocksectors) == 1
        for c in blocksectors(tx)
            VectorInterface.add!(block(ty, c), block(tx, c), α, β)
        end
    elseif numthreads == -1
        Threads.@sync for c in blocksectors(tx)
            Threads.@spawn VectorInterface.add!(block(ty, c), block(tx, c), α, β)
        end
    else
        # producer
        taskref = Ref{Task}()
        ch = Channel(; taskref=taskref, spawn=true) do ch
            for c in vcat(blocksectors(tx), fill(nothing, numthreads))
                put!(ch, c)
            end
        end
        # consumers
        tasks = map(1:numthreads) do _
            task = Threads.@spawn while true
                c = take!(ch)
                VectorInterface.add!(block(ty, c), block(tx, c), α, β)
            end
            return errormonitor(task)
        end

        wait.(tasks)
        wait(taskref[])
    end
    return ty
end
function VectorInterface.add!!(ty::AbstractTensorMap, tx::AbstractTensorMap,
                               α::Number, β::Number)
    # spacecheck is done in add(!)
    T = VectorInterface.promote_add(ty, tx, α, β)
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
    InnerProductStyle(tx) === EuclideanProduct() || throw_invalid_innerproduct(:inner)
    T = VectorInterface.promote_inner(tx, ty)
    s = zero(T)
    for c in blocksectors(tx)
        s += convert(T, dim(c)) * dot(block(tx, c), block(ty, c))
    end
    return s
end
