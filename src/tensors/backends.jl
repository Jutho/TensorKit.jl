# Scheduler implementation
# ------------------------
const blockscheduler = ScopedValue{Scheduler}(SerialScheduler())
const subblockscheduler = ScopedValue{Scheduler}(SerialScheduler())

# Backend implementation
# ----------------------
# TODO: figure out a name
# TODO: what should be the default scheduler?
@kwdef struct TensorKitBackend{B<:AbstractBackend,BS,SBS} <: AbstractBackend
    arraybackend::B = TO.DefaultBackend()
    blockscheduler::BS = blockscheduler[]
    subblockscheduler::SBS = subblockscheduler[]
end

function TO.select_backend(::typeof(TO.tensoradd!), C::AbstractTensorMap,
                           A::AbstractTensorMap)
    return TensorKitBackend()
end
function TO.select_backend(::typeof(TO.tensortrace!), C::AbstractTensorMap,
                           A::AbstractTensorMap)
    return TensorKitBackend()
end
function TO.select_backend(::typeof(TO.tensorcontract!), C::AbstractTensorMap,
                           A::AbstractTensorMap, B::AbstractTensorMap)
    return TensorKitBackend()
end
