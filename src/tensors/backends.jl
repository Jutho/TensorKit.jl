# Scheduler implementation
# ------------------------
"""
    const blockscheduler = ScopedValue{Scheduler}(SerialScheduler())

The default scheduler used when looping over different blocks in the matrix representation of a
tensor.

For controlling this value, see also [`set_blockscheduler`](@ref) and [`with_blockscheduler`](@ref).
"""
const blockscheduler = ScopedValue{Scheduler}(SerialScheduler())

"""
    const subblockscheduler = ScopedValue{Scheduler}(SerialScheduler())

The default scheduler used when looping over different subblocks in a tensor.

For controlling this value, see also [`set_subblockscheduler`](@ref) and [`with_subblockscheduler`](@ref).
"""
const subblockscheduler = ScopedValue{Scheduler}(SerialScheduler())

function select_scheduler(scheduler=OhMyThreads.Implementation.NotGiven(); kwargs...)
    return if scheduler == OhMyThreads.Implementation.NotGiven() && isempty(kwargs)
        Threads.nthreads() == 1 ? SerialScheduler() : DynamicScheduler()
    else
        OhMyThreads.Implementation._scheduler_from_userinput(scheduler; kwargs...)
    end
end

"""
    set_blockscheduler!([scheduler]; kwargs...) -> previuos

Set the default scheduler used in looping over the different blocks in the matrix representation
of a tensor.
The arguments to this function are either an `OhMyThreads.Scheduler` or a `Symbol` with optional
set of keywords arguments. For a detailed description, consult the
[`OhMyThreads` documentation](https://juliafolds2.github.io/OhMyThreads.jl/stable/refs/api/#Schedulers).

See also [`with_blockscheduler`](@ref).
"""
function set_blockscheduler!(scheduler=OhMyThreads.Implementation.NotGiven(); kwargs...)
    previous = blockscheduler[]
    blockscheduler[] = select_scheduler(scheduler; kwargs...)
    return previous
end

"""
    with_blockscheduler(f, [scheduler]; kwargs...)

Run `f` in a scope where the `blockscheduler` is determined by `scheduler` and `kwargs...`.

See also [`set_blockscheduler!`](@ref).
"""
function with_blockscheduler(f, scheduler=OhMyThreads.Implementation.NotGiven(); kwargs...)
    @with blockscheduler => select_scheduler(scheduler; kwargs...) f()
end

"""
    set_subblockscheduler!([scheduler]; kwargs...) -> previous

Set the default scheduler used in looping over the different subblocks in a tensor.
The arguments to this function are either an `OhMyThreads.Scheduler` or a `Symbol` with optional
set of keywords arguments. For a detailed description, consult the
[`OhMyThreads` documentation](https://juliafolds2.github.io/OhMyThreads.jl/stable/refs/api/#Schedulers).

See also [`with_subblockscheduler`](@ref).
"""
function set_subblockscheduler!(scheduler=OhMyThreads.Implementation.NotGiven(); kwargs...)
    previous = subblockscheduler[]
    subblockscheduler[] = select_scheduler(scheduler; kwargs...)
    return previous
end

"""
    with_subblockscheduler(f, [scheduler]; kwargs...)

Run `f` in a scope where the [`subblockscheduler`](@ref) is determined by `scheduler` and `kwargs...`.

See also [`set_subblockscheduler!`](@ref).
"""
function with_subblockscheduler(f, scheduler=OhMyThreads.Implementation.NotGiven();
                                kwargs...)
    @with subblockscheduler => select_scheduler(scheduler; kwargs...) f()
end

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
