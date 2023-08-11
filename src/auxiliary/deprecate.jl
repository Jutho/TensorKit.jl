import Base: eltype, transpose
@deprecate eltype(T::Type{<:AbstractTensorMap}) scalartype(T)
@deprecate eltype(t::AbstractTensorMap) scalartype(t)

@deprecate permute(t::AbstractTensorMap, p1::IndexTuple, p2::IndexTuple; copy::Bool=false) permute(t, (p1, p2); copy=copy)
@deprecate transpose(t::AbstractTensorMap, p1::IndexTuple, p2::IndexTuple; copy::Bool=false) transpose(t, (p1, p2); copy=copy)
@deprecate braid(t::AbstractTensorMap, p1::IndexTuple, p2::IndexTuple, levels; copy::Bool=false) braid(t, (p1, p2), levels; copy=copy)
