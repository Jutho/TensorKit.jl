# Tensor network type hierarchy
#-------------------------------
abstract AbstractTensorNetwork{S<:IndexSpace,P<:TensorSpace,T}
# Any implementation of AbstractTensor should have method definitions for the
# same set of methods which are defined for the dense implementation Tensor
# defined in tensor.jl.

# tensor characteristics
spacetype{S}(::AbstractTensorNetwork{S})=S
spacetype{S<:IndexSpace}(::Type{AbstractTensorNetwork{S}})=S
spacetype{S<:IndexSpace,P<:TensorSpace}(::Type{AbstractTensorNetwork{S,P}})=S
spacetype{S<:IndexSpace,P<:TensorSpace,T}(::Type{AbstractTensorNetwork{S,P,T}})=S
spacetype{TN<:AbstractTensorNetwork}(::Type{TN})=spacetype(super(TN))

tensortype{S,P}(::AbstractTensorNetwork{S,P})=P
tensortype{S<:IndexSpace,P<:TensorSpace}(::Type{AbstractTensorNetwork{S,P}})=P
tensortype{S<:IndexSpace,P<:TensorSpace,T}(::Type{AbstractTensorNetwork{S,P,T}})=P
tensortype{TN<:AbstractTensorNetwork}(::Type{TN})=spacetype(super(TN))

Base.eltype{S,P,T}(::AbstractTensorNetwork{S,P,T}) = T
Base.eltype{S<:IndexSpace,P<:TensorSpace,T}(::Type{AbstractTensorNetwork{S,P,T}})=T
Base.eltype{TN<:AbstractTensorNetwork}(::Type{TN}) = eltype(super(TN))
