# tensormap.jl
#
# This file defines an AbstractTensorMap type for creating linear maps or transformations
# between tensors. Concrete subtypes should provide a method domain and codomain that describe
# the vector space in which the input and output tensor are living, and a method A_mul_B!
# (and optionally Ac_mul_B!) for describing the action of the map on a tensor. AbstractTensorMap
# inherits from AbstractLinearMap and provides functionality for reinterpreting a tensor map
# as a linear map acting on Julia's standard Vector type.

abstract AbstractTensorMap{T<:Number} <: AbstractLinearMap{T}

domain(A::AbstractTensorMap)=throw(MethodError()) # this should be implemented by subtypes; defined here to allow import
codomain(A::AbstractTensorMap)=throw(MethodError()) # this should be implemented by subtypes; defined here to allow import

# The following methods allow to multiply AbstractTensorMap objects with standard Julia Vector objects, and to use it in
# general methods such as eigs, ...

Base.size(A::AbstractTensorMap)=(dim(codomain(A)),dim(domain(A)))
Base.size(A::AbstractTensorMap,n::Int)=(n==1 ? dim(codomain(A)) : (n==2 ? dim(domain(A)) : error("AbstractLinearMap objects have only 2 dimensions")))

function Base.A_mul_B!(y::Vector,A::AbstractTensorMap,x::Vector)
    Y=tensor(y,codomain(A))
    X=tensor(x,domain(A))
    vec(A_mul_B!(Y,A,X))
end
function Base.Ac_mul_B!(y::Vector,A::AbstractTensorMap,x::Vector)
    Y=tensor(y,domain(A)')
    X=tensor(x,codomain(A)')
    vec(Ac_mul_B!(Y,A,X))
end

*(A::AbstractTensorMap,X::AbstractTensor)=(x in domain(A) ? Base.A_mul_B!(similar(x,promote_type(eltype(A),eltype(x)),codomain(A)),A,x) : throw(SpaceError("Tensor not in domain of map")))
Ac_mul_B(A::AbstractTensorMap,X::AbstractTensor)=(x in codomain(A)' ? Base.Ac_mul_B!(similar(x,promote_type(eltype(A),eltype(x)),domain(A)'),A,x) : throw(SpaceError("Tensor not in domain of map")))
