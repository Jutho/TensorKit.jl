# tensormap.jl
#
# This file defines an AbstractTensorMap type for creating linear maps or transformations
# between tensors. Concrete subtypes should provide a method domain and codomain that describe
# the vector space in which the input and output tensor are living, and a method A_mul_B!
# (and optionally Ac_mul_B!) for describing the action of the map on a tensor. AbstractTensorMap
# hooks into AbstractLinearMap and provides functionality for reinterpreting a tensor map
# as a linear map acting on Julia's standard Vector type.

abstract AbstractTensorMap{S<:IndexSpace,P<:TensorSpace,T} <: AbstractLinearMap{T}

domain(A::AbstractTensorMap)=throw(MethodError()) # this should be implemented by subtypes; defined here to allow import
codomain(A::AbstractTensorMap)=throw(MethodError()) # this should be implemented by subtypes; defined here to allow import

# The following methods allow to multiply AbstractTensorMap objects with standard Julia Vector objects, and to use it in
# general methods such as eigs, ...

Base.size(A::AbstractTensorMap)=(dim(codomain(A)),dim(domain(A)))
Base.size(A::AbstractTensorMap,n::Int)=(n==1 ? dim(codomain(A)) : (n==2 ? dim(domain(A)) : error("AbstractLinearMap objects have only 2 dimensions")))

function Base.A_mul_B!(y::Vector,A::AbstractTensorMap,x::Vector)
    X=tensor(x,domain(A))
    Y=tensor(y,codomain(A))
    vec(A_mul_B!(Y,A,X))
end
function Base.At_mul_B!(y::Vector,A::AbstractTensorMap,x::Vector)
    X=tensor(x,dual(codomain(A)))
    Y=tensor(y,dual(domain(A)))
    vec(At_mul_B!(Y,A,X))
end
function Base.Ac_mul_B!(y::Vector,A::AbstractTensorMap,x::Vector)
    X=tensor(x,conj(dual(codomain(A))))
    Y=tensor(y,conj(dual(domain(A))))
    vec(Ac_mul_B!(Y,A,X))
end

*{S,P}(A::AbstractTensorMap{S,P},X::AbstractTensor{S,P})=(x in domain(A) ? Base.A_mul_B!(similar(x,promote_type(eltype(A),eltype(x)),codomain(A)),A,x) : throw(SpaceError("Tensor not in domain of map")))
Base.At_mul_B{S,P}(A::AbstractTensorMap{S,P},X::AbstractTensor{S,P})=(x in dual(codomain(A)) ? Base.A_mul_B!(similar(x,promote_type(eltype(A),eltype(x)),dual(domain(A))),A,x) : throw(SpaceError("Tensor not in domain of map")))
Base.Ac_mul_B{S,P}(A::AbstractTensorMap{S,P},X::AbstractTensor{S,P})=(x in dual(conj(codomain(A)))s ? Base.A_mul_B!(similar(x,promote_type(eltype(A),eltype(x)),dual(conj(domain(A)))),A,x) : throw(SpaceError("Tensor not in domain of map")))


# Some convenient definitions
immutable DualTensorMap{S,P,T} <: AbstractTensorMap{S,P,T}
    map::AbstractTensorMap{S,P,T}
end
domain(A::DualTensorMap) = dual(codomain(A.map))
codomain(A::DualTensorMap) = dual(domain(A.map))


Base.transpose(A::AbstractTensorMap) = dual(A)

dual(A::AbstractTensorMap) = DualTensorMap(A)
dual(A::DualTensorMap) = A.map

A_mul_B!{S,P}(Y::AbstractTensor{S,P},A::DualTensorMap{S,P},X::AbstractTensor{S,P})=At_mul_B!(Y,A.map,X)
At_mul_B!{S,P}(Y::AbstractTensor{S,P},A::DualTensorMap{S,P},X::AbstractTensor{S,P})=A_mul_B!(Y,A.map,X)
Ac_mul_B!{S,P}(Y::AbstractTensor{S,P},A::DualTensorMap{S,P},X::AbstractTensor{S,P})=(A_mul_B!(Y,A.map,conj(X)); return conj!(Y))