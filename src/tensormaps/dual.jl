# dualtensormap.jl

immutable DualTensorMap{S,P,T} <: AbstractTensorMap{S,P,T}
    map::AbstractTensorMap{S,P,T}
end

# transposition behavior of AbstractTensorMap objects
Base.transpose(A::AbstractTensorMap) = DualTensorMap(A)
Base.transpose(A::DualTensorMap) = A.map

# properties
domain(A::DualTensorMap) = dual(codomain(A.map))
codomain(A::DualTensorMap) = dual(domain(A.map))

# comparison of DualTensorMap objects
==(A::DualTensorMap,B::DualTensorMap)=A.lmap==B.lmap

# multiplication
A_mul_B!{S,P}(Y::AbstractTensor{S,P},A::DualTensorMap{S,P},X::AbstractTensor{S,P})=At_mul_B!(Y,A.map,X)
At_mul_B!{S,P}(Y::AbstractTensor{S,P},A::DualTensorMap{S,P},X::AbstractTensor{S,P})=A_mul_B!(Y,A.map,X)
