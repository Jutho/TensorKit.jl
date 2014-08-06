# abstracttensormap.jl
#
# This file defines an AbstractTensorMap type for creating linear maps or transformations
# between tensors. Concrete subtypes should provide a method domain and codomain that describe
# the vector space in which the input and output tensor are living, and a method A_mul_B!
# (and optionally Ac_mul_B!) for describing the action of the map on a tensor. AbstractTensorMap
# provides functionality for acting multiplication with Julia's standard Vector type.

abstract AbstractTensorMap{S<:IndexSpace,P<:TensorSpace,T}

spacetype{S}(::AbstractTensorMap{S})=S
spacetype{S<:IndexSpace}(::Type{AbstractTensorMap{S}})=S
spacetype{S<:IndexSpace,P<:TensorSpace}(::Type{AbstractTensorMap{S,P}})=S
spacetype{S<:IndexSpace,P<:TensorSpace,T}(::Type{AbstractTensorMap{S,P,T}})=S
spacetype{TM<:AbstractTensorMap}(::Type{TM})=spacetype(super(TM))

tensortype{S,P}(::AbstractTensorMap{S,P})=P
tensortype{S<:IndexSpace,P<:TensorSpace}(::Type{AbstractTensorMap{S,P}})=P
tensortype{S<:IndexSpace,P<:TensorSpace,T}(::Type{AbstractTensorMap{S,P,T}})=P
tensortype{TM<:AbstractTensorMap}(::Type{TM})=spacetype(super(TM))

Base.eltype{S,P,T}(::AbstractTensorMap{S,P,T}) = T
Base.eltype{S<:IndexSpace,P<:TensorSpace,T}(::Type{AbstractTensorMap{S,P,T}})=T
Base.eltype{TM<:AbstractTensor}(::Type{TM}) = eltype(super(TM))

domain(A::AbstractTensorMap)=throw(MethodError(domain,(AbstractTensorMap))) # this should be implemented by subtypes; defined here to allow import
codomain(A::AbstractTensorMap)=throw(MethodError(codomain,(AbstractTensorMap))) # this should be implemented by subtypes; defined here to allow import

# Generate non-mutating multiplication methods for maps that only implement mutating methods
*{S,P}(A::AbstractTensorMap{S,P},X::AbstractTensor{S,P})=(x in domain(A) ? Base.A_mul_B!(similar(x,promote_type(eltype(A),eltype(x)),codomain(A)),A,x) : throw(SpaceError("Tensor not in domain of map")))
Base.At_mul_B{S,P}(A::AbstractTensorMap{S,P},X::AbstractTensor{S,P})=(x in dual(codomain(A)) ? Base.A_mul_B!(similar(x,promote_type(eltype(A),eltype(x)),dual(domain(A))),A,x) : throw(SpaceError("Tensor not in domain of map")))
Base.Ac_mul_B{S,P}(A::AbstractTensorMap{S,P},X::AbstractTensor{S,P})=(x in dual(conj(codomain(A)))s ? Base.A_mul_B!(similar(x,promote_type(eltype(A),eltype(x)),dual(conj(domain(A)))),A,x) : throw(SpaceError("Tensor not in domain of map")))

# Other properties
dual(A::AbstractTensorMap)=transpose(A)
adjoint{S<:ElementaryHilbertSpace}(A::AbstractTensorMap{S})=(domain(A)==dual(conj(codomain(A))) ? ctranspose(A) : throw(SpaceError("Not an operator, i.e. domain != codomain")))

# The following methods allow to multiply AbstractTensorMap objects with standard Julia Vector objects, and to use it in
# general methods such as eigs, ...
Base.size(A::AbstractTensorMap)=(dim(codomain(A)),dim(domain(A)))
Base.size(A::AbstractTensorMap,n::Int)=(n==1 ? dim(codomain(A)) : (n==2 ? dim(domain(A)) : 1))

Base.isreal{S<:IndexSpace,P<:TensorSpace,T<:Real}(::AbstractTensorMap{S,P,T}) = true
Base.isreal(::AbstractTensorMap) = false # default assumption
Base.issym(::AbstractTensorMap)=false # default assumption
Base.ishermitian(A::AbstractTensorMap)=false # default assumption
Base.isposdef(::AbstractTensorMap)=false # default assumption

Base.A_mul_B!(y::Vector,A::AbstractTensorMap,x::Vector)=vec(A_mul_B!(tensor(y,codomain(A)), A, tensor(x,domain(A))))
Base.At_mul_B!(y::Vector,A::AbstractTensorMap,x::Vector)=vec(At_mul_B!(tensor(y,dual(domain(A))), A, tensor(x,dual(codomain(A)))))
Base.Ac_mul_B!(y::Vector,A::AbstractTensorMap,x::Vector)=vec(Ac_mul_B!(tensor(y,conj(dual(domain(A)))), A, tensor(x,conj(dual(codomain(A))))))

*(A::AbstractTensorMap,x::Vector)=A_mul_B!(similar(x,promote_type(eltype(A),eltype(x)),dim(codomain(A))), A, x)
Base.At_mul_B(A::AbstractTensorMap,x::Vector)=At_mul_B!(similar(x,promote_type(eltype(A),eltype(x)),dim(domain(A))), A, x)
Base.Ac_mul_B(y::Vector,A::AbstractTensorMap,x::Vector)=Ac_mul_B!(similar(x,promote_type(eltype(A),eltype(x)),dim(domain(A))), A, x)

# Compute full matrix representation
function Base.full(A::AbstractTensorMap)
    M,N=size(A)
    T=eltype(A)
    mat=zeros(T,(M,N))
    v=Array(T,N)
    w=Array(T,M)
    for j=1:N
        fill!(v,zero(T))
        v[j]=one(T)
        x=tensor(v,domain(A))
        y=tensor(w,codomain(A))
        A_mul_B!(y,A,x)
        mat[:,j]=w
    end
    return mat
end

function Base.diag(A::AbstractTensorMap)
    domain(A)==codomain(A) || throw(SpaceError("diag only works for operators, i.e. maps with domain == codomain"))
    M=size(A,1)
    T=eltype(A)
    d=zeros(T,M)
    v=Array(T,M)
    w=Array(T,M)
    for i=1:M
        fill!(v,zero(T))
        v[j]=one(T)
        x=tensor(v,domain(A))
        y=tensor(w,codomain(A))
        A_mul_B!(y,A,x)
        d[i]=w[i]
    end
    return d
end