# abstracttensor.jl
#
# Abstract Tensor type
#----------------------
"""
    abstract type AbstractTensorMap{T, S<:IndexSpace, N₁, N₂} end

Abstract supertype of all tensor maps, i.e. linear maps between tensor products
of vector spaces of type `S<:IndexSpace`. An `AbstractTensorMap` maps from
an input space of type `ProductSpace{S,N₂}` to an output space of type
`ProductSpace{S,N₁}`.
"""
abstract type AbstractTensorMap{S<:IndexSpace, N₁, N₂} end
"""
    AbstractTensor{S<:IndexSpace, N} = AbstractTensorMap{T,S,N,0}

Abstract supertype of all tensors, i.e. elements in the tensor product space
of type `ProductSpace{S,N}`, built from elementary spaces of type `S<:IndexSpace`.

An `AbstractTensor{S,N}` is actually a special case `AbstractTensorMap{S,N,0}`,
i.e. a tensor map with only a non-trivial output space.
"""
const AbstractTensor{S<:IndexSpace, N} = AbstractTensorMap{S, N, 0}

# tensor characteristics
Base.eltype(t::AbstractTensorMap) = eltype(typeof(t))
spacetype(t::AbstractTensorMap) = spacetype(typeof(t))
sectortype(t::AbstractTensorMap) = sectortype(typeof(t))
fieldtype(t::AbstractTensorMap) = fieldtype(typeof(t))
numin(t::AbstractTensorMap) = numin(typeof(t))
numout(t::AbstractTensorMap) = numout(typeof(t))
numind(t::AbstractTensorMap) = numind(typeof(t))

spacetype(::Type{<:AbstractTensorMap{S}}) where {S<:IndexSpace} = S
sectortype(::Type{<:AbstractTensorMap{S}}) where {S<:IndexSpace} = sectortype(S)
fieldtype(::Type{<:AbstractTensorMap{S}}) where {S<:IndexSpace} = fieldtype(S)
numin(::Type{<:AbstractTensorMap{<:IndexSpace,N₁,N₂}}) where {N₁, N₂} = N₁
numout(::Type{<:AbstractTensorMap{<:IndexSpace,N₁,N₂}}) where {N₁, N₂} = N₂
numind(::Type{<:AbstractTensorMap{<:IndexSpace,N₁,N₂}}) where {N₁, N₂} = N₁ + N₂

const order = numind

# tensormap implementation should provide codomain(t) and domain(t)
codomain(t::AbstractTensorMap, i) = codomain(t)[i]
domain(t::AbstractTensorMap, i) = domain(t)[i]
space(t::AbstractTensor) = codomain(t)
space(t::AbstractTensor, i) = space(t)[i]

# Defining vector spaces:
#------------------------
const TensorSpace{S<:IndexSpace, N} = ProductSpace{S,N}
const TensorMapSpace{S<:IndexSpace, N₁, N₂} = Pair{ProductSpace{S,N₂},ProductSpace{S,N₁}}

# Little unicode hack to define TensorMapSpace
→(dom::ProductSpace{S}, codom::ProductSpace{S}) where {S<:IndexSpace} = dom => codom
→(dom::S, codom::ProductSpace{S}) where {S<:IndexSpace} = ProductSpace(dom) => codom
→(dom::ProductSpace{S}, codom::S) where {S<:IndexSpace} = dom => ProductSpace(codom)
→(dom::S, codom::S) where {S<:IndexSpace} = ProductSpace(dom) => ProductSpace(codom)

←(codom::ProductSpace{S}, dom::ProductSpace{S}) where {S<:IndexSpace} = dom => codom
←(codom::S, dom::ProductSpace{S}) where {S<:IndexSpace} = dom => ProductSpace(codom)
←(codom::ProductSpace{S}, dom::S) where {S<:IndexSpace} = ProductSpace(dom) => codom
←(codom::S, dom::S) where {S<:IndexSpace} = ProductSpace(dom) => ProductSpace(codom)

# Basic algebra
#---------------
Base.copy(t::AbstractTensorMap) = Base.copy!(similar(t), t)

Base.:-(t::AbstractTensorMap) = scale!(t, -one(eltype(t)))
function Base.:+(t1::AbstractTensorMap, t2::AbstractTensorMap)
    T = promote_type(eltype(t1), eltype(t2))
    return add!(copy!(similar(t1, T), t1), t2, one(T))
end
function Base.:-(t1::AbstractTensorMap, t2::AbstractTensorMap)
    T = promote_type(eltype(t1), eltype(t2))
    return add!(copy!(similar(t1,T), t1), t2, -one(T))
end

Base.:*(t::AbstractTensorMap, α::Number) = scale!(similar(t, promote_type(eltype(t), typeof(α))), t, α)
Base.:*(α::Number, t::AbstractTensorMap) = *(t, α)
Base.:/(t::AbstractTensorMap, α::Number) = *(t, one(α)/α)
Base.:\(α::Number, t::AbstractTensorMap) = *(t, one(α)/α)

Base.scale!(t::AbstractTensorMap, α::Number) = scale!(t, t, α)
Base.scale!(α::Number, t::AbstractTensorMap) = scale!(t, t, α)
Base.scale!(tdest::AbstractTensorMap, α::Number, tsrc::AbstractTensorMap) = scale!(tdest, tsrc, α)

Base.LinAlg.axpy!(α::Number, tx::AbstractTensorMap, ty::AbstractTensorMap) = add!(ty, 1, tx, α)

# Base.conj(t::AbstractTensor) = Base.conj!(similar(t, conj(space(t))), t)
# Base.transpose(t::AbstractTensor) = Base.transpose!(similar(t, space(t).'), t)
# Base.ctranspose(t::AbstractTensor) = Base.ctranspose!(similar(t, space(t)'), t)

# Tensor operations
#-------------------
# convenience definition which works for vectors and matrices but also sometimes useful in general case
# *{S,T1,T2,N1,N2}(t1::AbstractTensor{S,T1,N1},t2::AbstractTensor{S,T2,N2})=(t3=similar(t1,promote_type(T1,T2),space(t1)[1:N1-1] ⊗ space(t2)[2:N2]);tensorcontract!(1,t1,vcat(1:N1-1,0),'N',t2,vcat(0,numind(t1)-1+(1:N2-1)),'N',0,t3,1:(N1+N2-2)))
# Base.At_mul_B{S,T1,T2,N1,N2}(t1::AbstractTensor{S,T1,N1},t2::AbstractTensor{S,T2,N2})=(t3=similar(t1,promote_type(T1,T2),space(t1)[2:N1].' ⊗ space(t2)[2:N2]);tensorcontract!(1,t1,vcat(0,reverse(1:N1-1)),'N',t2,vcat(0,N1-1+(1:N2-1)),'N',0,t3,1:(numind(t1)+numind(t2)-2)))
# Base.Ac_mul_B{S,T1,T2,N1,N2}(t1::AbstractTensor{S,T1,N1},t2::AbstractTensor{S,T2,N2})=(t3=similar(t1,promote_type(T1,T2),space(t1)[2:N1]' ⊗ space(t2)[2:N2]);tensorcontract!(1,t1,vcat(0,reverse(1:N1-1)),'C',t2,vcat(0,N1-1+(1:N2-1)),'N',0,t3,1:(numind(t1)+numind(t2)-2)))
#
# ⊗{S}(t1::AbstractTensor{S},t2::AbstractTensor{S})=tensorproduct(t1,1:numind(t1),t2,numind(t1)+(1:numind(t2)))
# Base.trace{S,T}(t::AbstractTensor{S,T,2})=scalar(tensortrace(t,[1,1],[]))
#
# # general tensor operations: no error checking, pass to mutating methods
# function tensorcopy(A::AbstractTensor,labelsA,outputlabels=labelsA)
#     spaceA=space(A)
#     spaceC=spaceA[indexin(outputlabels,labelsA)]
#     C=similar(A,spaceC)
#     tensorcopy!(A,labelsA,C,outputlabels)
#     return C
# end
# function tensoradd{S,TA,TB,N}(A::AbstractTensor{S,TA,N},labelsA,B::AbstractTensor{S,TB,N},labelsB,outputlabels=labelsA)
#     spaceA=space(A)
#     spaceC=spaceA[indexin(outputlabels,labelsA)]
#     T=promote_type(TA,TB)
#     C=similar(A,T,spaceC)
#     tensorcopy!(A,labelsA,C,outputlabels)
#     tensoradd!(1,B,labelsB,1,C,outputlabels)
#     return C
# end
# function tensortrace(A::AbstractTensor,labelsA,outputlabels)
#     T=eltype(A)
#     spaceA=space(A)
#     spaceC=spaceA[indexin(outputlabels,labelsA)]
#     C=similar(A,spaceC)
#     tensortrace!(1,A,labelsA,0,C,outputlabels)
#     return C
# end
# function tensortrace(A::AbstractTensor,labelsA) # there is no one-line method to compute the default outputlabels
#     ulabelsA=unique(labelsA)
#     labelsC=similar(labelsA,0)
#     sizehint(labelsC,length(ulabelsA))
#     for j=1:length(ulabelsA)
#         ind=findfirst(labelsA,ulabelsA[j])
#         if findnext(labelsA,ulabelsA[j],ind+1)==0
#             push!(labelsC,ulabelsA[j])
#         end
#     end
#     return tensortrace(A,labelsA,labelsC)
# end
# function tensorcontract{S}(A::AbstractTensor{S},labelsA,conjA::Char,B::AbstractTensor{S},labelsB,conjB::Char,outputlabels=symdiff(labelsA,labelsB);method::Symbol=:BLAS,buffer::TCBuffer=defaultcontractbuffer)
#     spaceA=conjA=='C' ? conj(space(A)) : space(A)
#     spaceB=conjB=='C' ? conj(space(B)) : space(B)
#     spaceC=(spaceA ⊗ spaceB)[indexin(outputlabels,vcat(labelsA,labelsB))]
#     T=promote_type(eltype(A),eltype(B))
#     C=similar(A,T,spaceC)
#     tensorcontract!(1,A,labelsA,conjA,B,labelsB,conjB,0,C,outputlabels;method=method,buffer=buffer)
#     return C
# end
# tensorcontract{S}(A::AbstractTensor{S},labelsA,B::AbstractTensor{S},labelsB,outputlabels=symdiff(labelsA,labelsB);
#     method::Symbol=:BLAS,buffer::TCBuffer=defaultcontractbuffer)=tensorcontract(A,labelsA,'N',B,labelsB,'N',outputlabels;method=method,buffer=buffer)
#
# function tensorproduct{S}(A::AbstractTensor{S},labelsA,B::AbstractTensor{S},labelsB,outputlabels=vcat(labelsA,labelsB))
#     spaceA=space(A)
#     spaceB=space(B)
#
#     spaceC=(spaceA ⊗ spaceB)[indexin(outputlabels,vcat(labelsA,labelsB))]
#     T=promote_type(eltype(A),eltype(B))
#     C=similar(A,T,spaceC)
#     tensorproduct!(1,A,labelsA,B,labelsB,0,C,outputlabels)
#     return C
# end

# Factorization
#---------------
# Base.svd{S,T}(t::AbstractTensor{S,T,2}, truncation::TruncationScheme = notrunc()) = svd(t, 1, 2, truncation)
# leftorth{S,T}(t::AbstractTensor{S,T,2}, truncation::TruncationScheme = notrunc()) = leftorth(t, 1, 2, truncation)
# rightorth{S,T}(t::AbstractTensor{S,T,2}, truncation::TruncationScheme = notrunc()) = rightorth(t, 1, 2, truncation)
#
# # general tensor factorizations: permute to correct order and pass to in place methods
# """
#     svd(t::AbstractTensor, leftind, rightind = setdiff(1:numind(t),leftind), truncation::TruncationScheme = notrunc()) -> U,S,V'
#
# Create orthonormal basis `U` for indices in `leftind` and orthonormal basis `V'`for indices in
# `rightind`, and a diagonal tensor with singular values `S`, such that tensor `t`
# (permuted into index order `vcat(leftind,rightind)`) can be written as `U*S*V'`.
#
# A truncation parameter can be specified for the new internal dimension, in which case
# a singular value decomposition will be performed. See `svd(!)` for further information.
# """
# Base.svd(t::AbstractTensor,leftind,truncation::TruncationScheme=notrunc())=svd(t,leftind,setdiff(1:numind(t),leftind),truncation)
# function Base.svd(t::AbstractTensor,leftind,rightind,truncation::TruncationScheme=notrunc())
#     N = numind(t)
#     p = vcat(leftind,rightind)
#     (isperm(p) && length(p)==N) || throw(IndexError("Not a valid bipartation of the tensor indices"))
#     newt = tensorcopy(t, 1:N, p)
#     return svd!(newt, length(leftind), truncation)
# end
#
# """
#     leftorth(t::AbstractTensor, leftind, rightind = setdiff(1:numind(t),leftind), truncation::TruncationScheme = notrunc()) -> Q,R
#
# Create orthonormal basis `Q` for indices in `leftind`, and remainder `R` such that tensor `t`
# (permuted into index order `vcat(leftind,rightind)`) can be written as `Q*R`.
#
# This decomposition should be unique, such that it always returns the same result for the
# same input tensor `t`. The QR decomposition is fastest but only unique after correcting for
# phases. A truncation parameter can be specified for the new internal dimension, in which case
# a singular value decomposition will be performed. See `svd(!)` for further information.
# """
# leftorth(t::AbstractTensor, leftind, truncation::TruncationScheme) = leftorth(t, leftind, setdiff(1:numind(t),leftind), truncation)
# function leftorth(t::AbstractTensor, leftind, rightind=setdiff(1:numind(t),leftind), truncation::TruncationScheme = notrunc())
#     N = numind(t)
#     p = vcat(leftind,rightind)
#     (isperm(p) && length(p)==N) || throw(IndexError("Not a valid bipartation of the tensor indices"))
#     newt = tensorcopy(t, 1:N, p)
#     return leftorth!(newt, length(leftind), truncation)
# end
#
# """
#     rightorth(t::AbstractTensor, leftind, rightind = setdiff(1:numind(t),leftind), truncation::TruncationScheme = notrunc()) -> L,Q
#
# Create orthonormal basis `Q'` for indices in `rightind`, and remainder `L` such that tensor `t`
# (permuted into index order `vcat(leftind,rightind)`) can be written as `L*Q`.
#
# This decomposition should be unique, such that it always returns the same result for the
# same input tensor `t`. The LQ decomposition is fastest but only unique after correcting for
# phases. A truncation parameter can be specified for the new internal dimension, in which case
# a singular value decomposition will be performed. See `svd(!)` for further information.
# """
# rightorth(t::AbstractTensor, leftind, truncation::TruncationScheme) = rightorth(t, leftind, setdiff(1:numind(t), leftind), truncation)
# function rightorth(t::AbstractTensor, leftind, rightind = setdiff(1:numind(t), leftind), truncation::TruncationScheme = notrunc())
#     N = numind(t)
#     p = vcat(leftind, rightind)
#     (isperm(p) && length(p)==N) || throw(IndexError("Not a valid bipartation of the tensor indices"))
#     newt = tensorcopy(t, 1:N, p)
#     return rightorth!(newt, length(leftind), truncation)
# end
#
# svd!(t::AbstractTensor, n::Int) = svd!(t, n, notrunc())
#
# """
#     leftorth!(t::AbstractTensor, n::Int, truncation::TruncationScheme = notrunc()) -> Q,R
#
# Create orthonormal basis `Q` for the first `n` indices and remainder `R` such that tensor `t` can be written as `Q*R`.
#
# This decomposition should be unique, such that it always returns the same result for the
# same input tensor `t`. The QR decomposition is fastest but only unique after correcting for
# phases. A truncation parameter can be specified for the new internal dimension, in which case
# a singular value decomposition will be performed. See `svd(!)` for further information.
#
# The data in input tensor `t` is overwritten for the computation of `Q` and `R`.
# """
# leftorth!(t::AbstractTensor, n::Int) = leftorth!(t, n, notrunc())
#
# """
#     rightorth!(t::AbstractTensor, n::Int, truncation::TruncationScheme = notrunc()) -> Q,R
#
# Create orthonormal basis `Q'` for the indices `n+1:numind(t)` and remainder `L` such that tensor `t` can be written as `L*Q`.
#
# This decomposition should be unique, such that it always returns the same result for the
# same input tensor `t`. The LQ decomposition is fastest but only unique after correcting for
# phases. A truncation parameter can be specified for the new internal dimension, in which case
# a singular value decomposition will be performed. See `svd(!)` for further information.
#
# The data in input tensor `t` is overwritten for the computation of `L` and `Q`.
# """
# rightorth!(t::AbstractTensor, n::Int) = rightorth!(t, n, notrunc())
