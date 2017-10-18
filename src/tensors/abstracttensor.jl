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

Base.@pure spacetype(::Type{<:AbstractTensorMap{S}}) where {S<:IndexSpace} = S
Base.@pure sectortype(::Type{<:AbstractTensorMap{S}}) where {S<:IndexSpace} = sectortype(S)
Base.@pure fieldtype(::Type{<:AbstractTensorMap{S}}) where {S<:IndexSpace} = fieldtype(S)
Base.@pure numin(::Type{<:AbstractTensorMap{<:IndexSpace,N₁,N₂}}) where {N₁, N₂} = N₁
Base.@pure numout(::Type{<:AbstractTensorMap{<:IndexSpace,N₁,N₂}}) where {N₁, N₂} = N₂
Base.@pure numind(::Type{<:AbstractTensorMap{<:IndexSpace,N₁,N₂}}) where {N₁, N₂} = N₁ + N₂

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

function blocksectors(codom::ProductSpace{S,N₁}, dom::ProductSpace{S,N₂}) where {S,N₁,N₂}
    G = sectortype(S)
    if G == Trivial
        return (Trivial(),)
    end
    if N₁ == 0
        c1 = Set{G}((one(G),))
    elseif N₁ == 1
        c1 = Set{G}(first(s) for s in sectors(codom))
    else
        c1 = foldl(union!, Set{G}(), (⊗(s...) for s in sectors(codom)))
    end
    if N₂ == 0
        c2 = Set{G}((one(G),))
    elseif N₂ == 1
        c2 = Set{G}(first(s) for s in sectors(dom))
    else
        c2 = foldl(union!, Set{G}(), (⊗(s...) for s in sectors(dom)))
    end

    return intersect(c1,c2)
end

# Basic algebra
#---------------
Base.copy(t::AbstractTensorMap) = Base.copy!(similar(t), t)

Base.:-(t::AbstractTensorMap) = scale!(copy(t), -one(eltype(t)))
function Base.:+(t1::AbstractTensorMap, t2::AbstractTensorMap)
    T = promote_type(eltype(t1), eltype(t2))
    return Base.LinAlg.axpy!(one(T), t2, copy!(similar(t1, T), t1))
end
function Base.:-(t1::AbstractTensorMap, t2::AbstractTensorMap)
    T = promote_type(eltype(t1), eltype(t2))
    return Base.LinAlg.axpy!(-one(T), t2, copy!(similar(t1, T), t1))
end

Base.:*(t::AbstractTensorMap, α::Number) = scale!(similar(t, promote_type(eltype(t), typeof(α))), t, α)
Base.:*(α::Number, t::AbstractTensorMap) = *(t, α)
Base.:/(t::AbstractTensorMap, α::Number) = *(t, one(α)/α)
Base.:\(α::Number, t::AbstractTensorMap) = *(t, one(α)/α)

Base.scale!(t::AbstractTensorMap, α::Number) = scale!(t, t, α)
Base.scale!(α::Number, t::AbstractTensorMap) = scale!(t, t, α)
Base.scale!(tdest::AbstractTensorMap, α::Number, tsrc::AbstractTensorMap) = scale!(tdest, tsrc, α)

Base.:*(t1::AbstractTensorMap, t2::AbstractTensorMap) = Base.A_mul_B!(similar(t1, promote_type(eltype(t1),eltype(t2)), codomain(t1)←domain(t2)), t1, t2)

# Index manipulations
#---------------------
"""
    permuteind(tsrc::AbstractTensorMap, leftind::NTuple{N₁,Int}, rightind::NTuple{N₂,Int}) -> tdst

Permutes the indices of `tsrc::AbstractTensorMap` such that a new tensor
`tdst::AbstractTensorMap{spacetype(t),N₁,N₂}` is obtained, with indices in `leftind`
playing the role of the codomain or range of the map, and indicies in `rightind` indicating
the domain.
"""
function permuteind(t::AbstractTensorMap, p1::NTuple{N₁,Int},  p2::NTuple{N₂,Int}=()) where {N₁,N₂}
    cod = codomain(t)
    dom = domain(t)
    N₁ + N₂ == length(cod)+length(dom) || throw(ArgumentError("not a valid permutation of length $(numind(t)): $p1 & $p2"))
    p = linearizepermutation(p1, p2, length(cod), length(dom))
    isperm(p) || throw(ArgumentError("not a valid permutation of length $(N₁+N₂): $p1 & $p2"))

    newspace = (cod ⊗ dual(dom))[p]
    newcod = newspace[ntuple(n->n, StaticLength(N₁))]
    newdom = dual(newspace[ntuple(n->N₁+n, StaticLength(N₂))])

    permuteind!(similar(t, newcod←newdom), t, p1, p2)
end

# Factorization
#---------------
const IndexTuple{N} = NTuple{N,Int}

"""
    svd(t::AbstractTensorMap, leftind::Tuple, rightind::Tuple, truncation::TruncationScheme = notrunc(), p::Real = 2) -> U,S,V,truncerr'

Performs the singular value decomposition such that `permute(t,leftind,rightind) = U * S *V`.

If `leftind` and `rightind` are not specified, the current partition of left and right indices
of `t` is used. In that case, less memory is allocated if one allows the data in `t` to
be destroyed/overwritten, by using `svd!(t, truncation = notrunc(), p = 2)`.

A truncation parameter can be specified for the new internal dimension, in which case
a truncated singular value decomposition will be computed. Choices are:
*   `notrunc()`: no truncation (default);
*   `truncerr(ϵ)`: truncates such that the p-norm of the truncated singular values is smaller than `ϵ` times the p-norm of all singular values;
*   `truncdim(χ)`: truncates such that the equivalent total dimension of the internal vector space is no larger than `χ`;
*   `truncspace(V)`: truncates such that the dimension of the internal vector space is smaller than that of `V` in any sector.

The `svd` also returns the truncation error `truncerr`, computed as the `p` norm of the
singular values that were truncated.

Orthogonality requires `spacetype(t)<:InnerProductSpace`, and `svd(!)` is currently
only implemented for `spacetype(t)<:EuclideanSpace`.
"""
Base.svd(t::AbstractTensorMap, p1::IndexTuple, p2::IndexTuple, trunc::TruncationScheme = NoTruncation(), p::Real = 2) = svd!(permuteind(t, p1, p2), trunc, p)

"""
    leftorth(t::AbstractTensorMap, leftind::Tuple, rightind::Tuple, truncation::TruncationScheme = notrunc()) -> Q, R

Create orthonormal basis `Q` for indices in `leftind`, and remainder `R` such that
`permute(t,leftind,rightind) = Q*R`.

If `leftind` and `rightind` are not specified, the current partition of left and right indices
of `t` is used. In that case, less memory is allocated if one allows the data in `t` to
be destroyed/overwritten, by using `leftorth!(t)`.

This decomposition should be unique, such that it always returns the same result for the
same input tensor `t`. This uses a QR decomposition with correction for making the diagonal
elements of R positive.

Orthogonality requires `spacetype(t)<:InnerProductSpace`, and `leftorth(!)` is currently
only implemented for `spacetype(t)<:EuclideanSpace`.
"""
leftorth(t::AbstractTensorMap, p1::IndexTuple, p2::IndexTuple, alg::OrthogonalFactorizationAlgorithm = QRpos()) = leftorth!(permuteind(t, p1, p2), alg)

"""
    rightorth(t::AbstractTensorMap, leftind::Tuple, rightind::Tuple, truncation::TruncationScheme = notrunc()) -> L, Q

Create orthonormal basis `Q` for indices in `leftind`, and remainder `R` such that
`permute(t,leftind,rightind) = L*Q`.

If `leftind` and `rightind` are not specified, the current partition of left and right indices
of `t` is used. In that case, less memory is allocated if one allows the data in `t` to
be destroyed/overwritten, by using `rightorth!(t)`.

This decomposition should be unique, such that it always returns the same result for the
same input tensor `t`. This uses an LQ decomposition with correction for making the diagonal
elements of R positive.

Orthogonality requires `spacetype(t)<:InnerProductSpace`, and `rightorth(!)` is currently
only implemented for `spacetype(t)<:EuclideanSpace`.
"""
rightorth(t::AbstractTensorMap, p1::IndexTuple, p2::IndexTuple, alg::OrthogonalFactorizationAlgorithm = LQpos()) = rightorth!(permuteind(t, p1, p2), alg)

"""
    leftnull(t::AbstractTensor, leftind::Tuple, rightind::Tuple) -> N

Create orthonormal basis for the orthogonal complement of the support of the indices in
`leftind`, such that `N' * permute(t, leftind, rightind) = 0`.

If `leftind` and `rightind` are not specified, the current partition of left and right indices
of `t` is used. In that case, less memory is allocated if one allows the data in `t` to
be destroyed/overwritten, by using `leftnull!(t)`.

Orthogonality requires `spacetype(t)<:InnerProductSpace`, and `leftnull(!)` is currently
only implemented for `spacetype(t)<:EuclideanSpace`.
"""
leftnull(t::AbstractTensorMap, p1::IndexTuple, p2::IndexTuple, alg::OrthogonalFactorizationAlgorithm = QRpos()) = leftnull!(permuteind(t, p1, p2), alg)

"""
    rightnull(t::AbstractTensor, leftind::Tuple, rightind::Tuple) -> N

Create orthonormal basis for the orthogonal complement of the support of the indices in
`rightind`, such that `permute(t, leftind, rightind)*N' = 0`.

If `leftind` and `rightind` are not specified, the current partition of left and right indices
of `t` is used. In that case, less memory is allocated if one allows the data in `t` to
be destroyed/overwritten, by using `rightnull!(t)`.

Orthogonality requires `spacetype(t)<:InnerProductSpace`, and `rightnull(!)` is currently
only implemented for `spacetype(t)<:EuclideanSpace`.
"""
rightnull(t::AbstractTensorMap, p1::IndexTuple, p2::IndexTuple, alg::OrthogonalFactorizationAlgorithm = LQpos()) = rightnull!(permuteind(t, p1, p2), alg)

"""
    eig(t::AbstractTensor, leftind::Tuple, rightind::Tuple) -> D, V

Compute eigenvalue factorization of tensor `t` as linear map from `rightind` to `leftind`.

If `leftind` and `rightind` are not specified, the current partition of left and right indices
of `t` is used. In that case, less memory is allocated if one allows the data in `t` to
be destroyed/overwritten, by using `eig!(t)`.
"""
Base.eig(t::AbstractTensorMap, p1::IndexTuple, p2::IndexTuple) = eig!(permuteind(t, p1, p2))

Base.svd(t::AbstractTensorMap, trunc::TruncationScheme = NoTruncation(), p::Real = 2) = svd!(copy(t), trunc, p)
leftorth(t::AbstractTensorMap, alg::OrthogonalFactorizationAlgorithm = QRpos()) = leftorth!(copy(t), alg)
rightorth(t::AbstractTensorMap, alg::OrthogonalFactorizationAlgorithm = LQpos()) = rightorth!(copy(t), alg)
leftnull(t::AbstractTensorMap, alg::OrthogonalFactorizationAlgorithm = QRpos()) = leftnull!(copy(t), alg)
rightnull(t::AbstractTensorMap, alg::OrthogonalFactorizationAlgorithm = LQpos()) = rightnull!(copy(t), alg)
Base.eig(t::AbstractTensorMap) = eig!(copy(t))

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
