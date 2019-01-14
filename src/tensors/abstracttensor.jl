# abstracttensor.jl
#
# Abstract Tensor type
#----------------------
"""
    abstract type AbstractTensorMap{S<:IndexSpace, N₁, N₂} end

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
Base.@pure Base.eltype(T::Type{<:AbstractTensorMap}) = eltype(storagetype(T))
Base.@pure similarstoragetype(TT::Type{<:AbstractTensorMap}, ::Type{T}) where {T} =
    Core.Compiler.return_type(similar, Tuple{storagetype(TT), Type{T}})

storagetype(t::AbstractTensorMap) = storagetype(typeof(t))
similarstoragetype(t::AbstractTensorMap, T) = similarstoragetype(typeof(t), T)
Base.eltype(t::AbstractTensorMap) = eltype(typeof(t))
spacetype(t::AbstractTensorMap) = spacetype(typeof(t))
sectortype(t::AbstractTensorMap) = sectortype(typeof(t))
field(t::AbstractTensorMap) = field(typeof(t))
numout(t::AbstractTensorMap) = numout(typeof(t))
numin(t::AbstractTensorMap) = numin(typeof(t))
numind(t::AbstractTensorMap) = numind(typeof(t))

Base.@pure spacetype(::Type{<:AbstractTensorMap{S}}) where {S<:IndexSpace} = S
Base.@pure sectortype(::Type{<:AbstractTensorMap{S}}) where {S<:IndexSpace} = sectortype(S)
Base.@pure field(::Type{<:AbstractTensorMap{S}}) where {S<:IndexSpace} = field(S)
Base.@pure numout(::Type{<:AbstractTensorMap{<:IndexSpace,N₁,N₂}}) where {N₁, N₂} = N₁
Base.@pure numin(::Type{<:AbstractTensorMap{<:IndexSpace,N₁,N₂}}) where {N₁, N₂} = N₂
Base.@pure numind(::Type{<:AbstractTensorMap{<:IndexSpace,N₁,N₂}}) where {N₁, N₂} = N₁ + N₂

const order = numind

# tensormap implementation should provide codomain(t) and domain(t)
codomain(t::AbstractTensorMap, i) = codomain(t)[i]
domain(t::AbstractTensorMap, i) = domain(t)[i]
source(t::AbstractTensorMap) = domain(t) # categorical terminology
target(t::AbstractTensorMap) = codomain(t) # categorical terminology
space(t::AbstractTensorMap{<:IndexSpace,N₁,N₂}, i) where {N₁,N₂} = i <= N₁ ? codomain(t,i) : dual(domain(t, i-N₁))

space(t::AbstractTensor) = codomain(t)
space(t::AbstractTensor, i) = space(t)[i]

# some index manipulation utilities
codomainind(t::AbstractTensorMap{<:IndexSpace,N₁,N₂}) where {N₁,N₂} = ntuple(n->n, StaticLength(N₁))
domainind(t::AbstractTensorMap{<:IndexSpace,N₁,N₂}) where {N₁,N₂} = ntuple(n->N₁+n, StaticLength(N₂))

adjointtensorindex(t::AbstractTensorMap{<:IndexSpace,N₁,N₂}, i) where {N₁,N₂} = ifelse(i<=N₁, N₂+i, i-N₁)
# NOTE: do we still need this
tensor2spaceindex(t::AbstractTensorMap{<:IndexSpace,N₁,N₂}, i) where {N₁,N₂} = ifelse(i<=N₁, i, 2N₁+N₂+1-i)
space2tensorindex(t::AbstractTensorMap{<:IndexSpace,N₁,N₂}, i) where {N₁,N₂} = ifelse(i<=N₁, i, 2N₁+N₂+1-i)

# Defining vector spaces:
#------------------------
const TensorSpace{S<:IndexSpace} = Union{S, ProductSpace{S}}
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

# NOTE: do we still need this
function blocksectors(codom::ProductSpace{S,N₁}, dom::ProductSpace{S,N₂}) where {S,N₁,N₂}
    sectortype(S) == Trivial && return (Trivial(),)
    return intersect(blocksectors(codom), blocksectors(dom))
end

# Conversion to Array:
#----------------------
# probably not optimized for speed, only for checking purposes
function Base.convert(::Type{Array}, t::AbstractTensorMap{S,N₁,N₂}) where {S,N₁,N₂}
    G = sectortype(t)
    if G == Trivial
        convert(Array, t[])
    elseif FusionStyle(G) isa Abelian || FusionStyle(G) isa SimpleNonAbelian
        # TODO: Frobenius-Schur indicators!, and fermions!
        cod = codomain(t)
        dom = domain(t)
        A = fill(zero(eltype(t)), (dims(cod)..., dims(dom)...))
        for (f1,f2) in fusiontrees(t)
            F1 = convert(Array, f1)
            F2 = convert(Array, f2)
            for i = 1:N₁
                if isdual(cod[i])
                    a = f1.uncoupled[i]
                    Z = sqrt(dim(a))*permutedims(conj(reshape(fusiontensor(a,dual(a),one(a)), (dim(a),dim(a)))),(2,1))
                    indF = ntuple(k->(k == i ? -i : k), StaticLength(N₁)+StaticLength(1))
                    indout = ntuple(identity, StaticLength(N₁)+StaticLength(1))
                    F1 = TensorOperations.tensorcontract(Z,(i,-i), F1, indF, indout)
                end
            end
            for i = 1:N₂
                if isdual(dom[i])
                    a = f2.uncoupled[i]
                    Z = sqrt(dim(a))*permutedims(conj(reshape(fusiontensor(a,dual(a),one(a)), (dim(a),dim(a)))),(2,1))
                    indF = ntuple(k->(k == i ? -i : k), StaticLength(N₂)+StaticLength(1))
                    indout = ntuple(identity, StaticLength(N₂)+StaticLength(1))
                    F2 = TensorOperations.tensorcontract(Z,(i,-i), F2, indF, indout)
                end
            end
            sz1 = size(F1)
            sz2 = size(F2)
            d1 = TupleTools.front(sz1)
            d2 = TupleTools.front(sz2)
            F = reshape(reshape(F1, TupleTools.prod(d1), sz1[end])*reshape(F2, TupleTools.prod(d2), sz2[end])', (d1...,d2...))
            Aslice = StridedView(A)[axes(cod, f1.uncoupled)..., axes(dom, f2.uncoupled)...]
            axpy!(1, StridedView(_kron(convert(Array, t[f1,f2]), F)), Aslice)
        end
        return A
    else
        # TODO: implement DegenerateNonAbelian case
        throw(MethodError(convert, (Array, t)))
    end
end
# TODO: Reverse conversion
