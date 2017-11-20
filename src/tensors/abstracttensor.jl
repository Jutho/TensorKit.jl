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
Base.eltype(t::AbstractTensorMap) = eltype(typeof(t))
spacetype(t::AbstractTensorMap) = spacetype(typeof(t))
sectortype(t::AbstractTensorMap) = sectortype(typeof(t))
fieldtype(t::AbstractTensorMap) = fieldtype(typeof(t))
numout(t::AbstractTensorMap) = numin(typeof(t))
numin(t::AbstractTensorMap) = numout(typeof(t))
numind(t::AbstractTensorMap) = numind(typeof(t))

Base.@pure spacetype(::Type{<:AbstractTensorMap{S}}) where {S<:IndexSpace} = S
Base.@pure sectortype(::Type{<:AbstractTensorMap{S}}) where {S<:IndexSpace} = sectortype(S)
Base.@pure fieldtype(::Type{<:AbstractTensorMap{S}}) where {S<:IndexSpace} = fieldtype(S)
Base.@pure numout(::Type{<:AbstractTensorMap{<:IndexSpace,N₁,N₂}}) where {N₁, N₂} = N₁
Base.@pure numin(::Type{<:AbstractTensorMap{<:IndexSpace,N₁,N₂}}) where {N₁, N₂} = N₂
Base.@pure numind(::Type{<:AbstractTensorMap{<:IndexSpace,N₁,N₂}}) where {N₁, N₂} = N₁ + N₂

const order = numind

# tensormap implementation should provide codomain(t) and domain(t)
codomain(t::AbstractTensorMap, i) = codomain(t)[i]
domain(t::AbstractTensorMap, i) = domain(t)[i]
space(t::AbstractTensorMap{<:IndexSpace,N₁,N₂}, i) where {N₁,N₂} = i <= N₁ ? codomain(t,i) : dual(domain(t, i-N₁))

space(t::AbstractTensor) = codomain(t)
space(t::AbstractTensor, i) = space(t)[i]

tensor2spaceindex(t::AbstractTensorMap{<:IndexSpace,N₁,N₂}, i) where {N₁,N₂} = ifelse(i<=N₁, i, 2N₁+N₂+1-i)
space2tensorindex(t::AbstractTensorMap{<:IndexSpace,N₁,N₂}, i) where {N₁,N₂} = ifelse(i<=N₁, i, 2N₁+N₂+1-i)
adjointtensorindex(t::AbstractTensorMap{<:IndexSpace,N₁,N₂}, i) where {N₁,N₂} = ifelse(i<=N₁, N₂+i, i-N₁)

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

Base.normalize!(t::AbstractTensorMap, p::Real = 2) = scale!(t, inv(vecnorm(t,p)))
normalize(t::AbstractTensorMap, p::Real = 2) = normalize!(copy(t), p)

Base.:*(t1::AbstractTensorMap, t2::AbstractTensorMap) = Base.A_mul_B!(similar(t1, promote_type(eltype(t1),eltype(t2)), codomain(t1)←domain(t2)), t1, t2)

# Convert to Array
function Base.convert(::Type{Array}, t::AbstractTensorMap)
    G = sectortype(t)
    if G == Trivial
        convert(Array, t[])
    else
        # TODO: Frobenius-Schur indicators!, and fermions!
        cod = codomain(t)
        dom = domain(t)
        A = fill(zero(eltype(t)), (dims(cod)..., dims(dom)...))
        for (f1,f2) in fusiontrees(t)
            F1 = convert(Array, f1)
            F2 = convert(Array, f2)
            sz1 = size(F1)
            sz2 = size(F2)
            d1 = TupleTools.front(sz1)
            d2 = TupleTools.front(sz2)
            F = reshape(reshape(F1, TupleTools.prod(d1), sz1[end])*reshape(F2, TupleTools.prod(d2), sz2[end])', (d1...,d2...))
            Aslice = sview(A, indices(cod, f1.outgoing)..., indices(dom, f2.outgoing)...)
            Base.LinAlg.axpy!(1, StridedView(_kron(convert(Array,t[f1,f2]), F)), Aslice)
        end
        return A
    end
end

function _kron(A, B)
    sA = size(A)
    sB = size(B)
    s = map(*, sA, sB)
    C = Array{promote_type(eltype(A),eltype(B))}(s)
    for IA in eachindex(IndexCartesian(), A)
        for IB in eachindex(IndexCartesian(), B)
            I = CartesianIndex(IB.I .+ (IA.I .- 1) .* sB)
            C[I] = A[IA]*B[IB]
        end
    end
    return C
end