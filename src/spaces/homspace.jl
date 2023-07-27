"""
    struct HomSpace{S<:ElementarySpace, P1<:CompositeSpace{S}, P2<:CompositeSpace{S}}
        codomain::P1
        domain::P2
    end

Represents the linear space of morphisms with codomain of type `P1` and domain of type `P2`.
Note that HomSpace is not a subtype of VectorSpace, i.e. we restrict the latter to denote
certain categories and their objects, and keep HomSpace distinct.
"""
struct HomSpace{S<:ElementarySpace,P1<:CompositeSpace{S},P2<:CompositeSpace{S}}
    codomain::P1
    domain::P2
end
codomain(W::HomSpace) = W.codomain
domain(W::HomSpace) = W.domain

dual(W::HomSpace) = HomSpace(dual(W.domain), dual(W.codomain))
function Base.adjoint(W::HomSpace{S}) where {S}
    InnerProductStyle(S) === EuclideanProduct() ||
        throw(ArgumentError("adjoint requires Euclidean inner product"))
    return HomSpace(W.domain, W.codomain)
end

Base.hash(W::HomSpace, h::UInt) = hash(domain(W), hash(codomain(W), h))
function Base.:(==)(W₁::HomSpace, W₂::HomSpace)
    return (W₁.codomain == W₂.codomain) && (W₁.domain == W₂.domain)
end

spacetype(W::HomSpace) = spacetype(typeof(W))
sectortype(W::HomSpace) = sectortype(typeof(W))
field(W::HomSpace) = field(typeof(W))

spacetype(::Type{<:HomSpace{S}}) where {S} = S
field(L::Type{<:HomSpace}) = field(spacetype(L))
sectortype(L::Type{<:HomSpace}) = sectortype(spacetype(L))

const TensorSpace{S<:ElementarySpace} = Union{S,ProductSpace{S}}
const TensorMapSpace{S<:ElementarySpace,N₁,N₂} = HomSpace{S,ProductSpace{S,N₁},
                                                          ProductSpace{S,N₂}}

function Base.getindex(W::TensorMapSpace{<:IndexSpace,N₁,N₂}, i) where {N₁,N₂}
    return i <= N₁ ? codomain(W)[i] : dual(domain(W)[i - N₁])
end

function →(dom::TensorSpace{S}, codom::TensorSpace{S}) where {S<:ElementarySpace}
    return HomSpace(ProductSpace(codom), ProductSpace(dom))
end

function ←(codom::TensorSpace{S}, dom::TensorSpace{S}) where {S<:ElementarySpace}
    return HomSpace(ProductSpace(codom), ProductSpace(dom))
end

function Base.show(io::IO, W::HomSpace)
    if length(W.codomain) == 1
        print(io, W.codomain[1])
    else
        print(io, W.codomain)
    end
    print(io, " ← ")
    if length(W.domain) == 1
        print(io, W.domain[1])
    else
        print(io, W.domain)
    end
end

"""
    blocksectors(W::HomSpace)

Return an iterator over the different unique coupled sector labels, i.e. the intersection
of the different fusion outputs that can be obtained by fusing the sectors present in the
domain, as well as from the codomain.

See also [`hasblock`](@ref).
"""
function blocksectors(W::HomSpace)
    sectortype(W) === Trivial &&
        return OneOrNoneIterator(dim(domain(W)) != 0 && dim(codomain(W)) != 0, Trivial())

    codom = codomain(W)
    dom = domain(W)
    N₁ = length(codom)
    N₂ = length(dom)
    I = sectortype(W)
    if N₁ == 0 || N₂ == 0
        return (one(I),)
    elseif N₂ <= N₁
        return filter!(c->hasblock(codom, c), collect(blocksectors(dom)))
    else
        return filter!(c->hasblock(dom, c), collect(blocksectors(codom)))
    end
end

"""
    hasblock(W::HomSpace, c::Sector)

Query whether a coupled sector `c` appears in both the codomain and domain of `W`.

See also [`blocksectors`](@ref).
"""
hasblock(W::HomSpace, c::Sector) =  hasblock(codomain(W), c) && hasblock(domain(W), c)

"""
    dim(W::HomSpace)

Return the total dimension of a `HomSpace`, i.e. the number of linearly independent
morphisms that can be constructed within this space.
"""
function dim(W::HomSpace)
    d = 0
    for c in blocksectors(W)
        d += blockdim(codomain(W), c) * blockdim(domain(W), c)
    end
    return d
end
