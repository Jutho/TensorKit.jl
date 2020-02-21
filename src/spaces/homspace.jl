"""
    struct HomSpace{S<:ElementarySpace,P1<:CompositeSpace{S},P2<:CompositeSpace{S}}
        codomain::P1
        domain::P2
    end

Represents the linear space of morphisms with codomain of type `P1` and domain of type `P2`.
"""
struct HomSpace{S<:ElementarySpace,
                    P1<:CompositeSpace{S},
                    P2<:CompositeSpace{S}}
    codomain::P1
    domain::P2
end
codomain(W::HomSpace) = W.codomain
domain(W::HomSpace) = W.domain

Base.adjoint(W::HomSpace) = HomSpace(W.domain, W.codomain)

Base.hash(W::HomSpace, h::UInt) = hash(domain(W), hash(codomain(W), h))
Base.:(==)(W1::HomSpace, W2::HomSpace) =
    (W1.codomain == W2.codomain) & (W1.domain == W2.domain)

spacetype(::Type{<:HomSpace{S}}) where S = S
spacetype(W::HomSpace) = spacetype(typeof(W))
field(L::Type{<:HomSpace}) = field(spacetype(L))
sectortype(L::Type{<:HomSpace}) = sectortype(spacetype(L))

const TensorSpace{S} = Union{S, ProductSpace{S}}
const TensorMapSpace{S, N₁, N₂} = HomSpace{S, ProductSpace{S,N₁}, ProductSpace{S,N₂}}

Base.getindex(W::TensorMapSpace{<:IndexSpace,N₁,N₂}, i) where {N₁,N₂} =
    i <= N₁ ? codomain(W)[i] : dual(domain(W)[i-N₁])

→(dom::TensorSpace{S}, codom::TensorSpace{S}) where {S<:ElementarySpace} =
    HomSpace(ProductSpace(codom), ProductSpace(dom))

←(codom::TensorSpace{S}, dom::TensorSpace{S}) where {S<:ElementarySpace} =
    HomSpace(ProductSpace(codom), ProductSpace(dom))

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

# NOTE: do we need this
function blocksectors(W::HomSpace)
    sectortype(S) === Trivial && return (Trivial(),)
    return intersect(blocksectors(W.codom), blocksectors(W.dom))
end

# NOTE: do we need this
function blocksectors(codom::ProductSpace{S,N₁}, dom::ProductSpace{S,N₂}) where {S,N₁,N₂}
    sectortype(S) === Trivial && return (Trivial(),)
    return intersect(blocksectors(codom), blocksectors(dom))
end
