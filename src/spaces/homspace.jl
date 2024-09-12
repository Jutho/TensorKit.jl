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

function ←(codom::ProductSpace{S}, dom::ProductSpace{S}) where {S<:ElementarySpace}
    return HomSpace(codom, dom)
end
function ←(codom::S, dom::S) where {S<:ElementarySpace}
    return HomSpace(ProductSpace(codom), ProductSpace(dom))
end
←(codom::VectorSpace, dom::VectorSpace) = ←(promote(codom, dom)...)
→(dom::VectorSpace, codom::VectorSpace) = ←(codom, dom)

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
        return sort!(filter!(c -> hasblock(codom, c), collect(blocksectors(dom))))
    else
        return sort!(filter!(c -> hasblock(dom, c), collect(blocksectors(codom))))
    end
end

"""
    hasblock(W::HomSpace, c::Sector)

Query whether a coupled sector `c` appears in both the codomain and domain of `W`.

See also [`blocksectors`](@ref).
"""
hasblock(W::HomSpace, c::Sector) = hasblock(codomain(W), c) && hasblock(domain(W), c)

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

# Operations on HomSpaces
# -----------------------
function permute(W::HomSpace{S}, (p₁, p₂)::Index2Tuple{N₁,N₂}) where {S,N₁,N₂}
    cod = ProductSpace{S,N₁}(map(n -> W[n], p₁))
    dom = ProductSpace{S,N₂}(map(n -> dual(W[n]), p₂))
    return cod ← dom
end

"""
    compose(W::HomSpace, V::HomSpace)

Obtain the HomSpace that is obtained from composing the morphisms in `W` and `V`. For this
to be possible, the domain of `W` must match the codomain of `V`.
"""
function compose(W::HomSpace{S}, V::HomSpace{S}) where {S}
    domain(W) == codomain(V) || throw(SpaceMismatch("$(domain(W)) ≠ $(codomain(V))"))
    return HomSpace(codomain(W), domain(V))
end

# Block and fusion tree ranges: structure information for building tensors
#--------------------------------------------------------------------------
struct TensorStructure{I,F₁,F₂}
    totaldim::Int
    blockstructure::SectorDict{I,Tuple{Tuple{Int,Int},UnitRange{Int}}}
    fusiontreelist::Vector{Tuple{F₁,F₂}}
    fusiontreeranges::Vector{Tuple{UnitRange{Int},UnitRange{Int}}}
    fusiontreeindices::FusionTreeDict{Tuple{F₁,F₂},Int}
end

abstract type CacheStyle end
struct NoCache <: CacheStyle end
struct TaskLocalCache{D<:AbstractDict} <: CacheStyle end
struct GlobalCache <: CacheStyle end

function CacheStyle(I::Type{<:Sector})
    return GlobalCache()
    # if FusionStyle(I) === UniqueFusion()
    #     return TaskLocalCache{SectorDict{I,Any}}()
    # else
    #     return GlobalCache()
    # end
end

tensorstructure(W::HomSpace) = tensorstructure(W, CacheStyle(sectortype(W)))

function tensorstructure(W::HomSpace, ::NoCache)
    codom = codomain(W)
    dom = domain(W)
    N₁ = length(codom)
    N₂ = length(dom)
    I = sectortype(W)
    F₁ = fusiontreetype(I, N₁)
    F₂ = fusiontreetype(I, N₂)

    blockstructure = SectorDict{I,Tuple{Tuple{Int,Int},UnitRange{Int}}}()
    fusiontreelist = Vector{Tuple{F₁,F₂}}()
    fusiontreeranges = Vector{Tuple{UnitRange{Int},UnitRange{Int}}}()
    outer_offset = 0
    for c in blocksectors(W)
        inner_offset₂ = 0
        inner_offset₁ = 0
        for f₂ in fusiontrees(dom, c)
            s₂ = f₂.uncoupled
            d₂ = dim(dom, s₂)
            r₂ = (inner_offset₂ + 1):(inner_offset₂ + d₂)
            inner_offset₂ = last(r₂)
            # TODO:  # now we run the code below for every f₂; should we do this separately
            inner_offset₁ = 0 # reset here to avoid multiple counting
            for f₁ in fusiontrees(codom, c)
                s₁ = f₁.uncoupled
                d₁ = dim(codom, s₁)
                r₁ = (inner_offset₁ + 1):(inner_offset₁ + d₁)
                inner_offset₁ = last(r₁)
                push!(fusiontreelist, (f₁, f₂))
                push!(fusiontreeranges, (r₁, r₂))
            end
        end
        blocksize = (inner_offset₁, inner_offset₂)
        blocklength = blocksize[1] * blocksize[2]
        blockrange = (outer_offset + 1):(outer_offset + blocklength)
        outer_offset = last(blockrange)
        blockstructure[c] = (blocksize, blockrange)
    end
    fusiontreeindices = sizehint!(FusionTreeDict{Tuple{F₁,F₂},Int}(), length(fusiontreelist))
    for i = 1:length(fusiontreelist)
        fusiontreeindices[fusiontreelist[i]] = i
    end
    totaldim = outer_offset
    structure = TensorStructure(totaldim, blockstructure, fusiontreelist, fusiontreeranges, fusiontreeindices)
    return structure
end

function tensorstructure(W::HomSpace, ::TaskLocalCache{D}) where {D}
    cache::D = get!(task_local_storage(), :_local_tensorstructure_cache) do 
        return D()
    end
    N₁ = length(codomain(W))
    N₂ = length(domain(W))
    I = sectortype(W)
    F₁ = fusiontreetype(I, N₁)
    F₂ = fusiontreetype(I, N₂)
    structure::TensorStructure{I,F₁,F₂} = get!(cache, W) do
        tensorstructure(W, NoCache())
    end
    return structure
end

const GLOBAL_TENSORSTRUCTURE_CACHE = LRU{Any,Any}(; maxsize=10^4)
# 10^4 different tensor spaces should be enough for most purposes
function tensorstructure(W::HomSpace, ::GlobalCache)
    cache = GLOBAL_TENSORSTRUCTURE_CACHE
    N₁ = length(codomain(W))
    N₂ = length(domain(W))
    I = sectortype(W)
    F₁ = fusiontreetype(I, N₁)
    F₂ = fusiontreetype(I, N₂)
    structure::TensorStructure{I,F₁,F₂} = get!(cache, W) do
        return tensorstructure(W, NoCache())
    end
    return structure
end

