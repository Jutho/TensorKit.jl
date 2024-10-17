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
struct FusionBlockStructure{I,F₁,F₂}
    totaldim::Int
    blockstructure::SectorDict{I,Tuple{Tuple{Int,Int},UnitRange{Int}}}
    fusiontreelist::Vector{Tuple{F₁,F₂}}
    fusiontreestructure::Vector{Tuple{Tuple{Int,Int},Tuple{Int,Int},Int}}
    fusiontreeindices::FusionTreeDict{Tuple{F₁,F₂},Int}
end

abstract type CacheStyle end
struct NoCache <: CacheStyle end
struct TaskLocalCache{D<:AbstractDict} <: CacheStyle end
struct GlobalLRUCache <: CacheStyle end

function CacheStyle(I::Type{<:Sector})
    return GlobalLRUCache()
    # if FusionStyle(I) === UniqueFusion()
    #     return TaskLocalCache{SectorDict{I,Any}}()
    # else
    #     return GlobalCache()
    # end
end

fusionblockstructure(W::HomSpace) = fusionblockstructure(W, CacheStyle(sectortype(W)))

function fusionblockstructure(W::HomSpace, ::NoCache)
    codom = codomain(W)
    dom = domain(W)
    N₁ = length(codom)
    N₂ = length(dom)
    I = sectortype(W)
    F₁ = fusiontreetype(I, N₁)
    F₂ = fusiontreetype(I, N₂)

    # output structure
    blockstructure = SectorDict{I,Tuple{Tuple{Int,Int},UnitRange{Int}}}() # size, range
    fusiontreelist = Vector{Tuple{F₁,F₂}}()
    fusiontreestructure = Vector{Tuple{Tuple{Int,Int},Tuple{Int,Int},Int}}() # size, strides, offset

    # temporary data structures
    splittingtrees = Vector{F₁}()
    splittingstructure = Vector{Tuple{Int,Int}}()

    # main computational routine
    blockoffset = 0
    for c in blocksectors(W)
        empty!(splittingtrees)
        empty!(splittingstructure)

        offset₁ = 0
        for f₁ in fusiontrees(codom, c)
            push!(splittingtrees, f₁)
            d₁ = dim(codom, f₁.uncoupled)
            push!(splittingstructure, (offset₁, d₁))
            offset₁ += d₁
        end
        blockdim₁ = offset₁
        strides = (1, blockdim₁)

        offset₂ = 0
        for f₂ in fusiontrees(dom, c)
            s₂ = f₂.uncoupled
            d₂ = dim(dom, s₂)
            for (f₁, (offset₁, d₁)) in zip(splittingtrees, splittingstructure)
                push!(fusiontreelist, (f₁, f₂))
                totaloffset = blockoffset + offset₂ * blockdim₁ + offset₁
                push!(fusiontreestructure, ((d₁, d₂), strides, totaloffset))
            end
            offset₂ += d₂
        end
        blockdim₂ = offset₂
        blocksize = (blockdim₁, blockdim₂)
        blocklength = blockdim₁ * blockdim₂
        blockrange = (blockoffset + 1):(blockoffset + blocklength)
        blockoffset = last(blockrange)
        blockstructure[c] = (blocksize, blockrange)
    end

    fusiontreeindices = sizehint!(FusionTreeDict{Tuple{F₁,F₂},Int}(),
                                  length(fusiontreelist))
    for (i, f₁₂) in enumerate(fusiontreelist)
        fusiontreeindices[f₁₂] = i
    end
    totaldim = blockoffset
    structure = FusionBlockStructure{I,F₁,F₂}(totaldim, blockstructure,
                                              fusiontreelist, fusiontreestructure,
                                              fusiontreeindices)
    return structure
end

function fusionblockstructure(W::HomSpace, ::TaskLocalCache{D}) where {D}
    cache::D = get!(task_local_storage(), :_local_tensorstructure_cache) do
        return D()
    end
    N₁ = length(codomain(W))
    N₂ = length(domain(W))
    I = sectortype(W)
    F₁ = fusiontreetype(I, N₁)
    F₂ = fusiontreetype(I, N₂)
    structure::FusionBlockStructure{I,F₁,F₂} = get!(cache, W) do
        return fusionblockstructure(W, NoCache())
    end
    return structure
end

const GLOBAL_FUSIONBLOCKSTRUCTURE_CACHE = LRU{Any,Any}(; maxsize=10^4)
# 10^4 different tensor spaces should be enough for most purposes
function fusionblockstructure(W::HomSpace, ::GlobalLRUCache)
    cache = GLOBAL_FUSIONBLOCKSTRUCTURE_CACHE
    N₁ = length(codomain(W))
    N₂ = length(domain(W))
    I = sectortype(W)
    F₁ = fusiontreetype(I, N₁)
    F₂ = fusiontreetype(I, N₂)
    structure::FusionBlockStructure{I,F₁,F₂} = get!(cache, W) do
        return fusionblockstructure(W, NoCache())
    end
    return structure
end
