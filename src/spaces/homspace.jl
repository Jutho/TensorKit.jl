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
    InnerProductStyle(S) === EuclideanInnerProduct() ||
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

numout(W::HomSpace) = length(codomain(W))
numin(W::HomSpace) = length(domain(W))
numind(W::HomSpace) = numin(W) + numout(W)

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
"""
    permute(W::HomSpace, (p₁, p₂)::Index2Tuple{N₁,N₂})

Return the `HomSpace` obtained by permuting the indices of the domain and codomain of `W`
according to the permutation `p₁` and `p₂` respectively.
"""
function permute(W::HomSpace{S}, (p₁, p₂)::Index2Tuple{N₁,N₂}) where {S,N₁,N₂}
    p = (p₁..., p₂...)
    TupleTools.isperm(p) && length(p) == numind(W) ||
        throw(ArgumentError("$((p₁, p₂)) is not a valid permutation for $(W)"))
    return select(W, (p₁, p₂))
end

"""
    select(W::HomSpace, (p₁, p₂)::Index2Tuple{N₁,N₂})

Return the `HomSpace` obtained by a selection from the domain and codomain of `W` according
to the indices in `p₁` and `p₂` respectively.
"""
function select(W::HomSpace{S}, (p₁, p₂)::Index2Tuple{N₁,N₂}) where {S,N₁,N₂}
    cod = ProductSpace{S,N₁}(map(n -> W[n], p₁))
    dom = ProductSpace{S,N₂}(map(n -> dual(W[n]), p₂))
    return cod ← dom
end

"""
    flip(W::HomSpace, I)

Return a new `HomSpace` object by applying `flip` to each of the spaces in the domain and
codomain of `W` for which the linear index `i` satisfies `i ∈ I`.
"""
function flip(W::HomSpace{S}, I) where {S}
    cod′ = let cod = codomain(W)
        ProductSpace{S}(ntuple(i -> i ∈ I ? flip(cod[i]) : cod[i], numout(W)))
    end
    dom′ = let dom = domain(W)
        ProductSpace{S}(ntuple(i -> (i + numout(W)) ∈ I ? flip(dom[i]) : dom[i], numin(W)))
    end
    return cod′ ← dom′
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

"""
    insertleftunit(W::HomSpace, i::Int=numind(W) + 1; conj=false, dual=false)

Insert a trivial vector space, isomorphic to the underlying field, at position `i`.
More specifically, adds a left monoidal unit or its dual.

See also [`insertrightunit`](@ref), [`removeunit`](@ref).
"""
@constprop :aggressive function insertleftunit(W::HomSpace, i::Int=numind(W) + 1;
                                               conj::Bool=false, dual::Bool=false)
    if i ≤ numout(W)
        return insertleftunit(codomain(W), i; conj, dual) ← domain(W)
    else
        return codomain(W) ← insertleftunit(domain(W), i - numout(W); conj, dual)
    end
end

"""
    insertrightunit(W::HomSpace, i::Int=numind(W); conj=false, dual=false)

Insert a trivial vector space, isomorphic to the underlying field, after position `i`.
More specifically, adds a right monoidal unit or its dual.

See also [`insertleftunit`](@ref), [`removeunit`](@ref).
"""
@constprop :aggressive function insertrightunit(W::HomSpace, i::Int=numind(W);
                                                conj::Bool=false, dual::Bool=false)
    if i ≤ numout(W)
        return insertrightunit(codomain(W), i; conj, dual) ← domain(W)
    else
        return codomain(W) ← insertrightunit(domain(W), i - numout(W); conj, dual)
    end
end

"""
    removeunit(P::HomSpace, i::Int)

This removes a trivial tensor product factor at position `1 ≤ i ≤ N`.
For this to work, that factor has to be isomorphic to the field of scalars.

This operation undoes the work of [`insertleftunit`](@ref) or [`insertrightunit`](@ref).
"""
@constprop :aggressive function removeunit(P::HomSpace, i::Int)
    if i ≤ numout(P)
        return removeunit(codomain(P), i) ← domain(P)
    else
        return codomain(P) ← removeunit(domain(P), i - numout(P))
    end
end

# Block and fusion tree ranges: structure information for building tensors
#--------------------------------------------------------------------------
struct FusionBlockStructure{I,N,F₁,F₂}
    totaldim::Int
    blockstructure::SectorDict{I,Tuple{Tuple{Int,Int},UnitRange{Int}}}
    fusiontreelist::Vector{Tuple{F₁,F₂}}
    fusiontreestructure::Vector{Tuple{NTuple{N,Int},NTuple{N,Int},Int}}
    fusiontreeindices::FusionTreeDict{Tuple{F₁,F₂},Int}
end

abstract type CacheStyle end
struct NoCache <: CacheStyle end
struct TaskLocalCache{D<:AbstractDict} <: CacheStyle end
struct GlobalLRUCache <: CacheStyle end

function CacheStyle(I::Type{<:Sector})
    return GlobalLRUCache()
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
    fusiontreestructure = Vector{Tuple{NTuple{N₁ + N₂,Int},NTuple{N₁ + N₂,Int},Int}}() # size, strides, offset

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
                subsz = (dims(codom, f₁.uncoupled)..., dims(dom, f₂.uncoupled)...)
                @assert !any(isequal(0), subsz)
                substr = _subblock_strides(subsz, (d₁, d₂), strides)
                push!(fusiontreestructure, (subsz, substr, totaloffset))
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
    structure = FusionBlockStructure(totaldim, blockstructure,
                                     fusiontreelist, fusiontreestructure,
                                     fusiontreeindices)
    return structure
end

function _subblock_strides(subsz, sz, str)
    sz_simplify = Strided.StridedViews._simplifydims(sz, str)
    return Strided.StridedViews._computereshapestrides(subsz, sz_simplify...)
end

function fusionblockstructure(W::HomSpace, ::TaskLocalCache{D}) where {D}
    cache::D = get!(task_local_storage(), :_local_tensorstructure_cache) do
        return D()
    end
    N₁ = length(codomain(W))
    N₂ = length(domain(W))
    N = N₁ + N₂
    I = sectortype(W)
    F₁ = fusiontreetype(I, N₁)
    F₂ = fusiontreetype(I, N₂)
    structure::FusionBlockStructure{I,N,F₁,F₂} = get!(cache, W) do
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
    N = N₁ + N₂
    I = sectortype(W)
    F₁ = fusiontreetype(I, N₁)
    F₂ = fusiontreetype(I, N₂)
    structure::FusionBlockStructure{I,N,F₁,F₂} = get!(cache, W) do
        return fusionblockstructure(W, NoCache())
    end
    return structure
end

# Diagonal ranges
#----------------
# TODO: is this something we want to cache?
function diagonalblockstructure(W::HomSpace)
    structure = SectorDict{sectortype(W),UnitRange{Int}}() # range
    offset = 0
    dom = domain(W)[1]
    for c in blocksectors(W)
        d = dim(dom, c)
        structure[c] = offset .+ (1:d)
        offset += d
    end
    return structure
end
