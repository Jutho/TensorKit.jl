"""
    struct GenericRepresentationSpace{G<:Sector} <: AbstractRepresentationSpace{G}

Generic implementation of a representation space, i.e. a complex Euclidean space
with a direct sum structure corresponding to different superselection sectors of
type `G<:Sector`, e.g. the irreps of a compact or finite group, or the labels of
a unitary fusion category.
"""
struct GenericRepresentationSpace{G<:Sector} <: RepresentationSpace{G}
    dims::Dict{G,Int}
    dual::Bool
end
GenericRepresentationSpace{G}(dims::Dict{G,Int}; dual::Bool = false) where {G<:Sector} = GenericRepresentationSpace{G}(dims, dual)

"""
    struct ZNSpace{N} <: AbstractRepresentationSpace{ZNIrrep{N}}

Optimized mplementation of a graded `ℤ_N` space, i.e. a complex Euclidean space graded
by the irreps of type `ZNIrrep{N}`
"""
struct ZNSpace{N} <: RepresentationSpace{ZNIrrep{N}}
    dims::NTuple{N,Int}
    dual::Bool
end

# Never write GenericRepresentationSpace, just use RepresentationSpace
RepresentationSpace{G}(dims::Vararg{Pair{<:Any,Int}}; dual::Bool = false) where {G<:Sector} = GenericRepresentationSpace{G}(Dict{G,Int}(convert(G,s)=>d for (s,d) in dims if d!=0), dual)
RepresentationSpace(dims::Dict{G,Int}; dual::Bool = false) where {G<:Sector} = GenericRepresentationSpace{G}(dims, dual)
RepresentationSpace(dims::Vararg{Pair{G,Int}}; dual::Bool = false) where {G<:Sector} = GenericRepresentationSpace{G}(Dict{G,Int}(dims...), dual)

# You might want to write ZNSpace
ZNSpace{N}(dims::NTuple{N,Int}; dual::Bool = false) where {N} = ZNSpace{N}(dims, dual)
ZNSpace{N}(dims::Vararg{Int,N}; dual::Bool = false) where {N} = ZNSpace{N}(dims, dual)
ZNSpace(dims::NTuple{N,Int}; dual::Bool = false) where {N} = ZNSpace{N}(dims, dual)
ZNSpace(dims::Vararg{Int,N}; dual::Bool = false) where {N} = ZNSpace{N}(dims, dual)

ZNSpace{N}(dims::Vararg{Pair{<:Any,Int}}; dual::Bool = false) where {N} = ZNSpace{N}((ZNIrrep{N}(c)=>d for (c,d) in dims)...; dual = dual)
ZNSpace{N}(dims::Dict{<:Any,Int}; dual::Bool = false) where {N} = ZNSpace{N}(dims...; dual = dual)
function ZNSpace{N}(args::Vararg{Pair{ZNIrrep{N},Int}}; dual::Bool = false) where {N}
    dims = ntuple(n->0, Val(N))
    @inbounds for (c,d) in args
        dims = setindex(dims, d, c.n+1)
    end
    return ZNSpace{N}(dims, dual)
end

RepresentationSpace{ZNIrrep{N}}(dims::Vararg{Pair{<:Any,Int}}; dual::Bool = false) where {N} = ZNSpace{N}((ZNIrrep{N}(c)=>d for (c,d) in dims)...; dual = dual)
RepresentationSpace(dims::Dict{ZNIrrep{N},Int}; dual::Bool = false) where {N} = ZNSpace{N}(dims...; dual = dual)
RepresentationSpace(dims::Vararg{Pair{ZNIrrep{N},Int}}; dual::Bool = false) where {N} = ZNSpace{N}(dims...; dual = dual)

Base.getindex(::ComplexNumbers, G::Type{<:Sector}) = RepresentationSpace{G}
Base.getindex(::ComplexNumbers, d1::Pair{G,Int}, dims::Vararg{Pair{G,Int}}) where {G<:Sector} = RepresentationSpace{G}(d1, dims...)

# Corresponding methods:
"""
    sectors(V::ElementarySpace) -> sectortype(V)
    sectors(V::ProductSpace{S,N}) -> NTuple{N,sectortype{V}}

Iterate over the different sectors in the vector space.
"""
sectors(V::GenericRepresentationSpace) = keys(V.dims)
sectors(V::ZNSpace{N}) where {N} = SectorSet{ZNIrrep{N}}(n-1 for n=1:N if V.dims[n] != 0)

checksectors(V::GenericRepresentationSpace{G}, s::G) where {G<:Sector} = s in keys(V.dims) || throw(SectorMismatch())
checksectors(V::ZNSpace{N}, c::ZNIrrep{N}) where {N} = V.dims[c.n+1] != 0 || throw(SectorMismatch())

# properties
dim(V::GenericRepresentationSpace) = sum(dim(c)*V.dims[c] for c in keys(V.dims))
dim(V::GenericRepresentationSpace{G}, c::G) where {G<:Sector} = get(V.dims, c, 0)

dim(V::ZNSpace) = sum(V.dims)
dim(V::ZNSpace{N}, c::ZNIrrep{N}) where {N} = V.dims[c.n+1]

Base.conj(V::GenericRepresentationSpace) = GenericRepresentationSpace(Dict(dual(c)=>dim(V,c) for c in sectors(V)), !V.dual)
Base.conj(V::ZNSpace{N}) where {N} = ZNSpace{N}((V.dims[1], reverse(tail(V.dims))...), !V.dual)

# equality / comparison
Base.:(==)(V1::RepresentationSpace, V2::RepresentationSpace) = (V1.dims == V2.dims) && V1.dual == V2.dual

# indices
Base.indices(V::RepresentationSpace) = Base.OneTo(dim(V))
function Base.indices(V::RepresentationSpace{G}, c::G) where {G}
    offset = 0
    for c′ in sectors(V)
        c′ == c && break
        offset += dim(V, c′)
    end
    return offset + (1:dim(V, c))
end

const ParitySpace = ZNSpace{2}
const ℤ₂Space = ZNSpace{2}
const ℤ₃Space = ZNSpace{3}
const ℤ₄Space = ZNSpace{4}
const U₁Space = RepresentationSpace{U₁}
const SU₂Space = RepresentationSpace{SU₂}

# Show methods
function Base.show(io::IO, V::RepresentationSpace{G}) where {G<:Sector}
    print(io, "RepresentationSpace{", G, "}(")
    seperator = ""
    comma = ", "
    for c in sectors(V)
        print(io, seperator, sprint(showcompact, c), "=>", dim(V, c))
        seperator = comma
    end
    print(io, ")")
    V.dual && print(io, "'")
end

# direct sum of RepresentationSpaces
function ⊕(V1::RepresentationSpace{G}, V2::RepresentationSpace{G}) where {G<:Sector}
    V1.dual == V2.dual || throw(SpaceMismatch("Direct sum of a vector space and its dual do not exist"))
    dims = Dict{G,Int}()
    for c in union(sectors(V1), sectors(V2))
        dims[c] = dim(V1,c) + dim(V2,c)
    end
    return RepresentationSpace(dims; dual = V1.dual)
end
