"""
    struct RepresentationSpace{G<:Sector} <: AbstractRepresentationSpace{G}

Generic implementation of a representation space, i.e. a complex Euclidean space
with a direct sum structure corresponding to different superselection sectors of
type `G<:Sector`, e.g. the irreps of a compact or finite group, or the labels of
a unitary fusion category.
"""
struct RepresentationSpace{G<:Sector} <: AbstractRepresentationSpace{G}
    dims::Dict{G,Int}
    dual::Bool
end
RepresentationSpace{G}(dims::Dict{G,Int}; dual::Bool = false) where {G<:Sector} = RepresentationSpace{G}(dims, dual)
RepresentationSpace{G}(dims::Vararg{Pair{<:Any,Int}}; dual::Bool = false) where {G<:Sector} = RepresentationSpace{G}(Dict{G,Int}(convert(G,s)=>d for (s,d) in dims if d!=0), dual)

RepresentationSpace(dims::Dict{G,Int}; dual::Bool = false) where {G<:Sector} = RepresentationSpace{G}(dims, dual)
RepresentationSpace(dims::Vararg{Pair{G,Int}}; dual::Bool = false) where {G<:Sector} = RepresentationSpace{G}(Dict{G,Int}(dims...), dual)


# Corresponding methods:
"""
    sectors(V::ElementarySpace) -> sectortype(V)
    sectors(V::ProductSpace{S,N}) -> NTuple{N,sectortype{V}}

Iterate over the different sectors in the vector space.
"""
sectors(V::RepresentationSpace) = keys(V.dims)
checksectors(V::RepresentationSpace{G}, s::G) where {G<:Sector} = s in keys(V.dims) || throw(SectorMismatch())

dim(V::RepresentationSpace) = sum(dim(c)*V.dims[c] for c in keys(V.dims))
dim(V::RepresentationSpace{G}, c::G) where {G<:Sector} = get(V.dims, c, 0)

Base.conj(V::RepresentationSpace) = RepresentationSpace(Dict(dual(c)=>dim(V,c) for c in sectors(V)), !V.dual)

Base.getindex(::ComplexNumbers, G::Type{<:Sector}) = RepresentationSpace{G}
Base.getindex(::ComplexNumbers, d1::Pair{G,Int}, dims::Vararg{Pair{G,Int}}) where {G<:Sector} = RepresentationSpace{G}(d1, dims...)

Base.:(==)(V1::RepresentationSpace, V2::RepresentationSpace) = (V1.dims == V2.dims) && V1.dual == V2.dual

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
    return RepresentationSpace(dims, V1.dual)
end

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

Spin(j::Union{Int,Rational{Int}}) = RepresentationSpace{SU₂}(j=>1)
fSpin(j::Union{Int,Rational{Int}}) = RepresentationSpace{fSU₂}(j=>1)

# ZNSpace -- more efficient implemenation for small N
#-----------------------------------------------------
"""
    struct ZNSpace{N} <: AbstractRepresentationSpace{ZNIrrep{N}}

Optimized mplementation of a graded `ℤ_N` space, i.e. a complex Euclidean space graded
by the irreps of type `ZNIrrep{N}`
"""
struct ZNSpace{N} <: AbstractRepresentationSpace{ZNIrrep{N}}
    dims::NTuple{N,Int}
    dual::Bool
end

# various constructors for convenience
ZNSpace{N}(dims::NTuple{N,Int}) where {N} = ZNSpace{N}(dims, false)
ZNSpace{N}(dims::Vararg{Int,N}) where {N} = ZNSpace{N}(dims, false)
ZNSpace(dims::NTuple{N,Int}) where {N} = ZNSpace{N}(dims)
ZNSpace(dims::Vararg{Int,N}) where {N} = ZNSpace{N}(dims)

ZNSpace{N}(dims::Vararg{Pair{<:Any,Int}}) where {N} = ZNSpace{N}((ZNIrrep{N}(c)=>d for (c,d) in dims)...)
function ZNSpace{N}(args::Vararg{Pair{ZNIrrep{N},Int}}) where {N}
    dims = ntuple(n->0, Val(N))
    @inbounds for (c,d) in args
        dims = setindex(dims, d, c.n+1)
    end
    return ZNSpace{N}(dims)
end
ZNSpace{N}(dims::Dict{<:Any,Int}) where {N} = ZNSpace{N}(dims...)

Base.getindex(::ComplexNumbers, ::Type{ZNIrrep{N}}) where {N} = ZNSpace{N}
Base.getindex(::ComplexNumbers, d1::Pair{ZNIrrep{N},Int}, dims::Vararg{Pair{ZNIrrep{N},Int}}) where {N} = ZNSpace{N}(d1, dims...)

# properties
sectors(V::ZNSpace{N}) where {N} = SectorSet{ZNIrrep{N}}(n-1 for n=1:N if V.dims[n] != 0)
checksectors(V::ZNSpace{N}, c::ZNIrrep{N}) where {N} = V.dims[c.n+1] != 0 || throw(SectorMismatch())

dim(V::ZNSpace) = sum(V.dims)
dim(V::ZNSpace{N}, c::ZNIrrep{N}) where {N} = V.dims[c.n+1]

Base.conj(V::ZNSpace{N}) where {N} = ZNSpace{N}((V.dims[1], reverse(tail(V.dims))...), !V.dual)

# indices
Base.indices(V::ZNSpace) = Base.OneTo(dim(V))
function Base.indices(V::ZNSpace{N}, c::ZNIrrep{N}) where {N}
    dims = V.dims
    n = c.n
    offset = 0
    @inbounds for k = 1:n
        offset += dims[k]
    end
    return offset + (1:dims[n+1])
end

const ParitySpace = ZNSpace{2}
const ℤ₂Space = ZNSpace{2}
const ℤ₃Space = ZNSpace{3}
const ℤ₄Space = ZNSpace{4}

# Show methods
Base.show(io::IO, ::Type{ZNSpace{2}}) = print(io, "ℤ₂Space")
Base.show(io::IO, ::Type{ZNSpace{3}}) = print(io, "ℤ₃Space")
Base.show(io::IO, ::Type{ZNSpace{4}}) = print(io, "ℤ₄Space")

Base.show(io::IO, V::ZNSpace{2}) = print(io, "ℤ₂Space", V.dims, V.dual ? "'" : "")
Base.show(io::IO, V::ZNSpace{3}) = print(io, "ℤ₃Space", V.dims, V.dual ? "'" : "")
Base.show(io::IO, V::ZNSpace{4}) = print(io, "ℤ₄Space", V.dims, V.dual ? "'" : "")
Base.show(io::IO, V::ZNSpace{N}) where {N} = print(io, "ZNSpace{", N, "}", V.dims, V.dual ? "'" : "")
