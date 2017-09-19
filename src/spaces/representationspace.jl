"""
    struct RepresentationSpace{G<:Sector} <: AbstractRepresentationSpace{G}

Generic implementation of a representation space, i.e. a complex Euclidean space
with a direct sum structure corresponding to different superselection sections of
type `G<:Sector`, e.g. the irreps of a compact or finite group, or the labels of
a unitary fusion category.
"""
struct RepresentationSpace{G<:Sector} <: AbstractRepresentationSpace{G}
    dims::Dict{G,Int}
    dual::Bool
end
RepresentationSpace(dims::Vararg{Pair{G,Int}}; dual::Bool = false) where {G<:Sector} = RepresentationSpace{G}(Dict{G,Int}(dims...), dual)
RepresentationSpace{G}(dims::Vararg{Pair{<:Any,Int}}; dual::Bool = false) where {G<:Sector} = RepresentationSpace{G}(Dict{G,Int}(convert(G,s)=>d for (s,d) in dims), dual)

# Corresponding methods:
function sectors(V::RepresentationSpace)
    s = collect(keys(V.dims))
    V.dual ? s : map!(conj,s,s)
end

dim(V::RepresentationSpace) = sum(values(V.dims))
dim(V::RepresentationSpace{G}, c::G) where {G<:Sector} = get(V.dims, c, 0)

Base.conj(V::RepresentationSpace) = RepresentationSpace(copy(V.dims), !V.dual)

Base.getindex(::ComplexNumbers, G::Type{<:Sector}) = RepresentationSpace{G}
Base.getindex(::ComplexNumbers, d1::Pair{G,Int}, dims::Vararg{Pair{G,Int}}) where {G<:Sector} = RepresentationSpace{G}(d1, dims...)

# Show methods
function Base.show(io::IO, V::RepresentationSpace{G}) where {G<:Sector}
    print(io, "RepresentationSpace{", G, "}(")
    s = sectors(V)
    c = s[1]
    print(io, sprint(showcompact, c), "=>", dim(V,c))
    for k = 2:length(s)
        c = s[k]
        print(io, ", ", sprint(showcompact, c), "=>", dim(V,c))
    end
    print(io, ")")
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
    if V.dual
        c = conj(c)
    end
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
function ZNSpace{N}(dims::Vararg{Pair{ZNIrrep{N},Int}}) where {N}
    newdims = zeros(Int, N)
    @inbounds for (c,d) in dims
        newdims[c.n+1] = d
    end
    return ZNSpace{N}(ntuple(n->newdims[n], Val(N)))
end

Base.getindex(::ComplexNumbers, ::Type{ZNIrrep{N}}) where {N} = ZNSpace{N}
Base.getindex(::ComplexNumbers, d1::Pair{ZNIrrep{N},Int}, dims::Vararg{Pair{ZNIrrep{N},Int}}) where {N} = ZNSpace{N}(d1, dims...)

# properties
function sectors(V::ZNSpace{N}) where {N}
    s = ntuple(n->ZNIrrep{N}(n-1), Val(N))
    V.dual ? map(conj, s) : s
end

dim(V::ZNSpace) = sum(V.dims)
dim(V::ZNSpace{N}, c::ZNIrrep{N}) where {N} = V.dims[c.n+1]

Base.conj(V::ZNSpace{N}) where {N} = ZNSpace{N}(V.dims, !V.dual)

# indices
Base.indices(V::ZNSpace) = Base.OneTo(dim(V))
function Base.indices(V::ZNSpace{N}, c::ZNIrrep{N}) where {N}
    dims = V.dims
    n = V.dual ? conj(c).n : c.n
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
