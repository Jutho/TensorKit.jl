"""
    struct GenericRepresentationSpace{G<:Sector} <: RepresentationSpace{G}

Generic implementation of a representation space, i.e. a complex Euclidean space with a
direct sum structure corresponding to different superselection sectors of type `G<:Sector`,
e.g. the irreps of a compact or finite group, or the labels of a unitary fusion category.
"""
struct GenericRepresentationSpace{G<:Sector} <: RepresentationSpace{G}
    dims::SectorDict{G,Int}
    dual::Bool
end
GenericRepresentationSpace{G}(dims::SectorDict{G,Int}) where {G<:Sector} =
    GenericRepresentationSpace{G}(dims, false)

function GenericRepresentationSpace{G}(dims::Tuple{Vararg{Pair{G,Int}}},
                                        dual::Bool) where {G<:Sector}
    d = SectorDict{G,Int}()
    @inbounds for k = 1:length(dims)
        if dims[k][2] != 0
            push!(d, dims[k])
        end
    end
    GenericRepresentationSpace{G}(d, dual)
end

GenericRepresentationSpace{G}(dims::Tuple{Vararg{Pair{<:Any,Int}}};
                                dual::Bool = false) where {G<:Sector} =
    GenericRepresentationSpace{G}(map(d->convert(Pair{G,Int}, d), dims), dual)
GenericRepresentationSpace{G}(dims::Pair{<:Any,Int}...;
                                dual::Bool = false) where {G<:Sector} =
    GenericRepresentationSpace{G}(map(d->convert(Pair{G,Int}, d), dims), dual)
GenericRepresentationSpace{G}(g::Base.Generator; dual::Bool = false) where {G<:Sector} =
    GenericRepresentationSpace{G}(g...; dual=dual)

Base.:(==)(V1::GenericRepresentationSpace, V2::GenericRepresentationSpace) =
    keys(V1.dims) == keys(V2.dims) &&
    values(V1.dims) == values(V2.dims) &&
    V1.dual == V2.dual

"""
    struct ZNSpace{N} <: AbstractRepresentationSpace{ZNIrrep{N}}

Optimized implementation of a graded `ℤ_N` space, i.e. a complex Euclidean space graded by
the irreps of type [`ZNIrrep{N}`](@ref).
"""
struct ZNSpace{N} <: RepresentationSpace{ZNIrrep{N}}
    dims::NTuple{N,Int}
    dual::Bool
end

# Never write GenericRepresentationSpace, just use RepresentationSpace
RepresentationSpace{G}(dims::Pair{<:Any,Int}...; dual::Bool = false) where {G<:Sector} =
    GenericRepresentationSpace{G}(dims; dual = dual)
RepresentationSpace(dims::AbstractDict{G,Int}; dual::Bool = false) where {G<:Sector} =
    GenericRepresentationSpace{G}(dims...; dual = dual)
RepresentationSpace(dims::Pair{G,Int}...; dual::Bool = false) where {G<:Sector} =
    GenericRepresentationSpace{G}(dims; dual = dual)

# You might want to write ZNSpace
ZNSpace{N}(dims::NTuple{N,Int}; dual::Bool = false) where {N} = ZNSpace{N}(dims, dual)
ZNSpace{N}(dims::Vararg{Int,N}; dual::Bool = false) where {N} = ZNSpace{N}(dims, dual)
ZNSpace(dims::NTuple{N,Int}; dual::Bool = false) where {N} = ZNSpace{N}(dims, dual)
ZNSpace(dims::Vararg{Int,N}; dual::Bool = false) where {N} = ZNSpace{N}(dims, dual)

ZNSpace{N}(dims::Pair{<:Any,Int}...; dual::Bool = false) where {N} =
    ZNSpace{N}((ZNIrrep{N}(c)=>d for (c,d) in dims)...; dual = dual)
ZNSpace{N}(dims::AbstractDict{<:Any,Int}; dual::Bool = false) where {N} =
    ZNSpace{N}(dims...; dual = dual)
function ZNSpace{N}(args::Pair{ZNIrrep{N},Int}...; dual::Bool = false) where {N}
    dims = ntuple(n->0, StaticLength(N))
    @inbounds for (c,d) in args
        dims = TupleTools.setindex(dims, d, c.n+1)
    end
    return ZNSpace{N}(dims, dual)
end

RepresentationSpace{ZNIrrep{N}}(dims::Pair{<:Any,Int}...; dual::Bool = false) where {N} =
    ZNSpace{N}((ZNIrrep{N}(c)=>d for (c,d) in dims)...; dual = dual)
RepresentationSpace(dims::AbstractDict{ZNIrrep{N},Int}; dual::Bool = false) where {N} =
    ZNSpace{N}(dims...; dual = dual)
RepresentationSpace(dims::Pair{ZNIrrep{N},Int}...; dual::Bool = false) where {N} =
    ZNSpace{N}(dims...; dual = dual)

RepresentationSpace{G}(g::Base.Generator; dual::Bool = false) where {G<:Sector} =
    RepresentationSpace{G}(g...; dual = dual)
RepresentationSpace(g::Base.Generator; dual::Bool = false) =
    RepresentationSpace(g...; dual = dual)

Base.getindex(::ComplexNumbers, G::Type{<:Sector}) = RepresentationSpace{G}
Base.getindex(::ComplexNumbers, d1::Pair{G,Int}, dims::Pair{G,Int}...) where {G<:Sector} =
    RepresentationSpace{G}(d1, dims...)

# Corresponding methods:
sectors(V::GenericRepresentationSpace{G}) where {G<:Sector} =
    SectorSet{G}(s->isdual(V) ? dual(s) : s, keys(V.dims))
sectors(V::ZNSpace{N}) where {N} = SectorSet{ZNIrrep{N}}(n->isdual(V) ? -(n-1) : (n-1),
    Iterators.filter(n->V.dims[n]!=0, 1:N))

hassector(V::RepresentationSpace{G}, s::G) where {G<:Sector} = dim(V, s) != 0

# properties
dim(V::GenericRepresentationSpace) = sum(c->dim(c)*V.dims[c], keys(V.dims))
dim(V::GenericRepresentationSpace{G}, c::G) where {G<:Sector} =
    get(V.dims, isdual(V) ? dual(c) : c, 0)

dim(V::ZNSpace) = sum(V.dims)
dim(V::ZNSpace{N}, c::ZNIrrep{N}) where {N} = V.dims[(isdual(V) ? dual(c).n : c.n)+1]

Base.conj(V::GenericRepresentationSpace) = GenericRepresentationSpace(V.dims, !V.dual)
Base.conj(V::ZNSpace{N}) where {N} = ZNSpace{N}(V.dims, !V.dual)
isdual(V::RepresentationSpace) = V.dual

# equality / comparison
Base.:(==)(V1::RepresentationSpace, V2::RepresentationSpace) =
    (V1.dims == V2.dims) && V1.dual == V2.dual

# axes
Base.axes(V::RepresentationSpace) = Base.OneTo(dim(V))
function Base.axes(V::RepresentationSpace{G}, c::G) where {G}
    offset = 0
    for c′ in sectors(V)
        c′ == c && break
        offset += dim(c′)*dim(V, c′)
    end
    return (offset+1):(offset+dim(c)*dim(V, c))
end

const ParitySpace = ZNSpace{2}
const ℤ₂Space = ZNSpace{2}
const ℤ₃Space = ZNSpace{3}
const ℤ₄Space = ZNSpace{4}
const U₁Space = GenericRepresentationSpace{U₁}
const CU₁Space = GenericRepresentationSpace{CU₁}
const SU₂Space = GenericRepresentationSpace{SU₂}

# non-Unicode alternatives
const Z2Space = ℤ₂Space
const Z3Space = ℤ₃Space
const Z4Space = ℤ₄Space
const U1Space = U₁Space
const CU1Space = CU₁Space
const SU2Space = SU₂Space


Base.oneunit(::Type{<:RepresentationSpace{G}}) where {G<:Sector} =
    RepresentationSpace{G}(one(G)=>1)

function ⊕(V1::RepresentationSpace{G}, V2::RepresentationSpace{G}) where {G<:Sector}
    dual1 = isdual(V1)
    dual1 == isdual(V2) ||
        throw(SpaceMismatch("Direct sum of a vector space and a dual space does not exist"))
    dims = SectorDict{G,Int}()
    for c in union(sectors(V1), sectors(V2))
        cout = ifelse(dual1, dual(c), c)
        dims[cout] = dim(V1,c) + dim(V2,c)
    end
    return RepresentationSpace(dims; dual = dual1)
end

function flip(V::RepresentationSpace)
    if isdual(V)
        typeof(V)((c=>dim(V,c) for c in sectors(V))...)
    else
        typeof(V)((dual(c)=>dim(V,c) for c in sectors(V))...)'
    end
end

function fuse(V1::RepresentationSpace{G}, V2::RepresentationSpace{G}) where {G<:Sector}
    dims = SectorDict{G,Int}()
    for a in sectors(V1), b in sectors(V2)
        for c in a ⊗ b
            dims[c] = get(dims, c, 0) + Nsymbol(a,b,c)*dim(V1,a)*dim(V2,b)
        end
    end
    return RepresentationSpace(dims)
end

function Base.min(V1::RepresentationSpace{G}, V2::RepresentationSpace{G}) where {G}
    if V1.dual == V2.dual
        RepresentationSpace{G}(c=>min(dim(V1,c), dim(V2,c)) for c in
            union(sectors(V1), sectors(V2)), dual = V1.dual)
    else
        throw(SpaceMismatch("V1 and V2 should both be normal or dual spaces"))
    end
end

function Base.max(V1::RepresentationSpace{G}, V2::RepresentationSpace{G}) where {G}
    if V1.dual == V2.dual
        RepresentationSpace{G}(c=>max(dim(V1,c), dim(V2,c)) for c in
            union(sectors(V1), sectors(V2)), dual = V1.dual)
    else
        throw(SpaceMismatch("V1 and V2 should both be normal or dual spaces"))
    end
end

Base.show(io::IO, ::Type{ℤ₂Space}) = print(io, "ℤ₂Space")
Base.show(io::IO, ::Type{ℤ₃Space}) = print(io, "ℤ₃Space")
Base.show(io::IO, ::Type{ℤ₄Space}) = print(io, "ℤ₄Space")
Base.show(io::IO, ::Type{U₁Space}) = print(io, "U₁Space")
Base.show(io::IO, ::Type{SU₂Space}) = print(io, "SU₂Space")

function Base.show(io::IO, V::RepresentationSpace{G}) where {G<:Sector}
    show(io, typeof(V))
    print(io, "(")
    seperator = ""
    comma = ", "
    for c in sectors(V)
        if isdual(V)
            print(io, seperator,
                    sprint(show, dual(c); context = :compact=>true), "=>", dim(V, c))
        else
            print(io, seperator, sprint(show, c; context = :compact=>true), "=>", dim(V, c))
        end
        seperator = comma
    end
    print(io, ")")
    V.dual && print(io, "'")
    return nothing
end
