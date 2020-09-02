"""
    struct GradedSpace{I<:Sector, D} <: EuclideanSpace{ℂ}
        dims::D
        dual::Bool
    end

A complex Euclidean space with a direct sum structure corresponding to labels in a set `I`,
the objects of which have the structure of a monoid with respect to a monoidal product `⊗`.
In practice, we restrict the label set to be a set of superselection sectors of type
`I<:Sector`, e.g. the set of distinct irreps of a finite or compact group, or the
isomorphism classes of simple objects of a unitary and pivotal (pre-)fusion category.

Here `dims` represents the degeneracy or multiplicity of every sector.

The data structure `D` of `dims` will depend on the result `Base.IteratorElsize(values(I))`;
if the result is of type `HasLength` or `HasShape`, `dims` will be stored in a
`NTuple{N,Int}` with `N = length(values(I))`. This requires that a sector `s::I` can be
transformed into an index via `s == getindex(values(I), i)` and
`i == findindex(values(I), s)`. If `Base.IteratorElsize(values(I))` results `IsInfinite()`
or `SizeUnknown()`, a `SectorDict{I,Int}` is used to store the non-zero degeneracy
dimensions with the corresponding sector as key. The parameter `D` is hidden from the user
and should typically be of no concern
"""
struct GradedSpace{I<:Sector, D} <: EuclideanSpace{ℂ}
    dims::D
    dual::Bool
end
Base.@pure sectortype(::Type{<:GradedSpace{I}}) where {I<:Sector} = I

Base.getindex(::Type{GradedSpace}, ::Type{Trivial}) = ComplexSpace
function Base.getindex(::Type{GradedSpace}, ::Type{I}) where {I<:Sector}
    if Base.IteratorSize(values(I)) isa Union{HasLength, HasShape}
        N = length(values(I))
        return GradedSpace{I, NTuple{N, Int}}
    else
        return GradedSpace{I, SectorDict{I, Int}}
    end
end

const Rep{G} = GradedSpace{Irrep{G}} where {G<:Group}
Base.getindex(::Type{Rep}, ::Type{G}) where {G<:Group} = GradedSpace[Irrep[G]]

function GradedSpace{I, NTuple{N, Int}}(dims; dual::Bool = false) where {I, N}
    d = ntuple(n->0, N)
    isset = ntuple(n->false, N)
    for (c, dc) in dims
        i = findindex(values(I), convert(I, c))
        isset[i] && throw(ArgumentError("Sector $c appears multiple times"))
        isset = TupleTools.setindex(isset, true, i)
        d = TupleTools.setindex(d, dc, i)
    end
    return GradedSpace{I, NTuple{N, Int}}(d, dual)
end
GradedSpace{I, NTuple{N, Int}}(dims::Pair; dual::Bool = false) where {I, N} =
    GradedSpace{I, NTuple{N, Int}}((dims,); dual = dual)

function GradedSpace{I, SectorDict{I, Int}}(dims; dual::Bool = false) where {I}
    d = SectorDict{I, Int}()
    for (c, dc) in dims
        k = convert(I, c)
        haskey(d, k) && throw(ArgumentError("Sector $k appears multiple times"))
        !iszero(dc) && push!(d, k=>dc)
    end
    return GradedSpace{I, SectorDict{I, Int}}(d, dual)
end
GradedSpace{I, SectorDict{I, Int}}(dims::Pair; dual::Bool = false) where {I} =
    GradedSpace{I, SectorDict{I, Int}}((dims,); dual = dual)

GradedSpace{I,D}(; kwargs...) where {I<:Sector,D} = GradedSpace{I,D}((); kwargs...)
GradedSpace{I,D}(d1::Pair, d2::Pair, dims::Vararg{Pair}; kwargs...) where {I<:Sector,D} =
    GradedSpace{I,D}((d1, d2, dims...); kwargs...)

# TODO: do we also want to support this interface
function GradedSpace{I}(args...; kwargs...) where {I<:Sector}
    @warn "Preferred interface is `GradedSpace[$I](args...; kwargs...)`." maxlog=1
    GradedSpace[I](args..., kwargs...)
end

GradedSpace(dims::Tuple{Vararg{Pair{I, <:Integer}}}; dual::Bool = false) where {I<:Sector} =
    GradedSpace[I](dims; dual = dual)
GradedSpace(dims::Vararg{Pair{I, <:Integer}}; dual::Bool = false) where {I<:Sector} =
    GradedSpace[I](dims; dual = dual)
GradedSpace(dims::AbstractDict{I, <:Integer}; dual::Bool = false) where {I<:Sector} =
    GradedSpace[I](dims; dual = dual)
# not inferrable
GradedSpace(g::Base.Generator; dual::Bool = false) =
    GradedSpace(g...; dual = dual)
GradedSpace(g::AbstractDict; dual::Bool = false) =
    GradedSpace(g...; dual = dual)

Base.hash(V::GradedSpace, h::UInt) = hash(V.dual, hash(V.dims, h))

Base.getindex(::ComplexNumbers, ::Type{Trivial}) = ComplexSpace
Base.getindex(::ComplexNumbers, I::Type{<:Sector}) = GradedSpace[I]
Base.getindex(::ComplexNumbers, d1::Pair{I, Int}, dims::Pair{I, Int}...) where {I<:Sector} =
    ℂ[I](d1, dims...)

# Corresponding methods:
# properties
dim(V::GradedSpace) =
    reduce(+, dim(V, c) * dim(c) for c in sectors(V); init = zero(dim(one(sectortype(V)))))

dim(V::GradedSpace{I,<:AbstractDict}, c::I) where {I<:Sector} =
    get(V.dims, isdual(V) ? dual(c) : c, 0)
dim(V::GradedSpace{I,<:Tuple}, c::I) where {I<:Sector} =
    V.dims[findindex(values(I), isdual(V) ? dual(c) : c)]

sectors(V::GradedSpace{I,<:AbstractDict}) where {I<:Sector} =
    SectorSet{I}(s->isdual(V) ? dual(s) : s, keys(V.dims))
sectors(V::GradedSpace{I,NTuple{N,Int}}) where {I<:Sector, N} =
    SectorSet{I}(Iterators.filter(n->V.dims[n]!=0, 1:N)) do n
        isdual(V) ? dual(values(I)[n]) : values(I)[n]
    end

hassector(V::GradedSpace{I}, s::I) where {I<:Sector} = dim(V, s) != 0

Base.conj(V::GradedSpace) = typeof(V)(V.dims, !V.dual)
isdual(V::GradedSpace) = V.dual

# equality / comparison
Base.:(==)(V1::GradedSpace, V2::GradedSpace) =
    sectortype(V1) == sectortype(V2) && (V1.dims == V2.dims) && V1.dual == V2.dual

# axes
Base.axes(V::GradedSpace) = Base.OneTo(dim(V))
function Base.axes(V::GradedSpace{I}, c::I) where {I}
    offset = 0
    for c′ in sectors(V)
        c′ == c && break
        offset += dim(c′)*dim(V, c′)
    end
    return (offset+1):(offset+dim(c)*dim(V, c))
end

Base.oneunit(::Type{<:GradedSpace{I}}) where {I<:Sector} =
    GradedSpace[I](one(I)=>1)

# TODO: the following methods can probably be implemented more efficiently for
# `FiniteGradedSpace`, but we don't expect them to be used often in hot loops, so
# these generic definitions (which are still quite efficient) are good for now.
function ⊕(V1::GradedSpace{I}, V2::GradedSpace{I}) where {I<:Sector}
    dual1 = isdual(V1)
    dual1 == isdual(V2) ||
        throw(SpaceMismatch("Direct sum of a vector space and a dual space does not exist"))
    dims = SectorDict{I, Int}()
    for c in union(sectors(V1), sectors(V2))
        cout = ifelse(dual1, dual(c), c)
        dims[cout] = dim(V1, c) + dim(V2, c)
    end
    return GradedSpace[I](dims; dual = dual1)
end

function flip(V::GradedSpace{I}) where {I}
    if isdual(V)
        GradedSpace[I](c=>dim(V, c) for c in sectors(V))
    else
        GradedSpace[I](dual(c)=>dim(V, c) for c in sectors(V))'
    end
end

function fuse(V1::GradedSpace{I}, V2::GradedSpace{I}) where {I<:Sector}
    dims = SectorDict{I, Int}()
    for a in sectors(V1), b in sectors(V2)
        for c in a ⊗ b
            dims[c] = get(dims, c, 0) + Nsymbol(a, b, c)*dim(V1, a)*dim(V2, b)
        end
    end
    return GradedSpace[I](dims)
end

function infimum(V1::GradedSpace{I}, V2::GradedSpace{I}) where {I}
    if V1.dual == V2.dual
        GradedSpace[I](c=>min(dim(V1, c), dim(V2, c)) for c in
            union(sectors(V1), sectors(V2)), dual = V1.dual)
    else
        throw(SpaceMismatch("Infimum of space and dual space does not exist"))
    end
end

function supremum(V1::GradedSpace{I}, V2::GradedSpace{I}) where {I<:Sector}
    if V1.dual == V2.dual
        GradedSpace[I](c=>max(dim(V1, c), dim(V2, c)) for c in
            union(sectors(V1), sectors(V2)), dual = V1.dual)
    else
        throw(SpaceMismatch("Supremum of space and dual space does not exist"))
    end
end

function Base.show(io::IO, V::GradedSpace{I}) where {I<:Sector}
    show(io, typeof(V))
    print(io, "(")
    seperator = ""
    comma = ", "
    io2 = IOContext(io, :typeinfo => I)
    for c in sectors(V)
        if isdual(V)
            print(io2, seperator, dual(c), "=>", dim(V, c))
        else
            print(io2, seperator, c, "=>", dim(V, c))
        end
        seperator = comma
    end
    print(io, ")")
    V.dual && print(io, "'")
    return nothing
end

# Specific constructors for Z_N
const ZNSpace{N} = GradedSpace{ZNIrrep{N}, NTuple{N,Int}}
ZNSpace{N}(dims::NTuple{N, Int}; dual::Bool = false) where {N} = ZNSpace{N}(dims, dual)
ZNSpace{N}(dims::Vararg{Int, N}; dual::Bool = false) where {N} = ZNSpace{N}(dims, dual)
ZNSpace(dims::NTuple{N, Int}; dual::Bool = false) where {N} = ZNSpace{N}(dims, dual)
ZNSpace(dims::Vararg{Int, N}; dual::Bool = false) where {N} = ZNSpace{N}(dims, dual)

# More type aliases
const ℤ₂Space = ZNSpace{2}
const ℤ₃Space = ZNSpace{3}
const ℤ₄Space = ZNSpace{4}
const U₁Space = Rep[U₁]
const CU₁Space = Rep[CU₁]
const SU₂Space = Rep[SU₂]

Base.show(io::IO, ::Type{GradedSpace{I,D}}) where {I,D} = print(io, "GradedSpace[", I, "]")
Base.show(io::IO, ::Type{ℤ₂Space}) = print(io, "ℤ₂Space")
Base.show(io::IO, ::Type{ℤ₃Space}) = print(io, "ℤ₃Space")
Base.show(io::IO, ::Type{ℤ₄Space}) = print(io, "ℤ₄Space")
Base.show(io::IO, ::Type{U₁Space}) = print(io, "U₁Space")
Base.show(io::IO, ::Type{CU₁Space}) = print(io, "CU₁Space")
Base.show(io::IO, ::Type{SU₂Space}) = print(io, "SU₂Space")

# non-Unicode alternatives
const Z2Space = ℤ₂Space
const Z3Space = ℤ₃Space
const Z4Space = ℤ₄Space
const U1Space = U₁Space
const CU1Space = CU₁Space
const SU2Space = SU₂Space
