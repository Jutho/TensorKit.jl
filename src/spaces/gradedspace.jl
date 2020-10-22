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
sectortype(::Type{<:GradedSpace{I}}) where {I<:Sector} = I

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

function GradedSpace{I, SectorDict{I, Int}}(dims; dual::Bool = false) where {I<:Sector}
    d = SectorDict{I, Int}()
    for (c, dc) in dims
        k = convert(I, c)
        haskey(d, k) && throw(ArgumentError("Sector $k appears multiple times"))
        !iszero(dc) && push!(d, k=>dc)
    end
    return GradedSpace{I, SectorDict{I, Int}}(d, dual)
end
GradedSpace{I, SectorDict{I, Int}}(dims::Pair; dual::Bool = false) where {I<:Sector} =
    GradedSpace{I, SectorDict{I, Int}}((dims,); dual = dual)

GradedSpace{I,D}(; kwargs...) where {I<:Sector,D} = GradedSpace{I,D}((); kwargs...)
GradedSpace{I,D}(d1::Pair, d2::Pair, dims::Vararg{Pair}; kwargs...) where {I<:Sector,D} =
    GradedSpace{I,D}((d1, d2, dims...); kwargs...)

# TODO: do we also want to support this interface
function GradedSpace{I}(args...; kwargs...) where {I<:Sector}
    @warn "Preferred interface is `Vect[$I](args...; kwargs...)`." maxlog=1
    Vect[I](args..., kwargs...)
end

GradedSpace(dims::Tuple{Vararg{Pair{I, <:Integer}}}; dual::Bool = false) where {I<:Sector} =
    Vect[I](dims; dual = dual)
GradedSpace(dims::Vararg{Pair{I, <:Integer}}; dual::Bool = false) where {I<:Sector} =
    Vect[I](dims; dual = dual)
GradedSpace(dims::AbstractDict{I, <:Integer}; dual::Bool = false) where {I<:Sector} =
    Vect[I](dims; dual = dual)
# not inferrable
GradedSpace(g::Base.Generator; dual::Bool = false) =
    GradedSpace(g...; dual = dual)
GradedSpace(g::AbstractDict; dual::Bool = false) =
    GradedSpace(g...; dual = dual)

Base.hash(V::GradedSpace, h::UInt) = hash(V.dual, hash(V.dims, h))

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
function Base.axes(V::GradedSpace{I}, c::I) where {I<:Sector}
    offset = 0
    for c′ in sectors(V)
        c′ == c && break
        offset += dim(c′)*dim(V, c′)
    end
    return (offset+1):(offset+dim(c)*dim(V, c))
end

Base.oneunit(S::Type{<:GradedSpace{I}}) where {I<:Sector} = S(one(I)=>1)

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
    return typeof(V1)(dims; dual = dual1)
end

function flip(V::GradedSpace{I}) where {I<:Sector}
    if isdual(V)
        typeof(V)(c=>dim(V, c) for c in sectors(V))
    else
        typeof(V)(dual(c)=>dim(V, c) for c in sectors(V))'
    end
end

function fuse(V1::GradedSpace{I}, V2::GradedSpace{I}) where {I<:Sector}
    dims = SectorDict{I, Int}()
    for a in sectors(V1), b in sectors(V2)
        for c in a ⊗ b
            dims[c] = get(dims, c, 0) + Nsymbol(a, b, c)*dim(V1, a)*dim(V2, b)
        end
    end
    return typeof(V1)(dims)
end

function infimum(V1::GradedSpace{I}, V2::GradedSpace{I}) where {I<:Sector}
    if V1.dual == V2.dual
        typeof(V1)(c=>min(dim(V1, c), dim(V2, c)) for c in
            union(sectors(V1), sectors(V2)), dual = V1.dual)
    else
        throw(SpaceMismatch("Infimum of space and dual space does not exist"))
    end
end

function supremum(V1::GradedSpace{I}, V2::GradedSpace{I}) where {I<:Sector}
    if V1.dual == V2.dual
        typeof(V1)(c=>max(dim(V1, c), dim(V2, c)) for c in
            union(sectors(V1), sectors(V2)), dual = V1.dual)
    else
        throw(SpaceMismatch("Supremum of space and dual space does not exist"))
    end
end

function Base.show(io::IO, V::GradedSpace{I}) where {I<:Sector}
    print(io, type_repr(typeof(V)), "(")
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

function Base.getindex(::Type{GradedSpace}, ::Type{I}) where {I<:Sector}
    @warn "`getindex(::Type{GradedSpace}, I::Type{<:Sector})` is deprecated, use `ℂ[I]`, `Vect[I]`, or, if `I == Irrep[G]`, `Rep[G]` instead." maxlog = 1
    return Vect[I]
end

struct SpaceTable end
const Vect = SpaceTable()
Base.getindex(::SpaceTable) = ComplexSpace
Base.getindex(::SpaceTable, ::Type{Trivial}) = ComplexSpace
function Base.getindex(::SpaceTable, I::Type{<:Sector})
    if Base.IteratorSize(values(I)) isa Union{HasLength, HasShape}
        N = length(values(I))
        return GradedSpace{I, NTuple{N, Int}}
    else
        return GradedSpace{I, SectorDict{I, Int}}
    end
end

Base.getindex(::ComplexNumbers, I::Type{<:Sector}) = Vect[I]
Base.getindex(::ComplexNumbers, d1::Pair{I, Int}, dims::Pair{I, Int}...) where {I<:Sector} =
    Vect[I](d1, dims...)

struct RepTable end
const Rep = RepTable()
Base.getindex(::RepTable, G::Type{<:Group}) = Vect[Irrep[G]]

type_repr(::Type{<:GradedSpace{I}}) where {I<:Sector} =
    "Vect[" * type_repr(I) * "]"
type_repr(::Type{<:GradedSpace{<:AbstractIrrep{G}}}) where {G<:Group} =
    "Rep[" * type_repr(G) * "]"
function type_repr(::Type{<:GradedSpace{ProductSector{T}}}) where
                                                        {T<:Tuple{Vararg{AbstractIrrep}}}
    sectors = T.parameters
    s = "Rep["
    for i in 1:length(sectors)
        if i != 1
            s *= " × "
        end
        s *= type_repr(supertype(sectors[i]).parameters[1])
    end
    s *= "]"
    return s
end

# Specific constructors for Z_N
const ZNSpace{N} = GradedSpace{ZNIrrep{N}, NTuple{N,Int}}
ZNSpace{N}(dims::NTuple{N, Int}; dual::Bool = false) where {N} = ZNSpace{N}(dims, dual)
ZNSpace{N}(dims::Vararg{Int, N}; dual::Bool = false) where {N} = ZNSpace{N}(dims, dual)
ZNSpace(dims::NTuple{N, Int}; dual::Bool = false) where {N} = ZNSpace{N}(dims, dual)
ZNSpace(dims::Vararg{Int, N}; dual::Bool = false) where {N} = ZNSpace{N}(dims, dual)

# TODO: Do we still need all of those
# ASCII type aliases
const ZNSpace{N} = GradedSpace{ZNIrrep{N}, NTuple{N,Int}}
const Z2Space = ZNSpace{2}
const Z3Space = ZNSpace{3}
const Z4Space = ZNSpace{4}
const U1Space = Rep[U₁]
const CU1Space = Rep[CU₁]
const SU2Space = Rep[SU₂]

# Unicode alternatives
const ℤ₂Space = Z2Space
const ℤ₃Space = Z3Space
const ℤ₄Space = Z4Space
const U₁Space = U1Space
const CU₁Space = CU1Space
const SU₂Space = SU2Space
