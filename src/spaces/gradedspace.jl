"""
    struct GradedSpace{I<:Sector, D} <: ElementarySpace
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
and should typically be of no concern.

The concrete type `GradedSpace{I,D}` with correct `D` can be obtained as `Vect[I]`, or if
`I == Irrep[G]` for some `G<:Group`, as `Rep[G]`.
"""
struct GradedSpace{I<:Sector,D} <: ElementarySpace
    dims::D
    dual::Bool
end
sectortype(::Type{<:GradedSpace{I}}) where {I<:Sector} = I

function GradedSpace{I,NTuple{N,Int}}(dims; dual::Bool=false) where {I,N}
    d = ntuple(n -> 0, N)
    isset = ntuple(n -> false, N)
    for (c, dc) in dims
        i = findindex(values(I), convert(I, c))
        isset[i] && throw(ArgumentError("Sector $c appears multiple times"))
        isset = TupleTools.setindex(isset, true, i)
        d = TupleTools.setindex(d, dc, i)
    end
    return GradedSpace{I,NTuple{N,Int}}(d, dual)
end
function GradedSpace{I,NTuple{N,Int}}(dims::Pair; dual::Bool=false) where {I,N}
    return GradedSpace{I,NTuple{N,Int}}((dims,); dual=dual)
end

function GradedSpace{I,SectorDict{I,Int}}(dims; dual::Bool=false) where {I<:Sector}
    d = SectorDict{I,Int}()
    for (c, dc) in dims
        k = convert(I, c)
        haskey(d, k) && throw(ArgumentError("Sector $k appears multiple times"))
        !iszero(dc) && push!(d, k => dc)
    end
    return GradedSpace{I,SectorDict{I,Int}}(d, dual)
end
function GradedSpace{I,SectorDict{I,Int}}(dims::Pair; dual::Bool=false) where {I<:Sector}
    return GradedSpace{I,SectorDict{I,Int}}((dims,); dual=dual)
end

GradedSpace{I,D}(; kwargs...) where {I<:Sector,D} = GradedSpace{I,D}((); kwargs...)
function GradedSpace{I,D}(d1::Pair, d2::Pair, dims::Vararg{Pair};
                          kwargs...) where {I<:Sector,D}
    return GradedSpace{I,D}((d1, d2, dims...); kwargs...)
end

GradedSpace{I}(args...; kwargs...) where {I<:Sector} = Vect[I](args..., kwargs...)

function GradedSpace(dims::Tuple{Vararg{Pair{I,<:Integer}}};
                     dual::Bool=false) where {I<:Sector}
    return Vect[I](dims; dual=dual)
end
function GradedSpace(dims::Vararg{Pair{I,<:Integer}}; dual::Bool=false) where {I<:Sector}
    return Vect[I](dims; dual=dual)
end
function GradedSpace(dims::AbstractDict{I,<:Integer}; dual::Bool=false) where {I<:Sector}
    return Vect[I](dims; dual=dual)
end
# not inferrable
GradedSpace(g::Base.Generator; dual::Bool=false) = GradedSpace(g...; dual=dual)
GradedSpace(g::AbstractDict; dual::Bool=false) = GradedSpace(g...; dual=dual)

Base.hash(V::GradedSpace, h::UInt) = hash(V.dual, hash(V.dims, h))

# Corresponding methods:
# properties
field(::Type{<:GradedSpace}) = ℂ
InnerProductStyle(::Type{<:GradedSpace}) = EuclideanProduct()
function dim(V::GradedSpace)
    return reduce(+, dim(V, c) * dim(c) for c in sectors(V);
                  init=zero(dim(one(sectortype(V)))))
end
function dim(V::GradedSpace{I,<:AbstractDict}, c::I) where {I<:Sector}
    return get(V.dims, isdual(V) ? dual(c) : c, 0)
end
function dim(V::GradedSpace{I,<:Tuple}, c::I) where {I<:Sector}
    return V.dims[findindex(values(I), isdual(V) ? dual(c) : c)]
end

function sectors(V::GradedSpace{I,<:AbstractDict}) where {I<:Sector}
    return SectorSet{I}(s -> isdual(V) ? dual(s) : s, keys(V.dims))
end
function sectors(V::GradedSpace{I,NTuple{N,Int}}) where {I<:Sector,N}
    SectorSet{I}(Iterators.filter(n -> V.dims[n] != 0, 1:N)) do n
        return isdual(V) ? dual(values(I)[n]) : values(I)[n]
    end
end

hassector(V::GradedSpace{I}, s::I) where {I<:Sector} = dim(V, s) != 0

Base.conj(V::GradedSpace) = typeof(V)(V.dims, !V.dual)
isdual(V::GradedSpace) = V.dual

# equality / comparison
function Base.:(==)(V₁::GradedSpace, V₂::GradedSpace)
    return sectortype(V₁) == sectortype(V₂) && (V₁.dims == V₂.dims) && V₁.dual == V₂.dual
end

# axes
Base.axes(V::GradedSpace) = Base.OneTo(dim(V))
function Base.axes(V::GradedSpace{I}, c::I) where {I<:Sector}
    offset = 0
    for c′ in sectors(V)
        c′ == c && break
        offset += dim(c′) * dim(V, c′)
    end
    return (offset + 1):(offset + dim(c) * dim(V, c))
end

Base.oneunit(S::Type{<:GradedSpace{I}}) where {I<:Sector} = S(one(I) => 1)

# TODO: the following methods can probably be implemented more efficiently for
# `FiniteGradedSpace`, but we don't expect them to be used often in hot loops, so
# these generic definitions (which are still quite efficient) are good for now.
function ⊕(V₁::GradedSpace{I}, V₂::GradedSpace{I}) where {I<:Sector}
    dual1 = isdual(V₁)
    dual1 == isdual(V₂) ||
        throw(SpaceMismatch("Direct sum of a vector space and a dual space does not exist"))
    dims = SectorDict{I,Int}()
    for c in union(sectors(V₁), sectors(V₂))
        cout = ifelse(dual1, dual(c), c)
        dims[cout] = dim(V₁, c) + dim(V₂, c)
    end
    return typeof(V₁)(dims; dual=dual1)
end

function flip(V::GradedSpace{I}) where {I<:Sector}
    if isdual(V)
        typeof(V)(c => dim(V, c) for c in sectors(V))
    else
        typeof(V)(dual(c) => dim(V, c) for c in sectors(V))'
    end
end

function fuse(V₁::GradedSpace{I}, V₂::GradedSpace{I}) where {I<:Sector}
    dims = SectorDict{I,Int}()
    for a in sectors(V₁), b in sectors(V₂)
        for c in a ⊗ b
            dims[c] = get(dims, c, 0) + Nsymbol(a, b, c) * dim(V₁, a) * dim(V₂, b)
        end
    end
    return typeof(V₁)(dims)
end

function infimum(V₁::GradedSpace{I}, V₂::GradedSpace{I}) where {I<:Sector}
    if V₁.dual == V₂.dual
        typeof(V₁)(c => min(dim(V₁, c), dim(V₂, c))
                   for c in
                       union(sectors(V₁), sectors(V₂)), dual in V₁.dual)
    else
        throw(SpaceMismatch("Infimum of space and dual space does not exist"))
    end
end

function supremum(V₁::GradedSpace{I}, V₂::GradedSpace{I}) where {I<:Sector}
    if V₁.dual == V₂.dual
        typeof(V₁)(c => max(dim(V₁, c), dim(V₂, c))
                   for c in
                       union(sectors(V₁), sectors(V₂)), dual in V₁.dual)
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

struct SpaceTable end
"""
    const Vect

A constant of a singleton type used as `Vect[I]` with `I<:Sector` a type of sector, to
construct or obtain the concrete type `GradedSpace{I,D}` instances without having to
specify `D`.
"""
const Vect = SpaceTable()
Base.getindex(::SpaceTable) = ComplexSpace
Base.getindex(::SpaceTable, ::Type{Trivial}) = ComplexSpace
function Base.getindex(::SpaceTable, I::Type{<:Sector})
    if Base.IteratorSize(values(I)) isa Union{HasLength,HasShape}
        N = length(values(I))
        return GradedSpace{I,NTuple{N,Int}}
    else
        return GradedSpace{I,SectorDict{I,Int}}
    end
end

Base.getindex(::ComplexNumbers, I::Type{<:Sector}) = Vect[I]
struct RepTable end
"""
    const Rep

A constant of a singleton type used as `Rep[G]` with `G<:Group` a type of group, to
construct or obtain the concrete type `GradedSpace{Irrep[G],D}` instances without having to
specify `D`. Note that `Rep[G] == Vect[Irrep[G]]`.

See also [`Irrep`](@ref) and [`Vect`](@ref).
"""
const Rep = RepTable()
Base.getindex(::RepTable, G::Type{<:Group}) = Vect[Irrep[G]]

type_repr(::Type{<:GradedSpace{I}}) where {I<:Sector} = "Vect[" * type_repr(I) * "]"
function type_repr(::Type{<:GradedSpace{<:AbstractIrrep{G}}}) where {G<:Group}
    return "Rep[" * type_repr(G) * "]"
end
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
const ZNSpace{N} = GradedSpace{ZNIrrep{N},NTuple{N,Int}}
ZNSpace{N}(dims::NTuple{N,Int}; dual::Bool=false) where {N} = ZNSpace{N}(dims, dual)
ZNSpace{N}(dims::Vararg{Int,N}; dual::Bool=false) where {N} = ZNSpace{N}(dims, dual)
ZNSpace(dims::NTuple{N,Int}; dual::Bool=false) where {N} = ZNSpace{N}(dims, dual)
ZNSpace(dims::Vararg{Int,N}; dual::Bool=false) where {N} = ZNSpace{N}(dims, dual)

# TODO: Do we still need all of those
# ASCII type aliases
const ZNSpace{N} = GradedSpace{ZNIrrep{N},NTuple{N,Int}}
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
