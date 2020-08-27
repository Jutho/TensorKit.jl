"""
    struct GenericGradedSpace{I<:Sector} <: GradedSpace{I}

Generic implementation of a [`GradedSpace`](@ref), which is used when `IteratorSize(values(I)) == IsInfinite()` or `IteratorSize(values(I)) == SizeUnknown()`.
"""
struct GenericGradedSpace{I<:Sector} <: GradedSpace{I}
    dims::SectorDict{I, Int}
    dual::Bool
end

function GenericGradedSpace{I}(dims; dual::Bool = false) where {I<:Sector}
    d = SectorDict{I, Int}()
    for (c, dc) in dims
        k = convert(I, c)
        haskey(d, k) && throw(ArgumentError("Sector $k appears multiple times"))
        !iszero(dc) && push!(d, k=>dc)
    end
    return GenericGradedSpace{I}(d, dual)
end
GenericGradedSpace{I}(; dual::Bool = false) where {I<:Sector} =
    GenericGradedSpace{I}((); dual = dual)
GenericGradedSpace{I}(d1::Pair; dual::Bool = false) where {I<:Sector} =
    GenericGradedSpace{I}((d1,); dual = dual)
GenericGradedSpace{I}(d1::Pair, dims::Vararg{Pair};
                                dual::Bool = false) where {I<:Sector} =
    GenericGradedSpace{I}((d1, dims...); dual = dual)

Base.:(==)(V1::GenericGradedSpace, V2::GenericGradedSpace) =
    keys(V1.dims) == keys(V2.dims) &&
    values(V1.dims) == values(V2.dims) &&
    V1.dual == V2.dual

Base.hash(V::GenericGradedSpace, h::UInt) = hash(V.dual, hash(V.dims, h))

"""
    struct FiniteGradedSpace{I<:Sector, N} <: GradedSpace{I}

Optimized implementation of a [`GradedSpace`](@ref), which is used when `IteratorSize(values(I)) == HasLength()` or `IteratorSize(values(I)) == HasShape()`.
"""
struct FiniteGradedSpace{I<:Sector, N} <: GradedSpace{I}
    dims::NTuple{N, Int}
    dual::Bool
    function FiniteGradedSpace{I, N}(dims::Dims{N}, dual::Bool) where {I<:Sector, N}
        return new{I, N}(dims, dual)
    end
end

function FiniteGradedSpace{I}(dims; dual::Bool = false) where {I<:Sector}
    N = length(values(I))
    d = ntuple(n->0, N)
    isset = ntuple(n->false, N)
    for (c, dc) in dims
        i = findindex(values(I), convert(I, c))
        isset[i] && throw(ArgumentError("Sector $c appears multiple times"))
        isset = TupleTools.setindex(isset, true, i)
        d = TupleTools.setindex(d, dc, i)
    end
    return FiniteGradedSpace{I, N}(d, dual)
end
FiniteGradedSpace{I}(; dual::Bool = false) where {I<:Sector} =
    FiniteGradedSpace{I}((); dual = dual)
FiniteGradedSpace{I}(d1::Pair; dual::Bool = false) where {I<:Sector} =
    FiniteGradedSpace{I}((d1,); dual = dual)
FiniteGradedSpace{I}(d1::Pair, dims::Vararg{Pair};
                                dual::Bool = false) where {I<:Sector} =
    FiniteGradedSpace{I}((d1, dims...); dual = dual)
# get rid of N parameter
FiniteGradedSpace{I, N}(dims::Pair...; dual::Bool = false) where {I<:Sector, N} =
    FiniteGradedSpace{I, N}(dims; dual = dual)
function FiniteGradedSpace{I, N}(dims; dual::Bool = false) where {I<:Sector, N}
    @assert N == length(values(I))
    FiniteGradedSpace{I}(dims; dual = dual)
end

# Never write GenericGradedSpace, just use GradedSpace
function GradedSpace{I}(args...; dual::Bool = false) where {I<:Sector}
    if Base.IteratorSize(values(I)) === IsInfinite()
        GenericGradedSpace{I}(args...; dual = dual)
    else
        FiniteGradedSpace{I}(args...; dual = dual)
    end
end
GradedSpace(dims::Tuple{Vararg{Pair{I, Int}}};
                        dual::Bool = false) where {I<:Sector} =
    GradedSpace{I}(dims; dual = dual)
GradedSpace(dims::Vararg{Pair{I, Int}}; dual::Bool = false) where {I<:Sector} =
    GradedSpace{I}(dims; dual = dual)
GradedSpace(dims::AbstractDict{I, Int}; dual::Bool = false) where {I<:Sector} =
    GradedSpace{I}(dims; dual = dual)
# not inferrable
GradedSpace(g::Base.Generator; dual::Bool = false) =
    GradedSpace(g...; dual = dual)
GradedSpace(g::AbstractDict; dual::Bool = false) =
    GradedSpace(g...; dual = dual)

Base.getindex(::ComplexNumbers, I::Type{<:Sector}) = GradedSpace{I}
Base.getindex(::ComplexNumbers, d1::Pair{I, Int}, dims::Pair{I, Int}...) where {I<:Sector} =
    GradedSpace{I}(d1, dims...)

# Corresponding methods:
# properties
dim(V::GenericGradedSpace) =
    mapreduce(c->dim(c)*V.dims[c], +, keys(V.dims); init = zero(dim(one(sectortype(V)))))
dim(V::GenericGradedSpace{I}, c::I) where {I<:Sector} =
    get(V.dims, isdual(V) ? dual(c) : c, 0)

dim(V::FiniteGradedSpace{I}) where {I<:Sector} =
    reduce(+, dc*dim(c) for (dc, c) in zip(V.dims, values(I));
            init = zero(dim(one(sectortype(V)))))
dim(V::FiniteGradedSpace{I}, c::I) where {I<:Sector} =
    V.dims[findindex(values(I), isdual(V) ? dual(c) : c)]

sectors(V::GenericGradedSpace{I}) where {I<:Sector} =
    SectorSet{I}(s->isdual(V) ? dual(s) : s, keys(V.dims))
sectors(V::FiniteGradedSpace{I, N}) where {I<:Sector, N} =
    SectorSet{I}(Iterators.filter(n->V.dims[n]!=0, 1:N)) do n
        isdual(V) ? dual(values(I)[n]) : values(I)[n]
    end

hassector(V::GradedSpace{I}, s::I) where {I<:Sector} = dim(V, s) != 0

Base.conj(V::GenericGradedSpace{I}) where {I<:Sector} =
    GenericGradedSpace{I}(V.dims, !V.dual)
Base.conj(V::FiniteGradedSpace{I, N}) where {I<:Sector, N} =
    FiniteGradedSpace{I, N}(V.dims, !V.dual)
isdual(V::GradedSpace) = V.dual

# equality / comparison
Base.:(==)(V1::GradedSpace, V2::GradedSpace) =
    (V1.dims == V2.dims) && V1.dual == V2.dual

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

# Specific constructors for Z_N
const ZNSpace{N} = FiniteGradedSpace{ZNIrrep{N}, N}
ZNSpace{N}(dims::NTuple{N, Int}; dual::Bool = false) where {N} = ZNSpace{N}(dims, dual)
ZNSpace{N}(dims::Vararg{Int, N}; dual::Bool = false) where {N} = ZNSpace{N}(dims, dual)
ZNSpace(dims::NTuple{N, Int}; dual::Bool = false) where {N} = ZNSpace{N}(dims, dual)
ZNSpace(dims::Vararg{Int, N}; dual::Bool = false) where {N} = ZNSpace{N}(dims, dual)

const ParitySpace = ZNSpace{2}
const ℤ₂Space = ZNSpace{2}
const ℤ₃Space = ZNSpace{3}
const ℤ₄Space = ZNSpace{4}
const U₁Space = GenericGradedSpace{U₁}
const CU₁Space = GenericGradedSpace{CU₁}
const SU₂Space = GenericGradedSpace{SU₂}

# non-Unicode alternatives
const Z2Space = ℤ₂Space
const Z3Space = ℤ₃Space
const Z4Space = ℤ₄Space
const U1Space = U₁Space
const CU1Space = CU₁Space
const SU2Space = SU₂Space

Base.oneunit(::Type{<:GradedSpace{I}}) where {I<:Sector} =
    GradedSpace{I}(one(I)=>1)

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
    return GradedSpace{I}(dims; dual = dual1)
end

function flip(V::GradedSpace{I}) where {I}
    if isdual(V)
        GradedSpace{I}(c=>dim(V, c) for c in sectors(V))
    else
        GradedSpace{I}(dual(c)=>dim(V, c) for c in sectors(V))'
    end
end

function fuse(V1::GradedSpace{I}, V2::GradedSpace{I}) where {I<:Sector}
    dims = SectorDict{I, Int}()
    for a in sectors(V1), b in sectors(V2)
        for c in a ⊗ b
            dims[c] = get(dims, c, 0) + Nsymbol(a, b, c)*dim(V1, a)*dim(V2, b)
        end
    end
    return GradedSpace{I}(dims)
end

function infimum(V1::GradedSpace{I}, V2::GradedSpace{I}) where {I}
    if V1.dual == V2.dual
        GradedSpace{I}(c=>min(dim(V1, c), dim(V2, c)) for c in
            union(sectors(V1), sectors(V2)), dual = V1.dual)
    else
        throw(SpaceMismatch("Infimum of space and dual space does not exist"))
    end
end

function supremum(V1::GradedSpace{I}, V2::GradedSpace{I}) where {I}
    if V1.dual == V2.dual
        GradedSpace{I}(c=>max(dim(V1, c), dim(V2, c)) for c in
            union(sectors(V1), sectors(V2)), dual = V1.dual)
    else
        throw(SpaceMismatch("Supremum of space and dual space does not exist"))
    end
end

Base.show(io::IO, ::Type{ℤ₂Space}) = print(io, "ℤ₂Space")
Base.show(io::IO, ::Type{ℤ₃Space}) = print(io, "ℤ₃Space")
Base.show(io::IO, ::Type{ℤ₄Space}) = print(io, "ℤ₄Space")
Base.show(io::IO, ::Type{U₁Space}) = print(io, "U₁Space")
Base.show(io::IO, ::Type{CU₁Space}) = print(io, "CU₁Space")
Base.show(io::IO, ::Type{SU₂Space}) = print(io, "SU₂Space")

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
