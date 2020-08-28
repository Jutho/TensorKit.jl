# Fusion trees:
#==============================================================================#
"""
    struct FusionTree{I, N, M, L, T}

Represents a fusion tree of sectors of type `I<:Sector`, fusing (or splitting) `N` uncoupled
sectors to a coupled sector. (It actually represents a splitting tree, but fusion tree
is a more common term). The `isdual` field indicates whether an isomorphism is present
(if the corresponding value is true) or not. The field `uncoupled` contains the sectors
coming out of the splitting trees, before the possible 𝑍 isomorphism. This fusion tree
has `M=max(0, N-2)` inner lines. Furthermore, for `FusionStyle(I) isa DegenerateNonAbelian`,
the `L=max(0, N-1)` corresponding vertices carry a label of type `T`. If `FusionStyle(I)
isa Union{Abelian, SimpleNonAbelian}`, `T = Nothing`.
"""
struct FusionTree{I<:Sector, N, M, L, T}
    uncoupled::NTuple{N, I}
    coupled::I
    isdual::NTuple{N, Bool}
    innerlines::NTuple{M, I} # M = N-2
    vertices::NTuple{L, T} # L = N-1
    function FusionTree{I, N, M, L, T}(uncoupled::NTuple{N, I},
                                            coupled::I,
                                            isdual::NTuple{N, Bool},
                                            innerlines::NTuple{M, I},
                                            vertices::NTuple{L, T}) where
                                            {I<:Sector, N, M, L, T}
        new{I, N, M, L, T}(uncoupled, coupled, isdual, innerlines, vertices)
    end
end
FusionTree{I}(uncoupled::NTuple{N, Any},
                coupled,
                isdual::NTuple{N, Bool},
                innerlines,
                vertices = ntuple(n->nothing, max(0, N-1))
                ) where {I<:Sector, N} =
    fusiontreetype(I, N)(map(s->convert(I, s), uncoupled),
        convert(I, coupled), isdual, map(s->convert(I, s), innerlines), vertices)
FusionTree(uncoupled::NTuple{N, I},
            coupled::I,
            isdual::NTuple{N, Bool},
            innerlines,
            vertices = ntuple(n->nothing, N-1)
            ) where {I<:Sector, N} =
    fusiontreetype(I, N)(uncoupled, coupled, isdual, innerlines, vertices)

function FusionTree{I}(uncoupled::NTuple{N}, coupled = one(I),
                        isdual = ntuple(n->false, N)) where {I<:Sector, N}
    FusionStyle(I) isa Abelian ||
        error("fusion tree requires inner lines if `FusionStyle(I) <: NonAbelian`")
    FusionTree{I}(map(s->convert(I, s), uncoupled), convert(I, coupled), isdual,
                    _abelianinner(map(s->convert(I, s), (uncoupled..., dual(coupled)))))
end
function FusionTree(uncoupled::NTuple{N, I}, coupled::I = one(I),
                        isdual = ntuple(n->false, N)) where {I<:Sector, N}
    FusionStyle(I) isa Abelian ||
        error("fusion tree requires inner lines if `FusionStyle(I) <: NonAbelian`")
    FusionTree{I}(uncoupled, coupled, isdual, _abelianinner((uncoupled..., dual(coupled))))
end

# Properties
sectortype(::Type{<:FusionTree{I}}) where {I<:Sector} = I
FusionStyle(::Type{<:FusionTree{I}}) where {I<:Sector} = FusionStyle(I)
BraidingStyle(::Type{<:FusionTree{I}}) where {I<:Sector} = BraidingStyle(I)
Base.length(::Type{<:FusionTree{<:Sector, N}}) where {N} = N

sectortype(f::FusionTree) = sectortype(typeof(f))
FusionStyle(f::FusionTree) = FusionStyle(typeof(f))
BraidingStyle(f::FusionTree) = BraidingStyle(typeof(f))
Base.length(f::FusionTree) = length(typeof(f))

# Hashing, important for using fusion trees as key in a dictionary
function Base.hash(f::FusionTree{I}, h::UInt) where {I}
    h = hash(f.isdual, hash(f.coupled, hash(f.uncoupled, h)))
    if FusionStyle(I) isa SimpleNonAbelian
        h = hash(f.innerlines, h)
    end
    if FusionStyle(I) isa DegenerateNonAbelian
        h = hash(f.vertices, h)
    end
    return h
end
function Base.isequal(f1::FusionTree{I, N}, f2::FusionTree{I, N}) where {I<:Sector, N}
    f1.coupled == f2.coupled || return false
    @inbounds for i = 1:N
        f1.uncoupled[i] == f2.uncoupled[i] || return false
        f1.isdual[i] == f2.isdual[i] || return false
    end
    if FusionStyle(I) isa SimpleNonAbelian
        @inbounds for i=1:N-2
            f1.innerlines[i] == f2.innerlines[i] || return false
        end
    end
    if FusionStyle(I) isa DegenerateNonAbelian
        @inbounds for i=1:N-1
            f1.vertices[i] == f2.vertices[i] || return false
        end
    end
    return true
end
Base.isequal(f1::FusionTree, f2::FusionTree) = false


# Facilitate getting correct fusion tree types
Base.@pure function fusiontreetype(::Type{I}, N::Int) where {I<:Sector}
    if N === 0
        FusionTree{I, 0, 0, 0, vertex_labeltype(I)}
    elseif N === 1
        FusionTree{I, 1, 0, 0, vertex_labeltype(I)}
    else
        FusionTree{I, N, N-2, N-1, vertex_labeltype(I)}
    end
end

# converting to actual array
function Base.convert(::Type{Array}, f::FusionTree{I, 0}) where {I}
    T = eltype(fusiontensor(one(I), one(I), one(I)))
    return fill(one(T), 1)
end
function Base.convert(::Type{Array}, f::FusionTree{I, 1}) where {I}
    c = f.coupled
    dc = dim(c)
    if f.isdual[1]
        Zcbartranspose = sqrt(dc)*reshape(fusiontensor(conj(c), c, one(c)), (dc, dc))
        return convert(Array, conj(Zcbartranspose))
    else
        convert(Array, reshape(fusiontensor(c, one(c), c), (dc, dc)))
    end
end

function Base.convert(::Type{Array}, f::FusionTree{I, 2}) where {I}
    a, b = f.uncoupled
    isduala, isdualb = f.isdual
    c = f.coupled
    da, db, dc = dim.((a, b, c))
    μ = f.vertices[1]
    X = reshape(fusiontensor(a, b, c, μ), da*db, dc)
    Za = convert(Array, FusionTree((a,), a, (isduala,), ()))
    Zb = convert(Array, FusionTree((b,), b, (isdualb,), ()))
    return convert(Array, reshape(kron(Zb, Za)*X, (da, db, dc)))
end

function Base.convert(::Type{Array}, f::FusionTree{I}) where {I}
    tailout = (f.innerlines[1], TupleTools.tail2(f.uncoupled)...)
    isdualout = (false, TupleTools.tail2(f.isdual)...)
    ftail = FusionTree(tailout, f.coupled, isdualout,
                        Base.tail(f.innerlines), Base.tail(f.vertices))
    Ctail = convert(Array, ftail)
    f1 = FusionTree((f.uncoupled[1], f.uncoupled[2]), f.innerlines[1],
                    (f.isdual[1], f.isdual[2]), (), (f.vertices[1],))
    C1 = convert(Array, f1)
    dtail = size(Ctail)
    d1 = size(C1)
    C = reshape(C1, d1[1]*d1[2], d1[3]) *
            reshape(Ctail, dtail[1], prod(Base.tail(dtail)))
    return reshape(C, (d1[1], d1[2], Base.tail(dtail)...))
end

# Show methods
function Base.show(io::IO, t::FusionTree{I, N, M, K, Nothing}) where {I<:Sector, N, M, K}
    print(IOContext(io, :typeinfo => I), "FusionTree{", I, "}(",
        t.uncoupled, ", ", t.coupled, ", ", t.isdual, ", ", t.innerlines, ")")
end
function Base.show(io::IO, t::FusionTree{I}) where {I<:Sector}
    print(IOContext(io, :typeinfo => I), "FusionTree{", I, "}(",
        t.uncoupled, ", ", t.coupled, ", ", t.isdual, ",",
        t.innerlines, ", ", t.vertices, ")")
end

# Manipulate fusion trees
include("manipulations.jl")

# Fusion tree iterators
include("iterator.jl")

# auxiliary routines
# _abelianinner: generate the inner indices for given outer indices in the abelian case
_abelianinner(outer::Tuple{}) = ()
_abelianinner(outer::Tuple{I}) where {I<:Sector} =
    outer[1] == one(I) ? () : throw(SectorMismatch())
_abelianinner(outer::Tuple{I, I}) where {I<:Sector} =
    outer[1] == dual(outer[2]) ? () : throw(SectorMismatch())
_abelianinner(outer::Tuple{I, I, I}) where {I<:Sector} =
    first(⊗(outer...)) == one(I) ? () : throw(SectorMismatch())
function _abelianinner(outer::NTuple{N, I}) where {I<:Sector, N}
    c = first(outer[1] ⊗ outer[2])
    return (c, _abelianinner((c, TupleTools.tail2(outer)...))...)
end
