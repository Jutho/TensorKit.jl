# Fusion trees:
#==============================================================================#
"""
    struct FusionTree{I, N, M, L, T}

Represents a fusion tree of sectors of type `I<:Sector`, fusing (or splitting) `N` uncoupled
sectors to a coupled sector. It actually represents a splitting tree, but fusion tree
is a more common term.

## Fields
- `uncoupled::NTuple{N,I}`: the uncoupled sectors coming out of the splitting tree, before
  the possible 𝑍 isomorphism (see `isdual`).
- `coupled::I`: the coupled sector.
- `isdual::NTuple{N,Bool}`: indicates whether a 𝑍 isomorphism is present (`true`) or not
  (`false`) for each uncoupled sector.
- `innerlines::NTuple{M,I}`: the labels of the `M=max(0, N-2)` inner lines of the splitting
  tree.
- `vertices::NTuple{L,T}`: the `L=max(0, N-1)` labels of type `T` of the vertices of the
  splitting tree. If `FusionStyle(I) isa MultiplicityFreeFusion`, then `T = Nothing`.
"""
struct FusionTree{I<:Sector,N,M,L,T}
    uncoupled::NTuple{N,I}
    coupled::I
    isdual::NTuple{N,Bool}
    innerlines::NTuple{M,I} # M = N-2
    vertices::NTuple{L,T} # L = N-1
    function FusionTree{I,N,M,L,T}(uncoupled::NTuple{N,I},
                                   coupled::I,
                                   isdual::NTuple{N,Bool},
                                   innerlines::NTuple{M,I},
                                   vertices::NTuple{L,T}) where
             {I<:Sector,N,M,L,T}
        # if N == 0
        #     @assert coupled == one(coupled)
        # elseif N == 1
        #     @assert coupled == uncoupled[1]
        # elseif N == 2
        #     @assert coupled ∈ ⊗(uncoupled...)
        # else
        #     @assert innerlines[1] ∈ ⊗(uncoupled[1], uncoupled[2])
        #     for n = 2:N-2
        #         @assert innerlines[n] ∈ ⊗(innerlines[n-1], uncoupled[n+1])
        #     end
        #     @assert coupled ∈ ⊗(innerlines[N-2], uncoupled[N])
        # end
        return new{I,N,M,L,T}(uncoupled, coupled, isdual, innerlines, vertices)
    end
end
function FusionTree{I}(uncoupled::NTuple{N,Any}, coupled,
                       isdual::NTuple{N,Bool}, innerlines,
                       vertices=ntuple(n -> nothing, max(0, N - 1))) where {I<:Sector,N}
    if FusionStyle(I) isa GenericFusion
        fusiontreetype(I, N)(map(s -> convert(I, s), uncoupled),
                             convert(I, coupled), isdual,
                             map(s -> convert(I, s), innerlines), vertices)
    else
        vertices′ = ntuple(n -> nothing, max(0, N - 1))
        if vertices == vertices′ || all(isone, vertices)
            fusiontreetype(I, N)(map(s -> convert(I, s), uncoupled),
                                 convert(I, coupled), isdual,
                                 map(s -> convert(I, s), innerlines), vertices)
        else
            throw(ArgumentError("Incorrect fusion vertices"))
        end
    end
end
function FusionTree(uncoupled::NTuple{N,I}, coupled::I,
                    isdual::NTuple{N,Bool}, innerlines,
                    vertices=ntuple(n -> nothing, max(0, N - 1))) where {I<:Sector,N}
    if FusionStyle(I) isa GenericFusion
        fusiontreetype(I, N)(uncoupled, coupled, isdual, innerlines, vertices)
    else
        vertices′ = ntuple(n -> nothing, max(0, N - 1))
        if vertices == vertices′ || all(isone, vertices)
            fusiontreetype(I, N)(uncoupled, coupled, isdual, innerlines, vertices′)
        else
            throw(ArgumentError("Incorrect fusion vertices"))
        end
    end
end

function FusionTree{I}(uncoupled::NTuple{N}, coupled=one(I),
                       isdual=ntuple(n -> false, N)) where {I<:Sector,N}
    FusionStyle(I) isa UniqueFusion ||
        error("fusion tree requires inner lines if `FusionStyle(I) <: MultipleFusion`")
    return FusionTree{I}(map(s -> convert(I, s), uncoupled), convert(I, coupled), isdual,
                         _abelianinner(map(s -> convert(I, s),
                                           (uncoupled..., dual(coupled)))))
end
function FusionTree(uncoupled::Tuple{I,Vararg{I}}, coupled::I=one(I),
                    isdual=ntuple(n -> false, length(uncoupled))) where {I<:Sector}
    FusionStyle(I) isa UniqueFusion ||
        error("fusion tree requires inner lines if `FusionStyle(I) <: MultipleFusion`")
    return FusionTree{I}(uncoupled, coupled, isdual,
                         _abelianinner((uncoupled..., dual(coupled))))
end

# Properties
sectortype(::Type{<:FusionTree{I}}) where {I<:Sector} = I
FusionStyle(::Type{<:FusionTree{I}}) where {I<:Sector} = FusionStyle(I)
BraidingStyle(::Type{<:FusionTree{I}}) where {I<:Sector} = BraidingStyle(I)
Base.length(::Type{<:FusionTree{<:Sector,N}}) where {N} = N

sectortype(f::FusionTree) = sectortype(typeof(f))
FusionStyle(f::FusionTree) = FusionStyle(typeof(f))
BraidingStyle(f::FusionTree) = BraidingStyle(typeof(f))
Base.length(f::FusionTree) = length(typeof(f))

# Hashing, important for using fusion trees as key in a dictionary
function Base.hash(f::FusionTree{I}, h::UInt) where {I}
    h = hash(f.isdual, hash(f.coupled, hash(f.uncoupled, h)))
    if FusionStyle(I) isa MultipleFusion
        h = hash(f.innerlines, h)
    end
    if FusionStyle(I) isa GenericFusion
        h = hash(f.vertices, h)
    end
    return h
end
function Base.isequal(f₁::FusionTree{I,N}, f₂::FusionTree{I,N}) where {I<:Sector,N}
    f₁.coupled == f₂.coupled || return false
    @inbounds for i in 1:N
        f₁.uncoupled[i] == f₂.uncoupled[i] || return false
        f₁.isdual[i] == f₂.isdual[i] || return false
    end
    if FusionStyle(I) isa MultipleFusion
        @inbounds for i in 1:(N - 2)
            f₁.innerlines[i] == f₂.innerlines[i] || return false
        end
    end
    if FusionStyle(I) isa GenericFusion
        @inbounds for i in 1:(N - 1)
            f₁.vertices[i] == f₂.vertices[i] || return false
        end
    end
    return true
end
Base.isequal(f₁::FusionTree, f₂::FusionTree) = false

# Facilitate getting correct fusion tree types
function fusiontreetype(::Type{I}, N::Int) where {I<:Sector}
    if N === 0
        FusionTree{I,0,0,0,vertex_labeltype(I)}
    elseif N === 1
        FusionTree{I,1,0,0,vertex_labeltype(I)}
    else
        FusionTree{I,N,N - 2,N - 1,vertex_labeltype(I)}
    end
end

# converting to actual array
function Base.convert(A::Type{<:AbstractArray}, f::FusionTree{I,0}) where {I}
    X = convert(A, fusiontensor(one(I), one(I), one(I)))[1, 1, :]
    return X
end
function Base.convert(A::Type{<:AbstractArray}, f::FusionTree{I,1}) where {I}
    c = f.coupled
    if f.isdual[1]
        sqrtdc = sqrtdim(c)
        Zcbartranspose = sqrtdc * convert(A, fusiontensor(conj(c), c, one(c)))[:, :, 1, 1]
        X = conj!(Zcbartranspose) # we want Zcbar^†
    else
        X = convert(A, fusiontensor(c, one(c), c))[:, 1, :, 1, 1]
    end
    return X
end

function Base.convert(A::Type{<:AbstractArray}, f::FusionTree{I,2}) where {I}
    a, b = f.uncoupled
    isduala, isdualb = f.isdual
    c = f.coupled
    da, db, dc = dim.((a, b, c))
    μ = (FusionStyle(I) isa GenericFusion) ? f.vertices[1] : 1
    C = convert(A, fusiontensor(a, b, c))[:, :, :, μ]
    X = C
    if isduala
        Xtemp = X
        X = similar(Xtemp)
        Za = convert(A, FusionTree((a,), a, (isduala,), ()))
        TO.tensorcontract!(X, ((1, 2, 3), ()), Za, ((1,), (2,)), :N, Xtemp, ((1,), (2, 3)),
                           :N, true, false)
    end
    if isdualb
        Xtemp = X
        X = similar(Xtemp)
        Zb = convert(A, FusionTree((b,), b, (isdualb,), ()))
        TO.tensorcontract!(X, ((2, 1, 3), ()), Zb, ((1,), (2,)), :N, Xtemp, ((2,), (1, 3)),
                           :N, true, false)
    end
    return X
end

function Base.convert(A::Type{<:AbstractArray}, f::FusionTree{I,N}) where {I,N}
    tailout = (f.innerlines[1], TupleTools.tail2(f.uncoupled)...)
    isdualout = (false, TupleTools.tail2(f.isdual)...)
    ftail = FusionTree(tailout, f.coupled, isdualout,
                       Base.tail(f.innerlines), Base.tail(f.vertices))
    Ctail = convert(A, ftail)
    f₁ = FusionTree((f.uncoupled[1], f.uncoupled[2]), f.innerlines[1],
                    (f.isdual[1], f.isdual[2]), (), (f.vertices[1],))
    C1 = convert(A, f₁)
    dtail = size(Ctail)
    d1 = size(C1)
    X = similar(C1, (d1[1], d1[2], Base.tail(dtail)...))
    trivialtuple = ntuple(identity, Val(N))
    return TO.tensorcontract!(X, ((trivialtuple..., N + 1), ()), C1, ((1, 2), (3,)), :N,
                              Ctail, ((1,), Base.tail(trivialtuple)), :N, true, false)
end

# TODO: is this piracy?
function Base.convert(A::Type{<:AbstractArray},
                      (f₁, f₂)::Tuple{FusionTree{I},FusionTree{I}}) where {I}
    F₁ = convert(A, f₁)
    F₂ = convert(A, f₂)
    sz1 = size(F₁)
    sz2 = size(F₂)
    d1 = TupleTools.front(sz1)
    d2 = TupleTools.front(sz2)

    return reshape(reshape(F₁, TupleTools.prod(d1), sz1[end]) *
                   reshape(F₂, TupleTools.prod(d2), sz2[end])', (d1..., d2...))
end

# Show methods
function Base.show(io::IO, t::FusionTree{I,N,M,K,Nothing}) where {I<:Sector,N,M,K}
    return print(IOContext(io, :typeinfo => I), "FusionTree{", type_repr(I), "}(",
                 t.uncoupled, ", ", t.coupled, ", ", t.isdual, ", ", t.innerlines, ")")
end
function Base.show(io::IO, t::FusionTree{I}) where {I<:Sector}
    return print(IOContext(io, :typeinfo => I), "FusionTree{", type_repr(I), "}(",
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
function _abelianinner(outer::Tuple{I}) where {I<:Sector}
    return outer[1] == one(I) ? () : throw(SectorMismatch())
end
function _abelianinner(outer::Tuple{I,I}) where {I<:Sector}
    return outer[1] == dual(outer[2]) ? () : throw(SectorMismatch())
end
function _abelianinner(outer::Tuple{I,I,I}) where {I<:Sector}
    return first(⊗(outer...)) == one(I) ? () : throw(SectorMismatch())
end
function _abelianinner(outer::Tuple{I,I,I,I,Vararg{I}}) where {I<:Sector}
    c = first(outer[1] ⊗ outer[2])
    return (c, _abelianinner((c, TupleTools.tail2(outer)...))...)
end
