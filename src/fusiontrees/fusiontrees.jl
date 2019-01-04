# Fusion trees:
#==============================================================================#
struct FusionTree{G<:Sector,N,M,L,T}
    uncoupled::NTuple{N,G}
    coupled::G
    innerlines::NTuple{M,G} # M = N-2
    vertices::NTuple{L,T} # L = N-1
end
FusionTree{G}(uncoupled::NTuple{N},
                coupled,
                innerlines,
                vertices = ntuple(n->nothing, StaticLength(N)-StaticLength(1))
                ) where {G<:Sector,N} =
    fusiontreetype(G, StaticLength(N))(map(s->convert(G,s),uncoupled),
        convert(G,coupled), map(s->convert(G,s), innerlines), vertices)
FusionTree(uncoupled::NTuple{N,G},
            coupled::G,
            innerlines,
            vertices = ntuple(n->nothing, StaticLength(N)-StaticLength(1))
            ) where {G<:Sector,N} =
    fusiontreetype(G, StaticLength(N))(uncoupled, coupled, innerlines, vertices)

function FusionTree{G}(uncoupled::NTuple{N}, coupled = one(G)) where {G<:Sector, N}
    FusionStyle(G) isa Abelian ||
        error("fusion tree requires inner lines if `FusionStyle(G) <: NonAbelian`")
    FusionTree{G}(map(s->convert(G,s), uncoupled), convert(G, coupled),
                    _abelianinner(map(s->convert(G,s),(uncoupled..., dual(coupled)))))
end
function FusionTree(uncoupled::NTuple{N,G}, coupled::G = one(G)) where {G<:Sector, N}
    FusionStyle(G) isa Abelian ||
        error("fusion tree requires inner lines if `FusionStyle(G) <: NonAbelian`")
    FusionTree{G}(uncoupled, coupled, _abelianinner((uncoupled..., dual(coupled))))
end

# Properties
sectortype(::Type{<:FusionTree{G}}) where {G<:Sector} = G
FusionStyle(::Type{<:FusionTree{G}}) where {G<:Sector} = FusionStyle(G)
Base.length(::Type{<:FusionTree{<:Sector,N}}) where {N} = N

sectortype(t::FusionTree) = sectortype(typeof(t))
FusionStyle(t::FusionTree) = FusionStyle(typeof(t))
Base.length(t::FusionTree) = length(typeof(t))

# Hashing, important for using fusion trees as key in Dict
function Base.hash(f::FusionTree{G}, h::UInt) where {G}
    if FusionStyle(G) isa Abelian
        hash(f.uncoupled, hash(f.coupled, h))
    elseif FusionStyle(G) isa SimpleNonAbelian
        hash(f.innerlines, hash(f.uncoupled, hash(f.coupled, h)))
    else
        hash(f.vertices, hash(f.innerlines, hash(f.uncoupled, hash(f.coupled, h))))
    end
end
function Base.isequal(f1::FusionTree{G,N}, f2::FusionTree{G,N}) where {G,N}
    f1.coupled == f2.coupled || return false
    @inbounds for i = 1:N
        f1.uncoupled[i] == f2.uncoupled[i] || return false
    end
    if FusionStyle(G) isa SimpleNonAbelian
        @inbounds for i=1:N-2
            f1.innerlines[i] == f2.innerlines[i] || return false
        end
    end
    if FusionStyle(G) isa DegenerateNonAbelian
        @inbounds for i=1:N-1
            f1.vertices[i] == f2.vertices[i] || return false
        end
    end
    return true
end
Base.isequal(f1::FusionTree{G1,N1}, f2::FusionTree{G2,N2}) where {G1,G2,N1,N2} = false

# Fusion tree methods
Base.@pure fusiontreetype(::Type{G}, ::StaticLength{0}) where {G<:Sector} =
    FusionTree{G, 0, 0, 0, vertex_labeltype(G)}
Base.@pure fusiontreetype(::Type{G}, ::StaticLength{1}) where {G<:Sector} =
    FusionTree{G, 1, 0, 0, vertex_labeltype(G)}
Base.@pure fusiontreetype(::Type{G}, ::StaticLength{2}) where {G<:Sector} =
    FusionTree{G, 2, 0, 1, vertex_labeltype(G)}
Base.@pure fusiontreetype(::Type{G}, ::StaticLength{N}) where {G<:Sector, N} =
    _fusiontreetype(G, StaticLength(N),
        StaticLength(N) - StaticLength(2), StaticLength(N) - StaticLength(1))
Base.@pure _fusiontreetype(::Type{G}, ::StaticLength{N}, ::StaticLength{M},
                            ::StaticLength{L}) where {G<:Sector, N, M, L} =
    FusionTree{G,N,M,L,vertex_labeltype(G)}

# converting to actual array
function Base.convert(::Type{Array}, f::FusionTree{G, 0}) where {G}
    T = eltype(fusiontensor(one(G), one(G), one(G)))
    return fill(one(T), 1)
end
function Base.convert(::Type{Array}, f::FusionTree{G, 1}) where {G}
    T = eltype(fusiontensor(one(G), one(G), one(G)))
    return copyto!(Matrix{T}(undef, dim(f.coupled), dim(f.coupled)), I)
end
Base.convert(::Type{Array}, f::FusionTree{G,2}) where {G} =
    fusiontensor(f.uncoupled[1], f.uncoupled[2], f.coupled, f.vertices[1])

function Base.convert(::Type{Array}, f::FusionTree{G}) where {G}
    tailout = (f.innerlines[1], TupleTools.tail2(f.uncoupled)...)
    ftail = FusionTree(tailout, f.coupled, Base.tail(f.innerlines), Base.tail(f.vertices))
    Ctail = convert(Array, ftail)
    C1 = fusiontensor(f.uncoupled[1], f.uncoupled[2], f.innerlines[1], f.vertices[1])
    dtail = size(Ctail)
    d1 = size(C1)
    C = reshape(C1, d1[1]*d1[2], d1[3]) *
            reshape(Ctail, dtail[1], prod(Base.tail(dtail)))
    return reshape(C, (d1[1], d1[2], Base.tail(dtail)...))
end

# permute fusion tree
"""
    function permute(t::FusionTree, i) -> <:AbstractDict{typeof(t),<:Number}

Perform a permutation of the uncoupled indices of the fusion tree `t` and returns the result
as a `<:AbstractDict` of output trees and corresponding coefficients.
"""
function permute(t::FusionTree{G,N}, p::NTuple{N,Int}) where {G<:Sector, N}
    @assert BraidingStyle(G) isa SymmetricBraiding
    if FusionStyle(G) isa Abelian
        coeff = Rsymbol(one(G), one(G), one(G))
        for i = 1:N
            for j = 1:i-1
                if p[j] > p[i]
                    a, b = t.uncoupled[j], t.uncoupled[i]
                    coeff *= Rsymbol(a, b, first(a ⊗ b))
                end
            end
        end
        t = FusionTree{G}(TupleTools._permute(t.uncoupled, p), t.coupled)
        return SingletonDict(t=>coeff)
    else
        coeff = Rsymbol(one(G), one(G), one(G))
        trees = FusionTreeDict(t=>coeff)
        newtrees = empty(trees)
        for s in permutation2swaps(p)
            for (t, c) in trees
                for (t′,c′) in braid(t, s)
                    newtrees[t′] = get(newtrees, t′, zero(coeff)) + c*c′
                end
            end
            trees, newtrees = newtrees, trees
            empty!(newtrees)
        end
        return trees
    end
end

"""
    function braid(t::FusionTree, i) -> <:AbstractDict{typeof(t),<:Number}

Perform a braid of neighbouring uncoupled indices `i` and `i+1` on a fusion tree `t`,
and returns the result as a linked list of output trees and corresponding coefficients.
"""
function braid(t::FusionTree{G,N}, i) where {G<:Sector, N}
    1 <= i < N ||
        throw(ArgumentError("Cannot swap outputs i=$i and i+1 out of only $N outputs"))
    outer = t.uncoupled
    inner = t.innerlines
    if i == 1
        a, b = outer[1], outer[2]
        c = N > 2 ? inner[1] : t.coupled
        outer = TupleTools.setindex(outer, b, 1)
        outer = TupleTools.setindex(outer, a, 2)
        if FusionStyle(G) isa Abelian
            return SingletonDict(FusionTree{G}(outer, t.coupled, inner, t.vertices) =>
                    Rsymbol(a, b, c))
        elseif FusionStyle(G) isa SimpleNonAbelian
            return FusionTreeDict(FusionTree{G}(outer, t.coupled, inner, t.vertices) =>
                    Rsymbol(a, b, c))
        end
    end
    # case i > 1:
    b = outer[i]
    d = outer[i+1]
    a = i == 2 ? outer[1] : inner[i-2]
    c = inner[i-1]
    e = i == N-1 ? t.coupled : inner[i]
    outer′ = TupleTools.setindex(outer, d, i)
    outer′ = TupleTools.setindex(outer′, b, i+1)
    if FusionStyle(G) isa Abelian
        inner′ = TupleTools.setindex(inner, first(a ⊗ d), i-1)
        return SingletonDict(FusionTree{G}(outer′, t.coupled, inner′) =>
            Rsymbol(b, d, first(b ⊗ d)))
    elseif FusionStyle(G) isa SimpleNonAbelian
        iter = a ⊗ d
        next = iterate(iter)
        next === nothing && error("Empty fusion channel $a and $d ?")
        c′, s = next
        while iszero(Nsymbol(b, c′, e))
            next = iterate(iter, s)
            next === nothing &&
                error("No valid fusion between $a ⊗ $d and dual(dual($e) ⊗ $b)?")
            c′, s = next
        end
        coeff = conj(Rsymbol(b,a,c))*Fsymbol(b,a,d,e,c,c′)*Rsymbol(b,c′,e)
        inner′ = TupleTools.setindex(inner, c′, i-1)
        output = FusionTreeDict(FusionTree{G}(outer′, t.coupled, inner′, t.vertices)=>coeff)
        next = iterate(iter, s)
        while next !== nothing
            c′, s = next
            next = iterate(iter, s)
            iszero(Nsymbol(b, c′, e)) && continue
            coeff = conj(Rsymbol(b,a,c))*Fsymbol(b,a,d,e,c,c′)*Rsymbol(b,c′,e)
            inner′ = TupleTools.setindex(inner, c′, i-1)
            if coeff != zero(coeff)
                push!(output, FusionTree{G}(outer′, t.coupled, inner′, t.vertices) => coeff)
            end
        end
        return output
    else
        # TODO: implement DegenerateNonAbelian case
        throw(MethodError(braid, (t, i)))
    end
end

# repartition double fusion tree
"""
    function repartition(t1::FusionTree{G,N₁},
                            t2::FusionTree{G,N₂},
                            ::StaticLength{N}) where {G,N₁,N₂,N}
        -> <:AbstractDict{Tuple{FusionTree{G,N}, FusionTree{G,N₁+N₂-N}},<:Number}

Input is a double fusion tree that describes the fusion of a set of incoming uncoupled
sectors to a set of outgoing uncoupled sectors, represented using the individual trees of
outgoing (`t1`) and incoming sectors (`t2`) respectively (with identical coupled sector
`t1.coupled == t2.coupled`). Computes new trees and corresponding coefficients obtained from
repartitioning the tree by bending incoming to outgoing sectors (or vice versa) in order to
have `N` outgoing sectors.
"""
function repartition(t1::FusionTree{G,N₁},
                        t2::FusionTree{G,N₂},
                        V::StaticLength{N}) where {G<:Sector, N₁, N₂, N}
    t1.coupled == t2.coupled || throw(SectorMismatch())
    @assert 0 <= N <= N₁+N₂
    V1 = V
    V2 = StaticLength(N₁)+StaticLength(N₂)-V

    if FusionStyle(t1) isa Abelian || FusionStyle(t1) isa SimpleNonAbelian
        coeff = sqrt(dim(one(G)))*Bsymbol(one(G), one(G), one(G))
        outer = (t1.uncoupled..., map(dual, reverse(t2.uncoupled))...)
        inner1ext = isa(StaticLength(N₁), StaticLength{0}) ? () :
                        (isa(StaticLength(N₁), StaticLength{1}) ? (one(G),) :
                            (one(G), first(outer), t1.innerlines...))
        inner2ext = isa(StaticLength(N₂), StaticLength{0}) ? () :
                        (isa(StaticLength(N₂), StaticLength{1}) ? (one(G),) :
                            (one(G), dual(last(outer)), t2.innerlines...))
        innerext = (inner1ext..., t1.coupled, reverse(inner2ext)...) # length N₁+N₂+1
        for n = N₁+1:N
             # map fusion vertex c<-(a,b) to splitting vertex (c,dual(b))<-a
            b = dual(outer[n])
            a = innerext[n+1]
            c = innerext[n]
            coeff *= inv(sqrt(dim(b))*Bsymbol(c, dual(b), a))
        end
        for n = N₁:-1:N+1
            # map splitting vertex (a,b)<-c to fusion vertex a<-(c,dual(b))
            b = outer[n]
            a = innerext[n]
            c = innerext[n+1]
            coeff *= sqrt(dim(b))*Bsymbol(a,b,c)
            # for Abelian: sqrt(dim(b)) = sqrt(1) is optimized out
        end
        uncoupled1 = TupleTools.getindices(outer, ntuple(n->n, V1))
        uncoupled2 = TupleTools.getindices(map(dual,outer), ntuple(n->N₁+N₂+1-n, V2))
        innerlines1 = TupleTools.getindices(innerext, ntuple(n->n+2, V1 - StaticLength(2)))
        innerlines2 = TupleTools.getindices(innerext, ntuple(n->N₁+N₂-n, V2 - StaticLength(2)))
        c = innerext[N+1]
        t1′ = FusionTree{G}(uncoupled1, c, innerlines1)
        t2′ = FusionTree{G}(uncoupled2, c, innerlines2)
        return SingletonDict((t1′,t2′)=>coeff)
    else
        # TODO: implement DegenerateNonAbelian case
        throw(MethodError(repartition, (t1, t2, V)))
    end
end

# permute double fusion tree
"""
    function permute(t1::FusionTree{G}, t2::FusionTree{G},
                        p1::NTuple{N₁,Int}, p2::NTuple{N₂,Int}) where {G,N₁,N₂}
        -> <:AbstractDict{Tuple{FusionTree{G,N₁}, FusionTree{G,N₂}},<:Number}

Input is a double fusion tree that describes the fusion of a set of incoming uncoupled
sectors to a set of outgoing uncoupled sectors, represented using the individual trees of
outgoing (`t1`) and incoming sectors (`t2`) respectively (with identical coupled sector
`t1.coupled == t2.coupled`). Computes new trees and corresponding coefficients obtained from
repartitioning and permuting the tree such that sectors `p1` become outgoing and sectors
`p2` become incoming.
"""
function permute(t1::FusionTree{G}, t2::FusionTree{G},
                    p1::NTuple{N₁,Int}, p2::NTuple{N₂,Int}) where {G<:Sector, N₁,N₂}
    @assert length(t1) + length(t2) == N₁ + N₂
    p = linearizepermutation(p1, p2, length(t1), length(t2))
    if FusionStyle(t1) isa Abelian
        (t,t0), coeff1 = first(repartition(t1, t2, StaticLength(N₁) + StaticLength(N₂)))
        t, coeff2 = first(permute(t, p))
        (t1′,t2′), coeff3 = first(repartition(t, t0, StaticLength(N₁)))
        return SingletonDict((t1′,t2′)=>coeff1*coeff2*coeff3)
    elseif FusionStyle(t1) isa SimpleNonAbelian
        (t,t0), coeff1 = first(repartition(t1, t2, StaticLength(N₁) + StaticLength(N₂)))
        trees = permute(t, p)
        next = iterate(trees)
        next === nothing && error("empty set of trees?")
        (t, coeff2), s = next
        (t1′, t2′), coeff3 = first(repartition(t, t0, StaticLength(N₁)))
        newtrees = Dict((t1′,t2′)=>coeff1*coeff2*coeff3)
        next = iterate(trees, s)
        while next !== nothing
            (t, coeff2), s = next
            (t1′, t2′), coeff3 = first(repartition(t, t0, StaticLength(N₁)))
            push!(newtrees, (t1′,t2′)=>coeff1*coeff2*coeff3)
            next = iterate(trees, s)
        end
        return newtrees
    else
        # TODO: implement DegenerateNonAbelian case
        throw(MethodError(permute, (t1, t2, p1, p2)))
    end
end
# TODO: Make permute a @generated function that computes the result ones and stores it
# TODO: take ideas from memoization

# Fusion tree iterators
include("iterator.jl")

# Show methods
function Base.show(io::IO, t::FusionTree{G,N,M,K,Nothing}) where {G<:Sector,N,M,K}
    print(io, "FusionTree{", G, "}(", t.uncoupled, ", ", t.coupled, ", ", t.innerlines, ")")
end
function Base.show(io::IO, t::FusionTree{G}) where {G<:Sector}
    print(io, "FusionTree{", G, "}(", t.uncoupled, ", ", t.coupled, ", ",
                                        t.innerlines, ", ", t.vertices, ")")
end

# auxiliary routines

# _abelianinner: generate the inner indices for given outer indices in the abelian case
_abelianinner(outer::Tuple{}) = ()
_abelianinner(outer::Tuple{G}) where {G<:Sector} =
    outer[1] == one(G) ? () : throw(SectorMismatch())
_abelianinner(outer::Tuple{G,G}) where {G<:Sector} =
    outer[1] == dual(outer[2]) ? () : throw(SectorMismatch())
_abelianinner(outer::Tuple{G,G,G}) where {G<:Sector} =
    first(⊗(outer...)) == one(G) ? () : throw(SectorMismatch())
function _abelianinner(outer::NTuple{N,G}) where {G<:Sector,N}
    c = first(outer[1] ⊗ outer[2])
    return (c, _abelianinner((c, TupleTools.tail2(outer)...))...)
end
