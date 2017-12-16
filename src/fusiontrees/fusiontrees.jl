# Fusion trees:
#==============================================================================#
struct FusionTree{G<:Sector,N,M,L,T}
    outgoing::NTuple{N,G}
    incoming::G
    innerlines::NTuple{M,G} # M = N-2
    vertices::NTuple{L,T} # L = N-1
end
FusionTree{G}(outgoing::NTuple{N}, incoming, innerlines, vertices = ntuple(n->nothing, StaticLength(N)-StaticLength(1))) where {G<:Sector,N} =
    fusiontreetype(G, StaticLength(N))(map(s->convert(G,s),outgoing), convert(G,incoming), map(s->convert(G,s),innerlines), vertices)
FusionTree(outgoing::NTuple{N,G}, incoming::G, innerlines, vertices = ntuple(n->nothing, StaticLength(N)-StaticLength(1))) where {G<:Sector,N} =
    fusiontreetype(G, StaticLength(N))(outgoing, incoming, innerlines, vertices)


function FusionTree{G}(outgoing::NTuple{N}, incoming = one(G)) where {G<:Sector, N}
    fusiontype(G) == Abelian || error("cannot create fusion tree without inner lines if `fusiontype(G) <: NonAbelian`")
    FusionTree{G}(map(s->convert(G,s), outgoing), convert(G, incoming), _abelianinner(map(s->convert(G,s),(outgoing..., dual(incoming)))))
end
function FusionTree(outgoing::NTuple{N,G}, incoming::G = one(G)) where {G<:Sector, N}
    fusiontype(G) == Abelian || error("cannot create fusion tree without inner lines if `fusiontype(G) <: NonAbelian`")
    FusionTree{G}(outgoing, incoming, _abelianinner((outgoing..., dual(incoming))))
end

# Properties
sectortype(::Type{<:FusionTree{G}}) where {G<:Sector} = G
fusiontype(::Type{<:FusionTree{G}}) where {G<:Sector} = fusiontype(G)
Base.length(::Type{<:FusionTree{<:Sector,N}}) where {N} = N

sectortype(t::FusionTree) = sectortype(typeof(t))
fusiontype(t::FusionTree) = fusiontype(typeof(t))
Base.length(t::FusionTree) = length(typeof(t))

# Hashing, important for using fusion trees as key in Dict
function Base.hash(f::FusionTree{G}, h::UInt) where {G}
    if fusiontype(G) == Abelian
        hash(f.outgoing, hash(f.incoming, h))
    elseif fusiontype(G) == SimpleNonAbelian
        hash(f.innerlines, hash(f.outgoing, hash(f.incoming, h)))
    else
        hash(f.vertices, hash(f.innerlines, hash(f.outgoing, hash(f.incoming, h))))
    end
end
function Base.isequal(f1::FusionTree, f2::FusionTree)
    sectortype(f1) == sectortype(f2) || return false
    G = sectortype(f1)
    if fusiontype(G) == Abelian
        f1.incoming == f2.incoming && f1.outgoing == f2.outgoing
    elseif fusiontype(G) == SimpleNonAbelian
        f1.incoming == f2.incoming && f1.outgoing == f2.outgoing && f1.innerlines == f2.innerlines
    else
        f1.incoming == f2.incoming && f1.outgoing == f2.outgoing && f1.innerlines == f2.innerlines && f1.vertices == f2.vertices
    end
end

# Fusion tree methods
fusiontreetype(::Type{G}, ::StaticLength{0}) where {G<:Sector} = FusionTree{G,0,0,0,vertex_labeltype(G)}
fusiontreetype(::Type{G}, ::StaticLength{1}) where {G<:Sector} = FusionTree{G,1,0,0,vertex_labeltype(G)}
fusiontreetype(::Type{G}, ::StaticLength{2}) where {G<:Sector} = FusionTree{G,2,0,1,vertex_labeltype(G)}
fusiontreetype(::Type{G}, ::StaticLength{N}) where {G<:Sector, N} = _fusiontreetype(G, StaticLength(N), StaticLength(N)-StaticLength(2), StaticLength(N)-StaticLength(1))
_fusiontreetype(::Type{G}, ::StaticLength{N}, ::StaticLength{M}, ::StaticLength{L}) where {G<:Sector, N, M, L} = FusionTree{G,N,M,L,vertex_labeltype(G)}

# converting to actual array
function Base.convert(::Type{Array}, f::FusionTree{G,0}) where {G}
    T = eltype(fusiontensor(one(G), one(G), one(G)))
    return fill(one(T), 1)
end
function Base.convert(::Type{Array}, f::FusionTree{G,1}) where {G}
    T = eltype(fusiontensor(one(G), one(G), one(G)))
    return copy!(Matrix{T}(uninitialized, dim(f.incoming), dim(f.incoming)), I)
end
Base.convert(::Type{Array}, f::FusionTree{G,2}) where {G} = fusiontensor(f.outgoing[1], f.outgoing[2], f.incoming, f.vertices[1])
function Base.convert(::Type{Array}, f::FusionTree{G}) where {G}
    tailout = (f.innerlines[1], TupleTools.tail2(f.outgoing)...)
    ftail = FusionTree(tailout, f.incoming, TupleTools.tail(f.innerlines), TupleTools.tail(f.vertices))
    Ctail = convert(Array, ftail)
    C1 = fusiontensor(f.outgoing[1], f.outgoing[2], f.innerlines[1], f.vertices[1])
    dtail = size(Ctail)
    d1 = size(C1)
    C = reshape(C1, d1[1]*d1[2], d1[3])*reshape(Ctail, dtail[1], TupleTools.prod(TupleTools.tail(dtail)))
    return reshape(C, (d1[1], d1[2], TupleTools.tail(dtail)...))
end

# permute fusion tree
"""
    function permute(t::FusionTree{<:Sector,N}, i) where {N} -> (Immutable)Dict{typeof(t),<:Number}

Performs a permutation of the outgoing indices of the fusion tree `t` and returns the result
as a `Dict` (or `ImmutableDict`) of output trees and corresponding coefficients.
"""
function Base.permute(t::FusionTree{G,N}, p::NTuple{N,Int}) where {G<:Sector, N}
    @assert braidingtype(G) <: SymmetricBraiding
    if fusiontype(G) == Abelian
        coeff = Rsymbol(one(G), one(G), one(G))
        for i = 1:N
            for j = 1:i-1
                if p[j] > p[i]
                    a, b = t.outgoing[j], t.outgoing[i]
                    coeff *= Rsymbol(a, b, first(a ⊗ b))
                end
            end
        end
        t = FusionTree{G}(TupleTools.permute(t.outgoing, p), t.incoming)
        return ImmutableDict(t=>coeff)
    else
        coeff = Rsymbol(one(G), one(G), one(G))
        trees = Dict(t=>coeff)
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
    function braid(t::FusionTree{<:Sector,N}, i) where {N} -> ImmutableDict{typeof(t),<:Number}

Performs a braid of neighbouring outgoing indices `i` and `i+1` on a fusion tree `t`,
and returns the result as a linked list of output trees and corresponding coefficients.
"""
function braid(t::FusionTree{G,N}, i) where {G<:Sector, N}
    1 <= i < N || throw(ArgumentError("Cannot swap outputs i=$i and i+1 out of only $N outputs"))
    outer = t.outgoing
    inner = t.innerlines
    if i == 1
        a, b = outer[1], outer[2]
        c = N > 2 ? inner[1] : t.incoming
        outer = TupleTools.setindex(outer, b, 1)
        outer = TupleTools.setindex(outer, a, 2)
        return ImmutableDict(FusionTree{G}(outer, t.incoming, inner, t.vertices) => Rsymbol(a, b, c))
    end
    # case i > 1:
    b = outer[i]
    d = outer[i+1]
    a = i == 2 ? outer[1] : inner[i-2]
    c = inner[i-1]
    e = i == N-1 ? t.incoming : inner[i]
    outer′ = TupleTools.setindex(outer, d, i)
    outer′ = TupleTools.setindex(outer′, b, i+1)
    if fusiontype(G) == Abelian
        inner′ = TupleTools.setindex(inner, first(a ⊗ d), i-1)
        return ImmutableDict(FusionTree{G}(outer′, t.incoming, inner′) => Rsymbol(b, d, first(b ⊗ d)))
    elseif fusiontype(G) == SimpleNonAbelian
        iter = a ⊗ d
        s = start(iter)
        c′, s = next(iter, s)
        while iszero(Nsymbol(b, c′, e))
            c′, s = next(iter, s)
        end
        coeff = conj(Rsymbol(b,a,c))*Fsymbol(b,a,d,e,c,c′)*Rsymbol(b,c′,e)
        inner′ = TupleTools.setindex(inner, c′, i-1)
        output = ImmutableDict(FusionTree{G}(outer′, t.incoming, inner′, t.vertices)=>coeff)
        while !done(iter, s)
            c′, s = next(iter, s)
            iszero(Nsymbol(b, c′, e)) && continue
            coeff = conj(Rsymbol(b,a,c))*Fsymbol(b,a,d,e,c,c′)*Rsymbol(b,c′,e)
            inner′ = TupleTools.setindex(inner, c′, i-1)
            output = ImmutableDict(output, FusionTree{G}(outer′, t.incoming, inner′, t.vertices) => coeff)
        end
        return output
    else
        # TODO: implement DegenerateNonAbelian case
        throw(MethodError(braid, (t, i)))
    end
end

# repartition outgoing and incoming fusion tree
"""
    function repartition(t1::FusionTree{G,N₁}, t2::FusionTree{G,N₂}, ::StaticLength{N}) where {G,N₁,N₂,N} -> (Immutable)Dict{Tuple{FusionTree{G,N}, FusionTree{G,N₁+N₂-N}},<:Number}

Input is a double fusion tree that describes the fusion of a set of `N₂` incoming charges to
a set of `N₁` outgoing charges, represented using the individual trees of outgoing (`t1`)
and incoming charges (`t2`) respectively (with `t1.incoming==t2.incoming`). Computes new trees
an corresponding coefficients obtained from repartitioning the tree by bending incoming
to outgoing charges (or vice versa) in order to have `N` outgoing charges.
"""
function repartition(t1::FusionTree{G,N₁}, t2::FusionTree{G,N₂}, V::StaticLength{N}) where {G<:Sector, N₁, N₂, N}
    t1.incoming == t2.incoming || throw(SectorMismatch())
    @assert 0 <= N <= N₁+N₂
    V1 = V
    V2 = StaticLength(N₁)+StaticLength(N₂)-V

    if fusiontype(t1) == Abelian || fusiontype(t1) == SimpleNonAbelian
        coeff = sqrt(dim(one(G)))*Bsymbol(one(G), one(G), one(G))
        outer = (t1.outgoing..., map(dual, reverse(t2.outgoing))...)
        inner1ext = isa(StaticLength(N₁), StaticLength{0}) ? () : (isa(StaticLength(N₁), StaticLength{1}) ? (one(G),) : (one(G), first(outer), t1.innerlines...))
        inner2ext = isa(StaticLength(N₂), StaticLength{0}) ? () : (isa(StaticLength(N₂), StaticLength{1}) ? (one(G),) : (one(G), dual(last(outer)), t2.innerlines...))
        innerext = (inner1ext..., t1.incoming, reverse(inner2ext)...) # length N₁+N₂+1
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
            coeff *= sqrt(dim(b))*Bsymbol(a,b,c) # for Abelian: sqrt(dim(b)) = sqrt(1) is optimized out
        end
        outgoing1 = TupleTools.getindices(outer, ntuple(n->n, V1))
        outgoing2 = TupleTools.getindices(map(dual,outer), ntuple(n->N₁+N₂+1-n, V2))
        innerlines1 = TupleTools.getindices(innerext, ntuple(n->n+2, V1 - StaticLength(2)))
        innerlines2 = TupleTools.getindices(innerext, ntuple(n->N₁+N₂-n, V2 - StaticLength(2)))
        c = innerext[N+1]
        t1′ = FusionTree{G}(outgoing1, c, innerlines1)
        t2′ = FusionTree{G}(outgoing2, c, innerlines2)
        return ImmutableDict((t1′,t2′)=>coeff)
    else
        # TODO: implement DegenerateNonAbelian case
        throw(MethodError(repartition, (t1, t2, V)))
    end
end

# permute double fusion tree
"""
    function permute(t1::FusionTree{G}, t2::FusionTree{G}, p1::NTuple{N₁,Int}, p2::NTuple{N₂,Int}) where {G,N₁,N₂} -> (Immutable)Dict{Tuple{FusionTree{G,N₁}, FusionTree{G,N₂}},<:Number}

Input is a double fusion tree that describes the fusion of a set of incoming charges to
a set of outgoing charges, represented using the individual trees of outgoing (`t1`)
and incoming charges (`t2`) respectively (with `t1.incoming==t2.incoming`). Computes new trees
and corresponding coefficients obtained from repartitioning and permuting the tree
such that charges `p1` become outgoing and charges `p2` become incoming.
"""
function Base.permute(t1::FusionTree{G}, t2::FusionTree{G}, p1::NTuple{N₁,Int}, p2::NTuple{N₂,Int}) where {G<:Sector, N₁,N₂}
    @assert length(t1) + length(t2) == N₁ + N₂
    p = linearizepermutation(p1, p2, length(t1), length(t2))
    @assert isperm(p)
    if fusiontype(t1) == Abelian
        (t,t0), coeff1 = first(repartition(t1, t2, StaticLength(N₁) + StaticLength(N₂)))
        t, coeff2 = first(permute(t, p))
        (t1′,t2′), coeff3 = first(repartition(t, t0, StaticLength(N₁)))
        return ImmutableDict((t1′,t2′)=>coeff1*coeff2*coeff3)
    elseif fusiontype(t1) == SimpleNonAbelian
        (t,t0), coeff1 = first(repartition(t1, t2, StaticLength(N₁) + StaticLength(N₂)))
        trees = permute(t, p)
        s = start(trees)
        (t, coeff2), s = next(trees, s)
        (t1′, t2′), coeff3 = first(repartition(t, t0, StaticLength(N₁)))
        newtrees = Dict((t1′,t2′)=>coeff1*coeff2*coeff3)
        while !done(trees, s)
            (t, coeff2), s = next(trees, s)
            (t1′, t2′), coeff3 = first(repartition(t, t0, StaticLength(N₁)))
            push!(newtrees, (t1′,t2′)=>coeff1*coeff2*coeff3)
        end
        return newtrees
    else
        # TODO: implement DegenerateNonAbelian case
        throw(MethodError(permute, (t1, t2, p1, p2)))
    end
end

# Fusion tree iterators
include("iterator.jl")

# Show methods
function Base.show(io::IO, t::FusionTree{G,N,M,K,Void}) where {G<:Sector,N,M,K}
    print(io, "FusionTree{", G, "}(", t.outgoing, ", ", t.incoming, ", ", t.innerlines, ")")
end
function Base.show(io::IO, t::FusionTree{G}) where {G<:Sector}
    print(io, "FusionTree{", G, "}(", t.outgoing, ", ", t.incoming, ", ", t.innerlines, ", ", t.vertices, ")")
end

# auxiliary routines

# _abelianinner: generate the inner indices for given outer indices in the abelian case
_abelianinner(outer::Tuple{}) = ()
_abelianinner(outer::Tuple{G}) where {G<:Sector} = outer[1] == one(G) ? () : throw(SectorMismatch())
_abelianinner(outer::Tuple{G,G}) where {G<:Sector} = outer[1] == dual(outer[2]) ? () : throw(SectorMismatch())
_abelianinner(outer::Tuple{G,G,G}) where {G<:Sector} = first(⊗(outer...)) == one(G) ? () : throw(SectorMismatch())
function _abelianinner(outer::NTuple{N,G}) where {G<:Sector,N}
    c = first(outer[1] ⊗ outer[2])
    return (c, _abelianinner((c, TupleTools.tail2(outer)...))...)
end
