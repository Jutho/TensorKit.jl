# Fusion trees:
#==============================================================================#
struct FusionTree{G<:Sector,N,M,L,T}
    outgoing::NTuple{N,G}
    incoming::G
    innerlines::NTuple{M,G} # M = N-2
    vertices::NTuple{L,T} # L = N-1
end
FusionTree{G,N}(outgoing::NTuple{N}, incoming, innerlines::NTuple{M}, vertices::NTuple{L}) where {G<:Sector,N,M,L} =
    FusionTree{G,N,M,L,vertex_labeltype(G)}(outgoing, incoming, innerlines, vertices)
FusionTree{G,N}(outgoing, incoming, innerlines) where {G<:Sector,N} =
    FusionTree{G,N}(outgoing, incoming, innerlines, ntuple(n->nothing, valsub(Val(N),Val(1))))

FusionTree(outgoing::NTuple{N,G}, incoming::G, innerlines::NTuple{M,G}) where {G<:Sector, N, M} = FusionTree{G,N}(outgoing, incoming, innerlines, ntuple(n->nothing,valsub(Val(N),Val(1))))

function FusionTree(outgoing::NTuple{N,G}, incoming::G) where {G<:Sector, N}
    fusiontype(G) == Abelian || error("cannot create fusion tree without inner lines if `fusiontype(G) <: NonAbelian`")
    FusionTree{G,N}(outgoing, incoming, _abelianinner((outgoing..., dual(incoming))))
end

# Properties
sectortype(::Type{<:FusionTree{G}}) where {G<:Sector} = G
fusiontype(::Type{<:FusionTree{G}}) where {G<:Sector} = fusiontype(G)
Base.length(::Type{<:FusionTree{<:Sector,N}}) where {N} = N

sectortype(t::FusionTree) = sectortype(typeof(t))
fusiontype(t::FusionTree) = fusiontype(typeof(t))
Base.length(t::FusionTree) = length(typeof(t))

# Fusion tree methods
fusiontreetype(::Type{G}, ::Val{0}) where {G<:Sector} = FusionTree{G,0,0,0,vertex_labeltype(G)}
fusiontreetype(::Type{G}, ::Val{1}) where {G<:Sector} = FusionTree{G,1,0,0,vertex_labeltype(G)}
fusiontreetype(::Type{G}, ::Val{2}) where {G<:Sector} = FusionTree{G,2,0,1,vertex_labeltype(G)}
fusiontreetype(::Type{G}, ::Val{N}) where {G<:Sector, N} = _fusiontreetype(G, Val(N), valsub(Val(N),Val(2)), valsub(Val(N),Val(1)))
_fusiontreetype(::Type{G}, ::Val{N}, ::Val{M}, ::Val{L}) where {G<:Sector, N, M, L} = FusionTree{G,N,M,L,vertex_labeltype(G)}

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
        t = FusionTree(tpermute(t.outgoing, p), t.incoming)
        return ImmutableDict(t=>coeff)
    else
        coeff = Rsymbol(one(G), one(G), one(G))
        trees = Dict(t=>coeff)
        newtrees = similar(trees)
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
function braid(t::FusionTree{<:Sector,N}, i) where {N}
    1 <= i < N || throw(ArgumentError("Cannot swap outputs i=$i and i+1 out of only $N outputs"))
    outer = t.outgoing
    inner = t.innerlines
    if i == 1
        a, b = outer[1], outer[2]
        c = N > 2 ? inner[1] : t.incoming
        outer = setindex(outer, b, 1)
        outer = setindex(outer, a, 2)
        return ImmutableDict(FusionTree(outer, t.incoming, inner, t.vertices) => Rsymbol(a, b, c))
    end
    # case i > 1:
    b = outer[i]
    d = outer[i+1]
    a = i == 2 ? outer[1] : inner[i-2]
    c = inner[i-1]
    e = i == N-1 ? t.incoming : inner[i]
    outer′ = setindex(outer, d, i)
    outer′ = setindex(outer′, b, i+1)
    if fusiontype(t) == Abelian
        inner′ = setindex(inner, first(a ⊗ d), i-1)
        return ImmutableDict(FusionTree(outer′, t.incoming, inner′) => Rsymbol(b, d, first(b ⊗ d)))
    elseif fusiontype(t) == SimpleNonAbelian
        iter = a ⊗ d
        s = start(iter)
        c′, s = next(iter, s)
        while iszero(Nsymbol(b, c′, e))
            c′, s = next(iter, s)
        end
        coeff = conj(Rsymbol(b,a,c))*Fsymbol(b,a,d,e,c,c′)*Rsymbol(b,c′,e)
        inner′ = setindex(inner, c′, i-1)
        output = ImmutableDict(FusionTree(outer′, t.incoming, inner′, t.vertices)=>coeff)
        while !done(iter, s)
            c′, s = next(iter, s)
            iszero(Nsymbol(b, c′, e)) && continue
            coeff = conj(Rsymbol(b,a,c))*Fsymbol(b,a,d,e,c,c′)*Rsymbol(b,c′,e)
            inner′ = setindex(inner, c′, i-1)
            output = ImmutableDict(output, FusionTree(outer′, t.incoming, inner′, t.vertices) => coeff)
        end
        return output
    else
        # TODO: implement DegenerateNonAbelian case
        throw(MethodError(braid, (t, i)))
    end
end

# repartition outgoing and incoming fusion tree tree
"""
    function repartition(t1::FusionTree{G,N₁}, t2::FusionTree{G,N₂}, ::Val{N}) where {G,N₁,N₂,N} -> (Immutable)Dict{Tuple{FusionTree{G,N}, FusionTree{G,N₁+N₂-N}},<:Number}

Input is a double fusion tree that describes the fusion of a set of `N₂` incoming charges to
a set of `N₁` outgoing charges, represented using the individual trees of outgoing (`t1`)
and incoming charges (`t2`) respectively (with `t1.incoming==t2.incoming`). Computes new trees
an corresponding coefficients obtained from repartitioning the tree by bending incoming
to outgoing charges (or vice versa) in order to have `N` outgoing charges.
"""
function repartition(t1::FusionTree{G,N₁}, t2::FusionTree{G,N₂}, V::Val{N}) where {G<:Sector, N₁, N₂, N}
    t1.incoming == t2.incoming || throw(SectorMismatch())
    @assert 0 <= N <= N₁+N₂
    V1 = V
    V2 = valsub(valadd(Val(N₁),Val(N₂)),Val(N))

    if fusiontype(t1) == Abelian || fusiontype(t1) == SimpleNonAbelian
        coeff = Bsymbol(one(G), one(G), one(G))
        outer = (t1.outgoing..., map(dual, reverse(t2.outgoing))...)
        inner1ext = isa(Val(N₁), Val{0}) ? () : (isa(Val(N₁), Val{1}) ? (one(G),) : (one(G), first(outer), t1.innerlines...))
        inner2ext = isa(Val(N₂), Val{0}) ? () : (isa(Val(N₂), Val{1}) ? (one(G),) : (one(G), first(outer), t2.innerlines...))
        innerext = (inner1ext..., t1.incoming, reverse(inner2ext)...) # length N₁+N₂+1
        for n = N₁+1:N
             # map fusion vertex c<-(a,b) to splitting vertex (c,dual(b))<-a
            b = dual(outer[n])
            a = innerext[n+1]
            c = innerext[n]
            coeff *= inv(Bsymbol(c, dual(b), a))
        end
        for n = N₁:-1:N+1
            # map splitting vertex (a,b)<-c to fusion vertex a<-(c,dual(b))
            b = outer[n]
            a = innerext[n]
            c = innerext[n+1]
            coeff *= Bsymbol(a,b,c)
        end
        outgoing1 = tselect(outer, ntuple(n->n, V1))
        outgoing2 = tselect(map(dual,outer), ntuple(n->N₁+N₂+1-n, V2))
        innerlines1 = tselect(innerext, ntuple(n->n+2, valsub(V1,Val(2))))
        innerlines2 = tselect(innerext, ntuple(n->N₁+N₂-n, valsub(V2,Val(2))))
        c = innerext[N+1]
        t1′ = FusionTree(outgoing1, c, innerlines1)
        t2′ = FusionTree(outgoing2, c, innerlines2)
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
    p = _linearizepermutation(p1, p2, length(t1), length(t2))
    @assert isperm(p)
    if fusiontype(t1) == Abelian
        (t,t0), coeff1 = first(repartition(t1, t2, valadd(Val(N₁),Val(N₂))))
        t, coeff2 = first(permute(t, p))
        (t1′,t2′), coeff3 = first(repartition(t, t0, Val(N₁)))
        return ImmutableDict((t1′,t2′)=>coeff1*coeff2*coeff3)
    elseif fusiontype(t1) == SimpleNonAbelian
        (t,t0), coeff1 = first(repartition(t1, t2, valadd(Val(N₁),Val(N₂))))
        trees = permute(t, p)
        s = start(trees)
        (t,coeff2), s = next(trees, s)
        (t1′, t2′), coeff3 = first(repartition(t, t0, Val(N₁)))
        newtrees = Dict((t1′,t2′)=>coeff1*coeff2*coeff3)
        while !done(trees, s)
            (t,coeff2), s = next(trees, s)
            (t1′, t2′), coeff3 = first(repartition(t, t0, Val(N₁)))
            push!(newtrees, (t1′,t2′)=>coeff1*coeff2*coeff3)
        end
        return newtrees
    else
        # TODO: implement DegenerateNonAbelian case
        throw(MethodError(permute, (t1, t2, p1, p2)))
    end
end
function _simplerepartition(t1, t2, coeff1, coeff2, ::Val{N}) where {N}
    (t1′,t2′), coeff3 = first(repartition(t1, t2, Val(N)))
    return (t1′,t2′)=>coeff1*coeff2*coeff3
end

function _linearizepermutation(p1::NTuple{N₁,Int}, p2::NTuple{N₂}, n₁::Int, n₂::Int) where {N₁,N₂}
    p1′ = ntuple(Val(N₁)) do n
        p1[n] > n₁ ? n₂+2n₁+1-p1[n] : p1[n]
    end
    p2′ = ntuple(Val(N₂)) do n
        p2[N₂+1-n] > n₁ ? n₂+2n₁+1-p2[N₂+1-n] : p2[N₂+1-n]
    end
    return (p1′..., p2′...)
end

# Fusion tree iterators
include("iterator.jl")

# Show methods
function Base.show(io::IO, t::FusionTree{G,0}) where {G<:Sector}
    if get(io, :compact, false)
        println(io, "FusionTree{", G, ", 0}((), (), (), ())")
    else
        println(io, "FusionTree{", G, ", 0}: (empty fusion tree)")
    end
end
function Base.show(io::IO, t::FusionTree{G,N,M,K,Void}) where {G<:Sector,N,M,K}
    if get(io, :compact, false)
        print(io, "FusionTree{", G, ", ", N, "}(", t.outgoing, ", ", t.incoming, ", ", t.innerlines, ")")
        return
    end
    println(io, "FusionTree{", G, ", ", N, "}:")
    cprint = x->sprint(showcompact, x)
    outlabels = map(cprint, t.outgoing)
    L = max(map(textwidth, outlabels)...)
    up = lpad(rpad(" ∧ ", L>>1), L)
    down = lpad(rpad(" ∨ ", L>>1), L)
    L = textwidth(up)
    f = lpad(rpad("⋅+-",L>>1,"-"), L, " ")
    m = lpad(rpad("-+-",L>>1,"-"), L, "-")
    a = "-<-"
    pad = " "^L
    line1 = ""
    line2 = ""
    line3 = ""
    line4 = ""
    n = 1
    while n <= N
        label = lpad(rpad(outlabels[n], 2), L)
        line4 *= label
        line3 *= down
        if n == 1
            line2 *= f
        else
            line2 *= m
        end
        line1 *= pad
        # print inner label
        if n == 1
            label = cprint(t.outgoing[1])
            label = lpad(rpad(label, 2), textwidth(a))
            line1 *= label
            ℓ = textwidth(label)
            line2 *= lpad(rpad(a, ℓ>>1, "-"), ℓ, "-")
            line3 *= " "^ℓ
            line4 *= " "^ℓ
        elseif n < N
            label = cprint(t.innerlines[n-1])
            label = lpad(rpad(label, 2), textwidth(a))
            line1 *= label
            ℓ = textwidth(label)
            line2 *= lpad(rpad(a, ℓ>>1, "-"), ℓ, "-")
            line3 *= " "^ℓ
            line4 *= " "^ℓ
        else
            line2 *= a
            line2 *= cprint(t.incoming)
        end
        n += 1
    end
    println(io, line1)
    println(io, line2)
    println(io, line3)
    println(io, line4)
end

# auxiliary routines

# _abelianinner: generate the inner indices for given outer indices in the abelian case
_abelianinner(outer::Tuple{}) = ()
_abelianinner(outer::Tuple{G}) where {G<:Sector} = outer[1] == one(G) ? () : throw(SectorMismatch())
_abelianinner(outer::Tuple{G,G}) where {G<:Sector} = outer[1] == dual(outer[2]) ? () : throw(SectorMismatch())
_abelianinner(outer::Tuple{G,G,G}) where {G<:Sector} = first(⊗(outer...)) == one(G) ? () : throw(SectorMismatch())
function _abelianinner(outer::NTuple{N,G}) where {G<:Sector,N}
    c = first(outer[1] ⊗ outer[2])
    return (c, _abelianinner((c, tail2(outer)...))...)
end
