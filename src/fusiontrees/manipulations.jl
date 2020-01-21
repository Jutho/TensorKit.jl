"""
    function artin_braid(t::FusionTree, i; inv::Bool = false)
        -> <:AbstractDict{typeof(t),<:Number}

Perform an elementary braid (Artin generator) of neighbouring uncoupled indices `i` and
`i+1` on a fusion tree `t`, and returns the result as a dictionary of output trees and
corresponding coefficients.

The keyword `inv` determines whether index `i` will braid above or below index `i+1`, i.e.
applying `artin_braid(t′, i; inv = true)` to all the outputs `t′` of
`artin_braid(t, i; inv = false)` and collecting the results should yield a single fusion
tree with non-zero coefficient, namely `t` with coefficient `1`. This keyword has no effect
if  `BraidingStyle(sectortype(t)) isa SymmetricBraiding`.
"""
function artin_braid(t::FusionTree{G,N}, i; inv::Bool = false) where {G<:Sector, N}
    1 <= i < N ||
        throw(ArgumentError("Cannot swap outputs i=$i and i+1 out of only $N outputs"))
    outer = t.uncoupled
    inner = t.innerlines
    if i == 1
        a, b = outer[1], outer[2]
        c = N > 2 ? inner[1] : t.coupled
        outer = TupleTools.setindex(outer, b, 1)
        outer = TupleTools.setindex(outer, a, 2)
        R = inv ? conj(Rsymbol(b,a,c)) : Rsymbol(a,b,c)
        if FusionStyle(G) isa Abelian
            return SingletonDict(FusionTree{G}(outer, t.coupled, inner, t.vertices) => R)
        elseif FusionStyle(G) isa SimpleNonAbelian
            return FusionTreeDict(FusionTree{G}(outer, t.coupled, inner, t.vertices) => R)
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
        R = inv ? conj(Rsymbol(d, b, first(b ⊗ d))) : Rsymbol(b, d, first(b ⊗ d))
        return SingletonDict(FusionTree{G}(outer′, t.coupled, inner′) => R)
    elseif FusionStyle(G) isa SimpleNonAbelian
        local newtrees
        for c′ in a ⊗ d
            coeff = if inv
                    Rsymbol(a,b,c)*Fsymbol(b,a,d,e,c,c′)*conj(Rsymbol(c′,b,e))
                else
                    conj(Rsymbol(b,a,c))*Fsymbol(b,a,d,e,c,c′)*Rsymbol(b,c′,e)
                end
            inner′ = TupleTools.setindex(inner, c′, i-1)
            t′ = FusionTree{G}(outer′, t.coupled, inner′)
            if coeff != zero(coeff)
                if @isdefined newtrees
                    newtrees[t′] = coeff
                else
                    newtrees = FusionTreeDict(t′ => coeff)
                end
            end
        end
        return newtrees
    else
        # TODO: implement DegenerateNonAbelian case
        throw(MethodError(artin_braid, (t, i)))
    end
end

# braid fusion tree
"""
    function braid(t::FusionTree, levels::NTuple{N,Int}, perm::NTuple{N,Int})
        -> <:AbstractDict{typeof(t),<:Number}

Perform a braiding of the uncoupled indices of the fusion tree `t` and returns the result
as a `<:AbstractDict` of output trees and corresponding coefficients. The braiding is
specified by specifying that index `i` goes to position `perm[i]` and assinging to every
index a distinct level `levels[i]`. This permutation is then decomposed into elementary
swaps between neighbouring indices, where the swaps are applied as braids such that if `i`
and `j` cross, `τ_{i,j}` is applied if `levels[i] < levels[j]` and `τ_{j,i}^{-1}` if
`levels[i] > levels[j]`. This does not allow to encode the most general braid, but a
general braid can be obtained by combining such operations.
"""
function braid(t::FusionTree{G,N}, levels::NTuple{N,Int},
                                    perm::NTuple{N,Int}) where {G<:Sector, N}
    if BraidingStyle(G) isa SymmetricBraiding
        return permute(t, perm) # over or under doesn't matter
    end
    coeff = Rsymbol(one(G), one(G), one(G))
    trees = FusionTreeDict(t=>coeff)
    newtrees = empty(trees)
    for s in permutation2swaps(perm)
        inv = levels[s] < levels[s+1]
        for (t, c) in trees
            for (t′,c′) in artin_braid(t, s; inv = inv)
                newtrees[t′] = get(newtrees, t′, zero(coeff)) + c*c′
            end
        end
        l = levels[s]
        levels = TupleTools.setindex(levels, levels[s+1], s)
        levels = TupleTools.setindex(levels, l, s+1)
        trees, newtrees = newtrees, trees
        empty!(newtrees)
    end
    return trees
end

# permute fusion tree
"""
    function permute(t::FusionTree, p::NTuple{N,Int}) -> <:AbstractDict{typeof(t),<:Number}

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
                for (t′,c′) in artin_braid(t, s)
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
    function split(t::FusionTree{G,N}, ::StaticLength(M))
        -> (::FusionTree{G,M}, ::FusionTree{G,N-M+1})

Split a fusion tree with the first M outgoing indices, and an incoming index corresponding
to the internal fusion tree index between outgoing indices N and N+1 of the original tree
`t`; and a second fusion tree whose first outgoing index is that same internal index. Its
remaining outgoing indices are the N-M outgoing indices of the original tree `t`, and also
the incoming index is the same. This is in the inverse of `insertat` in the sense that if
`t1, t2 = split(t, StaticLength(M)) ⇒ t == insertat(t2, 1, t1)`.
"""
Base.split(t::FusionTree{G,N}, ::StaticLength{N}) where {G,N} =
    (t, FusionTree{G}((t.coupled,), t.coupled, (), ()))
Base.split(t::FusionTree{G,N}, ::StaticLength{1}) where {G,N} =
    (FusionTree{G}((t.uncoupled[1],), t.uncoupled[1], (), ()), t)
function Base.split(t::FusionTree{G,N}, ::StaticLength{0}) where {G,N}
    t1 = FusionTree{G}((), one(G), (), ())
    uncoupled2 = (one(G), t.uncoupled...)
    coupled2 = t.coupled
    innerlines2 = N >= 2 ? (t.uncoupled[1], t.innerlines...) : ()
    if FusionStyle(G) isa DegenerateNonAbelian
        vertices2 = (1, t.vertices...)
        return t1, FusionTree(uncoupled2, coupled2, innerlines2, vertices2)
    else
        return t1, FusionTree(uncoupled2, coupled2, innerlines2)
    end
end
function Base.split(t::FusionTree{G,N}, ::StaticLength{M}) where {G,N,M}
    @assert 1 < M < N
    uncoupled1 = ntuple(n->t.uncoupled[n], Val(M))
    innerlines1 = M>2 ? ntuple(n->t.innerlines[n], Val(M-2)) : ()
    coupled1 = t.innerlines[M-1]
    vertices1 = ntuple(n->t.vertices[n], Val(M-1))
    t1 = FusionTree(uncoupled1, coupled1, innerlines1, vertices1)

    uncoupled2 = (coupled1, ntuple(n->t.uncoupled[M+n], Val(N-M))...)
    innerlines2 = ntuple(n->t.innerlines[M-1+n], Val(N-M-1))
    coupled2 = t.coupled
    vertices2 = ntuple(n->t.vertices[M-1+n], Val(N-M))
    t2 = FusionTree(uncoupled2, coupled2, innerlines2, vertices2)
    return t1, t2
end

"""
    function merge(t1::FusionTree{G,N₁}, t2::FusionTree{G,N₂}, c::G, μ = nothing)
        -> <:AbstractDict{<:FusionTree{G,N₁+N₂},<:Number}

Merge two fusion trees together to a linear combination of fusion trees whose uncoupled
sectors are those of `t1` followed by those of `t2`, and where the two coupled sectors of
`t1` and `t2` are further fused to `c`. In case of
`FusionStyle(G) == DegenerateNonAbelian()`, also a degeneracy label `μ` for the fusion of
the coupled sectors of `t1` and `t2` to `c` needs to be specified.
"""
function Base.merge(t1::FusionTree{G,N₁}, t2::FusionTree{G,N₂},
                    c::G, μ = nothing) where {G,N₁,N₂}
    if !(c in t1.coupled ⊗ t2.coupled)
        throw(SectorMismatch("cannot fuse sectors $(t1.coupled) and $(t2.coupled) to $c"))
    end
    t0 = FusionTree((t1.coupled, t2.coupled), c, (), (μ,))
    t, coeff = first(insertat(t0, 1, t1)) # takes fast path, single output
    @assert coeff == one(coeff)
    return insertat(t, N₁+1, t2)
end
function Base.merge(t1::FusionTree{G,0}, t2::FusionTree{G,0}, c::G, μ =nothing) where {G}
    c == one(G) ||
        throw(SectorMismatch("cannot fuse sectors $(t1.coupled) and $(t2.coupled) to $c"))
    return SingletonDict(t1=>Fsymbol(one(G),one(G),one(G),one(G),one(G),one(G)))
end

"""
    function insertat(t::FusionTree{G,N₁}, i, t2::FusionTree{G,N₂})
        -> <:AbstractDict{<:FusionTree{G,N₁+N₂-1},<:Number}

Attach a fusion tree `t2` to the uncoupled leg `i` of the fusion tree `t1` and bring it
into a linear combination of fusion trees in standard form. This requires that
`t2.coupled == t1.uncoupled[i]`.
"""
function insertat(t1::FusionTree{G}, i, t2::FusionTree{G,0}) where {G}
    # this actually removes uncoupled line i
    t1.uncoupled[i] == t2.coupled || # t2.coupled = one(G)
        throw(SectorMismatch("cannot connect $(t2.uncoupled) to $(t1.uncoupled[i])"))
    coeff = Fsymbol(one(G), one(G), one(G), one(G), one(G), one(G))

    outer = TupleTools.deleteat(t1.uncoupled, i)
    inner = TupleTools.deleteat(t1.innerlines, max(1,i-2))
    vertices = TupleTools.deleteat(t1.vertices, max(1, i-1))
    t = FusionTree(outer, t1.coupled, inner, vertices)
    if FusionStyle(G) isa Abelian
        return SingletonDict(t => coeff)
    elseif FusionStyle(G) isa SimpleNonAbelian
        return FusionTreeDict(t => coeff)
    end
end
function insertat(t1::FusionTree{G}, i, t2::FusionTree{G,1}) where {G}
    # identity operation
    t1.uncoupled[i] == t2.coupled ||
        throw(SectorMismatch("cannot connect $(t2.uncoupled) to $(t1.uncoupled[i])"))
    coeff = Fsymbol(one(G), one(G), one(G), one(G), one(G), one(G))
    if FusionStyle(G) isa Abelian
        return SingletonDict(t1 => coeff)
    elseif FusionStyle(G) isa SimpleNonAbelian
        return FusionTreeDict(t1 => coeff)
    end
end
function insertat(t1::FusionTree{G}, i, t2::FusionTree{G,2}) where {G}
    t1.uncoupled[i] == t2.coupled ||
        throw(SectorMismatch("cannot connect $(t2.uncoupled) to $(t1.uncoupled[i])"))
    return _insertat(t1, i, t2.uncoupled[1], t2.uncoupled[2], t2.vertices[1])
end
function insertat(t1::FusionTree{G}, i, t2::FusionTree{G}) where {G}
    t1.uncoupled[i] == t2.coupled ||
        throw(SectorMismatch("cannot connect $(t2.uncoupled) to $(t1.uncoupled[i])"))
    if length(t1) == 1
        coeff = Fsymbol(one(G), one(G), one(G), one(G), one(G), one(G))
        if FusionStyle(G) isa Abelian
            return SingletonDict(t2 => coeff)
        elseif FusionStyle(G) isa SimpleNonAbelian
            return FusionTreeDict(t2 => coeff)
        end
    end
    if i == 1
        outer = (t2.uncoupled..., tail(t1.uncoupled)...)
        inner = (t2.innerlines..., t2.coupled, t1.innerlines...)
        vertices = (t2.vertices..., t1.vertices...)
        coupled = t1.coupled
        t′ = FusionTree(outer, coupled, inner, vertices)
        coeff = Fsymbol(one(G), one(G), one(G), one(G), one(G), one(G))
        if FusionStyle(G) isa Abelian
            return SingletonDict(t′ => coeff)
        elseif FusionStyle(G) isa SimpleNonAbelian
            return FusionTreeDict(t′ => coeff)
        end
    end
    b = t2.innerlines[end]
    c = t2.uncoupled[end]
    v = t2.vertices[end]
    t2′ = FusionTree(front(t2.uncoupled), b, front(t2.innerlines), front(t2.vertices))
    if FusionStyle(G) isa Abelian
        t, coeff = first(_insertat(t1, i, b, c, v))
        t′, coeff′ = first(insertat(t, i, t2′))
        return SingletonDict(t′=>coeff*coeff′)
    else
        local newtrees
        for (t, coeff) in _insertat(t1, i, b, c, v)
            if @isdefined newtrees
                for (t′, coeff′) in insertat(t, i, t2′)
                    newtrees[t′] = get(newtrees, t′, zero(coeff′)) + coeff*coeff′
                end
            else
                newtrees = insertat(t, i, t2′)
                for (t′, coeff′) in newtrees
                    newtrees[t′] = coeff*coeff′
                end
            end
        end
        return newtrees
    end
end

function _insertat(t::FusionTree{G,N}, i, b::G, c::G, v = nothing) where {G,N}
    1 <= i <= N || throw(ArgumentError("Cannot attach to output $i of only $N outputs"))
    outer = t.uncoupled
    iszero(Nsymbol(b,c,outer[i])) &&
        throw(ArgumentError("Cannot attach $b and $c to $(outer[i])"))
    inner = t.innerlines
    if i == 1
        outer′ = (b, c, tail(outer)...)
        inner′ = (outer[1], inner...)
        vertices′ = (t.vertices..., v)
        coeff = Fsymbol(one(G), one(G), one(G), one(G), one(G), one(G))
        t′ = FusionTree(outer′, t.coupled,  inner′, vertices′)
        if FusionStyle(G) isa Abelian
            return SingletonDict(t′ => coeff)
        elseif FusionStyle(G) isa SimpleNonAbelian
            return FusionTreeDict(t′ => coeff)
        end
    end
    outer′ = TupleTools.insertafter(TupleTools.setindex(outer, b, i), i, (c,))
    a = i == 2 ? outer[1] : inner[i-2]
    d = i == N ? t.coupled : inner[i-1]
    f = outer[i]
    if FusionStyle(G) isa Abelian
        e = first(a ⊗ b)
        inner′ = TupleTools.insertafter(inner, i-2, (e,))
        t′ = FusionTree(outer′, t.coupled, inner′)
        coeff = conj(Fsymbol(a,b,c,d,e,f))
        return SingletonDict(t′ => coeff)
    elseif FusionStyle(G) isa SimpleNonAbelian
        local newtrees
        for e in a ⊗ b
            inner′ = TupleTools.insertafter(inner, i-2, (e,))
            t′ = FusionTree(outer′, t.coupled, inner′)
            coeff = conj(Fsymbol(a,b,c,d,e,f))
            if coeff != zero(coeff)
                if @isdefined newtrees
                    newtrees[t′] = coeff
                else
                    newtrees = FusionTreeDict(t′ => coeff)
                end
            end
        end
        return newtrees
    else
        # TODO: implement DegenerateNonAbelian case
        throw(MethodError(_insertat, (t, i, b, c, v)))
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
const permutecache = LRU{Any,Any}(; maxsize = 10^5)
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
                    p1::IndexTuple{N₁}, p2::IndexTuple{N₂}) where {G<:Sector,N₁,N₂}
    @assert length(t1) + length(t2) == N₁ + N₂
    @assert TupleTools.isperm((p1..., p2...))
    if FusionStyle(G) isa Abelian
        u = one(G)
        T = typeof(sqrt(dim(u))*Fsymbol(u,u,u,u,u,u))
        F₁ = fusiontreetype(G, StaticLength(N₁))
        F₂ = fusiontreetype(G, StaticLength(N₂))
        D = SingletonDict{Tuple{F₁,F₂}, T}
    else
        u = one(G)
        T = typeof(sqrt(dim(u))*Fsymbol(u,u,u,u,u,u))
        F₁ = fusiontreetype(G, StaticLength(N₁))
        F₂ = fusiontreetype(G, StaticLength(N₂))
        D = FusionTreeDict{Tuple{F₁,F₂}, T}
    end
    return _get_permute(D, (t1, t2, p1, p2))
end

function _get_permute(::Type{D}, @nospecialize(key)) where D
    d::D = get!(permutecache, key) do
        _permute(key)
    end
    return d
end

const PermuteFusionTreeKey{G<:Sector,N₁,N₂} =
    Tuple{<:FusionTree{G},<:FusionTree{G},IndexTuple{N₁},IndexTuple{N₂}}
function _permute((t1, t2, p1, p2)::PermuteFusionTreeKey{G,N₁,N₂}) where {G<:Sector,N₁,N₂}
    p = linearizepermutation(p1, p2, length(t1), length(t2))
    if FusionStyle(t1) isa Abelian
        (t,t0), coeff1 = first(repartition(t1, t2, StaticLength(N₁) + StaticLength(N₂)))
        t, coeff2 = first(permute(t, p))
        (t1′,t2′), coeff3 = first(repartition(t, t0, StaticLength(N₁)))
        return SingletonDict((t1′,t2′)=>coeff1*coeff2*coeff3)
    elseif FusionStyle(t1) isa SimpleNonAbelian
        (t,t0), coeff1 = first(repartition(t1, t2, StaticLength(N₁) + StaticLength(N₂)))
        local newtrees
        for (t, coeff2) in permute(t, p)
            (t1′, t2′), coeff3 = first(repartition(t, t0, StaticLength(N₁)))
            if @isdefined newtrees
                newtrees[(t1′,t2′)] = coeff1*coeff2*coeff3
            else
                newtrees = FusionTreeDict((t1′,t2′)=>coeff1*coeff2*coeff3)
            end
        end
        return newtrees
    else
        # TODO: implement DegenerateNonAbelian case
        throw(MethodError(permute, (t1, t2, p1, p2)))
    end
end

function braid(t1::FusionTree{G}, t2::FusionTree{G},
                    levels1::IndexTuple, levels2::IndexTuple,
                    p1::IndexTuple{N₁}, p2::IndexTuple{N₂}) where {G<:Sector,N₁,N₂}
    @assert length(t1) + length(t2) == N₁ + N₂
    p = linearizepermutation(p1, p2, length(t1), length(t2))
    levels = (levels1..., reverse(levels2)...)
    if FusionStyle(t1) isa Abelian
        (t,t0), coeff1 = first(repartition(t1, t2, StaticLength(N₁) + StaticLength(N₂)))
        t, coeff2 = first(braid(t, p, levels))
        (t1′,t2′), coeff3 = first(repartition(t, t0, StaticLength(N₁)))
        return SingletonDict((t1′,t2′)=>coeff1*coeff2*coeff3)
    elseif FusionStyle(t1) isa SimpleNonAbelian
        (t,t0), coeff1 = first(repartition(t1, t2, StaticLength(N₁) + StaticLength(N₂)))
        local newtrees
        for (t, coeff2) in braid(t, levels, p)
            (t1′, t2′), coeff3 = first(repartition(t, t0, StaticLength(N₁)))
            if @isdefined newtrees
                newtrees[(t1′,t2′)] = coeff1*coeff2*coeff3
            else
                newtrees = FusionTreeDict((t1′,t2′)=>coeff1*coeff2*coeff3)
            end
        end
        return newtrees
    else
        # TODO: implement DegenerateNonAbelian case
        throw(MethodError(permute, (t1, t2, p1, p2)))
    end
end
