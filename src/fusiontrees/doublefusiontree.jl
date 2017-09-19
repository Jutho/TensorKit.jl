# Fusion trees:
#==============================================================================#
struct FusionTree{G<:Sector,N₁,N₂,M,L,T}
    outgoing::NTuple{N₁,G}
    incoming::NTuple{N₂,G}
    innerlines::NTuple{M,G} # M = N₁+N₂-3
    vertices::NTuple{L,T} # L = N₁+N₂-2
end
FusionTree{G,N₁,N₂}(outgoing::NTuple{N₁}, incoming::NTuple{N₂}, innerlines::NTuple{M}, vertices::NTuple{L}) where {G<:Sector,N₁,N₂,M,L} =
    FusionTree{G,N₁,N₂,M,L,vertex_labeltype(G)}(outgoing, incoming, innerlines, vertices)
FusionTree{G,N₁,N₂}(outgoing, incoming, innerlines) where {G<:Sector,N₁,N₂} =
    FusionTree{G,N₁,N₂}(outgoing, incoming, innerlines, ntuple(n->nothing,valsub(valadd(Val(N₁),Val(N₂)),Val(2))))

FusionTree{G}(outgoing::NTuple{N₁}, incoming::NTuple{N₂}, innerlines::NTuple{M}, vertices::NTuple{L}) where {G<:Sector,N₁,N₂,M,L} =
    FusionTree{G,N₁,N₂,M,L,vertex_labeltype(G)}(outgoing, incoming, innerlines, vertices)
FusionTree{G}(outgoing::NTuple{N₁}, incoming::NTuple{N₂}, innerlines) where {G<:Sector,N₁,N₂} =
    FusionTree{G,N₁,N₂}(outgoing, incoming, innerlines, ntuple(n->nothing,valsub(valadd(Val(N₁),Val(N₂)),Val(2))))

FusionTree(outgoing::NTuple{N₁,G}, incoming::NTuple{N₂,G}, innerlines::NTuple{M,G}) where {G<:Sector, N₁, N₂, M} = FusionTree{G,N₁,N₂}(outgoing,incoming,innerlines,ntuple(n->nothing,valsub(valadd(Val(N₁),Val(N₂)),Val(2))))
FusionTree(outgoing::NTuple{N₁,G}, incoming::NTuple{N₂,G}, innerlines::Tuple{}, vertices::Tuple{}) where {G<:Sector, N₁, N₂} = FusionTree{G,N₁,N₂}(outgoing,incoming,innerlines,vertices)

# TODO: don't require innerlines for fusiontype(G) <: Abelian, remove dependence on fusiontrees in _permute(,,,::Type{Abelian})


# Properties
sectortype(::Type{<:FusionTree{G}}) where {G<:Sector} = G
fusiontype(::Type{<:FusionTree{G}}) where {G<:Sector} = fusiontype(G)
numout(::Type{<:FusionTree{G,N₁,N₂}}) where {G<:Sector,N₁,N₂} = N₁
numin(::Type{<:FusionTree{G,N₁,N₂}}) where {G<:Sector,N₁,N₂} = N₂

sectortype(t::FusionTree) = sectortype(typeof(t))
fusiontype(t::FusionTree) = fusiontype(typeof(t))
numout(t::FusionTree) = numout(typeof(t))
numin(t::FusionTree) = numin(typeof(t))

# Fusion tree methods
# adjoint swaps in and out
Base.adjoint(t::FusionTree) = FusionTree(t.incoming, t.outgoing, reverse(t.innerlines), reverse(t.vertices))

Base.split(t::FusionTree{G,0,0}) where {G<:Sector} = (FusionTree{G,0,1}((),(one(G),),()), FusionTree{G,1,0}((one(G),),(),()))
Base.split(t::FusionTree{G,1,0}) where {G<:Sector} = (FusionTree{G,1,1}(t.outgoing,t.outgoing,()), t)
Base.split(t::FusionTree{G,0,1}) where {G<:Sector} = (t, FusionTree{G,1,1}(t.incoming,t.incoming,()))
Base.split(t::FusionTree{G,1,1}) where {G<:Sector} = (t, t)
Base.split(t::FusionTree{G,2,1}) where {G<:Sector} = (t, FusionTree{G,1,1}(t.incoming,t.incoming,()))
Base.split(t::FusionTree{G,1,2}) where {G<:Sector} = (FusionTree{G,1,1}(t.outgoing,t.outgoing,()), t)
function Base.split(t::FusionTree{<:Sector, N₁, N₂}) where {N₁,N₂} # N₁ + N₂ > 3 and N₁,N₂ >= 2
    inner = t.innerlines
    c = inner[N₁-1]
    inner1 = ntuple(n->inner[n], valsub(Val(N₁),Val(2)))
    inner2 = ntuple(n->inner[N₁-1+n], valsub(Val(N₂),Val(2)))
    vert1 = ntuple(n->t.vertices[n], valsub(Val(N₁),Val(1)))
    vert2 = ntuple(n->t.vertices[N₁-1+n], valsub(Val(N₂),Val(1)))
    return FusionTree(t.outgoing, (c,), inner1, vert1), FusionTree((c,), t.incoming, inner2, vert2)
end

function Base.join(t1::FusionTree{G,N₁,N}, t2::FusionTree{G,N,N₂}) where {G<:Sector,N₁,N,N₂}
    t1a, t1b = split(t1)
    t2a, t2b = split(t2)
    t1b == adjoint(t2a) || throw(SectorMismatch())
    t = FusionTree{G,N₁,N₂}(t1a.outgoing, t2b.incoming, (t1a.innerlines..., t1a.incoming[1], t2b.innerlines...), (t1a.vertices..., t2b.vertices...))
    return t=>sqrt(prod(map(dim,t1b.incoming))/dim(t1b.outgoing[1]))
end

# repartition fusion tree
repartition(t::FusionTree, ::Val{N}) where {N} = _repartition(t, Val(N), fusiontype(t))
function _repartition(t::FusionTree{G,N₁,N₂}, ::Val{N}, F::Union{Type{Abelian},Type{SimpleNonAbelian}}) where {G<:Sector,N₁,N₂,N}
    coeff = Bsymbol(one(G), one(G), one(G))
    outer = (t.outgoing..., map(dual, reverse(t.incoming))...)
    innerext = (one(G), first(outer), t.innerlines..., dual(last(outer)), one(G))
    for n = N₁+1:N
         # map fusion vertex c<-(a,b) to splitting vertex (c,dual(b))<-a
        b = dual(outer[n])
        a = innerext[n+1]
        c = innerext[n]
        coeff *= conj(Bsymbol(c, dual(b), a)) # should be a pure phase
    end
    for n = N₁:-1:N+1
        # map splitting vertex (a,b)<-c to fusion vertex a<-(c,dual(b))
        b = outer[n]
        a = innerext[n]
        c = innerext[n+1]
        coeff *= Bsymbol(a,b,c)
    end
    outgoing = ntuple(n->outer[n], Val(N))
    incoming = ntuple(n->dual(outer[N₁+N₂+1-n]), valsub(valadd(Val(N₁),Val(N₂)),Val(N)))
    return ImmutableDict(FusionTree(outgoing,incoming,t.innerlines,t.vertices)=>coeff)
end
# TODO: implementation for DegenerateNonAbelian

# permute fusion tree
function Base.permute(t::FusionTree, p1::NTuple{N₁,Int}, p2::NTuple{N₂,Int}) where {N₁,N₂}
    @assert numin(t) + numout(t) == N₁ + N₂
    @assert braidingtype(sectortype(t)) <: SymmetricBraiding
    return _permute(t::FusionTree, p1, p2, fusiontype(t))
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

function _permute(tsrc::FusionTree, p1::NTuple{N₁,Int}, p2::NTuple{N₂,Int}, ::Type{Abelian}) where {N₁,N₂}
    p = _linearizepermutation(p1, p2, numout(tsrc), numin(tsrc))
    t, coeff = first(repartition(tsrc, valadd(Val(N₁),Val(N₂))))
    for i = 1:(N₁+N₂)
        for j = 1:i-1
            if p[j] > p[i]
                a, b = t.outgoing[j], t.outgoing[i]
                c = first(a ⊗ b)
                coeff *= Rsymbol(a, b, c)
            end
        end
    end
    t = first(fusiontrees(tpermute(t.outgoing, p),()))
    tdst, coeff2 = first(repartition(t, Val(N₁)))
    return ImmutableDict(tdst=>(coeff*coeff2))
end

function _permute(tsrc::FusionTree, p1::NTuple{N₁,Int}, p2::NTuple{N₂,Int}, ::Type{<:NonAbelian}) where {N₁,N₂}
    p = _linearizepermutation(p1, p2, numout(tsrc), numin(tsrc))
    trees = Dict(t=>c for (t,c) in repartition(tsrc, valadd(Val(N₁),Val(N₂))))
    newtrees = similar(trees)
    for s in permutation2swaps(p)
        for (t, c) in trees
            for (t′,c′) in swap(t, s)
                newtrees[t′] = get(newtrees, t′, zero(c)) + c*c′
            end
        end
        trees, newtrees = newtrees, trees
        empty!(newtrees)
    end
    s = start(trees)
    (t, c), s = next(trees, s)
    tlist = repartition(t, Val(N₁))
    output = Dict{keytype(tlist),valtype(tlist)}()
    for (t′, c′) in tlist
        output[t′] = c*c′
    end
    while !done(trees, s)
        v, s = next(trees, s)
        t, c = v
        for (t′,c′) in repartition(t, Val(N₁))
            output[t′] = get(output, t′, zero(c)) + c*c′
        end
    end
    return output
end

"""
    function swap(t::FusionTree{<:Sector,N,0}, i) where {N} -> ImmutableDict{typeof(t),<:Number}

Performs a swap of neighbouring outgoing indices `i` and `i+1` on a fusion tree `t`,
and returns the result as a linked list of output trees and corresponding coefficients.
"""
function swap(t::FusionTree{<:Sector,N,0}, i) where {N}
    1 <= i < N || throw(ArgumentError("Cannot swap outputs i=$i and i+1 out of only $N outputs"))
    _swap(t, i, fusiontype(t))
end

function _swap(t::FusionTree{<:Sector,N,0}, i, ::Type{Abelian}) where {N}
    outer = t.outgoing
    inner = t.innerlines
    b, d = outer[i], outer[i+1]
    outer = setindex(outer, d, i)
    outer = setindex(outer, b, i+1)
    if i > 1 && i < N-1
        e = (i == N-2 ? dual(outer[N]) : inner[i])
        inner = setindex(inner, first(e ⊗ dual(b)), i-1)
    end
    return ImmutableDict(FusionTree(outer, (), inner) => Rsymbol(b, d, first(b ⊗ d)))
end

function _swap(t::FusionTree{<:Sector,N,0}, i, ::Type{SimpleNonAbelian}) where {N}
    outer = t.outgoing
    inner = t.innerlines
    b = outer[i]
    d = outer[i+1]
    a = (i == 1 ? one(b) : (i == 2 ? outer[1] : inner[i-2]))
    c = (i == 1 ? b : (i == N-1 ? dual(d) : inner[i-1]))
    e = (i == N-1 ? one(b) : (i == N-2 ? dual(outer[N]) : inner[i]))
    outer = setindex(outer, d, i)
    outer = setindex(outer, b, i+1)
    iter = a ⊗ d
    s = start(iter)
    c′, s = next(iter, s)
    while !Nsymbol(b, c′, e)
        c′, s = next(iter, s)
    end
    coeff = conj(Rsymbol(b,a,c))*Fsymbol(b,a,d,e,c,c′)*Rsymbol(b,c′,e)
    inner′ = 1 < i < N-1 ? setindex(inner, c′, i-1) : inner
    output = ImmutableDict(FusionTree(outer, t.incoming, inner′, t.vertices)=>coeff)
    while !done(iter, s)
        c′, s = next(iter, s)
        Nsymbol(b, c′, e) || continue
        coeff = conj(Rsymbol(b,a,c))*Fsymbol(b,a,d,e,c,c′)*Rsymbol(b,c′,e)
        inner′ = 1 < i < N-1 ? setindex(inner, c′, i-1) : inner
        output = ImmutableDict(output, FusionTree(outer, t.incoming, inner′, t.vertices)=>coeff)
    end
    return output
end

# TODO: implementation for DegenerateNonAbelian

# Fusion tree iterators
include("iterator.jl")

# Show methods
function Base.show(io::IO, t::FusionTree{G,0,0}) where {G<:Sector}
    if get(io, :compact, false)
        println(io, "FusionTree{", G, ", 0, 0}((), (), (), ())")
    else
        println(io, "FusionTree{", G, ", 0, 0}: (empty fusion tree)")
    end
end
function Base.show(io::IO, t::FusionTree{G,1,0}) where {G<:Sector}
    if get(io, :compact, false)
        println(io, "FusionTree{", G, ", 1, 0}(", t.outgoing ,"(), (), ())")
    else
        println(io, "FusionTree{", G, ", 1, 0}: (vacuum line)")
    end
end
function Base.show(io::IO, t::FusionTree{G,0,1}) where {G<:Sector}
    if get(io, :compact, false)
        println(io, "FusionTree{", G, ", 0, 1}((), ", t.incoming ,"(), ())")
    else
        println(io, "FusionTree{", G, ", 0, 1}: (dual vacuum line)")
    end
end

function Base.show(io::IO, t::FusionTree{G,N₁,N₂,M,K,Void}) where {G<:Sector,N₁,N₂,M,K}
    if get(io, :compact, false)
        print(io, "FusionTree{", G, ", ", N₁, ", ", N₂, "}(", t.outgoing, ", ", t.incoming, ", ", t.innerlines, ")")
        return
    end
    println(io, "FusionTree{", G, ", ", N₁, ", ", N₂, "}:")
    cprint = x->sprint(showcompact, x)
    outlabels = map(cprint, t.outgoing)
    inlabels = map(cprint, t.incoming)
    L = max(map(strwidth, outlabels)..., map(strwidth, inlabels)...)
    up = lpad(rpad(" ∧ ", L>>1), L)
    down = lpad(rpad(" ∨ ", L>>1), L)
    L = strwidth(up)
    f = lpad(rpad("⋅+-",L>>1,"-"), L, " ")
    l = lpad(rpad("-+⋅",L>>1," "), L, "-")
    m = lpad(rpad("-+-",L>>1,"-"), L, "-")
    a = "-<-"
    pad = " "^L
    line1 = ""
    line2 = ""
    line3 = ""
    line4 = ""
    n = 1
    while n <= N₁ + N₂
        # print incoming or outgoing label
        if n <= N₁
            label = lpad(rpad(outlabels[n], 2), L)
            line4 *= label
            line3 *= down
        else
            label = lpad(rpad(inlabels[N₁+N₂+1-n], 2), L)
            line4 *= label
            line3 *= up
        end
        if n == 1
            line2 *= f
        elseif n == N₁+N₂
            line2 *= l
        else
            line2 *= m
        end
        line1 *= pad
        # print inner label
        if n == 1
            label = N₁ > 0 ? cprint(t.outgoing[1]) : cprint(dual(t.incoming[N₂]))
            label = lpad(rpad(label, 2), strwidth(a))
            line1 *= label
            ℓ = strwidth(label)
            line2 *= lpad(rpad(a, ℓ>>1, "-"), ℓ, "-")
            line3 *= " "^ℓ
            line4 *= " "^ℓ
        elseif n == N₁+N₂-1
            label = N₂ > 0 ? cprint(t.incoming[1]) : cprint(dual(t.outgoing[N₁]))
            label = lpad(rpad(label, 2), strwidth(a))
            line1 *= label
            ℓ = strwidth(label)
            line2 *= lpad(rpad(a, ℓ>>1, "-"), ℓ, "-")
            line3 *= " "^ℓ
            line4 *= " "^ℓ
        elseif n != N₁+N₂
            label = cprint(t.innerlines[n-1])
            label = lpad(rpad(label, 2), strwidth(a))
            line1 *= label
            ℓ = strwidth(label)
            line2 *= lpad(rpad(a, ℓ>>1, "-"), ℓ, "-")
            line3 *= " "^ℓ
            line4 *= " "^ℓ
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
_abelianinner(outer::Tuple{G}) where {G<:Sector} = ()
_abelianinner(outer::Tuple{G,G}) where {G<:Sector} = ()
_abelianinner(outer::Tuple{G,G,G}) where {G<:Sector} = ()
function _abelianinner(outer::NTuple{N,G}) where {G<:Sector,N}
    c = first(outer[1] ⊗ outer[2])
    return (c, _abelianinner((c, tail2(outer)...)))
end
