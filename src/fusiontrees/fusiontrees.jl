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

FusionTree{G}(outgoing::NTuple{N}, incoming, innerlines::NTuple{M}, vertices::NTuple{L}) where {G<:Sector,N,M,L} =
    FusionTree{G,N,M,L,vertex_labeltype(G)}(outgoing, incoming, innerlines, vertices)
FusionTree{G}(outgoing::NTuple{N}, incoming, innerlines) where {G<:Sector,N} =
    FusionTree{G,N}(outgoing, incoming, innerlines, ntuple(n->nothing, valsub(Val(N),Val(1))))

FusionTree(outgoing::NTuple{N,G}, incoming::G, innerlines::NTuple{M,G}) where {G<:Sector, N, M} = FusionTree{G,N}(outgoing, incoming, innerlines, ntuple(n->nothing,valsub(Val(N),Val(1))))

function FusionTree(outgoing::NTuple{N,G}, incoming::G = first(⊗(outgoing...))) where {G<:Sector, N}
    @assert fusiontype(G) == Abelian
    FusionTree{G,N}(outgoing, incoming, _abelianinner((outgoing...,dual(incoming))))
end


# Properties
sectortype(::Type{<:FusionTree{G}}) where {G<:Sector} = G
fusiontype(::Type{<:FusionTree{G}}) where {G<:Sector} = fusiontype(G)
Base.length(::Type{<:FusionTree{<:Sector,N}}) where {N} = N

sectortype(t::FusionTree) = sectortype(typeof(t))
fusiontype(t::FusionTree) = fusiontype(typeof(t))
Base.length(t::FusionTree) = length(typeof(t))

# Fusion tree methods

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
        # TODO: DegenerateNonAbelian case
    end
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
    L = max(map(strwidth, outlabels)...)
    up = lpad(rpad(" ∧ ", L>>1), L)
    down = lpad(rpad(" ∨ ", L>>1), L)
    L = strwidth(up)
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
            label = lpad(rpad(label, 2), strwidth(a))
            line1 *= label
            ℓ = strwidth(label)
            line2 *= lpad(rpad(a, ℓ>>1, "-"), ℓ, "-")
            line3 *= " "^ℓ
            line4 *= " "^ℓ
        elseif n < N
            label = cprint(t.innerlines[n-1])
            label = lpad(rpad(label, 2), strwidth(a))
            line1 *= label
            ℓ = strwidth(label)
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
_abelianinner(outer::Tuple{G}) where {G<:Sector} = ()
_abelianinner(outer::Tuple{G,G}) where {G<:Sector} = ()
_abelianinner(outer::Tuple{G,G,G}) where {G<:Sector} = ()
function _abelianinner(outer::NTuple{N,G}) where {G<:Sector,N}
    c = first(outer[1] ⊗ outer[2])
    return (c, _abelianinner((c, tail2(outer)...))...)
end
