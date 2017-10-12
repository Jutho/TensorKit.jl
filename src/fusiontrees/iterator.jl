# FusionTreeIterator:
# iterate over fusion trees for fixed incoming and outgoing sector labels
#==============================================================================#
function fusiontrees(outgoing::NTuple{N,G}, incoming::G = one(G)) where {N,G<:Sector}
    FusionTreeIterator{G,N}(outgoing, incoming)
end

struct FusionTreeIterator{G<:Sector,N}
    outgoing::NTuple{N,G}
    incoming::G
end

Base.iteratorsize(::FusionTreeIterator) = Base.SizeUnknown()
Base.iteratoreltype(::FusionTreeIterator) = Base.HasEltype()
Base.eltype(T::Type{FusionTreeIterator{G,N}}) where {G<:Sector, N} = fusiontreetype(G, StaticLength(N))

# * Iterator methods: start, next, done
#   Start with special cases:
Base.start(it::FusionTreeIterator{G,0} where {G<:Sector}) = false
function Base.next(it::FusionTreeIterator{G,0}, state) where {G<:Sector}
    T = vertex_labeltype(G)
    tree = FusionTree{G,0,0,0,T}((), one(G), (), ())
    return tree, true
end
Base.done(it::FusionTreeIterator{G,0} where {G<:Sector}, state) = state

Base.start(it::FusionTreeIterator{G,1}) where {G<:Sector} = it.outgoing[1] != it.incoming
function Base.next(it::FusionTreeIterator{G,1}, state) where {G<:Sector}
    T = vertex_labeltype(G)
    tree = FusionTree{G,1,0,0,T}(it.outgoing, it.incoming, (), ())
    return tree, true
end
Base.done(it::FusionTreeIterator{G,1} where {G<:Sector}, state) = state

#   General case:
Base.start(it::FusionTreeIterator{G,N} where {N}) where {G<:Sector} = _start(it.outgoing, it.incoming)
function Base.next(it::FusionTreeIterator{G,N} where {N}, state) where {G<:Sector}
    lines, vertices = _nextval(it.outgoing, it.incoming, state)
    f = FusionTree(it.outgoing, it.incoming, lines, vertices)
    return f, _nextstate(it.outgoing, it.incoming, state)
end
Base.done(it::FusionTreeIterator{G,N} where {N}, state) where {G<:Sector} = _done(it.outgoing, it.incoming, state)

# Actual implementation
function _start(outgoing::NTuple{2,G}, incoming::G) where {G<:Sector}
    a, b = outgoing
    return ((1,),)
end
_nextstate(outgoing::NTuple{2,G}, incoming::G, state) where {G<:Sector} = ((state[1][1]+1,),)
_nextval(outgoing::NTuple{2,G}, incoming::G, state) where {G<:Sector}  = (), (vertex_ind2label(state[1][1], outgoing[1], outgoing[2], incoming),)
_done(outgoing::NTuple{2,G}, incoming::G, state) where {G<:Sector} = state[1][1] > Nsymbol(outgoing[1],outgoing[2], incoming)

function _start(outgoing::NTuple{N,G}, incoming::G) where {N, G<:Sector}
    a, b, = outgoing
    it = a ⊗ b
    s = start(it) # done(it1,s1) == false: there should always be at least one fusion output
    c, snext = next(it, s)
    resttree = tuple(c, tail2(outgoing)...)
    reststate = _start(resttree, incoming)
    while _done(resttree, incoming, reststate)
        s = snext
        done(it, s) && break
        c, snext = next(it, s)
        resttree = tuple(c, tail2(outgoing)...)
        reststate = _start(resttree, incoming)
    end
    return tuple((1, s), reststate...)
end
function _nextstate(outgoing::NTuple{N,G}, incoming::G, state) where {N, G<:Sector}
    a,b, = outgoing
    it = a ⊗ b
    n, s = state[1]
    c, snext = next(it, s)
    resttree = tuple(c, tail2(outgoing)...)
    reststate = _nextstate(resttree, incoming, tail(state))
    while _done(resttree, incoming, reststate)
        if n < Nsymbol(a, b, c)
            n += 1
        else
            s = snext
            done(it, s) && break
            c, snext = next(it, s)
            resttree = tuple(c, tail2(outgoing)...)
        end
        reststate = _start(resttree, incoming)
    end
    return ((n, s), reststate...)
end
function _nextval(outgoing::NTuple{N,G}, incoming::G, state) where {N,G<:Sector}
    a, b, = outgoing
    it = a ⊗ b
    n, s = state[1]
    c, = next(it, s)
    resttree = tuple(c, tail2(outgoing)...)
    reststate = tail(state)
    lines, vertices = _nextval(resttree, incoming, reststate)
    return tuple(c, lines...), tuple(vertex_ind2label(n, outgoing[1], outgoing[2],c), vertices...)
end
_done(outgoing::NTuple{N,G}, incoming::G, state) where {N, G<:Sector} = done(outgoing[1] ⊗ outgoing[2], state[1][2])
