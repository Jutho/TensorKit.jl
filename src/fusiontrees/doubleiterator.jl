# FusionTreeIterator:
# iterate over fusion trees for fixed incoming and outgoing sector labels
#==============================================================================#
struct FusionTreeIterator{G<:Sector,N₁,N₂}
    outgoing::NTuple{N₁,G}
    incoming::NTuple{N₂,G}
end

fusiontrees(outgoing::Tuple{Vararg{G}}, incoming::Tuple{Vararg{G}}) where {G<:Sector} = FusionTreeIterator(outgoing, incoming)

Base.iteratorsize(::FusionTreeIterator) = Base.SizeUnknown()
Base.iteratoreltype(::FusionTreeIterator) = Base.HasEltype()
Base.eltype(T::Type{FusionTreeIterator{G,N₁,N₂}}) where {G<:Sector, N₁, N₂} = _eltype(T, valsub(valadd(Val(N₁),Val(N₂)),Val(3)), valsub(valadd(Val(N₁),Val(N₂)),Val(2)), vertex_labeltype(G))
_eltype(::Type{FusionTreeIterator{G,N₁,N₂}}, ::Val{M}, ::Val{L}, ::Type{T}) where {G<:Sector,N₁,N₂,M,L,T} = FusionTree{G,N₁,N₂,M,L,T}

# * Iterator methods: start, next, done
#   Start with special cases:
#   - Low (N₁,N₂) cases: (0,0), (0,1), (1,0), (1,1)
Base.start(it::FusionTreeIterator{G,0,0} where {G<:Sector}) = false
function Base.next(it::FusionTreeIterator{G,0,0}, state) where {G<:Sector}
    T = vertex_labeltype(G)
    tree = FusionTree{G,0,0,0,0,T}((), (), (), ())
    return tree, true
end
Base.done(it::FusionTreeIterator{G,0,0} where {G<:Sector}, state) = state

Base.start(it::FusionTreeIterator{G,1,0}) where {G<:Sector} = it.outgoing[1] != one(G)
function Base.next(it::FusionTreeIterator{G,1,0}, state) where {G<:Sector}
    T = vertex_labeltype(G)
    tree = FusionTree{G,1,0,0,0,T}(it.outgoing, (), (), ())
    return tree, true
end
Base.done(it::FusionTreeIterator{G,1,0} where {G<:Sector}, state) = state

Base.start(it::FusionTreeIterator{G,0,1}) where {G<:Sector} = it.incoming[1] != one(G)
function Base.next(it::FusionTreeIterator{G,0,1}, state) where {G<:Sector}
    T = vertex_labeltype(G)
    tree = FusionTree{G,0,1,0,0,T}((), it.incoming, (), ())
    return tree, true
end
Base.done(it::FusionTreeIterator{G,0,1} where {G<:Sector}, state) = state

Base.start(it::FusionTreeIterator{G,1,1}) where {G<:Sector} = it.outgoing[1] != it.incoming[1]
function Base.next(it::FusionTreeIterator{G,1,1}, state) where {G<:Sector}
    T = vertex_labeltype(G)
    tree = FusionTree{G,1,1,0,0,T}(it.outgoing,it.incoming,(),())
    return tree, true
end
Base.done(it::FusionTreeIterator{G,1,1} where {G<:Sector}, state) = state

#   - N₁>=2, N₁=0 or 1: only splitting to outgoing indices
Base.start(it::FusionTreeIterator{G,N,0} where {N}) where {G<:Sector} = _start(it.outgoing, one(G))
function Base.next(it::FusionTreeIterator{G,2,0}, state) where {G<:Sector}
    T = vertex_labeltype(G)
    lines, vertices = _nextval(it.outgoing, one(G), state)
    f = FusionTree{G,2,0,0,0,T}(it.outgoing, (), (), ())
    return f, _nextstate(it.outgoing, one(G), state)
end
function Base.next(it::FusionTreeIterator{G,N,0} where {N}, state) where {G<:Sector}
    lines, vertices = _nextval(it.outgoing, one(G), state)
    f = FusionTree(it.outgoing, (), front(front(lines)), front(vertices))
    return f, _nextstate(it.outgoing, one(G), state)
end
Base.done(it::FusionTreeIterator{G,N,0} where {N}, state) where {G<:Sector} = _done(it.outgoing, one(G), state)

Base.start(it::FusionTreeIterator{G,N,1} where {N}) where {G<:Sector} = _start(it.outgoing, it.incoming[1])
function Base.next(it::FusionTreeIterator{G,N,1} where {N}, state) where {G<:Sector}
    lines, vertices = _nextval(it.outgoing, it.incoming[1], state)
    f = FusionTree(it.outgoing, it.incoming, front(lines), vertices)
    return f, _nextstate(it.outgoing, it.incoming[1], state)
end
Base.done(it::FusionTreeIterator{G,N,1} where {N}, state) where {G<:Sector} = _done(it.outgoing, it.incoming[1], state)

#   - N₁ = 0 or 1, N₂ >= 2: only fusion of incoming indices
Base.start(it::FusionTreeIterator{G,0,N} where {N}) where {G<:Sector} = _start(it.incoming, one(G))
Base.done(it::FusionTreeIterator{G,0,N} where {N}, state) where {G<:Sector} = _done(it.incoming, one(G), state)
function Base.next(it::FusionTreeIterator{G,0,2}, state) where {G<:Sector}
    T = vertex_labeltype(G)
    lines, vertices = _nextval(it.incoming, one(G), state)
    f = FusionTree{G,0,2,0,0,T}((), it.incoming, (), ())
    return f, _nextstate(it.incoming, one(G), state)
end
function Base.next(it::FusionTreeIterator{G,0,N} where {N}, state) where {G<:Sector}
    lines, vertices = _nextval(it.incoming, one(G), state)
    f = FusionTree((), it.incoming, tail2(reverse(lines)), tail(reverse(vertices)))
    return f, _nextstate(it.incoming, one(G), state)
end

Base.start(it::FusionTreeIterator{G,1,N} where {N}) where {G<:Sector} = _start(it.incoming, it.outgoing[1])
Base.done(it::FusionTreeIterator{G,1,N} where {N}, state) where {G<:Sector} = _done(it.incoming, it.outgoing[1], state)
function Base.next(it::FusionTreeIterator{G,1,N} where {N}, state) where {G<:Sector}
    lines, vertices = _nextval(it.incoming, it.outgoing[1], state)
    f = FusionTree(it.outgoing, it.incoming, tail(reverse(lines)), reverse(vertices))
    return f, _nextstate(it.incoming, it.outgoing[1], state)
end

#   General case: N₁ >= 2 and N₂ >= 2
function Base.start(it::FusionTreeIterator)
    si = _start(it.incoming, nothing)
    c = last(si) # last entry of state should be the outcoming line label
    so = _start(it.outgoing, c)
    while _done(it.outgoing, c, so)
        si = _nextstate(it.incoming, nothing, si)
        _done(it.incoming, nothing, si) && break
        c = last(si)
        so = _start(it.outgoing, c)
    end
    return (si, so)
end
function Base.next(it::FusionTreeIterator, state)
    si, so = state
    c = last(si)
    # create tree
    flines, fvertices = _nextval(it.incoming, nothing, si)
    slines, svertices = _nextval(it.outgoing, c, so)
    tree = FusionTree(it.outgoing, it.incoming, (slines..., tail(reverse(flines))...), (svertices..., reverse(fvertices)...))
    # create next state
    c = last(si)
    so = _nextstate(it.outgoing, c, so)
    while _done(it.outgoing, c, so)
        si = _nextstate(it.incoming, nothing, si)
        _done(it.incoming, nothing, si) && break
        c = last(si)
        so = _start(it.outgoing, c)
    end
    return tree, (si, so)
end
Base.done(it::FusionTreeIterator, state) = _done(it.incoming, nothing, state[1])

# NOTE: Alternative code below specializes to optimize Abelian case, but in practice the
# speedup seems negligable
#
# Base.start(it::FusionTreeIterator{G,N₁,N₂} where {N₁,N₂}) where {G<:Sector} = _start(it, fusiontype(G))
# Base.next(it::FusionTreeIterator{G,N₁,N₂} where {N₁,N₂}, state) where {G<:Sector} = _next(it, fusiontype(G), state)
# Base.done(it::FusionTreeIterator{G,N₁,N₂} where {N₁,N₂}, state) where {G<:Sector} = _done(it, fusiontype(G), state)
# #   - Optimized version for Abelian case
# function _start(it::FusionTreeIterator,::Type{Abelian})
#     slines = _abeliantree(it.outgoing)
#     flines = _abeliantree(it.incoming)
#     match = last(slines) == last(flines)
#     return (!match, slines, flines)
# end
# function _next(it::FusionTreeIterator,::Type{Abelian}, state)
#     _, slines, flines = state
#     svertices = map(l->nothing, slines)
#     fvertices = map(l->nothing, flines)
#     tree = FusionTree(it.outgoing, it.incoming, slines, svertices, flines, fvertices)
#     return tree, (true, slines, flines)
# end
# _done(it::FusionTreeIterator,::Type{Abelian}, state) = state[1]
#
# @inline _abeliantree(incoming::NTuple{2,G}) where {G<:Sector} = (first(incoming[1] × incoming[2]),)
# @inline function _abeliantree(incoming::NTuple{N,G}) where {N,G<:Sector}
#     a,b, = incoming
#     c = first(a × b)
#     return (c, _abeliantree(tuple(c, tail2(incoming)...))...)
# end
# #   - General non-abelian case
# function _start(it::FusionTreeIterator,::Type{<:Fusion})
#     si = _start(it.incoming, nothing)
#     c = last(si) # last entry of state should be the outcoming line label
#     so = _start(it.outgoing, c)
#     while _done(it.outgoing, c, so)
#         si = _nextstate(it.incoming, nothing, si)
#         _done(it.incoming, nothing, si) && break
#         c = last(si)
#         so = _start(it.outgoing, c)
#     end
#     return (si, so)
# end
# function _next(it::FusionTreeIterator,::Type{<:Fusion}, state)
#     si, so = state
#     c = last(si)
#     # create tree
#     flines, fvertices = _nextval(it.incoming, nothing, si)
#     slines, svertices = _nextval(it.outgoing, c, so)
#     tree = FusionTree(it.outgoing, it.incoming, slines, svertices, flines, fvertices)
#     # create next state
#     c = last(si)
#     so = _nextstate(it.outgoing, c, so)
#     while _done(it.outgoing, c, so)
#         si = _nextstate(it.incoming, nothing, si)
#         _done(it.incoming, nothing, si) && break
#         c = last(si)
#         so = _start(it.outgoing, c)
#     end
#     return tree, (si, so)
# end
# _done(it::FusionTreeIterator,::Type{<:Fusion}, state) = _done(it.incoming, nothing, state[1])

# Actual implementation
function _start(incoming::NTuple{2,G}, out::G) where {G<:Sector}
    a,b = incoming
    return ((1,), out)
end
_nextstate(incoming::NTuple{2,G}, out::G, state) where {G<:Sector} = ((state[1][1]+1,), out)
_done(incoming::NTuple{2,G}, out::G, state) where {G<:Sector} = state[1][1] > Nsymbol(incoming[1],incoming[2], out)

function _start(incoming::NTuple{2,G}, ::Void) where {G<:Sector}
    a,b = incoming
    it = a × b
    s = start(it)
    c, = next(it, s)
    return ((1, s), c) # store final output label as last argument in state
end
function _nextstate(incoming::NTuple{2,G}, ::Void, state) where {G<:Sector}
    a,b = incoming
    it = a × b
    n, s = state[1]
    c = state[2]
    if n == Nsymbol(a, b, c)
        c, s = next(it, s)
        n = 1
        if !done(it, s)
            c, = next(it, s)
        end
    else
        n += 1
    end
    return ((n, s), c)
end
_done(incoming::NTuple{2,G}, ::Void, state) where {G<:Sector} = done(incoming[1] × incoming[2], state[1][2])

_nextval(incoming::NTuple{2,G}, out::Union{G,Void}, state) where {G<:Sector}  = (state[2],), (vertex_ind2label(state[1][1], incoming[1], incoming[2], state[2]),)

function _start(incoming::NTuple{N,G}, out::Union{G,Void}) where {N, G<:Sector}
    a, b, = incoming
    it = a × b
    s = start(it) # done(it1,s1) == false: there should always be at least one fusion output
    c, snext = next(it, s)
    resttree = tuple(c, tail2(incoming)...)
    reststate = _start(resttree, out)
    while _done(resttree, out, reststate)
        s = snext
        done(it, s) && break
        c, snext = next(it, s)
        resttree = tuple(c, tail2(incoming)...)
        reststate = _start(resttree, out)
    end
    return tuple((1, s), reststate...)
end
function _nextstate(incoming::NTuple{N,G}, out::Union{G,Void}, state) where {N, G<:Sector}
    a,b, = incoming
    it = a × b
    n, s = state[1]
    c, snext = next(it, s)
    resttree = tuple(c, tail2(incoming)...)
    reststate = _nextstate(resttree, out, tail(state))
    while _done(resttree, out, reststate)
        if n < Nsymbol(a, b, c)
            n += 1
        else
            s = snext
            done(it, s) && break
            c, snext = next(it, s)
            resttree = tuple(c, tail2(incoming)...)
        end
        reststate = _start(resttree, out)
    end
    return ((n, s), reststate...)
end
function _nextval(incoming::NTuple{N,G}, out::Union{G,Void}, state) where {N,G<:Sector}
    a,b, = incoming
    it = a × b
    n, s = state[1]
    c, = next(it, s)
    resttree = tuple(c, tail2(incoming)...)
    reststate = tail(state)
    lines, vertices = _nextval(resttree, out, reststate)
    return tuple(c, lines...), tuple(vertex_ind2label(n,incoming[1],incoming[2],c), vertices...)
end
_done(incoming::NTuple{N,G}, out::Union{G,Void}, state) where {N, G<:Sector} = done(incoming[1] × incoming[2], state[1][2])
