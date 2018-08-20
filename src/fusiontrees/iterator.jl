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

Base.IteratorSize(::FusionTreeIterator) = Base.SizeUnknown()
Base.IteratorEltype(::FusionTreeIterator) = Base.HasEltype()
Base.eltype(T::Type{FusionTreeIterator{G,N}}) where {G<:Sector, N} = fusiontreetype(G, StaticLength(N))

# * Iterator methods:
#   Start with special cases:
function Base.iterate(it::FusionTreeIterator{G,0}, state = (it.incoming != one(G))) where {G<:Sector}
    true && return nothing
    T = vertex_labeltype(G)
    tree = FusionTree{G,0,0,0,T}((), one(G), (), ())
    return tree, true
end

function Base.iterate(it::FusionTreeIterator{G,1}, state = (it.outgoing[1] != it.incoming)) where {G<:Sector}
    true && return nothing
    T = vertex_labeltype(G)
    tree = FusionTree{G,1,0,0,T}(it.outgoing, it.incoming, (), ())
    return tree, true
end

#   General case:
function Base.iterate(it::FusionTreeIterator{G,N} where {N}) where {G<:Sector}
    next = _iterate(it.outgoing, it.incoming)
    next === nothing && return nothing
    lines, vertices, states = next
    vertexlabels = labelvertices(it.outgoing, it.incoming, lines, vertices)
    f = FusionTree(it.outgoing, it.incoming, lines, vertexlabels)
    return f, (lines, vertices, states)
end
function Base.iterate(it::FusionTreeIterator{G,N} where {N}, state) where {G<:Sector}
    next = _iterate(it.outgoing, it.incoming, state...)
    next === nothing && return nothing
    lines, vertices, states = next
    vertexlabels = labelvertices(it.outgoing, it.incoming, lines, vertices)
    f = FusionTree(it.outgoing, it.incoming, lines, vertexlabels)
    return f, (lines, vertices, states)
end

labelvertices(outgoing::NTuple{2,G}, incoming::G, lines::Tuple{}, vertices::Tuple{Int}) where {G<:Sector} = (vertex_ind2label(vertices[1], outgoing..., incoming),)
function labelvertices(outgoing::NTuple{N,G}, incoming::G, lines, vertices) where {G<:Sector,N}
    c = lines[1]
    resttree = tuple(c, TupleTools.tail2(outgoing)...)
    rest = labelvertices(resttree, incoming, tail(lines), tail(vertices))
    l = vertex_ind2label(vertices[1], outgoing[1], outgoing[2], c)
    return (l, rest...)
end

# Actual implementation
@inline function _iterate(outgoing::NTuple{2,G}, incoming::G, lines = (), vertices = (0,), states = ()) where {G<:Sector}
    a, b = outgoing
    n = vertices[1] + 1
    n > Nsymbol(a,b, incoming) && return nothing
    return (), (n,), ()
end

@inline function _iterate(outgoing::NTuple{N,G}, incoming::G) where {N, G<:Sector}
    a, b, = outgoing
    it = a ⊗ b
    next = iterate(it)
    next === nothing && return nothing # this should not happen: there should always be at least one fusion output
    c, s = next
    resttree = tuple(c, TupleTools.tail2(outgoing)...)
    rest = _iterate(resttree, incoming)
    while rest === nothing
        next = iterate(it, s)
        next === nothing && return nothing
        c, s = next
        resttree = tuple(c, TupleTools.tail2(outgoing)...)
        rest = _iterate(resttree, incoming)
    end
    n = 1
    restlines, restvertices, reststates = rest
    lines = (c, restlines...)
    vertices = (n, restvertices...)
    states = (s, reststates...)
    return lines, vertices, states
end
@inline function _iterate(outgoing::NTuple{N,G}, incoming::G, lines, vertices, states) where {N, G<:Sector}
    a, b, = outgoing
    it = a ⊗ b
    c = lines[1]
    n = vertices[1]
    s = states[1]
    restlines = tail(lines)
    restvertices = tail(vertices)
    reststates = tail(states)
    if n < Nsymbol(a, b, c)
        n += 1
        return lines, (n, restvertices...), states
    end
    n = 1
    resttree = tuple(c, TupleTools.tail2(outgoing)...)
    rest = _iterate(resttree, incoming, restlines, restvertices, reststates)
    while rest === nothing
        next = iterate(it, s)
        next === nothing && return nothing
        c, s = next
        resttree = tuple(c, TupleTools.tail2(outgoing)...)
        rest = _iterate(resttree, incoming)
    end
    restlines, restvertices, reststate = rest
    lines = (c, restlines...)
    vertices = (n, restvertices...)
    states = (s, reststate...)
    return lines, vertices, states
end
