# taken from Julia's src/base/iterators.jl on recent master, makes this available
# in case of Julia v0.6
module Product

import Base: start, done, next, isempty, length, size, eltype, iteratorsize, iteratoreltype, indices, ndims
using Base: tail, tuple_type_head, tuple_type_tail, tuple_type_cons, SizeUnknown, HasLength, HasShape,
            IsInfinite, EltypeUnknown, HasEltype, OneTo, @propagate_inbounds

export product

struct ProductIterator{T<:Tuple}
    iterators::T
end

"""
    product(iters...)
Returns an iterator over the product of several iterators. Each generated element is
a tuple whose `i`th element comes from the `i`th argument iterator. The first iterator
changes the fastest.
# Examples
```jldoctest
julia> collect(Iterators.product(1:2,3:5))
2×3 Array{Tuple{Int64,Int64},2}:
 (1, 3)  (1, 4)  (1, 5)
 (2, 3)  (2, 4)  (2, 5)
```
"""
product(iters...) = ProductIterator(iters)

iteratorsize(::Type{ProductIterator{Tuple{}}}) = HasShape()
iteratorsize(::Type{ProductIterator{T}}) where {T<:Tuple} =
    prod_iteratorsize( iteratorsize(tuple_type_head(T)), iteratorsize(ProductIterator{tuple_type_tail(T)}) )

prod_iteratorsize(::Union{HasLength,HasShape}, ::Union{HasLength,HasShape}) = HasShape()
# products can have an infinite iterator
prod_iteratorsize(::IsInfinite, ::IsInfinite) = IsInfinite()
prod_iteratorsize(a, ::IsInfinite) = IsInfinite()
prod_iteratorsize(::IsInfinite, b) = IsInfinite()
prod_iteratorsize(a, b) = SizeUnknown()

size(P::ProductIterator) = _prod_size(P.iterators)
_prod_size(::Tuple{}) = ()
_prod_size(t::Tuple) = (_prod_size1(t[1], iteratorsize(t[1]))..., _prod_size(tail(t))...)
_prod_size1(a, ::HasShape)  = size(a)
_prod_size1(a, ::HasLength) = (length(a),)
_prod_size1(a, A) =
    throw(ArgumentError("Cannot compute size for object of type $(typeof(a))"))

indices(P::ProductIterator) = _prod_indices(P.iterators)
_prod_indices(::Tuple{}) = ()
_prod_indices(t::Tuple) = (_prod_indices1(t[1], iteratorsize(t[1]))..., _prod_indices(tail(t))...)
_prod_indices1(a, ::HasShape)  = indices(a)
_prod_indices1(a, ::HasLength) = (OneTo(length(a)),)
_prod_indices1(a, A) =
    throw(ArgumentError("Cannot compute indices for object of type $(typeof(a))"))

ndims(p::ProductIterator) = length(indices(p))
length(P::ProductIterator) = prod(size(P))
_length(p::ProductIterator) = prod(map(unsafe_length, indices(p)))

iteratoreltype(::Type{ProductIterator{Tuple{}}}) = HasEltype()
iteratoreltype(::Type{ProductIterator{Tuple{I}}}) where {I} = iteratoreltype(I)
function iteratoreltype(::Type{ProductIterator{T}}) where {T<:Tuple}
    I = tuple_type_head(T)
    P = ProductIterator{tuple_type_tail(T)}
    iteratoreltype(I) == EltypeUnknown() ? EltypeUnknown() : iteratoreltype(P)
end

eltype(P::ProductIterator) = _prod_eltype(P.iterators)
_prod_eltype(::Tuple{}) = Tuple{}
_prod_eltype(t::Tuple) = Base.tuple_type_cons(eltype(t[1]),_prod_eltype(tail(t)))

start(::ProductIterator{Tuple{}}) = false
next(::ProductIterator{Tuple{}}, state) = (), true
done(::ProductIterator{Tuple{}}, state) = state

function start(P::ProductIterator)
    iterators = P.iterators
    iter1 = first(iterators)
    state1 = start(iter1)
    d, states, nvalues = _prod_start(tail(iterators))
    d |= done(iter1, state1)
    return (d, (state1, states...), nvalues)
end
function next(P::ProductIterator, state)
    iterators = P.iterators
    d, states, nvalues = state
    iter1 = first(iterators)
    value1, state1 = next(iter1, states[1])
    tailstates = tail(states)
    values = (value1, map(unsafe_get, state[3])...) # safe if not done(P, state)
    if done(iter1, state1)
        d, tailstates, nvalues = _prod_next(tail(iterators), tailstates, nvalues)
        if !d # only restart iter1 if not completely done
            state1 = start(iter1)
        end
    end
    return values, (d, (state1, tailstates...), nvalues)
end
done(P::ProductIterator, state) = state[1]

_prod_start(iterators::Tuple{}) = false, (), ()
function _prod_start(iterators)
    iter1 = first(iterators)
    state1 = start(iter1)
    d, tailstates, tailnvalues = _prod_start(tail(iterators))
    if done(iter1, state1)
        d = true
        nvalue1 = Nullable{eltype(iter1)}()
    else
        value1, state1 = next(iter1, state1)
        nvalue1 = Nullable{eltype(iter1)}(value1)
    end
    return (d, (state1, tailstates...), (nvalue1, tailnvalues...))
end

_prod_next(iterators::Tuple{}, states, nvalues) = true, (), ()
function _prod_next(iterators, states, nvalues)
    iter1 = first(iterators)
    state1 = first(states)
    if !done(iter1, state1)
        value1, state1 = next(iter1, state1)
        nvalue1 = Nullable{eltype(iter1)}(value1)
        return false, (state1, tail(states)...), (nvalue1, tail(nvalues)...)
    else
        d, tailstates, tailnvalues = _prod_next(tail(iterators), tail(states), tail(nvalues))
        if d # all iterators are done
            nvalue1 = Nullable{eltype(iter1)}()
        else
            value1, state1 = next(iter1, start(iter1)) # iter cannot be done immediately
            nvalue1 = Nullable{eltype(iter1)}(value1)
        end
        return d, (state1, tailstates...), (nvalue1, tailnvalues...)
    end
end

end
