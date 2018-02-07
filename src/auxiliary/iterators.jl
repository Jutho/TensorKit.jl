struct ProductIterator{T<:Tuple}
    iterators::T
end

product(iters...) = ProductIterator(iters)

IteratorSize(::Type{ProductIterator{Tuple{}}}) = HasShape()
IteratorSize(::Type{ProductIterator{T}}) where {T<:Tuple} =
    prod_iteratorsize( IteratorSize(tuple_type_head(T)), IteratorSize(ProductIterator{tuple_type_tail(T)}) )

prod_iteratorsize(::HasLength, ::HasLength) = HasShape()
prod_iteratorsize(::HasLength, ::HasShape) = HasShape()
prod_iteratorsize(::HasShape, ::HasLength) = HasShape()
prod_iteratorsize(::HasShape, ::HasShape) = HasShape()

# products can have an infinite iterator
prod_iteratorsize(::IsInfinite, ::IsInfinite) = IsInfinite()
prod_iteratorsize(a, ::IsInfinite) = IsInfinite()
prod_iteratorsize(::IsInfinite, b) = IsInfinite()
prod_iteratorsize(a, b) = SizeUnknown()

Base.size(P::ProductIterator) = _prod_size(P.iterators)
_prod_size(::Tuple{}) = ()
_prod_size(t::Tuple) = (_prod_size1(t[1], IteratorSize(t[1]))..., _prod_size(tail(t))...)
_prod_size1(a, ::HasShape)  = size(a)
_prod_size1(a, ::HasLength) = (length(a),)
_prod_size1(a, A) =
    throw(ArgumentError("Cannot compute size for object of type $(typeof(a))"))

axes(P::ProductIterator) = _prod_indices(P.iterators)
_prod_indices(::Tuple{}) = ()
_prod_indices(t::Tuple) = (_prod_indices1(t[1], IteratorSize(t[1]))..., _prod_indices(tail(t))...)
_prod_indices1(a, ::HasShape)  = axes(a)
_prod_indices1(a, ::HasLength) = (OneTo(length(a)),)
_prod_indices1(a, A) =
    throw(ArgumentError("Cannot compute indices for object of type $(typeof(a))"))

Base.ndims(p::ProductIterator) = length(axes(p))
Base.length(P::ProductIterator) = prod(size(P))
_length(p::ProductIterator) = prod(map(unsafe_length, axes(p)))

IteratorEltype(::Type{ProductIterator{Tuple{}}}) = HasEltype()
IteratorEltype(::Type{ProductIterator{Tuple{I}}}) where {I} = IteratorEltype(I)
function IteratorEltype(::Type{ProductIterator{T}}) where {T<:Tuple}
    I = tuple_type_head(T)
    P = ProductIterator{tuple_type_tail(T)}
    IteratorEltype(I) == EltypeUnknown() ? EltypeUnknown() : IteratorEltype(P)
end

Base.eltype(::Type{<:ProductIterator{I}}) where {I} = _prod_eltype(I)
_prod_eltype(::Type{Tuple{}}) = Tuple{}
_prod_eltype(::Type{I}) where {I<:Tuple} =
    Base.tuple_type_cons(eltype(tuple_type_head(I)),_prod_eltype(tuple_type_tail(I)))

Base.start(::ProductIterator{Tuple{}}) = false
Base.next(::ProductIterator{Tuple{}}, state) = (), true
Base.done(::ProductIterator{Tuple{}}, state) = state

@inline function Base.start(P::ProductIterator)
    iterators = P.iterators
    iter1 = first(iterators)
    state1 = start(iter1)
    d, states, nvalues = _prod_start(tail(iterators))
    d |= done(iter1, state1)
    return (d, (state1, states...), nvalues)
end
@inline function Base.next(P::ProductIterator, state)
    iterators = P.iterators
    d, states, nvalues = state
    iter1 = first(iterators)
    value1, state1 = next(iter1, states[1])
    tailstates = tail(states)
    values = (value1, map(unsafe_get, nvalues)...) # safe if not done(P, state)
    if done(iter1, state1)
        d, tailstates, nvalues = _prod_next(tail(iterators), tailstates, nvalues)
        if !d # only restart iter1 if not completely done
            state1 = start(iter1)
        end
    end
    return values, (d, (state1, tailstates...), nvalues)
end
Base.done(P::ProductIterator, state) = state[1]

struct MaybeValue{T}
    x::T
    MaybeValue{T}() where {T} = new{T}()
    MaybeValue{T}(x::T) where {T} = new{T}(x)
end

unsafe_get(v::MaybeValue) = v.x

@inline _prod_start(iterators::Tuple{}) = false, (), ()
@inline function _prod_start(iterators)
    iter1 = first(iterators)
    state1 = start(iter1)
    d, tailstates, tailnvalues = _prod_start(tail(iterators))
    if done(iter1, state1)
        d = true
        nvalue1 = MaybeValue{eltype(iter1)}()
    else
        value1, state1 = next(iter1, state1)
        nvalue1 = MaybeValue{eltype(iter1)}(value1)
    end
    return (d, (state1, tailstates...), (nvalue1, tailnvalues...))
end

@inline _prod_next(iterators::Tuple{}, states, nvalues) = true, (), ()
@inline function _prod_next(iterators, states, nvalues)
    iter1 = first(iterators)
    state1 = first(states)
    if !done(iter1, state1)
        value1, state1 = next(iter1, state1)
        nvalue1 = MaybeValue{eltype(iter1)}(value1)
        return false, (state1, tail(states)...), (nvalue1, tail(nvalues)...)
    else
        d, tailstates, tailnvalues = _prod_next(tail(iterators), tail(states), tail(nvalues))
        if d # all iterators are done
            nvalue1 = MaybeValue{eltype(iter1)}()
        else
            value1, state1 = next(iter1, start(iter1)) # iter cannot be done immediately
            nvalue1 = MaybeValue{eltype(iter1)}(value1)
        end
        return d, (state1, tailstates...), (nvalue1, tailnvalues...)
    end
end

struct FilterIterator{F,I}
    flt::F
    itr::I
end

filter(flt, itr) = FilterIterator(flt, itr)

Base.start(f::FilterIterator) = start_filter(f.flt, f.itr)
@inline function start_filter(pred, itr)
    s = start(itr)
    while !done(itr,s)
        v,t = next(itr,s)
        if pred(v)
            return (false, MaybeValue{eltype(itr)}(v), t)
        end
        s=t
    end
    return (true, MaybeValue{eltype(itr)}(), s)
end

Base.next(f::FilterIterator, s) = advance_filter(f.flt, f.itr, s)
@inline function advance_filter(pred, itr, st)
    _, vn, s = st
    v = unsafe_get(vn)
    while !done(itr,s)
        w,t = next(itr,s)
        if pred(w)
            return v, (false, MaybeValue{eltype(itr)}(w), t)
        end
        s=t
    end
    v, (true, MaybeValue{eltype(itr)}(), s)
end

Base.done(f::FilterIterator, s) = s[1]

Base.eltype(::Type{FilterIterator{F,I}}) where {F,I} = eltype(I)
IteratorEltype(::Type{FilterIterator{F,I}}) where {F,I} = IteratorEltype(I)
IteratorSize(::Type{<:FilterIterator}) = Base.SizeUnknown()
