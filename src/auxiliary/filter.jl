# taken from Julia's src/base/iterators.jl on recent master, makes this available
# in case of Julia v0.6
module Filter

import Base: start, done, next, eltype, iteratorsize, iteratoreltype
using Base: SizeUnknown


struct FilterIterator{F,I}
    flt::F
    itr::I
end

filter(flt, itr) = FilterIterator(flt, itr)

start(f::FilterIterator) = start_filter(f.flt, f.itr)
function start_filter(pred, itr)
    s = start(itr)
    while !done(itr,s)
        v,t = next(itr,s)
        if pred(v)
            return (false, Nullable{eltype(itr)}(v), t)
        end
        s=t
    end
    return (true, Nullable{eltype(itr)}(), s)
end

next(f::FilterIterator, s) = advance_filter(f.flt, f.itr, s)
function advance_filter(pred, itr, st)
    _, vn, s = st
    v = unsafe_get(vn)
    while !done(itr,s)
        w,t = next(itr,s)
        if pred(w)
            return v, (false, Nullable{eltype(itr)}(w), t)
        end
        s=t
    end
    v, (true, Nullable{eltype(itr)}(), s)
end

done(f::FilterIterator, s) = s[1]

eltype(::Type{FilterIterator{F,I}}) where {F,I} = eltype(I)
iteratoreltype(::Type{FilterIterator{F,I}}) where {F,I} = iteratoreltype(I)
iteratorsize(::Type{<:FilterIterator}) = SizeUnknown()

end
