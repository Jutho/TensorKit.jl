struct TensorTreeIterator{G<:Sector,F₁,F₂}
    rowr::VectorDict{G,VectorDict{F₁,UnitRange{Int}}}
    colr::VectorDict{G,VectorDict{F₂,UnitRange{Int}}}
end

Base.iteratorsize(::TensorTreeIterator) = Base.SizeUnknown()
Base.iteratoreltype(::TensorTreeIterator) = Base.HasEltype()
Base.eltype(T::Type{TensorTreeIterator{G,F₁,F₂}}) where {G<:Sector, F₁,F₂} = Tuple{F₁,F₂}

Base.start(it::TensorTreeIterator) = (1,1,1)
function Base.next(it::TensorTreeIterator, s)
    i,j,k = s
    f1 = it.rowr.values[i].keys[j]
    f2 = it.colr.values[i].keys[k]

    if j < length(it.rowr.values[i])
        j += 1
    elseif k < length(it.colr.values[i])
        j = 1
        k += 1
    else
        j = 1
        k = 1
        i += 1
    end

    return (f1,f2), (i,j,k)
end
Base.done(it::TensorTreeIterator, s) = length(it.rowr) < s[1]
