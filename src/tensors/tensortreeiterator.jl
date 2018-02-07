struct TensorKeyIterator{G<:Sector,F₁<:FusionTree{G},F₂<:FusionTree{G}}
    rowr::SectorDict{G, FusionTreeDict{F₁, UnitRange{Int}}}
    colr::SectorDict{G, FusionTreeDict{F₂, UnitRange{Int}}}
end
struct TensorPairIterator{G<:Sector, F₁<:FusionTree{G}, F₂<:FusionTree{G}, A<:DenseMatrix}
    rowr::SectorDict{G, FusionTreeDict{F₁, UnitRange{Int}}}
    colr::SectorDict{G, FusionTreeDict{F₂, UnitRange{Int}}}
    data::SectorDict{G, A}
end

const TensorIterator{G<:Sector,F₁<:FusionTree{G},F₂<:FusionTree{G}} = Union{TensorKeyIterator{G,F₁,F₂},TensorPairIterator{G,F₁,F₂}}

IteratorSize(::Type{<:TensorIterator}) = Base.HasLength()
IteratorEltype(::Type{<:TensorIterator}) = Base.HasEltype()
Base.eltype(T::Type{TensorKeyIterator{G,F₁,F₂}}) where {G,F₁,F₂} = Tuple{F₁,F₂}

function Base.length(t::TensorKeyIterator)
    l = 0
    @inbounds for i = 1:length(t.rowr)
        rr = values(t.rowr)[i]
        cr = values(t.colr)[i]
        l += length(rr)*length(cr)
    end
    return l
end
function Base.getindex(t::TensorKeyIterator, i)
    i > 0 || throw(BoundsError)
    i -= 1
    @inbounds for k = 1:length(t.rowr)
        rr = values(t.rowr)[k]
        cr = values(t.colr)[k]
        l = length(rr)*length(cr)
        if i < l
            i2, i1 = divrem(i, length(rr))
            i1 += 1
            i2 += 1
            return keys(rr)[i1], keys(cr)[i2]
        else
            i -= l
        end
    end
    throw(BoundsError())
end

Base.start(it::TensorKeyIterator) = (1,1,1)
function Base.next(it::TensorKeyIterator, s)
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
Base.done(it::TensorKeyIterator, s) = length(it.rowr) < s[1]
