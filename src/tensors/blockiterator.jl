"""
    struct BlockIterator{T<:AbstractTensorMap,S}

Iterator over the blocks of type `T`, possibly holding some pre-computed data of type `S`
"""
struct BlockIterator{T<:AbstractTensorMap,S}
    t::T
    structure::S
end

Base.IteratorSize(::BlockIterator) = Base.HasLength()
Base.IteratorEltype(::BlockIterator) = Base.HasEltype()
Base.eltype(::Type{<:BlockIterator{T}}) where {T} = blocktype(T)
Base.length(iter::BlockIterator) = length(iter.structure)
Base.isdone(iter::BlockIterator, state...) = Base.isdone(iter.structure, state...)

Base.haskey(iter::BlockIterator, c) = haskey(iter.structure, c)
