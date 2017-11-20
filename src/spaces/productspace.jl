"""
    struct ProductSpace{S<:ElementarySpace, N} <: CompositeSpace{S}

A ProductSpace is a tensor product space of `N` vector spaces of type
`S<:ElementarySpace`. Only tensor products between `ElementarySpace` objects of
the same type are allowed.
"""
struct ProductSpace{S<:ElementarySpace, N} <: CompositeSpace{S}
    spaces::NTuple{N, S}
end
ProductSpace(spaces::Vararg{S,N}) where {S<:ElementarySpace, N} = ProductSpace{S,N}(spaces)
ProductSpace{S,N}(spaces::Vararg{S,N}) where {S<:ElementarySpace, N} = ProductSpace{S,N}(spaces)

# Corresponding methods
#-----------------------
dims(P::ProductSpace) = map(dim, P.spaces)
dim(P::ProductSpace, n::Int) = dim(P.spaces[n])
dim(P::ProductSpace) = reduce(*, 1, dims(P))

Base.indices(P::ProductSpace) = map(indices, P.spaces)
Base.indices(P::ProductSpace, n::Int) = indices(P.spaces[n])

dual(P::ProductSpace{<:ElementarySpace,0}) = P
dual(P::ProductSpace) = ProductSpace(map(dual, reverse(P.spaces)))

# Base.conj(P::ProductSpace) = ProductSpace(map(conj, P.spaces))

function Base.show(io::IO, P::ProductSpace{S}) where {S<:ElementarySpace}
    spaces = P.spaces
    if length(spaces) == 0
        print(io,"ProductSpace{", S, ",0}")
    end
    if length(spaces) == 1
        print(io,"ProductSpace")
    end
    print(io,"(")
    for i in 1:length(spaces)
        i==1 || print(io," ⊗ ")
        show(io, spaces[i])
    end
    print(io,")")
end

# more specific methods
sectors(P::ProductSpace) = _sectors(P, sectortype(P))
_sectors(P::ProductSpace{<:ElementarySpace, N}, ::Type{Trivial}) where {N} = (ntuple(n->Trivial(), StaticLength{N}()),) # speed up sectors for ungraded spaces
_sectors(P::ProductSpace{<:ElementarySpace, N}, ::Type{<:Sector}) where {N} = product(map(sectors, P.spaces)...)

checksectors(V::ProductSpace{<:ElementarySpace,N}, s::NTuple{N}) where {N} = reduce(&, true, map(checksectors, V.spaces, s))

dims(P::ProductSpace{<:ElementarySpace, N}, sector::NTuple{N, Sector}) where {N} = map(dim, P.spaces, sector)
dim(P::ProductSpace{<:ElementarySpace, N}, sector::NTuple{N, Sector}) where {N} = reduce(*, 1, dims(P, sector))

Base.indices(P::ProductSpace{<:ElementarySpace,N}, sectors::NTuple{N, <:Sector}) where {N} = map(indices, P.spaces, sectors)

Base.:(==)(P1::ProductSpace, P2::ProductSpace) = (P1.spaces == P2.spaces)

# Default construction from product of spaces
#---------------------------------------------
⊗(V1::S, V2::S) where {S<:ElementarySpace}= ProductSpace((V1, V2))
⊗(P1::ProductSpace{S}, V2::S) where {S<:ElementarySpace} = ProductSpace(tuple(P1.spaces..., V2))
⊗(V1::S, P2::ProductSpace{S}) where {S<:ElementarySpace} = ProductSpace(tuple(V1, P2.spaces...))
⊗(P1::ProductSpace{S}, P2::ProductSpace{S}) where {S<:ElementarySpace} = ProductSpace(tuple(P1.spaces..., P2.spaces...))
⊗(P::ProductSpace{S,0}, ::ProductSpace{S,0}) where {S<:ElementarySpace} = P
⊗(P::ProductSpace{S}, ::ProductSpace{S,0}) where {S<:ElementarySpace} = P
⊗(::ProductSpace{S,0}, P::ProductSpace{S}) where {S<:ElementarySpace} = P
⊗(V::ElementarySpace) = ProductSpace((V,))
⊗(P::ProductSpace) = P

# unit element with respect to the monoidal structure of taking tensor products
Base.one(::Type{<:ProductSpace{S}}) where {S<:ElementarySpace} = ProductSpace{S,0}(())
Base.one(::Type{S}) where {S<:ElementarySpace} = ProductSpace{S,0}(())
Base.one(V::VectorSpace) = one(typeof(V))

Base.convert(::Type{<:ProductSpace}, V::ElementarySpace) = ProductSpace((V,))
if VERSION <= v"0.6.99"
    Base.literal_pow(::typeof(^), V::ElementarySpace, p::Type{Val{N}}) where {N} = ProductSpace(ntuple(n->V, p))
else
    Base.literal_pow(::typeof(^), V::ElementarySpace, p::Val) = ProductSpace(ntuple(n->V, p))
end

# Functionality for extracting and iterating over spaces
#--------------------------------------------------------
Base.length(P::ProductSpace) = length(P.spaces)
Base.getindex(P::ProductSpace, n::Integer) = P.spaces[n]
Base.getindex(P::ProductSpace{S}, I::NTuple{N,Integer}) where {S<:ElementarySpace,N} = ProductSpace{S,N}(TupleTools.getindices(P.spaces, I))
#
Base.start(P::ProductSpace) = start(P.spaces)
Base.next(P::ProductSpace, state) = next(P.spaces, state)
Base.done(P::ProductSpace, state) = done(P.spaces, state)

Base.iteratoreltype(P::ProductSpace) = Base.iteratoreltype(P.spaces)
Base.iteratorsize(P::ProductSpace) = Base.iteratorsize(P.spaces)