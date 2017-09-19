"""
    struct ProductSpace{S<:ElementarySpace, N} <: CompositeSpace{S}

A ProductSpace is a tensor product space of `N` vector spaces of type
`S<:ElementarySpace`. Only tensor products between `ElementarySpace` objects of
the same type are allowed.
"""
struct ProductSpace{S<:ElementarySpace, N} <: CompositeSpace{S}
    spaces::NTuple{N, S}
end
ProductSpace(spaces::NTuple{N,S}) where {S<:ElementarySpace, N} = ProductSpace{S,N}(spaces)
ProductSpace(spaces::Vararg{S,N}) where {S<:ElementarySpace, N} = ProductSpace{S,N}(spaces)

# Corresponding methods
#-----------------------
dim(P::ProductSpace{<:ElementarySpace,0}) = 1
dim(P::ProductSpace) = prod(dim, P.spaces)
dim(P::ProductSpace, n::Int) = dim(P.spaces[n], args...)

Base.indices(P::ProductSpace) = CartesianRange(map(indices, P.spaces))
Base.indices(P::ProductSpace, n::Int) = indices(P.spaces[n])

dual(P::ProductSpace{<:ElementarySpace,0}) = P
dual(P::ProductSpace) = ProductSpace(map(dual, reverse(P.spaces)))

# Base.conj(P::ProductSpace) = ProductSpace(map(conj, P.spaces))

function Base.show(io::IO, P::ProductSpace)
    spaces = P.spaces
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
_sectors(P::ProductSpace{<:ElementarySpace, N}, ::Type{Trivial}) where {N} = (ntuple(n->Trivial(),Val{N}()),) # speed up sectors for ungraded spaces
_sectors(P::ProductSpace{<:ElementarySpace, N}, ::Type{<:Sector}) where {N} = product(map(sectors, P.spaces)...)
dim(P::ProductSpace{<:ElementarySpace, N}, sector::NTuple{N, Sector}) where {N} = prod(map(dim, P.spaces, sector))

Base.indices(P::ProductSpace{<:AbstractRepresentationSpace{G}, N}, sectors::NTuple{N, G}) where {G<:Sector, N} =
        CartesianRange(map(indices, P.spaces, sectors))

# Default construction from product of spaces
#---------------------------------------------
⊗(V1::S, V2::S) where {S<:ElementarySpace}= ProductSpace((V1, V2))
⊗(P1::ProductSpace{S}, V2::S) where {S<:ElementarySpace} = ProductSpace(tuple(P1.spaces..., V2))
⊗(V1::S, P2::ProductSpace{S}) where {S<:ElementarySpace} = ProductSpace(tuple(V1, P2.spaces...))
⊗(P1::ProductSpace{S}, P2::ProductSpace{S}) where {S<:ElementarySpace} = ProductSpace(tuple(P1.spaces..., P2.spaces...))
⊗(P::ProductSpace{S}, ::ProductSpace{S,0}) where {S<:ElementarySpace} = P1

# Functionality for extracting and iterating over spaces
#--------------------------------------------------------
Base.length(P::ProductSpace) = length(P.spaces)
Base.getindex(P::ProductSpace, n::Integer) = P.spaces[n]
Base.getindex(P::ProductSpace, r::AbstractVector{<:Integer}) = ProductSpace(P.spaces[r])
#
Base.start(P::ProductSpace) = start(P.spaces)
Base.next(P::ProductSpace, state) = next(P.spaces, state)
Base.done(P::ProductSpace, state) = done(P.spaces, state)

Base.iteratoreltype(P::ProductSpace) = Base.iteratoreltype(P.spaces)
Base.iteratorsize(P::ProductSpace) = Base.iteratorsize(P.spaces)
