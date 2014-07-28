# ProductSpace
#--------------
# Tensor product of several ElementarySpace objects
immutable ProductSpace{N, S<:ElementarySpace} <: VectorSpace
  spaces::NTuple{N, S}
end

# Functionality for extracting and iterating over spaces
Base.length{N}(P::ProductSpace{N}) = N
Base.endof{N}(P::ProductSpace{N}) = N
Base.getindex(P::ProductSpace,n::Integer) = P.spaces[n]
Base.getindex{N,S}(P::ProductSpace{N,S},r)=ProductSpace{length(r),S}(P.spaces[r])

Base.reverse{N,S}(P::ProductSpace{N,S})=ProductSpace{N,S}(reverse(P.spaces))
Base.map(f::Base.Callable,P::ProductSpace) = map(f,P.spaces) # required to make map(dim,P) efficient

Base.start(P::ProductSpace) = start(P.spaces)
Base.next(P::ProductSpace, state) = next(P.spaces, state)
Base.done(P::ProductSpace, state) = done(P.spaces, state)

# Corresponding methods
dim(P::ProductSpace{0}) = 1
dim(P::ProductSpace) = prod(dim, P.spaces)
dual{N,S}(P::ProductSpace{N,S}) = ProductSpace{N,S}(reverse(map(dual, P.spaces)))
iscnumber(P::ProductSpace) = length(P)==0 || all(iscnumber,P.spaces)

*{S<:ElementarySpace}(V1::S, V2::S) = ProductSpace{2,S}((V1, V2))
*{N,S<:ElementarySpace}(P1::ProductSpace{N,S}, V2::S) = ProductSpace{N+1,S}(tuple(P1.spaces..., V2))
*{N,S<:ElementarySpace}(V1::S, P2::ProductSpace{N,S}) = ProductSpace{N+1,S}(tuple(V1, P2.spaces...))
*{N1,N2,S}(P1::ProductSpace{N1,S}, P2::ProductSpace{N2,S}) = ProductSpace{N1+N2,S}(tuple(P1.spaces..., P2.spaces...))

# Show method
function Base.show(io::IO, P::ProductSpace)
  for i in 1:length(P)
    i==1 || print(io," ⊗ ")
    showcompact(io, P[i])
  end
end

# basis and basisvector
typealias ProductBasisVector{N,S} BasisVector{ProductSpace{N,S},Int} # use integer from 1 to dim as identifier
typealias ProductBasis{N,S} Basis{ProductSpace{N,S}}

Base.length{N,S}(B::ProductBasis{N,S}) = dim(space(B))
Base.start{N,S}(B::ProductBasis{N,S}) = 1
Base.next{N,S}(B::ProductBasis{N,S}, state::Int) = (ProductBasisVector{N,S}(space(B),state),state+1)
Base.done{N,S}(B::ProductBasis{N,S}, state::Int) = state>length(B)

Base.to_index{N,S}(b::ProductBasisVector{N,S}) = b.identifier # use linear indexing as long as we cannot efficiently generate a cartesian iterator
Base.show{N,S}(io::IO,b::ProductBasisVector{N,S}) = print(io, "BasisVector($(b.space),$(ind2sub(map(dim,b.space),b.identifier)))")
