# ProductSpace
#--------------
# Tensor product of several ElementarySpace objects
immutable ProductSpace{S<:ElementarySpace,N} <: TensorSpace{S,N}
    spaces::NTuple{N, S}
end

# Additional constructors
ProductSpace{S<:ElementarySpace}(V::S,Vlist::S...) = ProductSpace(tuple(V,Vlist...))
ProductSpace(P::ProductSpace) = P

# Corresponding methods
dim(P::ProductSpace) = (d::Int=1;for V in P;d*=dim(V);end;return d)
iscnumber(P::ProductSpace) = all(iscnumber,P)

dim{S<:UnitaryRepresentationSpace,G<:Sector,N}(P::ProductSpace{S,N},s::NTuple{N,G})=_dim(P.spaces,s)
sectors{S<:UnitaryRepresentationSpace}(P::ProductSpace{S})=_sectors(P.spaces)
sectortype{S<:UnitaryRepresentationSpace}(P::ProductSpace{S})=sectortype(S)
sectortype{S<:UnitaryRepresentationSpace,N}(::Type{ProductSpace{S,N}})=sectortype(S)

# Convention on dual, conj, transpose and ctranspose of tensor product spaces
dual{S,N}(P::ProductSpace{S,N}) = ProductSpace{S,N}(ntuple(N,n->dual(P[n])))
Base.conj{S,N}(P::ProductSpace{S,N}) = ProductSpace{S,N}(ntuple(N,n->conj(P[n])))

Base.transpose(P::ProductSpace) = reverse(P)
Base.ctranspose(P::ProductSpace) = reverse(conj(P))

# Default construction from product of spaces:
⊗{S<:ElementarySpace}(V1::S, V2::S) = ProductSpace((V1, V2))
⊗{S<:ElementarySpace}(P1::ProductSpace{S}, V2::S) = ProductSpace(tuple(P1.spaces..., V2))
⊗{S<:ElementarySpace}(V1::S, P2::ProductSpace{S}) = ProductSpace(tuple(V1, P2.spaces...))
⊗{S<:ElementarySpace}(P1::ProductSpace{S}, P2::ProductSpace{S}) = ProductSpace(tuple(P1.spaces..., P2.spaces...))

# Functionality for extracting and iterating over spaces
Base.length{S,N}(P::ProductSpace{S,N}) = N
Base.endof(P::ProductSpace) = length(P)
Base.getindex(P::ProductSpace, n::Integer) = P.spaces[n]
Base.getindex(P::ProductSpace, r)=ProductSpace(P.spaces[r])

Base.reverse(P::ProductSpace)=ProductSpace(reverse(P.spaces))
Base.map(f::Base.Callable,P::ProductSpace) = map(f,P.spaces) # required to make map(dim,P) efficient

Base.start(P::ProductSpace) = start(P.spaces)
Base.next(P::ProductSpace, state) = next(P.spaces, state)
Base.done(P::ProductSpace, state) = done(P.spaces, state)

# Comparison
==(P1::ProductSpace,P2::ProductSpace) = P1.spaces==P2.spaces
==(P::ProductSpace,V::ElementarySpace) = length(P)==1 && P[1] == V
==(V::ElementarySpace,P::ProductSpace) = length(P)==1 && P[1] == V

# Show method
function Base.show(io::IO, P::ProductSpace)
    for i in 1:length(P)
        i==1 || print(io," ⊗ ")
        show(io, P[i])
    end
end

# # Promotion and conversion
# Base.convert{S<:ElementarySpace}(::Type{ProductSpace{S,1}}, V::S) = ProductSpace(V)
# Base.convert{S<:ElementarySpace}(::Type{ProductSpace{S}}, V::S) = ProductSpace(V)
# Base.convert(::Type{ProductSpace}, V::ElementarySpace) = ProductSpace(V)
#
# Base.promote_rule{S<:ElementarySpace,N}(::Type{ProductSpace{S,N}},::Type{S}) = ProductSpace{S}
# Base.promote_rule{S<:ElementarySpace}(::Type{ProductSpace{S}},::Type{S}) = ProductSpace{S}
#

# # basis and basisvector
# typealias ProductBasisVector{S,N} BasisVector{ProductSpace{S,N},Int} # use integer from 1 to dim as identifier
# typealias ProductBasis{S,N} Basis{ProductSpace{S,N}}
#
# Base.length{S,N}(B::ProductBasis{S,N}) = dim(space(B))
# Base.start{S,N}(B::ProductBasis{S,N}) = 1
# Base.next{S,N}(B::ProductBasis{S,N}, state::Int) = (ProductBasisVector{S,N}(space(B),state),state+1)
# Base.done{S,N}(B::ProductBasis{S,N}, state::Int) = state>length(B)
#
# Base.to_index{S,N}(b::ProductBasisVector{S,N}) = b.identifier # use linear indexing as long as we cannot efficiently generate a cartesian iterator
# Base.show{S,N}(io::IO,b::ProductBasisVector{S,N}) = print(io, "BasisVector($(b.space),$(ind2sub(map(dim,b.space),b.identifier)))")
