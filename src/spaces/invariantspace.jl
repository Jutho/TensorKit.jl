# InvariantSpace
#--------------
# Tensor product of several ElementarySpace objects
immutable InvariantSpace{S<:UnitaryRepresentationSpace,N} <: TensorSpace{S,N}
    spaces::NTuple{N, S}
    _dims::Dict{NTuple{N,Sector},Int}
    InvariantSpace(spaces::NTuple{N,S})=new(spaces,_computedims(spaces))
end

dim{G<:Sector}(spaces::NTuple{1,UnitaryRepresentationSpace{G}},sectors::NTuple{1,G})=dim(spaces[1],sectors[1])
dim{G<:Sector,N}(spaces::NTuple{N,UnitaryRepresentationSpace{G}},sectors::NTuple{N,G})=dim(spaces[1],sectors[1])*dim(spaces[2:end],sectors[2:end])
function _computesectors(spaces::NTuple{N,UnitaryRepresentationSpace{G}})
    sectors=[tuple(c,s...) for c in sectors(spaces[1]), s in _computesectors(spaces[2:end])]
end


# # Additional constructors
# InvariantSpace{S<:UnitaryRepresentationSpace}(V::S,Vlist::S...) = InvariantSpace(tuple(V,Vlist...))
# InvariantSpace(P::UnitaryRepresentationSpace) = P
# InvariantSpace{S<:UnitaryRepresentationSpace,N}(P::ProductSpace{S,N}) = InvariantSpace{S,N}(P.spaces)

# # Default construction:
# invariant(P::ProductSpace)=InvariantSpace(P)

# # Functionality for extracting and iterating over spaces
# Base.length{S,N}(P::InvariantSpace{S,N}) = N
# Base.endof(P::InvariantSpace) = length(P)
# Base.getindex(P::InvariantSpace, n::Integer) = P.spaces[n]
# Base.getindex{S,N}(P::InvariantSpace{S,N}, r)=ProductSpace{S,length(r)}(P.spaces[r])

# Base.reverse{S,N}(P::InvariantSpace{S,N})=InvariantSpace{S,N}(reverse(P.spaces))
# Base.map(f::Base.Callable,P::InvariantSpace) = map(f,P.spaces) # required to make map(dim,P) efficient

# Base.start(P::InvariantSpace) = start(P.spaces)
# Base.next(P::InvariantSpace, state) = next(P.spaces, state)
# Base.done(P::InvariantSpace, state) = done(P.spaces, state)

# # Corresponding methods
# dim(P::InvariantSpace) = (d=1;for V in P;d*=dim(V);end;return d)
# iscnumber(P::InvariantSpace) = length(P)==0 || all(iscnumber,P)

# # Convention on dual, conj, transpose and ctranspose of tensor product spaces
# dual{S,N}(P::InvariantSpace{S,N}) = InvariantSpace{S,N}(ntuple(N,n->dual(P[n])))
# Base.conj{S,N}(P::InvariantSpace{S,N}) = InvariantSpace{S,N}(ntuple(N,n->conj(P[n])))

# Base.transpose{S,N}(P::InvariantSpace{S,N}) = reverse(P)
# Base.ctranspose{S,N}(P::InvariantSpace{S,N}) = reverse(conj(P))

# # Promotion and conversion
# Base.convert{S<:ElementarySpace}(::Type{InvariantSpace{S,1}}, V::S) = InvariantSpace(V)
# Base.convert{S<:ElementarySpace}(::Type{InvariantSpace{S}}, V::S) = InvariantSpace(V)
# Base.convert(::Type{InvariantSpace}, V::ElementarySpace) = InvariantSpace(V)

# Base.promote_rule{S<:ElementarySpace,N}(::Type{InvariantSpace{S,N}},::Type{S}) = InvariantSpace{S}
# Base.promote_rule{S<:ElementarySpace}(::Type{InvariantSpace{S}},::Type{S}) = InvariantSpace{S}

# ==(P::InvariantSpace,V::ElementarySpace) = length(P) ==1 && P[1] == V
# ==(V::ElementarySpace,P::InvariantSpace) = length(P) ==1 && P[1] == V

# # Show method
# function Base.show(io::IO, P::InvariantSpace)
#   for i in 1:length(P)
#     i==1 || print(io," ⊗ ")
#     show(io, P[i])
#   end
# end

# # basis and basisvector
# typealias ProductBasisVector{S,N} BasisVector{InvariantSpace{S,N},Int} # use integer from 1 to dim as identifier
# typealias ProductBasis{S,N} Basis{InvariantSpace{S,N}}

# Base.length{S,N}(B::ProductBasis{S,N}) = dim(space(B))
# Base.start{S,N}(B::ProductBasis{S,N}) = 1
# Base.next{S,N}(B::ProductBasis{S,N}, state::Int) = (ProductBasisVector{S,N}(space(B),state),state+1)
# Base.done{S,N}(B::ProductBasis{S,N}, state::Int) = state>length(B)

# Base.to_index{S,N}(b::ProductBasisVector{S,N}) = b.identifier # use linear indexing as long as we cannot efficiently generate a cartesian iterator
# Base.show{S,N}(io::IO,b::ProductBasisVector{S,N}) = print(io, "BasisVector($(b.space),$(ind2sub(map(dim,b.space),b.identifier)))")
