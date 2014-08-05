# ProductSpace
#--------------
# Tensor product of several ElementarySpace objects
immutable ProductSpace{S<:ElementarySpace,N} <: TensorSpace{S,N}
  spaces::NTuple{N, S}
end

# Functionality for extracting and iterating over spaces
Base.length{S,N}(P::ProductSpace{S,N}) = N
Base.endof(P::ProductSpace) = length(P)
Base.getindex(P::ProductSpace, n::Integer) = P.spaces[n]
Base.getindex{S,N}(P::ProductSpace{S,N}, r)=ProductSpace{S,length(r)}(P.spaces[r])

Base.reverse{S,N}(P::ProductSpace{S,N})=ProductSpace{S,N}(reverse(P.spaces))
Base.map(f::Base.Callable,P::ProductSpace) = map(f,P.spaces) # required to make map(dim,P) efficient

Base.start(P::ProductSpace) = start(P.spaces)
Base.next(P::ProductSpace, state) = next(P.spaces, state)
Base.done(P::ProductSpace, state) = done(P.spaces, state)

# Corresponding methods
dim(P::ProductSpace) = (d=1;for V in P;d*=dim(V);end;return d)
iscnumber(P::ProductSpace) = length(P)==0 || all(iscnumber,P)

# Convention on dual, conj, transpose and ctranspose of tensor product spaces
dual{S,N}(P::ProductSpace{S,N}) = ProductSpace{S,N}(reverse(ntuple(N,n->dual(P[n]))))
Base.conj{S,N}(P::ProductSpace{S,N}) = ProductSpace{S,N}(ntuple(N,n->conj(P[n])))
Base.transpose{S,N}(P::ProductSpace{S,N}) = reverse(P)
Base.ctranspose{S,N}(P::ProductSpace{S,N}) = reverse(conj(P))

# Construct from product of spaces
⊗{S<:ElementarySpace}(V1::S, V2::S) = ProductSpace{S,2}((V1, V2))
⊗{S<:ElementarySpace,N}(P1::ProductSpace{S,N}, V2::S) = ProductSpace{S,N+1}(tuple(P1.spaces..., V2))
⊗{S<:ElementarySpace,N}(V1::S, P2::ProductSpace{S,N}) = ProductSpace{S,N+1}(tuple(V1, P2.spaces...))
⊗{S,N1,N2}(P1::ProductSpace{S,N1}, P2::ProductSpace{S,N2}) = ProductSpace{S,N1+N2}(tuple(P1.spaces..., P2.spaces...))

⊗{S<:ElementarySpace}(V::S) = ProductSpace{S,1}((V,))
⊗{S<:ElementarySpace}(V1::S, V2::S, V3::S) = ProductSpace{S,3}(tuple(V1,V2,V3))
⊗{S<:ElementarySpace}(V1::S, V2::S, V3::S, V4::S) = ProductSpace{S,4}(tuple(V1,V2,V3,V4))
⊗{S<:ElementarySpace}(V1::S, V2::S, V3::S, V4::S, V5::S) = ProductSpace{S,5}(tuple(V1,V2,V3,V4,V5))
⊗{S<:ElementarySpace}(V1::S, V2::S, V3::S, V4::S, V5::S, V6::S) = ProductSpace{S,6}(tuple(V1,V2,V3,V4,V5,V6))
⊗{S<:ElementarySpace}(V1::S, V2::S, V3::S...) = ProductSpace{S,2+length(V3)}(tuple(V1,V2,V3...))

# Promotion and conversion
Base.convert{S<:ElementarySpace}(::Type{ProductSpace{S,1}}, V::S) = prod(V)
Base.convert{S<:ElementarySpace}(::Type{ProductSpace{S}}, V::S) = prod(V)
Base.convert(::Type{ProductSpace}, V::ElementarySpace) = prod(V)

Base.promote_rule{S<:ElementarySpace,N}(::Type{ProductSpace{S,N}},::Type{S}) = ProductSpace{S}
Base.promote_rule{S<:ElementarySpace}(::Type{ProductSpace{S}},::Type{S}) = ProductSpace{S}

==(P::ProductSpace,V::ElementarySpace) = length(P) ==1 && P[1] == V
==(V::ElementarySpace,P::ProductSpace) = length(P) ==1 && P[1] == V

# Show method
function Base.show(io::IO, P::ProductSpace)
  for i in 1:length(P)
    i==1 || print(io," ⊗ ")
    show(io, P[i])
  end
end

# basis and basisvector
typealias ProductBasisVector{S,N} BasisVector{ProductSpace{S,N},Int} # use integer from 1 to dim as identifier
typealias ProductBasis{S,N} Basis{ProductSpace{S,N}}

Base.length{S,N}(B::ProductBasis{S,N}) = dim(space(B))
Base.start{S,N}(B::ProductBasis{S,N}) = 1
Base.next{S,N}(B::ProductBasis{S,N}, state::Int) = (ProductBasisVector{S,N}(space(B),state),state+1)
Base.done{S,N}(B::ProductBasis{S,N}, state::Int) = state>length(B)

Base.to_index{S,N}(b::ProductBasisVector{S,N}) = b.identifier # use linear indexing as long as we cannot efficiently generate a cartesian iterator
Base.show{S,N}(io::IO,b::ProductBasisVector{S,N}) = print(io, "BasisVector($(b.space),$(ind2sub(map(dim,b.space),b.identifier)))")
