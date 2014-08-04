# Cspace.jl
#
# Defines the immutable ComplexSpace for a standard complex vector space C^n and is
# characterized by its dimension and whether or not it is the dual space. Tensors with
# ComplexSpace as index spaces make a distinction between covariant and contravariant indices.

# EuclideanSpace:
#-----------------
immutable ComplexSpace <: EuclideanSpace{ℂ}
  d::Int
  dual::Bool
  ComplexSpace(d::Int, dual::Bool=false) = (d>0 ? new(d, dual) : throw(ArgumentError("Dimension of a vector space should be bigger than zero")))
end
^(::Type{ℂ},d::Int) = ComplexSpace(d)

# Corresponding methods:
dim(V::ComplexSpace) = V.d
dual(V::ComplexSpace) = ComplexSpace(V.d, !V.dual)
cnumber(V::ComplexSpace) = ComplexSpace(1,V.dual)
cnumber(::Type{ComplexSpace}) = ComplexSpace(1)
iscnumber(V::ComplexSpace) = dim(V)==1

# Show methods
Base.show(io::IO, V::ComplexSpace) = print(io, V.dual ? "ℂ^$(V.d)*" : "ℂ^$(V.d)")

# direct sum of ComplexSpaces
directsum(V1::ComplexSpace, V2::ComplexSpace) = (V1.dual==V2.dual ? ComplexSpace(V1.d+V2.d, V1.dual) : throw(SpaceError("Direct sum of a vector space and its dual do not exist")))

# fusing and splitting ComplexSpaces
fuse(V1::ComplexSpace,V2::ComplexSpace,V::ComplexSpace) = dim(V1)*dim(V2)==dim(V)

# basis and basisvector
typealias ComplexBasisVector BasisVector{ComplexSpace,Int} # use integer from 1 to dim as identifier
typealias ComplexBasis Basis{ComplexSpace}

Base.length(B::ComplexBasis) = dim(space(B))
Base.start(B::ComplexBasis) = 1
Base.next(B::ComplexBasis, state::Int) = (EuclideanBasisVector(space(B),state),state+1)
Base.done(B::ComplexBasis, state::Int) = state>length(B)

Base.to_index(b::ComplexBasisVector) = b.identifier
