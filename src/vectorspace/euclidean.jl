# euclidean.jl
#
# Defines the immutible EuclideanSpace for a euclidean (real or complex) vector space  and is
# characterized by its dimension and whether or not it is the dual space. Tensors with
# EuclideanSpace as index spaces make a distinction between covariant and contravariant indices.

# EuclideanSpace:
#-----------------
immutable EuclideanSpace <: ElementarySpace
  d::Int
  dual::Bool
  EuclideanSpace(d::Int, dual::Bool=false) = (d>0 ? new(d, dual) : throw(ArgumentError("Dimension of a vector space should be bigger than zero")))
end

# Corresponding methods:
dim(V::EuclideanSpace) = V.d
dual(V::EuclideanSpace) = EuclideanSpace(V.d, !V.dual)
iscnumber(V::EuclideanSpace) = dim(V)==1

# Show methods
Base.show(io::IO, V::EuclideanSpace) = print(io, V.dual ? "EuclideanSpace($(V.d))*" : "EuclideanSpace($(V.d))")
Base.showcompact(io::IO, V::EuclideanSpace) = print(io, V.dual ? "E($(V.d))*" : "E($(V.d))")

# direct sum of EuclideanSpace vector spaces
directsum(V1::EuclideanSpace, V2::EuclideanSpace) = (V1.dual==V2.dual ? EuclideanSpace(V1.d+V2.d, V1.dual) : throw(SpaceError("Direct sum of a vector space and its dual do not exist")))

# fusing and splitting EuclideanSpace
fuse(V1::EuclideanSpace,V2::EuclideanSpace,V::EuclideanSpace) = dim(V1)*dim(V2)==dim(V)

# basis and basisvector
typealias EuclideanBasisVector BasisVector{EuclideanSpace,Int} # use integer from 1 to dim as identifier
typealias EuclideanBasis Basis{EuclideanSpace}

Base.length(B::EuclideanBasis) = dim(space(B))
Base.start(B::EuclideanBasis) = 1
Base.next(B::EuclideanBasis, state::Int) = (EuclideanBasisVector(space(B),state),state+1)
Base.done(B::EuclideanBasis, state::Int) = state>length(B)

Base.to_index(b::EuclideanBasisVector) = b.identifier