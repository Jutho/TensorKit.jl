# cartesianspace.jl
#
# Defines the immutable CartesianSpace for denoting a self-dual vector space. CartesianSpace is completely
# characterised by its dimension. Tensors with CartesianSpace as index space cannot distinguish between
# covariant and contravariant indices. This allows to easily wrap standard Arrays.

# CartesianSpace:
#----------------
immutable CartesianSpace <: EuclideanSpace{ℝ}
    d::Int
end
^(::Type{ℝ},d::Int) = CartesianSpace(d)

# Corresponding methods:
dim(V::CartesianSpace) = V.d
cnumber(V::CartesianSpace) = CartesianSpace(1)
cnumber(::Type{CartesianSpace}) = CartesianSpace(1)
iscnumber(V::CartesianSpace) = dim(V)==1

# Show methods
Base.show(io::IO, V::CartesianSpace) = print(io, "ℝ^$(V.d)")

# direct sum of CartesianSpace vector spaces
directsum(V1::CartesianSpace, V2::CartesianSpace) = CartesianSpace(V1.d+V2.d)

# fusing CartesianSpace: return Bool indicating whether V1 and V2 can be fused to V
fuse(V1::CartesianSpace,V2::CartesianSpace,V::CartesianSpace) = dim(V1)*dim(V2)==dim(V)

# # basis and basisvector
# typealias CartesianBasisVector BasisVector{CartesianSpace,Int} # use integer from 1 to dim as identifier
# typealias CartesianBasis Basis{CartesianSpace}
#
# Base.length(B::CartesianBasis) = dim(space(B))
# Base.start(B::CartesianBasis) = 1
# Base.next(B::CartesianBasis, state::Int) = (CartesianBasisVector(space(B),state),state+1)
# Base.done(B::CartesianBasis, state::Int) = state>length(B)
#
# Base.to_index(b::CartesianBasisVector) = b.identifier
