# tensortoolbox.jl
#
# Main file for module TensorToolbox, a Julia package for working with
# with tensors and tensor operations

module TensorToolbox

using Debug

# Exports
#---------
# Types:
export VectorSpace, ElementarySpace, ElementaryHilbertSpace, EuclideanSpace
export ComplexSpace, CartesianSpace, GeneralSpace
export CompositeSpace, ProductSpace
export IndexSpace, TensorSpace, AbstractTensor, Tensor
export TruncationScheme
export AbstractTensorMap
export SpaceError, IndexError

# general vector space methods
export space, issubspace, dim, dual, cnumber, iscnumber, directsum, fuse, basis
export ⊗, ℂ, ℝ # some unicode

# vector spaces with symmetries
export sectors, invariant

# tensor characteristics
export spacetype, tensortype, numind, order
# tensor constructors
export tensor, tensorcat
# index manipulations
export insertind, deleteind, fuseind, splitind
# tensor operations
export tensorcopy, tensoradd, tensortrace, tensorcontract, tensorproduct
export tensorcopy!, tensoradd!, tensortrace!, tensorcontract!, tensorproduct!
# tensor factorizations
export leftorth, rightorth
export leftorth!, rightorth!, svd!

# truncation schemes
export notrunc, truncerr, truncdim, truncspace

# tensor maps
export domain, codomain, hermitian, posdef, tensormap

# Exception types:
#------------------
abstract TensorException <: Exception

type SpaceError <: TensorException
    message::String
end
SpaceError()=SpaceError("")
# Exception type for all errors related to vector space mismatch

type IndexError <: TensorException
    message::String
end
IndexError()=IndexError("")
# Exception type for all errors related to invalid tensor index specification.

# Types and methods for vector spaces and corresponding bases
#-------------------------------------------------------------
abstract VectorSpace
space(V::VectorSpace) = V # just returns the space
issubspace(V1::VectorSpace,V2::VectorSpace) = V1==V2 # default, only identical spaces are subspaces
*(V1::VectorSpace,V2::VectorSpace) = ⊗(V1,V2) # for convenience, product of vector spaces is tensor product

include("basis.jl") # defining basis of a vector space
include("elementaryspace.jl") # elementary finite-dimensional vector spaces
include("compositespace.jl") # composing elementary vector spaces

# Types and methods for tensors
#-------------------------------
import TensorOperations
# intentionally shadow original TensorOperation methods for StridedArray objects

# define truncation schemes for tensors
include("tensors/truncation.jl")

# general definitions
include("tensors/abstracttensor.jl")

# Implementations:
include("tensors/tensor.jl") # generic tensor living in a ProductSpace without special properties

# Tensor maps: linear maps acting on tensors
#--------------------------------------------
include("tensormaps/abstracttensormap.jl")
include("tensormaps/linearcombination.jl")
include("tensormaps/composition.jl")
include("tensormaps/shifted.jl")
include("tensormaps/dual.jl")
#include("tensormaps/adjoint.jl")
include("tensormaps/hermitian.jl") # guaranteed self adjoint operator
include("tensormaps/posdef.jl") # guaranteed positive definite operator
include("tensormaps/tensormap.jl") # a dense tensor map implemented using a tensor

end
