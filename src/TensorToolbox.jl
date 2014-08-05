# tensortoolbox.jl
#
# Main file for module TensorToolbox, a Julia package for working with
# with tensors and tensor operations

module TensorToolbox

# Exports
#---------
# Types:
export VectorSpace, ElementarySpace, ElementaryHilbertSpace, EuclideanSpace
export ComplexSpace, CartesianSpace, GeneralSpace
export CompositeSpace, ProductSpace
export IndexSpace, TensorSpace, AbstractTensor, Tensor
export AbstractTensorMap, DualTensorMap
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
export leftorth, rightorth, svdtrunc
export leftorth!, rightorth!, svd!, svdtrunc!

# tensor maps
export domain, codomain

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

# General definitions:
#----------------------
⊗(V1,V2,V3...)=⊗(⊗(V1,V2),V3...)

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

include("tensors/abstracttensor.jl")

# Implementations:
include("tensors/tensor.jl") # generic tensor living in a ProductSpace without special properties

# Linear maps acting on tensors
#-------------------------------
using LinearMaps

include("tensormap.jl")

end
