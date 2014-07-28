# tensortoolbox.jl
#
# Main file for module TensorToolbox, a Julia package for working with
# with tensors and tensor operations

module TensorToolbox

using Reexport
using LinearMaps
@reexport using TensorOperations
#using Cartesian

# Exports
#---------
# Types:
export VectorSpace, EuclideanSpace, CartesianSpace, ProductSpace
export AbstractTensor, Tensor
export TensorMap

# general vector space methods
export dim, dual, iscnumber, issubspace, directsum, fuse, basis
# vector spaces with symmetries
export sectors, invariant

# tensor characteristics
export numind, order, space, spacetype
# tensor constructors
export tensor, tensorcat
# index manipulations
export insertind, deleteind, fuseind, splitind
# tensor operations
export tensorcopy, tensoradd, tensortrace, tensorcontract
# tensor factorizations
export leftorth, rightorth, svdtrunc

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

# Types and methods for vector spaces and corresponding bases
#-------------------------------------------------------------
include("vectorspace.jl")

# Types and methods for tensors
#-------------------------------
# New abstract type for defining tensors, multilinear objects whose indices
# take values in an IndexSpace <: VectorSpace.
include("abstracttensor.jl")

# Implementations:

# generic tensor living in a ProductSpace{N,IndexSpace} without special properties
include("tensor.jl")

# Linear maps acting on tensors
#-------------------------------
include("tensormap.jl")

end
