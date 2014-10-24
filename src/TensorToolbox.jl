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
export UnitaryRepresentationSpace, AbelianSpace
export U1Charge, ZNCharge, Parity
export CompositeSpace, ProductSpace, InvariantSpace
export IndexSpace, TensorSpace, AbstractTensor, Tensor, InvariantTensor
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

# tensor networks
export AbstractTensorNetwork, TensorNetwork
export network, optimizecontract

# General imports:
#------------------
using Cartesian

# Exception types:
#------------------
abstract TensorException <: Exception

type SpaceError <: TensorException
    message::String
end
SpaceError()=SpaceError("Space mismatch")
# Exception type for all errors related to vector space mismatch

type IndexError <: TensorException
    message::String
end
IndexError()=IndexError("Invalid index specification")
# Exception type for all errors related to invalid tensor index specification.

# Types and methods for vector spaces
#-------------------------------------
abstract VectorSpace
space(V::VectorSpace) = V # just returns the space
issubspace(V1::VectorSpace,V2::VectorSpace) = V1==V2 # default, only identical spaces are subspaces
⊗(a,b,c...)=⊗(a,⊗(b,c...)) # introduce ⊗ as operator
*(V1::VectorSpace,V2::VectorSpace) = ⊗(V1,V2) # for convenience, product of vector spaces is tensor product

#include("basis.jl") # basis: iterator over a set of basis vectors spanning the space

include("elementaryspace.jl") # elementary finite-dimensional vector spaces
include("compositespace.jl") # composing elementary vector spaces

# Types and methods for tensors
#-------------------------------
import TensorOperations
import TensorOperations: TCBuffer, defaultcontractbuffer
# intentionally shadow original TensorOperation methods for StridedArray objects

# define truncation schemes for tensors
include("tensors/truncation.jl")

# general definitions
include("tensors/abstracttensor.jl")

# Implementations:
include("tensors/tensor.jl") # generic tensor living in a ProductSpace without special properties
include("tensors/invarianttensor.jl") # generic tensor living in a ProductSpace without special properties

# Tensor networks: contract a network of tensors
#------------------------------------------------
include("tensornetworks/abstracttensornetwork.jl")
include("tensornetworks/tensornetwork.jl")
include("tensornetworks/optimize.jl")

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
