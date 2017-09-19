# tensortoolbox.jl
#
# Main file for module TensorToolbox, a Julia package for working with
# with tensors and tensor operations

# module TensorToolbox
#
# # Exports
# #---------
# # Types:
# export VectorSpace, Field, ElementarySpace, InnerProductSpace, EuclideanSpace
# export ComplexSpace, CartesianSpace, GeneralSpace, GradedSpace, ZNSpace
# export Parity, ZNCharge, U1Charge
# export CompositeSpace, ProductSpace, InvariantSpace
# export IndexSpace, TensorSpace, AbstractTensor, Tensor, InvariantTensor
# export TruncationScheme
# export AbstractTensorMap
# export SpaceError, IndexError
#
# # general vector space methods
# export space, dim, dual, fieldtype, sectortype #, cnumber, iscnumber, directsum, fuse
# export ⊗, ⊕, ℂ, ℝ # some unicode
#
# # vector spaces with symmetries
# export sectors, invariant
#
# # tensor characteristics
# export spacetype, tensortype, numind, order
# # tensor constructors
# export tensor, tensorcat
# # index manipulations
# export insertind, deleteind, fuseind, splitind
# # tensor operations
# export tensorcopy, tensoradd, tensortrace, tensorcontract, tensorproduct
# export tensorcopy!, tensoradd!, tensortrace!, tensorcontract!, tensorproduct!
# # tensor factorizations
# export leftorth, rightorth
# export leftorth!, rightorth!, svd!
#
# # truncation schemes
# export notrunc, truncerr, truncdim, truncspace
#
# # tensor maps
# export domain, codomain, hermitian, posdef, tensormap
#
# # tensor networks
# export AbstractTensorNetwork, TensorNetwork
# export network, optimizecontract

using Base: tuple_type_head, tuple_type_tail, tuple_type_cons, tail, front, setindex,
            Iterators.product, ImmutableDict
include("auxiliary/auxiliary.jl")

# Exception types:
#------------------
abstract type TensorException <: Exception end

# Exception type for all errors related to sector mismatch
struct SectorMismatch{S<:Union{Void,String}} <: TensorException
    message::S
end
SectorMismatch()=SectorMismatch{Void}(nothing)
Base.show(io::IO, ::SectorMismatch{Void}) = print(io, "SectorMismatch()")

# Exception type for all errors related to vector space mismatch
struct SpaceMismatch{S<:Union{Void,String}} <: TensorException
    message::S
end
SpaceMismatch()=SpaceMismatch{Void}(nothing)
Base.show(io::IO, ::SpaceMismatch{Void}) = print(io, "SpaceMismatch()")

# Exception type for all errors related to invalid tensor index specification.
struct IndexError{S<:Union{Void,String}} <: TensorException
    message::S
end
IndexError()=IndexError{Void}(nothing)
Base.show(io::IO, ::IndexError{Void}) = print(io, "IndexError()")

# Tensor product operator
#-------------------------
⊗(a, b, c, d...)=⊗(a, ⊗(b, c, d...))

# Definitions and methods for superselection sectors (quantum numbers)
#----------------------------------------------------------------------
include("sectors/sectors.jl")

# Definitions and methods for vector spaces
#-------------------------------------------
include("spaces/vectorspaces.jl")

# Constructing and manipulating fusion trees and iterators thereof
#------------------------------------------------------------------
include("fusiontrees/fusiontrees.jl")

# # Definitions and methods for tensors
# #-------------------------------------
# import TensorOperations
# intentionally shadow original TensorOperation methods for StridedArray objects

# define truncation schemes for tensors
# include("tensors/truncation.jl")

# general definitions
#include("tensors/abstracttensor.jl")

# specific implementation
# include("tensors/tensor.jl")

#end
