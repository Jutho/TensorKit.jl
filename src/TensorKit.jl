# TensorKit.jl
#
# Main file for module TensorKit, a Julia package for working with
# with tensors, tensor operations and tensor factorizations

module TensorKit

# Exports
#---------
# Types:
export Sector, AbstractIrrep, Irrep
export FusionStyle, UniqueFusion, MultipleFusion, MultiplicityFreeFusion,
       SimpleFusion, GenericFusion
export BraidingStyle, SymmetricBraiding, Bosonic, Fermionic, Anyonic, NoBraiding
export Trivial, Z2Irrep, Z3Irrep, Z4Irrep, ZNIrrep, U1Irrep, SU2Irrep, CU1Irrep
export ProductSector
export FermionParity, FermionNumber, FermionSpin
export FibonacciAnyon, IsingAnyon

export VectorSpace, Field, ElementarySpace # abstract vector spaces
export InnerProductStyle, NoInnerProduct, HasInnerProduct, EuclideanProduct
export ComplexSpace, CartesianSpace, GeneralSpace, GradedSpace # concrete spaces
export ZNSpace, Z2Space, Z3Space, Z4Space, U1Space, CU1Space, SU2Space
export Vect, Rep # space constructors
export CompositeSpace, ProductSpace # composite spaces
export FusionTree
export IndexSpace, TensorSpace, TensorMapSpace
export AbstractTensorMap, AbstractTensor, TensorMap, Tensor, TrivialTensorMap # tensors and tensor properties
export TruncationScheme
export SpaceMismatch, SectorMismatch, IndexError # error types

# general vector space methods
export space, field, dual, dim, dims, fuse, flip, isdual, insertunit, oplus

# partial order for vector spaces
export infimum, supremum, isisomorphic, ismonomorphic, isepimorphic

# methods for sectors and properties thereof
export sectortype, sectors, hassector, Nsymbol, Fsymbol, Rsymbol, Bsymbol,
       frobeniusschur, twist, otimes
export fusiontrees, braid, permute, transpose
export ZNSpace, SU2Irrep, U1Irrep, CU1Irrep
# other fusion tree manipulations, should not be exported:
# export insertat, split, merge, repartition, artin_braid,
#        bendleft, bendright, foldleft, foldright, cycleclockwise, cycleanticlockwise

# some unicode
export ⊕, ⊗, ×, ⊠, ℂ, ℝ, ℤ, ←, →, ≾, ≿, ≅, ≺, ≻
export ℤ₂, ℤ₃, ℤ₄, U₁, SU, SU₂, CU₁
export fℤ₂, fU₁, fSU₂
export ℤ₂Space, ℤ₃Space, ℤ₄Space, U₁Space, CU₁Space, SU₂Space

# tensor maps
export domain, codomain, numind, numout, numin, domainind, codomainind, allind
export spacetype, sectortype, storagetype, scalartype, tensormaptype
export blocksectors, blockdim, block, blocks

# random methods for constructor
export randuniform, randnormal, randisometry, randhaar

# special purpose constructors
export zero, one, one!, id, isomorphism, unitary, isometry

# reexport most of VectorInterface and some more tensor algebra
export zerovector, zerovector!, zerovector!!, scale, scale!, scale!!, add, add!, add!!
export inner, dot, norm, normalize, normalize!, tr

# factorizations
export mul!, lmul!, rmul!, adjoint!, pinv, axpy!, axpby!
export leftorth, rightorth, leftnull, rightnull,
       leftorth!, rightorth!, leftnull!, rightnull!,
       tsvd!, tsvd, eigen, eigen!, eig, eig!, eigh, eigh!, exp, exp!,
       isposdef, isposdef!, ishermitian, sylvester
export braid, braid!, permute, permute!, transpose, transpose!, twist, twist!, repartition,
       repartition!
export catdomain, catcodomain

export OrthogonalFactorizationAlgorithm, QR, QRpos, QL, QLpos, LQ, LQpos, RQ, RQpos,
       SVD, SDD, Polar

# tensor operations
export @tensor, @tensoropt, @ncon, ncon, @planar, @plansor
export scalar, add!, contract!

# truncation schemes
export notrunc, truncerr, truncdim, truncspace, truncbelow

# Imports
#---------
using TupleTools
using TupleTools: StaticLength

using Strided

using VectorInterface

using TensorOperations: TensorOperations, @tensor, @tensoropt, @ncon, ncon
using TensorOperations: IndexTuple, Index2Tuple, linearize, Backend
const TO = TensorOperations

using LRUCache

using HalfIntegers
using WignerSymbols

using Base: @boundscheck, @propagate_inbounds, OneTo, tail, front,
            tuple_type_head, tuple_type_tail, tuple_type_cons,
            SizeUnknown, HasLength, HasShape, IsInfinite, EltypeUnknown, HasEltype
using Base.Iterators: product, filter

using LinearAlgebra: LinearAlgebra
using LinearAlgebra: norm, dot, normalize, normalize!, tr,
                     axpy!, axpby!, lmul!, rmul!, mul!,
                     adjoint, adjoint!, transpose, transpose!,
                     pinv, sylvester,
                     eigen, eigen!, svd, svd!,
                     isposdef, isposdef!, ishermitian,
                     Diagonal, Hermitian
import Base.Meta

using PackageExtensionCompat

# Auxiliary files
#-----------------
include("auxiliary/auxiliary.jl")
include("auxiliary/dicts.jl")
include("auxiliary/iterators.jl")
include("auxiliary/linalg.jl")
include("auxiliary/random.jl")

#--------------------------------------------------------------------
# experiment with different dictionaries
const SectorDict{K,V} = SortedVectorDict{K,V}
const FusionTreeDict{K,V} = Dict{K,V}
#--------------------------------------------------------------------

# Exception types:
#------------------
abstract type TensorException <: Exception end

# Exception type for all errors related to sector mismatch
struct SectorMismatch{S<:Union{Nothing,String}} <: TensorException
    message::S
end
SectorMismatch() = SectorMismatch{Nothing}(nothing)
Base.show(io::IO, ::SectorMismatch{Nothing}) = print(io, "SectorMismatch()")
Base.show(io::IO, e::SectorMismatch) = print(io, "SectorMismatch(\"", e.message, "\")")

# Exception type for all errors related to vector space mismatch
struct SpaceMismatch{S<:Union{Nothing,String}} <: TensorException
    message::S
end
SpaceMismatch() = SpaceMismatch{Nothing}(nothing)
Base.show(io::IO, ::SpaceMismatch{Nothing}) = print(io, "SpaceMismatch()")
Base.show(io::IO, e::SpaceMismatch) = print(io, "SpaceMismatch(\"", e.message, "\")")

# Exception type for all errors related to invalid tensor index specification.
struct IndexError{S<:Union{Nothing,String}} <: TensorException
    message::S
end
IndexError() = IndexError{Nothing}(nothing)
Base.show(io::IO, ::IndexError{Nothing}) = print(io, "IndexError()")
Base.show(io::IO, e::IndexError) = print(io, "IndexError(", e.message, ")")

# typerepr
type_repr(T::Type) = repr(T)

# Definitions and methods for superselection sectors (quantum numbers)
#----------------------------------------------------------------------
include("sectors/sectors.jl")

# Constructing and manipulating fusion trees and iterators thereof
#------------------------------------------------------------------
include("fusiontrees/fusiontrees.jl")

# Definitions and methods for vector spaces
#-------------------------------------------
include("spaces/vectorspaces.jl")

# # Definitions and methods for tensors
# #-------------------------------------
# # general definitions
include("tensors/abstracttensor.jl")
include("tensors/tensortreeiterator.jl")
include("tensors/tensor.jl")
include("tensors/adjoint.jl")
include("tensors/linalg.jl")
include("tensors/vectorinterface.jl")
include("tensors/tensoroperations.jl")
include("tensors/indexmanipulations.jl")
include("tensors/truncation.jl")
include("tensors/factorizations.jl")
include("tensors/braidingtensor.jl")

# # Planar macros and related functionality
# #-----------------------------------------
@nospecialize
using Base.Meta: isexpr
include("planar/analyzers.jl")
include("planar/preprocessors.jl")
include("planar/postprocessors.jl")
include("planar/macros.jl")
@specialize
include("planar/planaroperations.jl")

# deprecations: to be removed in version 1.0 or sooner
include("auxiliary/deprecate.jl")

# Extensions
# ----------
function __init__()
    @require_extensions
end

end
