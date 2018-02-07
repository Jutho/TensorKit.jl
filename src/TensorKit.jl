# TensorKit.jl
#
# Main file for module TensorKit, a Julia package for working with
# with tensors, tensor operations and tensor factorizations

module TensorKit

# Exports
#---------
# Types:
export StaticLength
export Sector, Irrep
export Abelian, SimpleNonAbelian, DegenerateNonAbelian, SymmetricBraiding, Bosonic, Fermionic, Anyonic # sector properties
export Parity, ZNIrrep, U1Irrep, SU2Irrep, FermionParity, FermionNumber, FermionSpin # specific sectors
export VectorSpace, Field, ElementarySpace, InnerProductSpace, EuclideanSpace # abstract vector spaces
export ComplexSpace, CartesianSpace, GeneralSpace, RepresentationSpace, ZNSpace # concrete spaces
export Z2Space, Z3Space, Z4Space, U1Space, CU1Space, SU2Space
export CompositeSpace, ProductSpace # composite spaces
export FusionTree
export IndexSpace, TensorSpace, AbstractTensorMap, AbstractTensor, TensorMap, Tensor # tensors and tensor properties
export TruncationScheme
export SpaceMismatch, SectorMismatch, IndexError # error types

# general vector space methods
export space, dual, dim, dims, fuse, flip
# methods for sectors and properties thereof
export sectortype, fusiontype, braidingtype, sectors, Nsymbol, Fsymbol, Rsymbol, Bsymbol, frobeniusschur
export Trivial, ZNSpace, SU2Irrep, U1Irrep, CU1Irrep # Fermion
export fusiontrees, braid, repartition

# some unicode
export ⊕, ⊗, ×, ℂ, ℝ, ←, →
export ℤ₂, ℤ₃, ℤ₄, U₁, SU₂, CU₁
export ℤ₂Space, ℤ₃Space, ℤ₄Space, U₁Space, CU₁Space, SU₂Space

# tensor maps
export domain, codomain
export blocksectors, block, blocks

# random methods for constructor
export randuniform, randnormal, randisometry
export one!

# tensor algebra and factorizations
export vecnorm, vecdot
export leftorth, rightorth, leftnull, rightnull, leftorth!, rightorth!, leftnull!, rightnull!, svd!, svd, exp, exp!
export permuteind, fuseind, splitind, permuteind!, fuseind!, splitind!

export OrthogonalFactorizationAlgorithm, QR, QRpos, QL, QLpos, LQ, LQpos, RQ, RQpos, SVD, Polar

# tensor operations
export @tensor, @tensoropt
export scalar, add!, contract!

# truncation schemes
export notrunc, truncerr, truncdim, truncspace, truncbelow

# Imports
#---------
using TupleTools
using TupleTools: StaticLength
import TupleTools: permute

using Strided

import TensorOperations
import TensorOperations: @tensor, @tensoropt

using WignerSymbols
using WignerSymbols: HalfInteger

using Base: @boundscheck, @propagate_inbounds, OneTo
using Base: tail, tuple_type_head, tuple_type_tail, tuple_type_cons,
            SizeUnknown, HasLength, HasShape, IsInfinite, EltypeUnknown, HasEltype

const IndexTuple{N} = NTuple{N,Int}

#--------------------------------------------------------------------
# ALL OF THIS CAN GO ON JULIA 0.7 or 1.0
@static if VERSION >= v"0.7-"
    using LinearAlgebra

    import Base: IteratorSize, IteratorEltype, axes

    import Strided: mul!, axpy!, axpby!, adjoint, adjoint!
    import LinearAlgebra: exp!, svd, eig, normalize, normalize!, vecnorm, vecdot, ×

    import Base: empty

    using Base.Iterators: product, filter
end
@static if VERSION < v"0.7-" # julia 0.6
    const LinearAlgebra = Base.LinAlg

    const IteratorSize = Base.iteratorsize
    const IteratorEltype = Base.iteratoreltype
    const Nothing = Base.Void
    const axes = Base.indices

    copyto!(dst, src) = copy!(dst, src)

    import Strided: mul!, axpy!, axpby!, adjoint, adjoint!
    export adjoint

    const exp! = TensorKit.LinearAlgebra.expm!
    import TensorKit.LinearAlgebra: scale!, svd, eig, normalize, normalize!, vecnorm, vecdot, ×

    const AbstractDict = Base.Associative

    const ComplexF32 = Complex64
    const ComplexF64 = Complex128

    empty(a::AbstractDict) = empty(a, keytype(a), valtype(a))
    empty(a::AbstractDict, ::Type{V}) where {V} = empty(a, keytype(a), V)
    empty(a::AbstractDict, ::Type{K}, ::Type{V}) where {K, V} = Dict{K, V}()

    Base.Array{T}(s::UniformScaling, dims::Base.Dims{2}) where {T} = Matrix{T}(s, dims)
    Base.Array{T}(s::UniformScaling, m::Integer, n::Integer) where {T} = Matrix{T}(s, m, n)

    Base.Matrix{T}(s::UniformScaling, dims::Base.Dims{2}) where {T}= setindex!(zeros(T, dims), T(s.λ), diagind(dims...))
    Base.Matrix{T}(s::UniformScaling, m::Integer, n::Integer) where {T} = Matrix{T}(s, Dims((m, n)))

    struct Uninitialized end
    Base.Array{T}(::Uninitialized, args...) where {T} = Array{T}(args...)
    Base.Array{T,N}(::Uninitialized, args...) where {T,N} = Array{T,N}(args...)
    Base.Vector(::Uninitialized, args...) = Vector(args...)
    Base.Matrix(::Uninitialized, args...) = Matrix(args...)

    const uninitialized = Uninitialized()
    export uninitialized

    struct EqualTo{T} <: Function
        x::T
        EqualTo(x::T) where {T} = new{T}(x)
    end
    (f::EqualTo)(y) = isequal(f.x, y)
    const equalto = EqualTo

    print_array(io, a) = Base.showarray(io, a, false; header=false)

    include("auxiliary/iterators.jl")
end

# Auxiliary files
#-----------------
include("auxiliary/auxiliary.jl")
include("auxiliary/dicts.jl")
include("auxiliary/linalg.jl")
include("auxiliary/random.jl")

# include("auxiliary/juarray.jl")
# export JuArray

#--------------------------------------------------------------------
# experiment with different dictionaries
const SectorDict{K,V} = VectorDict{K,V}
const FusionTreeDict{K,V} = VectorDict{K,V}
#--------------------------------------------------------------------

# Exception types:
#------------------
abstract type TensorException <: Exception end

# Exception type for all errors related to sector mismatch
struct SectorMismatch{S<:Union{Nothing,String}} <: TensorException
    message::S
end
SectorMismatch()=SectorMismatch{Nothing}(nothing)
Base.show(io::IO, ::SectorMismatch{Nothing}) = print(io, "SectorMismatch()")

# Exception type for all errors related to vector space mismatch
struct SpaceMismatch{S<:Union{Nothing,String}} <: TensorException
    message::S
end
SpaceMismatch()=SpaceMismatch{Nothing}(nothing)
Base.show(io::IO, ::SpaceMismatch{Nothing}) = print(io, "SpaceMismatch()")

# Exception type for all errors related to invalid tensor index specification.
struct IndexError{S<:Union{Nothing,String}} <: TensorException
    message::S
end
IndexError()=IndexError{Nothing}(nothing)
Base.show(io::IO, ::IndexError{Nothing}) = print(io, "IndexError()")

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

# Definitions and methods for tensors
#-------------------------------------
# general definitions
include("tensors/abstracttensor.jl")
include("tensors/tensortreeiterator.jl")
include("tensors/tensor.jl")
include("tensors/adjoint.jl")
include("tensors/linalg.jl")
include("tensors/tensoroperations.jl")
include("tensors/indexmanipulations.jl")
include("tensors/truncation.jl")
include("tensors/factorizations.jl")

end
