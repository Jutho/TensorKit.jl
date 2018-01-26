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

# tensor factorizations
export leftorth, rightorth, leftnull, rightnull, leftorth!, rightorth!, leftnull!, rightnull!, svd!
export permuteind, fuseind, splitind, permuteind!, fuseind!, splitind!

export OrthogonalFactorizationAlgorithm, QR, QRpos, QL, QLpos, LQ, LQpos, RQ, RQpos, SVD, Polar

# tensor operations
export @tensor, @tensoropt
export scalar, add!, contract!

# truncation schemes
export notrunc, truncerr, truncdim, truncspace

# Imports
#---------
using TupleTools
using TupleTools: StaticLength

using Strided

using Base: @boundscheck, @propagate_inbounds
using Base: ImmutableDict

import Base: permute

include("auxiliary/filter.jl")
using .Filter.filter

# if VERSION <= v"0.6.1"
    include("auxiliary/product.jl")
    using .Product.product
# else
#     using Base.Iterators.product
# end

#--------------------------------------------------------------------
# ALL OF THIS CAN GO ON JULIA 0.7 or 1.0
if VERSION < v"0.7.0-DEV.2377"
    Base.Matrix{T}(s::UniformScaling, dims::Base.Dims{2}) where {T}= setindex!(zeros(T, dims), T(s.λ), diagind(dims...))
    Base.Matrix{T}(s::UniformScaling, m::Integer, n::Integer) where {T} = Matrix{T}(s, Dims((m, n)))
end
if VERSION < v"0.7.0-DEV.2543"
    Base.Array{T}(s::UniformScaling, dims::Base.Dims{2}) where {T} = Matrix{T}(s, dims)
    Base.Array{T}(s::UniformScaling, m::Integer, n::Integer) where {T} = Matrix{T}(s, m, n)
end

@static if !isdefined(Base, :AbstractDict)
    const AbstractDict = Base.Associative
end

@static if !isdefined(Base, :ComplexF32)
    const ComplexF32 = Complex64
    const ComplexF64 = Complex128
end

@static if isdefined(Base.LinAlg, :mul!)
    import Base.LinAlg.mul!
else
    const mul! = Base.A_mul_B!
end

@static if !isdefined(Base, :empty)
    empty(a::Associative) = empty(a, keytype(a), valtype(a))
    empty(a::Associative, ::Type{V}) where {V} = empty(a, keytype(a), V)
    empty(a::Associative, ::Type{K}, ::Type{V}) where {K, V} = Dict{K, V}()
end

@static if !isdefined(Base, :adjoint)
    const adjoint = Base.ctranspose
    const adjoint! = Base.ctranspose!
    export adjoint, adjoint!
else
    import Base: adjoint, adjoint!
end

@static if !isdefined(Base, Symbol("@__MODULE__"))
    # 0.7
    macro __MODULE__()
        return current_module()
    end
    Base.expand(mod::Module, x::ANY) = eval(mod, :(expand($(QuoteNode(x)))))
    Base.macroexpand(mod::Module, x::ANY) = eval(mod, :(macroexpand($(QuoteNode(x)))))
    Base.include_string(mod::Module, code::String, fname::String) =
        eval(mod, :(include_string($code, $fname)))
    Base.include_string(mod::Module, code::AbstractString, fname::AbstractString="string") =
        eval(mod, :(include_string($code, $fname)))
end

@static if !isdefined(Base, :Uninitialized)
    include_string(@__MODULE__, """
        struct Uninitialized end
        Base.Array{T}(::Uninitialized, args...) where {T} = Array{T}(args...)
        Base.Array{T,N}(::Uninitialized, args...) where {T,N} = Array{T,N}(args...)
        Base.Vector(::Uninitialized, args...) = Vector(args...)
        Base.Matrix(::Uninitialized, args...) = Matrix(args...)
    """)
    const uninitialized = Uninitialized()
end

@static if !isdefined(Base, :EqualTo)
    if VERSION >= v"0.6.0"
        include_string(@__MODULE__, """
            struct EqualTo{T} <: Function
                x::T
                EqualTo(x::T) where {T} = new{T}(x)
            end
        """)
    else
        include_string(@__MODULE__, """
            immutable EqualTo{T} <: Function
                x::T
            end
        """)
    end
    (f::EqualTo)(y) = isequal(f.x, y)
    const equalto = EqualTo
end

@static if isdefined(Base, :print_array)
    using Base.print_array
else
    print_array(io, a) = Base.showarray(io, a, false; header=false)
end

# if VERSION >= v"0.6.99"
#     finalizer(f, o) = Base.finalizer(f,o)
# else
#     finalizer(f, o) = Base.finalizer(o,f)
# end


#--------------------------------------------------------------------

import TensorOperations
import TensorOperations: @tensor, @tensoropt

const IndexTuple{N} = NTuple{N,Int}

# Auxiliary files
#-----------------
include("auxiliary/auxiliary.jl")
include("auxiliary/dicts.jl")
include("auxiliary/halfinteger.jl")
include("auxiliary/linalg.jl")
include("auxiliary/random.jl")
# include("auxiliary/unsafe_similar.jl")

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
# define truncation schemes for tensors
include("tensors/truncation.jl")
# general definitions
include("tensors/abstracttensor.jl")
include("tensors/tensortreeiterator.jl")
include("tensors/tensor.jl")
include("tensors/adjoint.jl")
include("tensors/tensoroperations.jl")
include("tensors/indexmanipulations.jl")
include("tensors/factorizations.jl")

@static if isdefined(Base.LinAlg, :Adjoint)
    Base.LinAlg.Adjoint(t::AbstractTensorMap) = adjoint(t)
    Base.LinAlg.Adjoint(V::VectorSpace) = adjoint(V)
end

# include("auxiliary/juarray.jl")
# export JuArray


end
