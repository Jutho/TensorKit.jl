# vectorspace.jl
#
# Types and methods for defining and working with vector spaces and corresponding basis vectors.
#
# Written by Jutho Haegeman

#++++++++++++++
# VectorSpace:
#++++++++++++++
abstract VectorSpace
# Start a type hierarchy for defining vector spaces

# general definitions
space(V::VectorSpace) = V # just returns the space

# Elementary vector spaces
#-------------------------
abstract ElementarySpace{F} <: VectorSpace
# Elementary finite-dimensional vector space over a field F that can be used as the index space corresponding to the indices
# of a tensor. Every elementary vector space should respond to the methods conj and dual, returning the complex conjugate space
# and the dual space respectively. The complex conjugate of the dual space is obtained as dual(conj(V)) === conj(dual(V)). These
# different spaces should be of the same type, so that a tensor can be defined as an element of a homogeneous tensor product
# of these spaces

Base.eltype{F}(V::ElementarySpace{F}) = F
Base.eltype{F}(::Type{ElementarySpace{F}}) = F
Base.eltype{S<:ElementarySpace}(::Type{S}) = eltype(super(S))

# The complex conjugate vector space of a real vector space is equal to itself
Base.conj{F<:Real}(V::ElementarySpace{F}) = V

abstract ElementaryHilbertSpace{F} <: ElementarySpace{F}
typealias ElementaryInnerProductSpace{F} ElementaryHilbertSpace{F}
# An inner product space, has the possibility to raise or lower indices of tensors

const R=Real
const C=Complex{Real}

abstract EuclideanSpace{F<:Union(R,C)} <: ElementaryHilbertSpace{F}
# Elementary finite-dimensional space R^d or C^d with standard (Euclidean) inner product

Base.conj(V::EuclideanSpace) = dual(V)
# The complex conjugate space of a Hilbert space is naturally isomorphic to its dual space

dual(V::EuclideanSpace{R}) = V
# For a real Euclidean space, the dual space is naturally isomorphic to the vector space

# Functionality for extracting and iterating over elementary space
Base.length(V::ElementarySpace) = 1
Base.endof(V::ElementarySpace) = 1
Base.getindex(V::ElementarySpace, n::Integer) = (n == 1 ? V : throw(BoundsError()))

Base.start(V::ElementarySpace) = false
Base.next(V::ElementarySpace, state) = (V,true)
Base.done(V::ElementarySpace, state) = state

# Composite vector spaces
#-------------------------
# Obtained in some way or another from composing elementary vector spaces
abstract CompositeSpace{S<:ElementarySpace} <: VectorSpace

Base.eltype{S}(V::CompositeSpace{S}) = eltype(S)
Base.eltype{S}(::Type{CompositeSpace{S}}) = eltype(S)
Base.eltype{S<:CompositeSpace}(::Type{S}) = eltype(super(S))

# BasisVector and Basis
#-----------------------
# immutable type for specifying basis vectors: every vector space should have a canonical basis, where the basis
# vectors can be specified using an identifier of some custom type T (e.g. user defined)
immutable BasisVector{S<:VectorSpace,T}
    space::S
    identifier::T
end

space(b::BasisVector) = b.space
in(b::BasisVector,V::VectorSpace) = space(b) == V

Base.show(io::IO, b::BasisVector) = print(io, "BasisVector($(b.space),$(b.identifier))")

# a basis is just an iterator over the canonical basis vectors of the space 
immutable Basis{S<:VectorSpace}
    space::S
end

space(B::Basis) = B.space
basis(V::VectorSpace) = Basis(V) # lowercase convenience method

Base.show(io::IO, B::Basis) = print(io, "Basis($(B.space))")

# Explicit realizations:
#------------------------
# CartesianSpace: standard real space R^N with euclidean structure (canonical orthonormal/cartesian basis)
include("vectorspace/cartesianspace.jl")

# ComplexSpace: standard complex space C^N with euclidean structure (canonical orthonormal basis)
include("vectorspace/complexspace.jl")

# GeneralSpace: a general elementary vector space (without inner product structure)
include("vectorspace/generalspace.jl")

# ProductSpace: type and methods for tensor products of ElementarySpace objects
include("vectorspace/productspace.jl")

# Graded elementary vector spaces
#---------------------------------
# vector spaces which have a natural decomposition as a direct sum of several sectors

abstract Sector
# hierarchy of types for labelling different sectors, e.g. irreps, quantum numbers, anyon types, ...
abstract Abelian <: Sector
# sectors that have an abelian fusion structure

#include("sectors/parity.jl")
#include("sectors/zncharge.jl")
#include("sectors/u1charge.jl")

# vector spaces graded with abelian sectors
#include("vectorspace/abelian.jl")


# invariant subspace of tensor product of abelian spaces
#include("vectorspace/invariant.jl")