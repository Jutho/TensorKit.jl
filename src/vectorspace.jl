# vectorspace.jl
#
# Types and methods for defining and working with vector spaces and corresponding basis vectors.
#
# Written by Jutho Haegeman

#++++++++++++++
# VectorSpace:
#++++++++++++++
abstract VectorSpace
# Start a type hierarchy for defining vector spaces (actually Hilbert spaces) and corresponding basis vectors

# general definition
Base.ctranspose(V::VectorSpace) = dual(V)

abstract ElementarySpace <: VectorSpace
# Elementary finite-dimensional vector spaces that can be used as the index space corresponding to the indices of a tensor

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

# Elementary vector spaces with simple euclidean structure
#----------------------------------------------------------
# CartesianSpace: self-dual vector space (e.g. cartesian basis)
include("vectorspace/cartesian.jl")
# EuclideanSpace: standard euclidean space
include("vectorspace/euclidean.jl")

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

# Composite vector spaces
#-------------------------
# ProductSpace: type and methods for tensor products of ElementarySpace objects
include("vectorspace/product.jl")

# invariant subspace of tensor product of abelian spaces
#include("vectorspace/invariant.jl")

# Fermionic vector spaces
#-------------------------
# Z2 graded spaces with fermionic anticommutation rules under permutation (braiding)
#include("vectorspace/fermion.jl")