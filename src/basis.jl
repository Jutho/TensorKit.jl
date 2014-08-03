# basis.jl
#
# Types and methods for defining and working with basis vectors of vector spaces
#
# Written by Jutho Haegeman

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