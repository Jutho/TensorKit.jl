"""
    struct ComplexSpace <: EuclideanSpace{ℂ}

A standard complex vector space ℂ^d with Euclidean inner product and no additional
structure. It is completely characterised by its dimension and whether its the normal space
or its dual (which is canonically isomorphic to the conjugate space).
"""
struct ComplexSpace <: EuclideanSpace{ℂ}
  d::Int
  dual::Bool
end
ComplexSpace(d::Int) = ComplexSpace(d, false)

# convenience constructor
Base.:^(::ComplexNumbers, d::Int) = ComplexSpace(d)
Base.getindex(::ComplexNumbers) = ComplexSpace
Base.getindex(::ComplexNumbers, d::Int) = ComplexSpace(d)

# Corresponding methods:
#------------------------
dim(V::ComplexSpace) = V.d
Base.axes(V::ComplexSpace) = Base.OneTo(dim(V))

Base.conj(V::ComplexSpace) = ComplexSpace(V.d, !V.dual)
isdual(V::ComplexSpace) = V.dual

Base.oneunit(::Type{ComplexSpace}) = ComplexSpace(1)
⊕(V1::ComplexSpace, V2::ComplexSpace) = (V1.dual==V2.dual ?
    ComplexSpace(V1.d+V2.d, V1.dual) :
    throw(SpaceMismatch("Direct sum of a vector space and its dual does not exist")))
fuse(V1::ComplexSpace, V2::ComplexSpace) = ComplexSpace(V1.d*V2.d)
flip(V::ComplexSpace) = dual(V)

Base.min(V1::ComplexSpace, V2::ComplexSpace) = V1.dual == V2.dual ?
    ComplexSpace(min(V1.d, V2.d), V1.dual) :
    throw(SpaceMismatch("V1 and V2 should both be normal or dual spaces"))
Base.max(V1::ComplexSpace, V2::ComplexSpace) = V1.dual == V2.dual ?
    ComplexSpace(max(V1.d, V2.d), V1.dual) :
    throw(SpaceMismatch("V1 and V2 should both be normal or dual spaces"))

Base.show(io::IO, V::ComplexSpace) = print(io, V.dual ? "(ℂ^$(V.d))'" : "ℂ^$(V.d)")
Base.show(io::IO, ::Type{ComplexSpace}) = print(io, "ComplexSpace")
