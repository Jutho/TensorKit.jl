# general.jl
#
# Defines the immutible GeneralSpace for a general finite dimensional vector space over an arbitrary
# field, characterized by its field, its dimension and whether or not it is the dual space and/or the
# complex conjugate space. Tensors with GeneralSpace as index spaces make a distinction between covariant
# and contravariant indices, and covariant and contravariant dotted/barred indices.

# EuclideanSpace:
#-----------------
immutable GeneralSpace{F} <: ElementarySpace{F}
    field::Type{F}
    d::Int
    dual::Bool
    conj::Bool
    GeneralSpace(d::Int,dual::Bool=false,conj::Bool=false) = (d>0 ? new(F, d, dual, !(F<:Real) && conj) : throw(ArgumentError("Dimension of a vector space should be bigger than zero")))
end
GeneralSpace{F}(::Type{F},d::Int,dual::Bool=false,conj::Bool=false) = GeneralSpace{F}(d,dual,conj)

# Corresponding methods:
dim(V::GeneralSpace) = V.d
dual(V::GeneralSpace) = GeneralSpace(V.field,V.d, !V.dual, V.conj)
Base.conj(V::GeneralSpace) = GeneralSpace(V.field,V.d, V.dual, !V.conj)
iscnumber(V::ElementarySpace) = dim(V)==1 && V.field <: Number

# Show methods
function Base.show(io::IO, V::GeneralSpace)
    if V.conj
        print(io,"conj(")
    end
    print(io,"$(V.field)($(V.d))")
    if V.dual
        print(io,"*")
    end
    if V.conj
        print(io,")")
    end
end
