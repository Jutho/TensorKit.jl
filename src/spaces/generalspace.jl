"""
    struct GeneralSpace{k} <: ElementarySpace{k}

A finite-dimensional space over an arbitrary field `F` without additional structure. It is thus
characterized by its dimension, and whether or not it is the dual and/or conjugate space. For a
real field `F`, the space and its conjugate are the same.
"""
struct GeneralSpace{k} <: ElementarySpace{k}
    d::Int
    dual::Bool
    conj::Bool
    GeneralSpace{k}(d::Int, dual::Bool = false, conj::Bool = false) where {k} =
        (d >= 0 ? new{k}(d, dual, conj) : throw(ArgumentError("Dimension of a vector space should be bigger than zero")))
end

dim(V::GeneralSpace) = V.d
Base.axes(V::GeneralSpace) = Base.OneTo(dim(V))

dual(V::GeneralSpace{k}) where {k} = GeneralSpace{k}(V.d, !V.dual, V.conj)
isdual(V::GeneralSpace) = V.dual
Base.conj(V::GeneralSpace{k}) where {k} = GeneralSpace{k}(V.d, V.dual, !V.conj)

function Base.show(io::IO, V::GeneralSpace{k}) where {k}
    if V.conj
        print(io, "conj(")
    end
    print(io, "GeneralSpace{", k, "}(", V.d, ")")
    if V.dual
        print(io, "'")
    end
    if V.conj
        print(io, ")")
    end
end
Base.show(io::IO, ::Type{GeneralSpace}) = print(io, "GeneralSpace")
