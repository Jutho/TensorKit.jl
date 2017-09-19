# truncation.jl
#
# Implements truncation schemes for truncating a tensor with svd, leftorth or rightorth

abstract type TruncationScheme end

struct NoTruncation <: TruncationScheme
end
notrunc() = NoTruncation()

struct TruncationError <: TruncationScheme
    epsilon::Real
end
truncerr(epsilon::Real) = TruncationError(epsilon)
Base.eps(t::TruncationError) = t.epsilon

struct TruncationDimension <: TruncationScheme
    D::Integer
    epsilon::Real
end
truncdim(D::Int) = TruncationDimension(D, 0)
dim(t::TruncationDimension) = t.D
Base.eps(t::TruncationDimension) = t.epsilon

struct TruncationSpace{S<:ElementarySpace} <: TruncationScheme
    space::S
    epsilon::Real
end
truncspace(space::ElementarySpace) = TruncationSpace(space, 0)
space(s::TruncationSpace) = s.space
dim(s::TruncationSpace) = dim(s.space)
Base.eps(t::TruncationSpace) = t.epsilon

#
#
# # in order not having to specify all possibilities:
# Base.:|(trunc1::TruncationScheme, trunc2::TruncationScheme) = |(trunc2, trunc1)
# Base.:&(trunc1::TruncationScheme, trunc2::TruncationScheme) = &(trunc2, trunc1)
# Base.:|(trunc1::NoTruncation, trunc2::NoTruncation) = trunc1
# Base.:|(trunc1::NoTruncation, trunc2::TruncationScheme) = trunc2
#
# Base.:&(trunc1::NoTruncation, trunc2::NoTruncation) = trunc1
# Base.:&(trunc1::NoTruncation, trunc2::TruncationScheme) = trunc1
#
# Base.:|(trunc1::TruncationError, trunc2::TruncationError) = TruncationError(max(eps(trunc1), eps(trunc2)))
# Base.:&(trunc1::TruncationError, trunc2::TruncationError) = TruncationError(min(eps(trunc1), eps(trunc2)))
#
#
# Base.:|(trunc1::TruncationDimension, trunc2::TruncationDimension) = TruncationDimension(min(dim(trunc1), dim(trunc2)), max(eps(trunc1), eps(trunc2)))
# Base.:&(trunc1::TruncationDimension, trunc2::TruncationDimension) = TruncationDimension(max(dim(trunc1), dim(trunc2)), min(eps(trunc1), eps(trunc2)))
#
# Base.:|(trunc1::TruncationDimension, trunc2::TruncationError) = TruncationDimension(dim(trunc1), max(eps(trunc1), eps(trunc2)))
# Base.:&(trunc1::TruncationDimension, trunc2::TruncationError) = TruncationError(min(eps(trunc1), eps(trunc2)))
#
# Base.&(trunc1::TruncationSpace, trunc2::TruncationError) = TruncationSpace(space(trunc1), max(eps(trunc1), eps(trunc2)))
# Base.&(trunc2::TruncationError, trunc1::TruncationSpace) = TruncationSpace(space(trunc1), max(eps(trunc1), eps(trunc2)))
