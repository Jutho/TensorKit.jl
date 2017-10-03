# truncation.jl
#
# Implements truncation schemes for truncating a tensor with svd, leftorth or rightorth

abstract type TruncationScheme end

struct NoTruncation <: TruncationScheme
end
notrunc() = NoTruncation()

struct TruncationError{T<:Real, S<:Real} <: TruncationScheme
    ϵ::T
    p::S
end
truncerr(epsilon::Real, p::Real = 2) = TruncationError(epsilon, p)

struct TruncationDimension <: TruncationScheme
    dim::Int
end
truncdim(d::Int) = TruncationDimension(d)

struct TruncationSpace{S<:ElementarySpace} <: TruncationScheme
    space::S
end
truncspace(space::ElementarySpace) = TruncationSpace(space)
