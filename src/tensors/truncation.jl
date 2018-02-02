# truncation.jl
#
# Implements truncation schemes for truncating a tensor with svd, leftorth or rightorth

abstract type TruncationScheme end

struct NoTruncation <: TruncationScheme
end
notrunc() = NoTruncation()

struct TruncationError{T<:Real} <: TruncationScheme
    ϵ::T
end
truncerr(epsilon::Real) = TruncationError(epsilon)

struct TruncationDimension <: TruncationScheme
    dim::Int
end
truncdim(d::Int) = TruncationDimension(d)

struct TruncationSpace{S<:ElementarySpace} <: TruncationScheme
    space::S
end
truncspace(space::ElementarySpace) = TruncationSpace(space)

struct TruncateBelow{T<:Real} <: TruncationScheme
    ϵ::T
end
truncbelow(epsilon::Real) = TruncateBelow(epsilon)
