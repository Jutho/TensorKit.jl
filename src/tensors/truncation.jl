# truncation.jl
#
# Implements truncation schemes for truncating a tensor with svd, leftorth or rightorth

abstract TruncationScheme

immutable NoTruncation <: TruncationScheme
end
notrunc()=NoTruncation()

immutable TruncationError <: TruncationScheme
    epsilon::Real
end
truncerr(epsilon::Real)=TruncationError(epsilon)
Base.eps(t::TruncationError)=t.epsilon

immutable TruncationDimension <: TruncationScheme
    D::Integer
end
truncdim(D::Int)=TruncationDimension(D)
dim(t::TruncationDimension)=t.D

immutable TruncationSpace{S<:ElementarySpace} <: TruncationScheme
    space::S
end
truncspace(space::ElementarySpace) = TruncationSpace(space)
space(s::TruncationSpace) = s.space
dim(s::TruncationSpace) = dim(s.space)
