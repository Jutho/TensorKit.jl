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
    epsilon::Real
end
truncdim(D::Int)=TruncationDimension(D,0)
dim(t::TruncationDimension)=t.D
Base.eps(t::TruncationDimension)=t.epsilon

Base.&(trunc1::TruncationDimension,trunc2::TruncationError)=TruncationDimension(dim(trunc1),max(eps(trunc1),eps(trunc2)))
Base.&(trunc2::TruncationError,trunc1::TruncationDimension)=TruncationDimension(dim(trunc1),max(eps(trunc1),eps(trunc2)))

immutable TruncationSpace{S<:ElementarySpace} <: TruncationScheme
    space::S
    epsilon::Real
end
truncspace(space::ElementarySpace) = TruncationSpace(space,0)
space(s::TruncationSpace) = s.space
dim(s::TruncationSpace) = dim(s.space)
Base.eps(t::TruncationSpace)=t.epsilon

Base.&(trunc1::TruncationSpace,trunc2::TruncationError)=TruncationSpace(space(trunc1),max(eps(trunc1),eps(trunc2)))
Base.&(trunc2::TruncationError,trunc1::TruncationSpace)=TruncationSpace(space(trunc1),max(eps(trunc1),eps(trunc2)))
