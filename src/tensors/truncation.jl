# truncation.jl
#
# Implements truncation strategies for truncating a tensor with svdtrunc!, leftorthtrunc! or rightorthtrunc!

abstract TruncationScheme

immutable NoTruncation <: TruncationScheme
end
notrunc()=NoTruncation()

immutable MaximalTruncationError <: TruncationScheme
    epsilon::Real
end
maxtruncerr(epsilon::Real)=MaximalTruncationError(epsilon)
Base.eps(t::MaximalTruncationError)=t.epsilon

immutable MaximalTruncationDimension <: TruncationScheme
    D::Integer
end
maxtruncdim(D::Int)=MaximalTruncationDimension(D)
dim(t::MaximalTruncationDimension)=t.D

immutable TruncationSpace{S<:ElementarySpace} <: TruncationScheme
    space::S
end
truncspace(space::ElementarySpace) = TruncationSpace(space)

space(s::TruncationSpace) = s.space
dim(s::TruncationSpace) = dim(s.space)
