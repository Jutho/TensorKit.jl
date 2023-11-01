"""
    randuniform([::Type{T}=Float64], dims::Dims{N}) -> Array{T,N}

Create an array of size `dims` with random entries uniformly distributed in the allowed
values of `T`.

See also [`randnormal`](@ref), [`randisometry`](@ref) and[`randhaar`](@ref).
"""
randuniform(dims::Base.Dims) = randuniform(Float64, dims)
randuniform(::Type{T}, dims::Base.Dims) where {T<:Number} = rand(T, dims)

"""
    randnormal([::Type{T}=Float64], dims::Dims{N}) -> Array{T,N}

Create an array of size `dims` with random entries normally distributed.

See also [`randuniform`](@ref), [`randisometry`](@ref) and[`randhaar`](@ref).
"""
randnormal(dims::Base.Dims) = randnormal(Float64, dims)
randnormal(::Type{T}, dims::Base.Dims) where {T<:Number} = randn(T, dims)

"""
    randisometry([::Type{T}=Float64], dims::Dims{2}) -> Array{T,2}
    randhaar([::Type{T}=Float64], dims::Dims{2}) -> Array{T,2}

Create a random isometry of size `dims`.

See also [`randuniform`](@ref) and [`randnormal`](@ref).
"""
randisometry(dims::Base.Dims{2}) = randisometry(Float64, dims)
function randisometry(::Type{T}, dims::Base.Dims{2}) where {T<:Number}
    return dims[1] >= dims[2] ? _leftorth!(randnormal(T, dims), QRpos(), 0)[1] :
           throw(DimensionMismatch("cannot create isometric matrix with dimensions $dims; isometry needs to be tall or square"))
end

const randhaar = randisometry
