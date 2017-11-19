randuniform(dims::Base.Dims) = randuniform(Float64, dims)
randuniform(::Type{T}, dims::Base.Dims) where {T<:Number} = rand(T, dims)

randnormal(dims::Base.Dims) = randnormal(Float64, dims)
function randnormal(::Type{T}, dims::Base.Dims) where {T<:Number}
    if T <: Real
        return randn(T, dims)
    else
        return complex(randn(real(T), dims), randn(real(T), dims))
    end
end

randisometry(dims::Base.Dims) = randisometry(Float64, dims)
randisometry(::Type{T}, dims::Base.Dims) where {T<:Number} = leftorth!(randnormal(T, dims), QRpos())[1]
