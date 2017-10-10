struct StridedView{T,N,A<:DenseArray{T}} <: DenseArray{T,N}
    data::A
    size::NTuple{N,Int}
    strides::NTuple{N,Int}
    offset::Int
    conj::Bool
end
StridedView(a::A,
            size::NTuple{N,Int} = size(a),
            strides::NTuple{N,Int} = strides(a),
            offset::Int = 0, conj::Bool = false) where {T,N,A<:DenseArray{T}} =
    StridedView{T,N,A}(a, size, strides, offset, conj)

function StridedView(a::SubArray)
    @assert isa(a, StridedArray)
    return StridedView(parent(a), size(a), strides(a), Base.first_index(a)-1)
end

Base.parent(a::StridedView) = a.size
Base.first_index(a::StridedView) = a.offset+1
Base.size(a::StridedView) = a.size
Base.strides(a::StridedView) = a.strides
@inline function Base.getindex(a::StridedView{<:Any,N}, I::Vararg{Int,N}) where {N}
    @boundscheck checkbounds(a, I...)
    @inbounds r = a.data[a.offset+_computeind(I, a.strides)]
    return r
end
@inline function Base.setindex!(a::StridedView{<:Any,N}, v, I::Vararg{Int,N}) where {N}
    @boundscheck checkbounds(a, I...)
    @inbounds a.data[a.offset+_computeind(I, a.strides)] = v
    return a
end

Base.similar(a::StridedView, args...) = similar(a.data, args...)
Base.unsafe_convert(::Type{Ptr{T}}, a::StridedView{T}) where {T} = pointer(a.data, a.offset+1)

function Base.permutedims(a::StridedView{<:Any,N}, p) where {N}
    (length(p) == N && isperm(p)) || throw(ArgumentError("Invalid permutation of length $N: $p"))
    newsize = ntuple(n->a.size[p[n]], Val(N))
    newstrides = ntuple(n->a.strides[p[n]], Val(N))
    return StridedView(a.data, newsize, newstrides, a.offset)
end

const SizeType = Union{Int, Tuple{Vararg{Int}}}

function splitdims(a::DenseArray{<:Any,N}, newsizes::Vararg{SizeType,N}) where {N}
    map(prod, newsizes) == size(a) || throw(DimensionMismatch())
    newstrides = _computestrides(strides(a), newsizes)
    newsize = _flatten(newsizes...)
    return StridedView(a, newsize, newstrides, 0)
end
function splitdims(a::SubArray{<:Any,N}, newsizes::Vararg{SizeType,N}) where {N}
    @assert isa(a, StridedArray)
    map(prod, newsizes) == size(a) || throw(DimensionMismatch())
    newstrides = _computestrides(strides(a), newsizes)
    newsize = _flatten(newsizes...)
    return StridedView(Base.parent(a), newsize, newstrides, Base.first_index(a)-1)
end
function splitdims(a::StridedView{<:Any,N}, newsizes::Vararg{SizeType,N}) where {N}
    map(prod, newsizes) == size(a) || throw(DimensionMismatch())
    newstrides = _computestrides(strides(a), newsizes)
    newsize = _flatten(newsizes...)
    return StridedView(a.data, newsize, newstrides, a.offset)
end


using Base.tail

@inline _computeind(indices::Tuple{}, strides::Tuple{}) = 1
@inline _computeind(indices::NTuple{N,Int}, strides::NTuple{N,Int}) where {N} = (indices[1]-1)*strides[1] + _computeind(tail(indices), tail(strides))

@inline _computestrides(stride::Int, size::Int) = (stride,)
@inline _computestrides(stride::Int, size::Tuple{}) = ()
@inline _computestrides(stride::Int, size::Tuple{Vararg{Int}}) = (stride, _computestrides(stride*size[1], tail(size))...)

@inline _computestrides(strides::Tuple{Int}, sizes::Tuple{SizeType}) = _computestrides(strides[1], sizes[1])
@inline _computestrides(strides::NTuple{N,Int}, sizes::NTuple{N,SizeType}) where {N} =
    (_computestrides(strides[1], sizes[1])..., _computestrides(tail(strides), tail(sizes))...)

@inline _flatten(t::Tuple) = t
@inline _flatten(t) = (t,)
@inline _flatten(a, args...) = (_flatten(a)..., _flatten(args...)...)
