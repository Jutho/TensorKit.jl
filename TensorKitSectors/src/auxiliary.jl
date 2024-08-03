# kronecker product with arbitrary number of dimensions
_kron(A, B, C, D...) = _kron(_kron(A, B), C, D...)
function _kron(A, B)
    sA = size(A)
    sB = size(B)
    s = map(*, sA, sB)
    C = similar(A, promote_type(eltype(A), eltype(B)), s)
    for IA in eachindex(IndexCartesian(), A)
        for IB in eachindex(IndexCartesian(), B)
            I = CartesianIndex(IB.I .+ (IA.I .- 1) .* sB)
            C[I] = A[IA] * B[IB]
        end
    end
    return C
end

# Manhattan based distance enumeration: I is supposed to be one-based index
# TODO: is there any way to make this faster?

# forward mapping from multidimensional to single Manhattan index
@inline function num_manhattan_points(d::Int, sz::Dims{1})
    return Int(sz[1] > d)
end
@inline function num_manhattan_points(d::Int, sz::Dims{N}) where {N}
    d == 0 && return 1
    num = 0
    for i in 1:min(sz[1], d + 1)
        num += num_manhattan_points(d - i + 1, Base.tail(sz))
    end
    return num
end

@inline localoffset(d, I::Tuple{Int}, sz::Tuple{Int}) = 0
@inline function localoffset(d, I, sz)
    offset = 0
    for i in 1:(I[1] - 1)
        offset += num_manhattan_points(d - i + 1, Base.tail(sz))
    end
    offset += localoffset(d - I[1] + 1, Base.tail(I), Base.tail(sz))
    return offset
end

to_manhattan_index(::Tuple{}, ::Tuple{}) = 1
to_manhattan_index(I::Tuple{Int}, ::Tuple{Int}) = I[1]
function to_manhattan_index(I::Dims{N}, sz::Dims{N}) where {N}
    d = sum(I) - N # Manhattan distance to origin for one-based indices
    d == 0 && return 1
    index = 1
    # count all the points with smaller Manhatten distance
    for k in 0:(d - 1)
        index += num_manhattan_points(k, sz)
    end
    return index + localoffset(d, I, sz)
end

# inverse mapping
@inline invertlocaloffset(d, offset, sz::Tuple{Int}) = (d + 1,)
@inline function invertlocaloffset(d, offset, sz)
    i₁ = 1
    while i₁ < sz[1]
        jump = num_manhattan_points(d - i₁ + 1, Base.tail(sz))
        if offset < jump
            break
        end
        offset -= jump
        i₁ += 1
    end
    return (i₁, invertlocaloffset(d - i₁ + 1, offset, Base.tail(sz))...)
end

manhattan_to_multidimensional_index(index::Int, ::Dims{0}) = ()
manhattan_to_multidimensional_index(index::Int, ::Dims{1}) = (index,)
function manhattan_to_multidimensional_index(index::Int, sz::Dims{N}) where {N}
    # find the layer of the Manhattan distance
    index == 1 && return ntuple(one, Val(N))
    offset = 1
    d = 1
    while true
        currentlayer = num_manhattan_points(d, sz)
        if index <= offset + currentlayer
            break
        end
        d += 1
        offset += currentlayer
    end

    # find the position within the layer
    index -= offset
    return invertlocaloffset(d, index - 1, sz)
end
