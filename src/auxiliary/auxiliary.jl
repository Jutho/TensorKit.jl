using Base: tuple_type_head, tuple_type_tail, tuple_type_cons, tail, front, setindex

@inline argtail2(a, b, c...) = c
@inline tail2(x::Tuple) = argtail2(x...)
@inline unsafe_tail(t::Tuple{}) = t
@inline unsafe_tail(t::Tuple) = tail(t)

@inline unsafe_front(t::Tuple{}) = t
@inline unsafe_front(t::Tuple) = front(t)

tpermute(t::NTuple{N,T}, p::NTuple{N,Int}) where {T,N} = tselect(t, p)
tselect(t::Tuple, ind::NTuple{N,Int}) where {N} = ntuple(n->t[ind[n]], Val(N))

Base.@pure valadd(::Val{N1},::Val{N2}) where {N1,N2} = Val{N1+N2}()
Base.@pure valsub(::Val{N1},::Val{N2}) where {N1,N2} = N1 > N2 ? Val{N1-N2}() : Val{0}()


function permutation2swaps(perm)
    p = collect(perm)
    swaps = Vector{Int}()
    N = length(p)
    for k = 1:N-1
        append!(swaps, p[k]-1:-1:k)
        for l = k+1:N
            if p[l] < p[k]
                p[l] += 1
            end
        end
        p[k] = k
    end
    return swaps
end

mul!(C::StridedMatrix, β::Number, A::StridedMatrix, B::StridedMatrix, α::Number) =
    LinAlg.BLAS.gemm!('N', 'N', convert(eltype(C), α), A, B, convert(eltype(C), β), C)
