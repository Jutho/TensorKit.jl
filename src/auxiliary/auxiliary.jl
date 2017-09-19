Base.@pure valadd(::Val{N1},::Val{N2}) where {N1,N2} = Val{N1+N2}()
Base.@pure valsub(::Val{N1},::Val{N2}) where {N1,N2} = N1 > N2 ? Val{N1-N2}() : Val{0}()

tpermute(t::NTuple{N,T}, p::NTuple{N,Int}) where {T,N} = tselect(t, p)
tselect(t::Tuple, ind::NTuple{N,Int}) where {N} = ntuple(n->t[ind[n]], Val(N))

@inline argtail2(a, b, c...) = c
@inline tail2(x::Tuple) = argtail2(x...)

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
