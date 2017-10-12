using Base: tuple_type_head, tuple_type_tail, tuple_type_cons, tail, front, setindex


struct StaticLength{N}
end
Base.@pure StaticLength(N::Int) = StaticLength{N}()
Base.@pure Base.:+(::StaticLength{N₁}, ::StaticLength{N₂}) where {N₁,N₂} = StaticLength(N₁+N₂)
Base.@pure Base.:-(::StaticLength{N₁}, ::StaticLength{N₂}) where {N₁,N₂} = StaticLength(max(0,N₁-N₂))

if VERSION < v"0.7.0-DEV.843"
    @inline Base.ntuple(f, ::StaticLength{N}) where {N} = ntuple(f, Val{N})
else
    @inline Base.ntuple(f, ::StaticLength{N}) where {N} = ntuple(f, Val{N}())
end

@inline argtail2(a, b, c...) = c
@inline tail2(x::Tuple) = argtail2(x...)
@inline unsafe_tail(t::Tuple{}) = t
@inline unsafe_tail(t::Tuple) = tail(t)

@inline unsafe_front(t::Tuple{}) = t
@inline unsafe_front(t::Tuple) = front(t)

tpermute(t::NTuple{N,T}, p::NTuple{N,Int}) where {T,N} = tselect(t, p)
tselect(t::Tuple, ind::NTuple{N,Int}) where {N} = ntuple(n->t[ind[n]], StaticLength(N))

function linearizepermutation(p1::NTuple{N₁,Int}, p2::NTuple{N₂}, n₁::Int, n₂::Int) where {N₁,N₂}
    p1′ = ntuple(StaticLength(N₁)) do n
        p1[n] > n₁ ? n₂+2n₁+1-p1[n] : p1[n]
    end
    p2′ = ntuple(StaticLength(N₂)) do n
        p2[N₂+1-n] > n₁ ? n₂+2n₁+1-p2[N₂+1-n] : p2[N₂+1-n]
    end
    return (p1′..., p2′...)
end

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
