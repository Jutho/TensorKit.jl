# planar versions of tensor operations add!, trace! and contract!
function planaradd!(C::AbstractTensorMap{S,N₁,N₂},
                    A::AbstractTensorMap{S},
                    p::Index2Tuple{N₁,N₂},
                    α::Number,
                    β::Number,
                    backend::Backend...) where {S,N₁,N₂}
    return add_transpose!(C, A, p, α, β, backend...)
end

function planartrace!(C::AbstractTensorMap{S,N₁,N₂},
                      A::AbstractTensorMap{S},
                      p::Index2Tuple{N₁,N₂},
                      q::Index2Tuple{N₃,N₃},
                      α::Number,
                      β::Number,
                      backend::Backend...) where {S,N₁,N₂,N₃}
    if BraidingStyle(sectortype(S)) == Bosonic()
        return trace_permute!(C, A, p, q, α, β, backend...)
    end

    @boundscheck begin
        all(i -> space(A, p[1][i]) == space(C, i), 1:N₁) ||
            throw(SpaceMismatch("trace: A = $(codomain(A))←$(domain(A)),
                    C = $(codomain(C))←$(domain(C)), p1 = $(p1), p2 = $(p2)"))
        all(i -> space(A, p[2][i]) == space(C, N₁ + i), 1:N₂) ||
            throw(SpaceMismatch("trace: A = $(codomain(A))←$(domain(A)),
                    C = $(codomain(C))←$(domain(C)), p1 = $(p1), p2 = $(p2)"))
        all(i -> space(A, q[1][i]) == dual(space(A, q[2][i])), 1:N₃) ||
            throw(SpaceMismatch("trace: A = $(codomain(A))←$(domain(A)),
                    q1 = $(q1), q2 = $(q2)"))
    end

    if iszero(β)
        fill!(C, β)
    elseif !isone(β)
        rmul!(C, β)
    end
    for (f₁, f₂) in fusiontrees(A)
        for ((f₁′, f₂′), coeff) in planar_trace(f₁, f₂, p..., q...)
            TO.tensortrace!(C[f₁′, f₂′], p, A[f₁, f₂], q, :N, α * coeff, true, backend...)
        end
    end
    return C
end

function planarcontract!(C::AbstractTensorMap,
                         A::AbstractTensorMap, pA::Index2Tuple, conjA::Symbol,
                         B::AbstractTensorMap, pB::Index2Tuple, conjB::Symbol,
                         pAB::Index2Tuple,
                         α::Number, β::Number, backend::Backend...)
    # get rid of conj arguments by going to adjoint tensormaps
    if conjA == :N
        A′ = A
        pA′ = pA
    elseif conjA == :C
        A′ = A'
        pA′ = adjointtensorindices(A, pA)
    else
        throw(ArgumentError("unknown conjugation flag $conjA"))
    end
    if conjB == :N
        B′ = B
        pB′ = pB
    elseif conjB == :C
        B′ = B'
        pB′ = adjointtensorindices(B, pB)
    else
        throw(ArgumentError("unknown conjugation flag $conjB"))
    end

    return planarcontract!(C, A, pA′, B, pB′, pAB, α, β, backend...)
end

function _isplanar(inds::Index2Tuple, p::Index2Tuple)
    return iscyclicpermutation((inds[1]..., inds[2]...),
                               (p[1]..., reverse(p[2])...))
end

function planarcontract!(C::AbstractTensorMap{S},
                         A::AbstractTensorMap{S},
                         pA::Index2Tuple{N₁,N₃},
                         B::AbstractTensorMap{S},
                         pB::Index2Tuple{N₃,N₂},
                         pAB::Index2Tuple,
                         α::Number,
                         β::Number,
                         backend::Backend...) where {S,N₁,N₂,N₃}
    indA = (codomainind(A), reverse(domainind(A)))
    indB = (codomainind(B), reverse(domainind(B)))
    indC = (codomainind(C), reverse(domainind(C)))
    pA′, pB′, pAB′ = reorder_planar_indices(indA, pA, indB, pB, pAB)

    @assert _isplanar(indA, pA′) "not a planar contraction (indA = $indA, pA′ = $pA′)"
    @assert _isplanar(indB, pB′) "not a planar contraction (pB′ = $pB′)"
    @assert _isplanar(indC, pAB′) "not a planar contraction (pAB′ = $pAB′)"

    if BraidingStyle(sectortype(spacetype(C))) == Bosonic()
        return contract!(C, A, pA′, B, pB′, pAB′, α, β, backend...)
    end

    if pA′ == (codomainind(A), domainind(A))
        A′ = A
    else
        A′ = TO.tensoralloc_add(scalartype(A), pA′, A, :N, true)
        add_transpose!(A′, A, pA′, true, false, backend...)
    end

    if pB′ == (codomainind(B), domainind(B))
        B′ = B
    else
        B′ = TO.tensoralloc_add(scalartype(B), pB′, B, :N, true)
        add_transpose!(B′, B, pB′, true, false, backend...)
    end

    ipAB = TupleTools.invperm(linearize(pAB′))
    oindAinC = getindices(ipAB, ntuple(n -> n, N₁))
    oindBinC = getindices(ipAB, ntuple(n -> n + N₁, N₂))

    if has_shared_permute(C, (oindAinC, oindBinC))
        C′ = transpose(C, (oindAinC, oindBinC))
        mul!(C′, A′, B′, α, β)
    else
        C′ = A′ * B′
        add_transpose!(C, C′, pAB′, α, β)
    end

    pA′ == (codomainind(A), domainind(A)) || TO.tensorfree!(A′)
    pB′ == (codomainind(B), domainind(B)) || TO.tensorfree!(B′)

    return C
end

function planarcontract(A::AbstractTensorMap{S}, pA::Index2Tuple,
                        B::AbstractTensorMap{S}, pB::Index2Tuple,
                        pAB::Index2Tuple{N₁,N₂},
                        α::Number, backend::Backend...) where {S,N₁,N₂}
    TC = TO.promote_contract(scalartype(A), scalartype(B), scalartype(α))
    C = TO.tensoralloc_contract(TC, pAB, A, pA, :N, B, pB, :N)
    return planarcontract!(C, A, pA, B, pB, pAB, α, VectorInterface.Zero(), backend...)
end

# auxiliary routines
_cyclicpermute(t::Tuple) = (Base.tail(t)..., t[1])
_cyclicpermute(t::Tuple{}) = ()

_circshift(::Tuple{}, ::Int) = ()
_circshift(t::Tuple, n::Int) = ntuple(i -> t[mod1(i - n, length(t))], length(t))

_indexin(v1, v2) = ntuple(n -> findfirst(isequal(v1[n]), v2), length(v1))

function _iscyclicpermutation(v1, v2)
    length(v1) == length(v2) || return false
    return iscyclicpermutation(_indexin(v1, v2))
end

function _findsetcircshift(p_cyclic, p_subset)
    N = length(p_cyclic)
    M = length(p_subset)
    N == M == 0 && return 0
    i = findfirst(0:(N - 1)) do i
        return issetequal(getindices(p_cyclic, ntuple(n -> mod1(n + i, N), M)),
                          p_subset)
    end
    isnothing(i) &&
        throw(ArgumentError("no cyclic permutation of $p_cyclic that matches $p_subset"))
    return i - 1::Int
end
