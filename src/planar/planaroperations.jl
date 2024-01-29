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

function planarcontract!(C::AbstractTensorMap{S},
                         A::AbstractTensorMap{S},
                         pA::Index2Tuple{N₁,N₃},
                         B::AbstractTensorMap{S},
                         pB::Index2Tuple{N₃,N₂},
                         pAB::Index2Tuple,
                         α::Number,
                         β::Number,
                         backend::Backend...) where {S,N₁,N₂,N₃}
    if BraidingStyle(sectortype(S)) == Bosonic()
        return contract!(C, A, pA, B, pB, pAB, α, β, backend...)
    end

    indA = (codomainind(A), reverse(domainind(A)))
    indB = (codomainind(B), reverse(domainind(B)))
    pA′, pB′, pAB′ = reorder_indices(indA, pA, indB, pB, pAB)


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
    oindAinC = TupleTools.getindices(ipAB, ntuple(n -> n, N₁))
    oindBinC = TupleTools.getindices(ipAB, ntuple(n -> n + N₁, N₂))

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


function _iscyclicpermutation(v1, v2)
    length(v1) == length(v2) || return false
    return iscyclicpermutation(_indexin(v1, v2))
end

function _findsetcircshift(p_cyclic, p_subset)
    N = length(p_cyclic)
    M = length(p_subset)
    i = findfirst(0:(N - 1)) do i
        return issetequal(TupleTools.getindices(p_cyclic, ntuple(n -> mod1(n + i, N), M)),
                          p_subset)
    end
    isnothing(i) && throw(ArgumentError("no cyclic permutation of $p_cyclic that matches $p_subset"))
    return i-1::Int
end

function reorder_planar_indices(indA, pA, indB, pB, pAB)
    NA₁ = length(pA[1])
    NA₂ = length(pA[2])
    NA = NA₁ + NA₂
    NB₁ = length(pB[1])
    NB₂ = length(pB[2])
    NB = NB₁ + NB₂
    NAB₁ = length(pAB[1])
    NAB₂ = length(pAB[2])
    NAB = NAB₁ + NAB₂
    
    # input checks
    @assert NA == length(indA[1]) + length(indA[2])
    @assert NB == length(indB[1]) + length(indB[2])
    @assert NA₂ == NB₁
    @assert NAB == NA₁ + NB₂

    # find circshift index of pAB if considered as shifting sets
    indAB = (ntuple(identity, NAB₁),
             reverse(ntuple(n -> n + NAB₁, NAB₂)))
    indAB_lin = (indAB[1]..., indAB[2]...)
    iAB = _findsetcircshift(indAB_lin, pAB[1])
    @assert iAB == _findsetcircshift(indAB_lin, pAB[2]) - NAB₁ "sanity check"

    # migrate permutations from pAB to pA and pB
    permA = TupleTools.getindices((pAB[1]..., reverse(pAB[2])...),
                                        ntuple(n -> mod1(n + iAB, NAB), NA₁))
    permB = reverse(TupleTools.getindices((pAB[1]..., reverse(pAB[2])...),
                                          ntuple(n -> mod1(n + iAB + NA₁, NAB), NB₂)) .-
                    NA₁)

    pA′ = (TupleTools.getindices(pA[1], permA), pA[2])
    pB′ = (pB[1], reverse(TupleTools.getindices(reverse(pB[2]), permB)))
    pAB′ = (ntuple(n -> n + iAB, NAB₁),
           ntuple(n -> n + iAB + NAB₁, NAB₂))

    # fix permutations of contracted indices
    if NA₂ > 0
        indA_lin = (indA[1]..., indA[2]...)
        iA = _findsetcircshift(indA_lin, pA′[1])
        @assert iA == _findsetcircshift(indA_lin, pA′[2]) - NA₁ "sanity check"
        pA′ = (pA′[1],
               reverse(TupleTools.getindices(linearize(indA),
                                             ntuple(n -> mod1(n + iA + NA₁, NA), NA₂))))

        indB_lin = (indB[1]..., indB[2]...)
        iB = _findsetcircshift(indB_lin, pB′[1])
        @assert iB == _findsetcircshift(indB_lin, pB′[2]) - NB₁ "sanity check"
        pB′ = (TupleTools.getindices(linearize(indB),
                                     ntuple(n -> mod1(n + iB, NB), NB₁)),
               pB′[2])
    end

    # make sure this is still the same contraction
    @assert issetequal(pA[1], pA′[1]) && issetequal(pA[2], pA′[2])
    @assert issetequal(pB[1], pB′[1]) && issetequal(pB[2], pB′[2])
    @assert issetequal(pAB[1], pAB′[1]) && issetequal(pAB[2], pAB′[2])
    @assert issetequal(tuple.(pA[2], pB[1]), tuple.(pA′[2], pB′[1]))
    
    # make sure that everything is now planar
    @assert _iscyclicpermutation((indA[1]..., (indA[2])...),
                                 (pA′[1]..., reverse(pA′[2])...))
    @assert _iscyclicpermutation((indB[1]..., (indB[2])...),
                               (pB′[1]..., reverse(pB′[2])...))
    @assert _iscyclicpermutation((indAB[1]..., (indAB[2])...),
                               (pAB′[1]..., reverse(pAB′[2])...))
    
    return pA′, pB′, pAB′
end
