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
    pA′, pB′, pAB′ = reorder_planar_indices(indA, pA, indB, pB, pAB)
    
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
    indAB = (ntuple(identity, NA₁), reverse(ntuple(n -> n + NA₁, NB₂)))

    if NAB > 0
        indAB_lin = (indAB[1]..., indAB[2]...)
        iAB = _findsetcircshift(indAB_lin, pAB[1])
        @assert iAB == mod(_findsetcircshift(indAB_lin, pAB[2]) - NAB₁, NAB) "sanity check"
        indAB_lin = _circshift(indAB_lin, -iAB)
        # migrate permutations from pAB to pA and pB
        
        pAB_lin = (pAB[1]..., reverse(pAB[2])...)
        permA = getindices(pAB_lin,
                           ntuple(n -> mod1(n - iAB, NAB), NA₁))
        permB = reverse(getindices(pAB_lin,
                                   ntuple(n -> mod1(n - iAB + NA₁, NAB), NB₂)) .- NA₁)

        pA′ = (getindices(pA[1], permA), pA[2])
        pB′ = (pB[1], getindices(pB[2], permB))
        pAB′ = (getindices(indAB_lin, ntuple(n -> mod1(n, NAB), NAB₁)), 
               reverse(getindices(indAB_lin, ntuple(n -> mod1(n + NAB₁, NAB), NAB₂))))
    else
        pA′ = pA
        pB′ = pB
        pAB′ = pAB
    end

    # cycle indA to be of the form (oindA..., reverse(cindA)...)
    indA_lin = (indA[1]..., indA[2]...)
    if NA₁ != 0
        iA = findfirst(==(first(pA′[1])), indA_lin) - 1
        indA_lin = _circshift(indA_lin, -iA)
    end
    pc = ntuple(identity, NA₂)
    @assert all(getindices(indA_lin, ntuple(identity, NA₁)) .== pA′[1]) "sanity check"
    pA′ = (pA′[1], reverse(getindices(indA_lin, pc .+ NA₁)))

    # cycle indB to be of the form (cindB..., reverse(oindB)...)
    indB_lin = (indB[1]..., indB[2]...)
    if NB₂ != 0
        iB = findfirst(==(first(pB′[2])), indB_lin)
        indB_lin = _circshift(indB_lin, -iB)
    end
    @assert all(getindices(indB_lin, ntuple(identity, NB₂) .+ NB₁) .== reverse(pB′[2])) "sanity check"
    pB′ = (getindices(indB_lin, pc), pB′[2])

    # if uncontracted indices are empty, we can still make cyclic adjustments
    if NA₁ == 0 && NA₂ != 0
        hit = pA[2][findfirst(==(first(pB′[1])), pB[1])]
        shiftA = findfirst(==(hit), pA′[2]) - 1
        pA′ = (pA′[1], _circshift(pA′[2], -shiftA))
    end

    if NB₂ == 0 && NB₁ != 0
        hit = pB[1][findfirst(==(first(pA′[2])), pA[2])]
        shiftB = findfirst(==(hit), pB′[1]) - 1
        pB′ = (_circshift(pB′[1], -shiftB), pB′[2])
    end

    # make sure this is still the same contraction
    @assert issetequal(pA[1], pA′[1]) && issetequal(pA[2], pA′[2])
    @assert issetequal(pB[1], pB′[1]) && issetequal(pB[2], pB′[2])
    # @assert issetequal(pAB[1], pAB′[1]) && issetequal(pAB[2], pAB′[2]) "pAB = $pAB, pAB′ = $pAB′"
    @assert issetequal(tuple.(pA[2], pB[1]), tuple.(pA′[2], pB′[1])) "$pA $pB $pA′ $pB′"

    # make sure that everything is now planar
    @assert _iscyclicpermutation((indA[1]..., (indA[2])...),
                                 (pA′[1]..., reverse(pA′[2])...)) "indA = $indA, pA′ = $pA′"
    @assert _iscyclicpermutation((indB[1]..., (indB[2])...),
                                 (pB′[1]..., reverse(pB′[2])...)) "indB = $indB, pB′ = $pB′"
    @assert _iscyclicpermutation((indAB[1]..., (indAB[2])...),
                                 (pAB′[1]..., reverse(pAB′[2])...)) "indAB = $indAB, pAB′ = $pAB′"

    return pA′, pB′, pAB′
end
