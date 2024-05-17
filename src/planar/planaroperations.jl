# ----------
# CONJ FLAGS
# ----------

function planaradd!(C::AbstractTensorMap{S},
                    A::AbstractTensorMap{S}, pA::Index2Tuple, conjA::Symbol,
                    α::Number, β::Number, backend::Backend...) where {S}
    if conjA == :N
        A′ = A
        p′ = _canonicalize(pA, C)
    elseif conjA == :C
        A′ = adjoint(A)
        p′ = adjointtensorindices(A, _canonicalize(pA, C))
    else
        throw(ArgumentError("unknown conjugation flag $conjA"))
    end
    return add_transpose!(C, A′, p′, α, β, backend...)
end

function planartrace!(C::AbstractTensorMap{S},
                      A::AbstractTensorMap{S}, p::Index2Tuple, q::Index2Tuple,
                      conjA::Symbol,
                      α::Number, β::Number, backend::Backend...) where {S}
    if conjA == :N
        A′ = A
        p′ = _canonicalize(p, C)
        q′ = q
    elseif conjA == :C
        A′ = A'
        p′ = adjointtensorindices(A, _canonicalize(p, C))
        q′ = adjointtensorindices(A, q)
    else
        throw(ArgumentError("unknown conjugation flag $conjA"))
    end

    return trace_transpose!(C, A′, p′, q′, α, β, backend...)
end

function planarcontract!(C::AbstractTensorMap,
                         A::AbstractTensorMap, pA::Index2Tuple, conjA::Symbol,
                         B::AbstractTensorMap, pB::Index2Tuple, conjB::Symbol,
                         pAB::Index2Tuple,
                         α::Number, β::Number, backend::Backend...)
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

    return _planarcontract!(C, A′, pA′, B′, pB′, pAB, α, β, backend...)
end

# ---------------
# IMPLEMENTATIONS
# ---------------

function trace_transpose!(tdst::AbstractTensorMap{S,N₁,N₂},
                          tsrc::AbstractTensorMap{S},
                          (p₁, p₂)::Index2Tuple{N₁,N₂}, (q₁, q₂)::Index2Tuple{N₃,N₃},
                          α::Number, β::Number, backend::Backend...) where {S,N₁,N₂,N₃}
    @boundscheck begin
        space(tdst) == permute(space(tsrc), (p₁, p₂)) ||
            throw(SpaceMismatch("trace: tsrc = $(codomain(tsrc))←$(domain(tsrc)),
                    tdst = $(codomain(tdst))←$(domain(tdst)), p₁ = $(p₁), p₂ = $(p₂)"))
        all(i -> space(tsrc, q₁[i]) == dual(space(tsrc, q₂[i])), 1:N₃) ||
            throw(SpaceMismatch("trace: tsrc = $(codomain(tsrc))←$(domain(tsrc)),
                    q₁ = $(q₁), q₂ = $(q₂)"))
        # TODO: check planarity?
    end

    # TODO: not sure if this is worth it
    if BraidingStyle(sectortype(S)) == Bosonic()
        return @inbounds trace_permute!(tdst, tsrc, p, q, α, β, backend...)
    end

    scale!(tdst, β)
    β′ = One()

    for (f₁, f₂) in fusiontrees(tsrc)
        @inbounds A = tsrc[f₁, f₂]
        for ((f₁′, f₂′), coeff) in planar_trace(f₁, f₂, p₁, p₂, q₁, q₂)
            @inbounds C = tdst[f₁′, f₂′]
            TO.tensortrace!(C, (p₁, p₂), A, (q₁, q₂), :N, α * coeff, β′, backend...)
        end
    end

    return tdst
end

# TODO: reuse the same memcost checks as in `contract!`
function _planarcontract!(C::AbstractTensorMap{S},
                          A::AbstractTensorMap{S}, pA::Index2Tuple{N₁,N₃},
                          B::AbstractTensorMap{S}, pB::Index2Tuple{N₃,N₂},
                          pAB::Index2Tuple,
                          α::Number, β::Number, backend::Backend...) where {S,N₁,N₂,N₃}
    indA = (codomainind(A), reverse(domainind(A)))
    indB = (codomainind(B), reverse(domainind(B)))
    indAB = (ntuple(identity, N₁), reverse(ntuple(i -> i + N₁, N₂)))

    # TODO: avoid this step once @planar is reimplemented
    pA′, pB′, pAB′ = reorder_planar_indices(indA, pA, indB, pB, pAB)

    @assert _isplanar(indA, pA′) "not a planar contraction (indA = $indA, pA′ = $pA′)"
    @assert _isplanar(indB, pB′) "not a planar contraction (indB = $indB, pB′ = $pB′)"
    @assert _isplanar(indAB, pAB′) "not a planar contraction (indAB = $indAB, pAB′ = $pAB′)"

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
