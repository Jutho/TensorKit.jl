# planar versions of tensor operations add!, trace! and contract!
function planaradd!(C::AbstractTensorMap,
                    A::AbstractTensorMap,
                    p::Index2Tuple,
                    α::Number, β::Number,
                    backend::Backend...)
    return add_transpose!(C, A, p, α, β, backend...)
end

function planartrace!(C::AbstractTensorMap,
                      A::AbstractTensorMap,
                      (p₁, p₂)::Index2Tuple,
                      (q₁, q₂)::Index2Tuple,
                      α::Number,
                      β::Number,
                      backend::Backend...)
    (S = spacetype(C)) == spacetype(A) ||
        throw(SpaceMismatch("incompatible spacetypes"))
    if BraidingStyle(sectortype(S)) == Bosonic()
        return trace_permute!(C, A, p, q, α, β, backend...)
    end
    (N₃ = length(q₁)) == length(q₂) ||
        throw(IndexError("number of trace indices does not match"))
    N₁, N₂ = length(p₁), length(p₂)

    @boundscheck begin
        numout(C) == N₁ || throw(IndexError("number of output indices does not match"))
        numin(C) == N₂ || throw(IndexError("number of input indices does not match"))
        all(i -> space(A, p₁[i]) == space(C, i), 1:N₁) ||
            throw(SpaceMismatch("trace: A = $(space(A)),
                    C = $(space(C)), p₁ = $(p₁), p₂ = $(p₂)"))
        all(i -> space(A, p₂[i]) == space(C, N₁ + i), 1:N₂) ||
            throw(SpaceMismatch("trace: A = $(space(A)),
                    C = $(space(C)), p₁ = $(p₁), p₂ = $(p₂)"))
        all(i -> space(A, q₁[i]) == dual(space(A, q₂[i])), 1:N₃) ||
            throw(SpaceMismatch("trace: A = $(space(A)),
                    q1 = $(q₁), q2 = $(q₂)"))
    end

    if iszero(β)
        fill!(C, β)
    elseif !isone(β)
        rmul!(C, β)
    end
    for (f₁, f₂) in fusiontrees(A)
        for ((f₁′, f₂′), coeff) in planar_trace(f₁, f₂, p₁, p₂, q₁, q₂)
            TO.tensortrace!(C[f₁′, f₂′], (p₁, p₂), A[f₁, f₂], (q₁, q₂), :N, α * coeff, true,
                            backend...)
        end
    end
    return C
end

function planarcontract!(C::AbstractTensorMap,
                         A::AbstractTensorMap,
                         pA::Index2Tuple,
                         B::AbstractTensorMap,
                         pB::Index2Tuple,
                         pAB::Index2Tuple,
                         α::Number,
                         β::Number,
                         backend::Backend...)
    if BraidingStyle(sectortype(C)) == Bosonic()
        return contract!(C, A, pA, B, pB, pAB, α, β, backend...)
    end

    codA, domA = codomainind(A), domainind(A)
    codB, domB = codomainind(B), domainind(B)
    oindA, cindA = pA
    cindB, oindB = pB
    oindA, cindA, oindB, cindB = reorder_indices(codA, domA, codB, domB, oindA, cindA,
                                                 oindB, cindB, pAB...)

    if oindA == codA && cindA == domA
        A′ = A
    else
        A′ = TO.tensoralloc_add(scalartype(A), (oindA, cindA), A, :N, true)
        add_transpose!(A′, A, (oindA, cindA), true, false, backend...)
    end

    if cindB == codB && oindB == domB
        B′ = B
    else
        B′ = TensorOperations.tensoralloc_add(scalartype(B), (cindB, oindB), B, :N, true)
        add_transpose!(B′, B, (cindB, oindB), true, false, backend...)
    end
    mul!(C, A′, B′, α, β)
    (oindA == codA && cindA == domA) || TO.tensorfree!(A′)
    (cindB == codB && oindB == domB) || TO.tensorfree!(B′)

    return C
end

# auxiliary routines
_cyclicpermute(t::Tuple) = (Base.tail(t)..., t[1])
_cyclicpermute(t::Tuple{}) = ()

function reorder_indices(codA, domA, codB, domB, oindA, oindB, p1, p2)
    N₁ = length(oindA)
    N₂ = length(oindB)
    @assert length(p1) == N₁ && all(in(p1), 1:N₁)
    @assert length(p2) == N₂ && all(in(p2), N₁ .+ (1:N₂))
    oindA2 = TupleTools.getindices(oindA, p1)
    oindB2 = TupleTools.getindices(oindB, p2 .- N₁)
    indA = (codA..., reverse(domA)...)
    indB = (codB..., reverse(domB)...)
    # cycle indA to be of the form (oindA2..., reverse(cindA2)...)
    while length(oindA2) > 0 && indA[1] != oindA2[1]
        indA = _cyclicpermute(indA)
    end
    # cycle indB to be of the form (cindB2..., reverse(oindB2)...)
    while length(oindB2) > 0 && indB[end] != oindB2[1]
        indB = _cyclicpermute(indB)
    end
    for i in 2:N₁
        @assert indA[i] == oindA2[i]
    end
    for j in 2:N₂
        @assert indB[end + 1 - j] == oindB2[j]
    end
    Nc = length(indA) - N₁
    @assert Nc == length(indB) - N₂
    pc = ntuple(identity, Nc)
    cindA2 = reverse(TupleTools.getindices(indA, N₁ .+ pc))
    cindB2 = TupleTools.getindices(indB, pc)
    return oindA2, cindA2, oindB2, cindB2
end

function reorder_indices(codA, domA, codB, domB, oindA, cindA, oindB, cindB, p1, p2)
    oindA2, cindA2, oindB2, cindB2 = reorder_indices(codA, domA, codB, domB, oindA, oindB,
                                                     p1, p2)

    #if oindA or oindB are empty, then reorder indices can only order it correctly up to a cyclic permutation!
    if isempty(oindA2) && !isempty(cindA)
        # isempty(cindA) is a cornercase which I'm not sure if we can encounter
        hit = cindA[findfirst(==(first(cindB2)), cindB)]
        while hit != first(cindA2)
            cindA2 = _cyclicpermute(cindA2)
        end
    end
    if isempty(oindB2) && !isempty(cindB)
        hit = cindB[findfirst(==(first(cindA2)), cindA)]
        while hit != first(cindB2)
            cindB2 = _cyclicpermute(cindB2)
        end
    end
    @assert TupleTools.sort(cindA) == TupleTools.sort(cindA2)
    @assert TupleTools.sort(tuple.(cindA2, cindB2)) == TupleTools.sort(tuple.(cindA, cindB))
    return oindA2, cindA2, oindB2, cindB2
end
