# methods/simple.jl
#
# Method-based access to planar operations using simple definitions.

# ------------------------------------------------------------------------------------------

function planarcopy(A, pA::Index2Tuple, conjA::Symbol, α::Number=One(), backend::Backend...)
    TC = TO.promote_add(scalartype(A), scalartype(α))
    C = tensoralloc_add(TC, pA, A, conjA)
    return planaradd!(C, A, pA, conjA, α, Zero(), backend...)
end

# ------------------------------------------------------------------------------------------

function planartrace(A, pA::Index2Tuple, qA::Index2Tuple, conjA::Symbol, α::Number=One(),
                     backend::Backend...)
    TC = TO.promote_contract(scalartype(A), scalartype(α))
    C = tensoralloc_add(TC, pA, A, conjA)
    return planartrace!(C, A, pA, qA, conjA, α, Zero(), backend...)
end

# ------------------------------------------------------------------------------------------

"""
    planarcontract(A, IA, [conjA], B, IB, [conjB], [IC], [α=1])
    planarcontract(A, pA::Index2Tuple, conjA, B, pB::Index2Tuple, conjB, pAB::Index2Tuple, α=1, [backend]) # expert mode

Contract indices of tensor `A` with corresponding indices in tensor `B` by assigning
them identical labels in the iterables `IA` and `IB`. The indices of the resulting
tensor correspond to the indices that only appear in either `IA` or `IB` and can be
ordered by specifying the optional argument `IC`. The default is to have all open
indices of `A` followed by all open indices of `B`. Note that inner contractions of an array
should be handled first with `tensortrace`, so that every label can appear only once in `IA`
or `IB` seperately, and once (for an open index) or twice (for a contracted index) in the
union of `IA` and `IB`.

Optionally, the symbols `conjA` and `conjB` can be used to specify that the input tensors
should be conjugated.

See also [`tensorcontract`](@ref).
"""
function planarcontract end

function planarcontract(A, IA::TensorLabels, conjA::Symbol, B, IB::TensorLabels,
                        conjB::Symbol, IC::TensorLabels,
                        α::Number=One())
    ia = canonicalize_labels(A, IA)
    ib = canonicalize_labels(B, IB)
    ic = canonicalize_labels(IC)
    pA, pB, pAB = planarcontract_indices(ia, ib, ic)
    return planarcontract(A, pA, conjA, B, pB, conjB, pAB, α)
end
# default `IC`
function planarcontract(A, IA::TensorLabels, conjA::Symbol, B, IB::TensorLabels,
                        conjB::Symbol, α::Number=One())
    ia = canonicalize_labels(A, IA)
    ib = canonicalize_labels(B, IB)
    pA, pB, pAB = planarcontract_indices(ia, ib)
    return planarcontract(A, pA, conjA, B, pB, conjB, pAB, α)
end
# default `conjA` and `conjB`
function planarcontract(A, IA, B, IB, IC, α::Number=One())
    return planarcontract(A, IA, :N, B, IB, :N, IC, α)
end
function planarcontract(A, IA, B, IB, α::Number=One())
    return planarcontract(A, IA, :N, B, IB, :N, α)
end

# expert mode
function planarcontract(A, pA::Index2Tuple, conjA::Symbol,
                        B, pB::Index2Tuple, conjB::Symbol,
                        pAB::Index2Tuple, α::Number=One(),
                        backend::Backend...)
    TC = TO.promote_contract(scalartype(A), scalartype(B), scalartype(α))
    C = TO.tensoralloc_contract(TC, pAB, A, pA, conjA, B, pB, conjB)
    return planarcontract!(C, A, pA, conjA, B, pB, conjB, pAB, α, Zero(), backend...)
end
