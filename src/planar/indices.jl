"""
    planarcontract_indices(IA, IB, IC)

Convert a set of tensor labels to a set of indices. Throws an error if this cannot be achieved in a planar manner.
"""
function planarcontract_indices(IA::Tuple{NTuple{NA1},NTuple{NA2}},
                                IB::Tuple{NTuple{NB1},NTuple{NB2}},
                                IC::Tuple{NTuple{NC1},NTuple{NC2}}) where {NA1,NA2,NB1,NB2,
                                                                           NC1,NC2}
    IA_linear = (IA[1]..., reverse(IA[2])...)
    IB_linear = (IB[1]..., reverse(IB[2])...)
    IC_linear = (IC[1]..., reverse(IC[2])...)
    IAB = (IA_linear..., IB_linear...)
    isodd(length(IAB) - length(IC_linear)) &&
        throw(IndexError("invalid contraction pattern: $IA and $IB to $IC"))

    Icontract = TO.tunique(TO.tsetdiff(IAB, IC_linear))
    IopenA = TO.tsetdiff(IA_linear, Icontract)
    IopenB = TO.tsetdiff(IB_linear, Icontract)

    # bring IA to the form (IopenA..., Icontract...) (as sets)
    IA′ = IA_linear
    ctr = 0
    while !issetequal(getindices(IA′, ntuple(identity, length(IopenA))), IopenA)
        IA′ = _cyclicpermute(IA′)
        ctr += 1
        ctr > length(IA′) &&
            throw(ArgumentError("no cyclic permutation of $IA that matches $IB"))
    end

    # bring IB to the form (Icontract..., IopenB...) (as sets)
    IB′ = IB_linear
    ctr = 0
    while !issetequal(getindices(IB′, ntuple(i -> i + length(Icontract), length(IopenB))),
                      IopenB)
        IB′ = _cyclicpermute(IB′)
        ctr += 1
        ctr > length(IB′) &&
            throw(ArgumentError("no cyclic permutation of $IB that matches $IA"))
    end

    # special case when IopenA is empty -> still have freedom to circshift IA
    if length(IopenA) == 0
        ctr = 0
        while !isequal(IA′, reverse(getindices(IB′, ntuple(identity, length(IA′)))))
            IA′ = _cyclicpermute(IA′)
            ctr += 1
            ctr > length(IA′) &&
                throw(ArgumentError("no cyclic permutation of $IA that matches $IB"))
        end
    end

    # special case when IopenB is empty -> still have freedom to circshift IB
    if length(IopenB) == 0
        ctr = 0
        while !isequal(IB′,
                       reverse(getindices(IA′,
                                          ntuple(i -> i + length(IopenA), length(IB′)))))
            IB′ = _cyclicpermute(IB′)
            ctr += 1
            ctr > length(IB′) &&
                throw(ArgumentError("no cyclic permutation of $IB that matches $IA"))
        end
    end

    # bring IC to the form (IopenA..., IopenB...) (as sets)
    IC′ = IC_linear
    IopenA
    ctr = 0
    while !issetequal(getindices(IC′, ntuple(identity, length(IopenA))), IopenA)
        IC′ = _cyclicpermute(IC′)
        ctr += 1
        ctr > length(IC′) &&
            throw(ArgumentError("no cyclic permutation of $IC that matches $IA and $IB"))
    end

    # special case when Icontract is empty -> still have freedom to circshift IA and IB to
    # match IC
    # TODO: this is not yet implemented
    @assert length(Icontract) != 0 "not yet implemented"

    IA_nonlinear = (IA[1]..., IA[2]...)
    pA = (_indexin(getindices(IA′, ntuple(identity, length(IopenA))), IA_nonlinear),
          reverse(_indexin(getindices(IA′,
                                      ntuple(i -> i + length(IopenA), length(Icontract))),
                           IA_nonlinear)))

    IB_nonlinear = (IB[1]..., IB[2]...)
    pB = (_indexin(getindices(IB′, ntuple(identity, length(Icontract))), IB_nonlinear),
          reverse(_indexin(getindices(IB′,
                                      ntuple(i -> i + length(Icontract), length(IopenB))),
                           IB_nonlinear)))

    IC″ = (ntuple(i -> IC′[i], length(IopenA))...,
           ntuple(i -> IC′[end + 1 - i], length(IopenB))...)
    invIC = _indexin(IC_linear, IC″)
    pC = (ntuple(i -> invIC[i], NC1),
          ntuple(i -> invIC[end + 1 - i], NC2))

    return pA, pB, pC
end
