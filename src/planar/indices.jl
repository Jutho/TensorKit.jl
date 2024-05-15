const TensorLabels = Union{Tuple,Vector}

function canonicalize_labels(A::AbstractTensorMap, IA::TensorLabels)
    numind(A) == length(IA) ||
        throw(ArgumentError("invalid labels for tensor: $IA for ($(numout(A)), $(numin(A)))"))
    return (ntuple(i -> IA[i], numout(A)), ntuple(i -> IA[numout(A) + i], numin(A)))
end
canonicalize_labels(IA::TensorLabels) = (tuple(IA...), ())

function _isplanar(inds::Index2Tuple, p::Index2Tuple)
    return iscyclicpermutation((inds[1]..., inds[2]...),
                               (p[1]..., reverse(p[2])...))
end

function planartrace_indices(IA::Tuple{Tuple,Tuple}, conjA, IC::Tuple{Tuple,Tuple})
    IA′ = conjA == :C ? reverse(IA) : IA
    
    IA_linear = (IA′[1]..., (IA′[2])...)
    IC_linear = (IC[1]..., (IC[2])...)
    
    p, q1, q2 = TO.trace_indices(IA_linear, IC_linear)
    
    if conjA == :C
        p′ = adjointtensorindices((length(IC[1]), length(IC[2])), p)
        q′ = adjointtensorindices((length(IA[2]), length(IA[1])), (q1, q2))
    else
        p′ = p
        q′ = (q1, q2)
    end

    return p′, q′
end
function planartrace_indices(IA::Tuple{Tuple,Tuple}, conjA)
    IA′ = conjA == :C ? reverse(IA) : IA
    
    IA_linear = (IA′[1]..., (IA′[2])...)
    IC_linear = tuple(TO.unique2(IA_linear)...)
    p, q1, q2 = TO.trace_indices(IA_linear, IC_linear)

    if conjA == :C
        p′ = adjointtensorindices((length(p[1]), length(p[2])), p)
        q′ = adjointtensorindices((length(IA[2]), length(IA[1])), (q1, q2))
    else
        p′ = p
        q′ = (q1, q2)
    end
    
    return p′, q′
end


"""
    planarcontract_indices(IA, IB, [IC])

Convert a set of tensor labels to a set of indices. Throws an error if this cannot be
achieved in a planar manner.
"""
function planarcontract_indices(IA::Tuple{Tuple,Tuple}, conjA::Symbol,
                                IB::Tuple{Tuple,Tuple}, conjB::Symbol,
                                IC::Tuple{Tuple,Tuple})
    
    IA′ = conjA == :C ? reverse(IA) : IA
    IB′ = conjB == :C ? reverse(IB) : IB
    
    pA, pB, pAB = planarcontract_indices(IA′, IB′, IC)
    
    # map indices back to original tensor
    pA′ = conjA == :C ? adjointtensorindices((length(IA[2]), length(IA[1])), pA) : pA
    pB′ = conjB == :C ? adjointtensorindices((length(IB[2]), length(IB[1])), pB) : pB
    
    return pA′, pB′, pAB
end
function planarcontract_indices(IA::Tuple{Tuple,Tuple}, conjA::Symbol,
                                IB::Tuple{Tuple,Tuple}, conjB::Symbol)
    # map indices to indices of adjoint tensor
    IA′ = conjA == :C ? reverse(IA) : IA
    IB′ = conjB == :C ? reverse(IB) : IB
    
    pA, pB, pAB = planarcontract_indices(IA′, IB′)
    
    # map indices back to original tensor
    pA′ = conjA == :C ? adjointtensorindices((length(IA[2]), length(IA[1])), pA) : pA
    pB′ = conjB == :C ? adjointtensorindices((length(IB[2]), length(IB[1])), pB) : pB

    return pA′, pB′, pAB
end

function planarcontract_indices(IA::Tuple{Tuple,Tuple},
                                IB::Tuple{Tuple,Tuple},
                                IC::Tuple{Tuple,Tuple})
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
    pC = (ntuple(i -> invIC[i], length(IC[1])),
          ntuple(i -> invIC[end + 1 - i], length(IC[2])))

    return pA, pB, pC
end
function planarcontract_indices(IA::Tuple{Tuple,Tuple},
                                IB::Tuple{Tuple,Tuple})
    IA_linear = (IA[1]..., reverse(IA[2])...)
    IB_linear = (IB[1]..., reverse(IB[2])...)
    IAB = (IA_linear..., IB_linear...)

    Icontract = TO.tunique(TO.tsetdiff(IAB, tuple(TO.unique2(IAB)...)))
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

    pC = (ntuple(identity, length(IopenA)), ntuple(i -> i + length(IopenA), length(IopenB)))

    return pA, pB, pC
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
    @assert all(getindices(indA_lin, ntuple(identity, NA₁)) .== pA′[1]) "sanity check: $indA $pA"
    pA′ = (pA′[1], reverse(getindices(indA_lin, pc .+ NA₁)))

    # cycle indB to be of the form (cindB..., reverse(oindB)...)
    indB_lin = (indB[1]..., indB[2]...)
    if NB₂ != 0
        iB = findfirst(==(first(pB′[2])), indB_lin)
        indB_lin = _circshift(indB_lin, -iB)
    end
    @assert all(getindices(indB_lin, ntuple(identity, NB₂) .+ NB₁) .== reverse(pB′[2])) "sanity check: $indB $pB"
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
