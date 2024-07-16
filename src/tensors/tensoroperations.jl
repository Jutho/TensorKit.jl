# Implement full TensorOperations.jl interface
#----------------------------------------------
TO.tensorstructure(t::AbstractTensorMap) = space(t)
function TO.tensorstructure(t::AbstractTensorMap, iA::Int, conjA::Bool)
    return !conjA ? space(t, iA) : conj(space(t, iA))
end

function TO.tensoralloc(::Type{TT}, structure::TensorMapSpace{S,N₁,N₂}, istemp::Val,
                        allocator=TO.DefaultAllocator()) where {E,S,N₁,N₂,A,
                                                                TT<:TrivialTensorMap{E,S,N₁,
                                                                                     N₂,
                                                                                     A}}
    data = TO.tensoralloc(A, (dim(codomain(structure)), dim(domain(structure))), istemp,
                          allocator)
    return TT(data, codomain(structure), domain(structure))
end

function TO.tensoralloc(::Type{TT}, structure::TensorMapSpace{S,N₁,N₂}, istemp::Val,
                        allocator=TO.DefaultAllocator()) where {E,S,N₁,N₂,
                                                                TT<:AbstractTensorMap{E,S,
                                                                                      N₁,
                                                                                      N₂}}
    blocksectoriterator = blocksectors(structure)
    rowr, rowdims = _buildblockstructure(codomain(structure), blocksectoriterator)
    colr, coldims = _buildblockstructure(domain(structure), blocksectoriterator)
    A = storagetype(TT)
    blockallocator(c) = TO.tensoralloc(A, (rowdims[c], coldims[c]), istemp, allocator)
    data = SectorDict(c => blockallocator(c) for c in blocksectoriterator)
    return TT(data, codomain(structure), domain(structure), rowr, colr)
end

function TO.tensorfree!(t::AbstractTensorMap, allocator=TO.DefaultAllocator())
    for (c, b) in blocks(t)
        TO.tensorfree!(b, allocator)
    end
    return nothing
end

TO.tensorscalar(t::AbstractTensorMap) = scalar(t)

function _canonicalize(p::Index2Tuple{N₁,N₂},
                       ::AbstractTensorMap{<:IndexSpace,N₁,N₂}) where {N₁,N₂}
    return p
end
_canonicalize(p::Index2Tuple, t::AbstractTensorMap) = _canonicalize(linearize(p), t)
function _canonicalize(p::IndexTuple, t::AbstractTensorMap)
    p₁ = TupleTools.getindices(p, codomainind(t))
    p₂ = TupleTools.getindices(p, domainind(t))
    return (p₁, p₂)
end

# tensoradd!
function TO.tensoradd!(C::AbstractTensorMap,
                       A::AbstractTensorMap, pA::Index2Tuple, conjA::Bool,
                       α::Number, β::Number, backend::AbstractBackend, allocator)
    if !conjA
        A′ = A
        pA′ = _canonicalize(pA, C)
    else
        A′ = adjoint(A)
        pA′ = adjointtensorindices(A, _canonicalize(pA, C))
    end
    add_permute!(C, A′, pA′, α, β, backend)
    return C
end

function TO.tensoradd_type(TC, A::AbstractTensorMap, ::Index2Tuple{N₁,N₂},
                           ::Bool) where {N₁,N₂}
    M = similarstoragetype(A, TC)
    return tensormaptype(spacetype(A), N₁, N₂, M)
end

function TO.tensoradd_structure(A::AbstractTensorMap, pA::Index2Tuple{N₁,N₂},
                                conjA::Bool) where {N₁,N₂}
    if !conjA
        return permute(space(A), pA)
    else
        return TO.tensoradd_structure(adjoint(A), adjointtensorindices(A, pA), false)
    end
end

# tensortrace!
function TO.tensortrace!(C::AbstractTensorMap,
                         A::AbstractTensorMap, p::Index2Tuple, q::Index2Tuple,
                         conjA::Bool,
                         α::Number, β::Number, backend::AbstractBackend, allocator)
    if !conjA
        A′ = A
        p′ = _canonicalize(p, C)
        q′ = q
    else
        A′ = adjoint(A)
        p′ = adjointtensorindices(A, _canonicalize(p, C))
        q′ = adjointtensorindices(A, q)
    end
    trace_permute!(C, A′, p′, q′, α, β, backend)
    return C
end

# tensorcontract!
function TO.tensorcontract!(C::AbstractTensorMap,
                            A::AbstractTensorMap, pA::Index2Tuple, conjA::Bool,
                            B::AbstractTensorMap, pB::Index2Tuple, conjB::Bool,
                            pAB::Index2Tuple, α::Number, β::Number,
                            backend::AbstractBackend, allocator)
    pAB′ = _canonicalize(pAB, C)
    if !conjA
        A′ = A
        pA′ = pA
    else
        A′ = A'
        pA′ = adjointtensorindices(A, pA)
    end
    if !conjB
        B′ = B
        pB′ = pB
    else
        B′ = B'
        pB′ = adjointtensorindices(B, pB)
    end
    contract!(C, A′, pA′, B′, pB′, pAB′, α, β, backend, allocator)
    return C
end

function TO.tensorcontract_type(TC,
                                A::AbstractTensorMap, ::Index2Tuple, ::Bool,
                                B::AbstractTensorMap, ::Index2Tuple, ::Bool,
                                ::Index2Tuple{N₁,N₂}) where {N₁,N₂}
    M = similarstoragetype(A, TC)
    M == similarstoragetype(B, TC) || throw(ArgumentError("incompatible storage types"))
    spacetype(A) == spacetype(B) || throw(SpaceMismatch("incompatible space types"))
    return tensormaptype(spacetype(A), N₁, N₂, M)
end

function TO.tensorcontract_structure(A::AbstractTensorMap, pA::Index2Tuple, conjA::Bool,
                                     B::AbstractTensorMap, pB::Index2Tuple, conjB::Bool,
                                     pAB::Index2Tuple{N₁,N₂}) where {N₁,N₂}
    sA = TO.tensoradd_structure(A, pA, conjA)
    sB = TO.tensoradd_structure(B, pB, conjB)
    return permute(compose(sA, sB), pAB)
end

function TO.checkcontractible(tA::AbstractTensorMap, iA::Int, conjA::Bool,
                              tB::AbstractTensorMap, iB::Int, conjB::Bool,
                              label)
    sA = TO.tensorstructure(tA, iA, conjA)'
    sB = TO.tensorstructure(tB, iB, conjB)
    sA == sB ||
        throw(SpaceMismatch("incompatible spaces for $label: $sA ≠ $sB"))
    return nothing
end

TO.tensorcost(t::AbstractTensorMap, i::Int) = dim(space(t, i))

#----------------
# IMPLEMENTATONS
#----------------

# Trace implementation
#----------------------
function trace_permute!(tdst::AbstractTensorMap,
                        tsrc::AbstractTensorMap,
                        (p₁, p₂)::Index2Tuple,
                        (q₁, q₂)::Index2Tuple,
                        α::Number,
                        β::Number,
                        backend::AbstractBackend=TO.DefaultBackend())
    # some input checks
    (S = spacetype(tdst)) == spacetype(tsrc) ||
        throw(SpaceMismatch("incompatible spacetypes"))
    if !(BraidingStyle(sectortype(S)) isa SymmetricBraiding)
        throw(SectorMismatch("only tensors with symmetric braiding rules can be contracted; try `@planar` instead"))
    end
    (N₃ = length(q₁)) == length(q₂) ||
        throw(IndexError("number of trace indices does not match"))

    N₁, N₂ = length(p₁), length(p₂)

    @boundscheck begin
        space(tdst) == permute(space(tsrc), (p₁, p₂)) ||
            throw(SpaceMismatch("trace: tsrc = $(codomain(tsrc))←$(domain(tsrc)),
                    tdst = $(codomain(tdst))←$(domain(tdst)), p₁ = $(p₁), p₂ = $(p₂)"))
        all(i -> space(tsrc, q₁[i]) == dual(space(tsrc, q₂[i])), 1:N₃) ||
            throw(SpaceMismatch("trace: tsrc = $(codomain(tsrc))←$(domain(tsrc)),
                    q₁ = $(q₁), q₂ = $(q₂)"))
    end

    I = sectortype(S)
    # TODO: is it worth treating UniqueFusion separately? Is it worth to add multithreading support?
    if I === Trivial
        cod = codomain(tsrc)
        dom = domain(tsrc)
        n = length(cod)
        TO.tensortrace!(tdst[], tsrc[], (p₁, p₂), (q₁, q₂), false, α, β, backend)
        # elseif FusionStyle(I) isa UniqueFusion
    else
        cod = codomain(tsrc)
        dom = domain(tsrc)
        n = length(cod)
        scale!(tdst, β)
        r₁ = (p₁..., q₁...)
        r₂ = (p₂..., q₂...)
        for (f₁, f₂) in fusiontrees(tsrc)
            for ((f₁′, f₂′), coeff) in permute(f₁, f₂, r₁, r₂)
                f₁′′, g₁ = split(f₁′, N₁)
                f₂′′, g₂ = split(f₂′, N₂)
                g₁ == g₂ || continue
                coeff *= dim(g₁.coupled) / dim(g₁.uncoupled[1])
                for i in 2:length(g₁.uncoupled)
                    if !(g₁.isdual[i])
                        coeff *= twist(g₁.uncoupled[i])
                    end
                end
                C = tdst[f₁′′, f₂′′]
                A = tsrc[f₁, f₂]
                α′ = α * coeff
                TO.tensortrace!(C, A, (p₁, p₂), (q₁, q₂), false, α′, One(), backend)
            end
        end
    end
    return tdst
end

# Contract implementation
#-------------------------
# TODO: contraction with either A or B a rank (1, 1) tensor does not require to
# permute the fusion tree and should therefore be special cased. This will speed
# up MPS algorithms
function contract!(C::AbstractTensorMap,
                   A::AbstractTensorMap, (oindA, cindA)::Index2Tuple,
                   B::AbstractTensorMap, (cindB, oindB)::Index2Tuple,
                   (p₁, p₂)::Index2Tuple,
                   α::Number, β::Number,
                   backend::AbstractBackend, allocator)
    length(cindA) == length(cindB) ||
        throw(IndexError("number of contracted indices does not match"))
    N₁, N₂ = length(oindA), length(oindB)

    # find optimal contraction scheme
    hsp = has_shared_permute
    ipAB = TupleTools.invperm((p₁..., p₂...))
    oindAinC = TupleTools.getindices(ipAB, ntuple(n -> n, N₁))
    oindBinC = TupleTools.getindices(ipAB, ntuple(n -> n + N₁, N₂))

    qA = TupleTools.sortperm(cindA)
    cindA′ = TupleTools.getindices(cindA, qA)
    cindB′ = TupleTools.getindices(cindB, qA)

    qB = TupleTools.sortperm(cindB)
    cindA′′ = TupleTools.getindices(cindA, qB)
    cindB′′ = TupleTools.getindices(cindB, qB)

    dA, dB, dC = dim(A), dim(B), dim(C)

    # keep order A en B, check possibilities for cind
    memcost1 = memcost2 = dC * (!hsp(C, (oindAinC, oindBinC)))
    memcost1 += dA * (!hsp(A, (oindA, cindA′))) +
                dB * (!hsp(B, (cindB′, oindB)))
    memcost2 += dA * (!hsp(A, (oindA, cindA′′))) +
                dB * (!hsp(B, (cindB′′, oindB)))

    # reverse order A en B, check possibilities for cind
    memcost3 = memcost4 = dC * (!hsp(C, (oindBinC, oindAinC)))
    memcost3 += dB * (!hsp(B, (oindB, cindB′))) +
                dA * (!hsp(A, (cindA′, oindA)))
    memcost4 += dB * (!hsp(B, (oindB, cindB′′))) +
                dA * (!hsp(A, (cindA′′, oindA)))

    if min(memcost1, memcost2) <= min(memcost3, memcost4)
        if memcost1 <= memcost2
            return _contract!(α, A, B, β, C, oindA, cindA′, oindB, cindB′, p₁, p₂)
        else
            return _contract!(α, A, B, β, C, oindA, cindA′′, oindB, cindB′′, p₁, p₂)
        end
    else
        p1′ = map(n -> ifelse(n > N₁, n - N₁, n + N₂), p₁)
        p2′ = map(n -> ifelse(n > N₁, n - N₁, n + N₂), p₂)
        if memcost3 <= memcost4
            return _contract!(α, B, A, β, C, oindB, cindB′, oindA, cindA′, p1′, p2′)
        else
            return _contract!(α, B, A, β, C, oindB, cindB′′, oindA, cindA′′, p1′, p2′)
        end
    end
end

# TODO: also transform _contract! into new interface, and add backend support
function _contract!(α, A::AbstractTensorMap, B::AbstractTensorMap,
                    β, C::AbstractTensorMap,
                    oindA::IndexTuple, cindA::IndexTuple,
                    oindB::IndexTuple, cindB::IndexTuple,
                    p₁::IndexTuple, p₂::IndexTuple)
    if !(BraidingStyle(sectortype(C)) isa SymmetricBraiding)
        throw(SectorMismatch("only tensors with symmetric braiding rules can be contracted; try `@planar` instead"))
    end
    N₁, N₂ = length(oindA), length(oindB)
    copyA = false
    if BraidingStyle(sectortype(A)) isa Fermionic
        for i in cindA
            if !isdual(space(A, i))
                copyA = true
            end
        end
    end
    A′ = permute(A, (oindA, cindA); copy=copyA)
    B′ = permute(B, (cindB, oindB))
    A′ = twist!(A′, filter(i -> !isdual(space(A′, i)), domainind(A′)))
    ipAB = TupleTools.invperm((p₁..., p₂...))
    oindAinC = TupleTools.getindices(ipAB, ntuple(n -> n, N₁))
    oindBinC = TupleTools.getindices(ipAB, ntuple(n -> n + N₁, N₂))
    if has_shared_permute(C, (oindAinC, oindBinC))
        C′ = permute(C, (oindAinC, oindBinC))
        mul!(C′, A′, B′, α, β)
    else
        C′ = A′ * B′
        add_permute!(C, C′, (p₁, p₂), α, β)
    end
    return C
end

# Scalar implementation
#-----------------------
function scalar(t::AbstractTensorMap)
    return dim(codomain(t)) == dim(domain(t)) == 1 ?
           first(blocks(t))[2][1, 1] : throw(DimensionMismatch())
end
