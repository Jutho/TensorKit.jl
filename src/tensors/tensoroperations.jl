# Implement full TensorOperations.jl interface
#----------------------------------------------
TO.tensorstructure(t::AbstractTensorMap) = space(t)
function TO.tensorstructure(t::AbstractTensorMap, iA::Int, conjA::Symbol)
    return conjA == :N ? space(t, iA) : conj(space(t, iA))
end

function TO.tensoralloc(::Type{TT}, structure, istemp=false,
                        backend::Backend...) where {TT<:AbstractTensorMap}
    function blockallocator(d)
        return TO.tensoralloc(storagetype(TT), d, istemp, backend...)
    end
    return TensorMap(blockallocator, structure)
end

function TO.tensorfree!(t::AbstractTensorMap, backend::Backend...)
    for (c, b) in blocks(t)
        TO.tensorfree!(b, backend...)
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
function TO.tensoradd!(C::AbstractTensorMap{S}, pC::Index2Tuple,
                       A::AbstractTensorMap{S}, conjA::Symbol,
                       α::Number, β::Number, backend::Backend...) where {S}
    if conjA == :N
        A′ = A
        pC′ = _canonicalize(pC, C)
    elseif conjA == :C
        A′ = adjoint(A)
        pC′ = adjointtensorindices(A, _canonicalize(pC, C))
    else
        throw(ArgumentError("unknown conjugation flag $conjA"))
    end
    add_permute!(C, A′, pC′, α, β, backend...)
    return C
end

function TO.tensoradd_type(TC, ::Index2Tuple{N₁,N₂}, A::AbstractTensorMap{S},
                           ::Symbol) where {S,N₁,N₂}
    M = similarstoragetype(A, TC)
    return tensormaptype(S, N₁, N₂, M)
end

function TO.tensoradd_structure(pC::Index2Tuple{N₁,N₂},
                                A::AbstractTensorMap{S}, conjA::Symbol) where {S,N₁,N₂}
    if conjA == :N
        return permute(space(A), pC)
    else
        return TO.tensoradd_structure(adjointtensorindices(A, pC), adjoint(A), :N)
    end
end

# tensortrace!
function TO.tensortrace!(C::AbstractTensorMap{S}, p::Index2Tuple,
                         A::AbstractTensorMap{S}, q::Index2Tuple, conjA::Symbol,
                         α::Number, β::Number, backend::Backend...) where {S}
    if conjA == :N
        A′ = A
        p′ = _canonicalize(p, C)
        q′ = q
    elseif conjA == :C
        A′ = adjoint(A)
        p′ = adjointtensorindices(A, _canonicalize(p, C))
        q′ = adjointtensorindices(A, q)
    else
        throw(ArgumentError("unknown conjugation flag $conjA"))
    end
    # TODO: novel syntax for tensortrace?
    # tensortrace!(C, pC′, A′, qA′, α, β, backend...)
    trace_permute!(C, A′, p′, q′, α, β, backend...)
    return C
end

# tensorcontract!
function TO.tensorcontract!(C::AbstractTensorMap{S,N₁,N₂}, pAB::Index2Tuple,
                            A::AbstractTensorMap{S}, pA::Index2Tuple, conjA::Symbol,
                            B::AbstractTensorMap{S}, pB::Index2Tuple, conjB::Symbol,
                            α::Number, β::Number, backend::Backend...) where {S,N₁,N₂}
    pAB′ = _canonicalize(pAB, C)
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
    contract!(C, A′, pA′, B′, pB′, pAB′, α, β, backend...)
    return C
end

function TO.tensorcontract_type(TC, ::Index2Tuple{N₁,N₂},
                                A::AbstractTensorMap{S}, pA, conjA,
                                B::AbstractTensorMap{S}, pB, conjB) where {S,N₁,N₂}
    M = similarstoragetype(A, TC)
    M == similarstoragetype(B, TC) || throw(ArgumentError("incompatible storage types"))
    return tensormaptype(S, N₁, N₂, M)
end

function TO.tensorcontract_structure(pC::Index2Tuple{N₁,N₂},
                                     A::AbstractTensorMap{S}, pA::Index2Tuple, conjA,
                                     B::AbstractTensorMap{S}, pB::Index2Tuple,
                                     conjB) where {S,N₁,N₂}
    sA = TO.tensoradd_structure(pA, A, conjA)
    sB = TO.tensoradd_structure(pB, B, conjB)
    return permute(compose(sA, sB), pC)
end

function TO.checkcontractible(tA::AbstractTensorMap{S}, iA::Int, conjA::Symbol,
                              tB::AbstractTensorMap{S}, iB::Int, conjB::Symbol,
                              label) where {S}
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
function trace_permute!(tdst::AbstractTensorMap{S,N₁,N₂},
                        tsrc::AbstractTensorMap{S},
                        (p₁, p₂)::Index2Tuple{N₁,N₂},
                        (q₁, q₂)::Index2Tuple{N₃,N₃},
                        α::Number,
                        β::Number,
                        backend::Backend...) where {S,N₁,N₂,N₃}
    if !(BraidingStyle(sectortype(S)) isa SymmetricBraiding)
        throw(SectorMismatch("only tensors with symmetric braiding rules can be contracted; try `@planar` instead"))
    end
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
        TO.tensortrace!(tdst[], (p₁, p₂), tsrc[], (q₁, q₂), :N, α, β)
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
                TO.tensortrace!(C, (p₁, p₂), A, (q₁, q₂), :N, α′, One(), backend...)
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
function contract!(C::AbstractTensorMap{S},
                   A::AbstractTensorMap{S},
                   (oindA, cindA)::Index2Tuple{N₁,N₃},
                   B::AbstractTensorMap{S},
                   (cindB, oindB)::Index2Tuple{N₃,N₂},
                   (p₁, p₂)::Index2Tuple,
                   α::Number,
                   β::Number,
                   backend::Backend...) where {S,N₁,N₂,N₃}

    # find optimal contraction scheme
    hsp = has_shared_permute
    ipC = TupleTools.invperm((p₁..., p₂...))
    oindAinC = TupleTools.getindices(ipC, ntuple(n -> n, N₁))
    oindBinC = TupleTools.getindices(ipC, ntuple(n -> n + N₁, N₂))

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
function _contract!(α, A::AbstractTensorMap{S}, B::AbstractTensorMap{S},
                    β, C::AbstractTensorMap{S},
                    oindA::IndexTuple{N₁}, cindA::IndexTuple,
                    oindB::IndexTuple{N₂}, cindB::IndexTuple,
                    p₁::IndexTuple, p₂::IndexTuple) where {S,N₁,N₂}
    if !(BraidingStyle(sectortype(S)) isa SymmetricBraiding)
        throw(SectorMismatch("only tensors with symmetric braiding rules can be contracted; try `@planar` instead"))
    end
    copyA = false
    if BraidingStyle(sectortype(S)) isa Fermionic
        for i in cindA
            if !isdual(space(A, i))
                copyA = true
            end
        end
    end
    A′ = permute(A, (oindA, cindA); copy=copyA)
    B′ = permute(B, (cindB, oindB))
    A′ = twist!(A′, filter(i -> !isdual(space(A′, i)), domainind(A′)))
    ipC = TupleTools.invperm((p₁..., p₂...))
    oindAinC = TupleTools.getindices(ipC, ntuple(n -> n, N₁))
    oindBinC = TupleTools.getindices(ipC, ntuple(n -> n + N₁, N₂))
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
function scalar(t::AbstractTensorMap{S}) where {S<:IndexSpace}
    return dim(codomain(t)) == dim(domain(t)) == 1 ?
           first(blocks(t))[2][1, 1] : throw(DimensionMismatch())
end
