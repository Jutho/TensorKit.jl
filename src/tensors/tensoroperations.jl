# Implement full TensorOperations.jl interface
#----------------------------------------------
TO.tensorstructure(t::AbstractTensorMap) = space(t)
function TO.tensorstructure(t::AbstractTensorMap, iA::Int, conjA::Symbol)
    return conjA == :N ? space(t, iA) : conj(space(t, iA))
end

function TO.tensoralloc(ttype::Type{<:AbstractTensorMap}, structure, istemp=false,
                        backend::Backend...)
    M = storagetype(ttype)
    return TensorMap(structure) do d
        return TO.tensoralloc(M, d, istemp, backend...)
    end
end

function TO.tensorfree!(t::AbstractTensorMap, backend::Backend...)
    for (c, b) in blocks(t)
        TO.tensorfree!(b, backend...)
    end
    return nothing
end

TO.tensorscalar(t::AbstractTensorMap) = scalar(t)

_canonicalize(p::Index2Tuple{N₁,N₂}, ::AbstractTensorMap{<:IndexSpace,N₁,N₂}) where {N₁,N₂} = p
function _canonicalize(p::Index2Tuple, ::AbstractTensorMap)
    p′ = linearize(p)
    p₁ = TupleTools.getindices(p′, codomainind(t))
    p₂ = TupleTools.getindices(p′, domainind(t))
    return (p₁, p₂)
end

# tensoradd!
function TO.tensoradd!(C::AbstractTensorMap{S},
                       A::AbstractTensorMap{S}, pC::Index2Tuple, conjA::Symbol,
                       α::Number, β::Number, backend::Backend...) where {S}
    if conjA == :N
        A′ = A
        pC′ = _canonicalize(pC, C)
    elseif conjA == :C
        A′ = adjoint(A)
        pC′ = adjointtensorindices(A, _canonicalize(pA, C))
    else
        throw(ArgumentError("unknown conjugation flag $conjA"))
    end
    # TODO: novel syntax for tensoradd!?
    # tensoradd!(C, A′, pC′, α, β, backend...)
    add!(α, A′, β, C, pC′[1], pC′[2])
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
        cod = ProductSpace{S,N₁}(space.(Ref(A), pC[1]))
        dom = ProductSpace{S,N₂}(dual.(space.(Ref(A), pC[2])))
        return dom → cod
    else
        return TO.tensoradd_structure(adjoint(A), adjointtensorindices(A, pC), :N)
    end
end

# tensortrace!
function TO.tensortrace!(C::AbstractTensorMap{S}, pC::Index2Tuple,
                         A::AbstractTensorMap{S}, qA::Index2Tuple, conjA::Symbol,
                         α::Number, β::Number, backend::Backend...) where {S}
    if conjA == :N
        A′ = A
        pC′ = _canonicalize(pC, C)
        qA′ = qA
    elseif conjA == :C
        A′ = adjoint(A)
        pC′ = adjointtensorindices(A, _canonicalize(pC, C))
        qA′ = adjointtensorindices(A, qA)
    else
        throw(ArgumentError("unknown conjugation flag $conjA"))
    end
    # TODO: novel syntax for tensortrace?
    # tensortrace!(C, pC′, A′, qA′, α, β, backend...)
    trace!(α, A′, β, C, pC′[1], pC′[2], qA′[1], qA′[2])
    return C
end

# tensorcontract!
function TO.tensorcontract!(C::AbstractTensorMap{S,N₁,N₂}, pC::Index2Tuple,
                            A::AbstractTensorMap{S}, pA::Index2Tuple, conjA::Symbol,
                            B::AbstractTensorMap{S}, pB::Index2Tuple, conjB::Symbol,
                            α::Number, β::Number, backend::Backend...) where {S,N₁,N₂}
    pC′ = _canonicalize(pC, C)
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
    # TODO: novel syntax for tensorcontract?
    # tensorcontract!(C, pC′, A′, pA′, B′, pB′, α, β, backend...)
    contract!(α, A′, B′, β, C, pA′[1], pA′[2], pB′[2], pB′[1], pC′[1], pC′[2])
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
                                     B::AbstractTensorMap{S}, pB::Index2Tuple, conjB) where {S,N₁,N₂}
    
    spaces1 = TO.flag2op(conjA).(space.(Ref(A), pA[1]))
    spaces2 = TO.flag2op(conjB).(space.(Ref(B), pB[2]))
    spaces = (spaces1..., spaces2...)
    cod = ProductSpace{S,N₁}(getindex.(Ref(spaces), pC[1]))
    dom = ProductSpace{S,N₂}(dual.(getindex.(Ref(spaces), pC[2])))
    return dom → cod
end

# Actual implementations
function cached_permute(sym::Symbol, t::TensorMap{S},
                        p1::IndexTuple{N₁}, p2::IndexTuple{N₂}=();
                        copy::Bool=false) where {S,N₁,N₂}
    cod = ProductSpace{S,N₁}(map(n -> space(t, n), p1))
    dom = ProductSpace{S,N₂}(map(n -> dual(space(t, n)), p2))
    # share data if possible
    if !copy
        if p1 === codomainind(t) && p2 === domainind(t)
            return t
        elseif has_shared_permute(t, p1, p2)
            return TensorMap(reshape(t.data, dim(cod), dim(dom)), cod, dom)
        end
    end
    # general case
    @inbounds begin
        tp = TO.cached_similar_from_indices(sym, scalartype(t), p1, p2, t, :N)
        return add!(true, t, false, tp, p1, p2)
    end
end

function cached_permute(sym::Symbol, t::AdjointTensorMap,
                        p1::IndexTuple, p2::IndexTuple=();
                        copy::Bool=false)
    p1′ = adjointtensorindices(t, p2)
    p2′ = adjointtensorindices(t, p1)
    return adjoint(cached_permute(sym, adjoint(t), p1′, p2′; copy=copy))
end

@propagate_inbounds function add!(α, tsrc::AbstractTensorMap{S},
                                  β, tdst::AbstractTensorMap{S},
                                  p1::IndexTuple, p2::IndexTuple) where {S}
    I = sectortype(S)
    if BraidingStyle(I) isa SymmetricBraiding
        add_permute!(α, tsrc, β, tdst, p1, p2)
    else
        throw(ArgumentError("add! without levels is defined only if `BraidingStyle(sectortype(...)) isa SymmetricBraiding`"))
    end
end
@propagate_inbounds function add!(α, tsrc::AbstractTensorMap{S},
                                  β, tdst::AbstractTensorMap{S},
                                  p1::IndexTuple, p2::IndexTuple,
                                  levels::IndexTuple) where {S}
    return add_braid!(α, tsrc, β, tdst, p1, p2, levels)
end

@propagate_inbounds function add_permute!(α, tsrc::AbstractTensorMap{S},
                                          β, tdst::AbstractTensorMap{S,N₁,N₂},
                                          p1::IndexTuple{N₁},
                                          p2::IndexTuple{N₂}) where {S,N₁,N₂}
    return _add!(α, tsrc, β, tdst, p1, p2, (f₁, f₂) -> permute(f₁, f₂, p1, p2))
end
@propagate_inbounds function add_braid!(α, tsrc::AbstractTensorMap{S},
                                        β, tdst::AbstractTensorMap{S,N₁,N₂},
                                        p1::IndexTuple{N₁},
                                        p2::IndexTuple{N₂},
                                        levels::IndexTuple) where {S,N₁,N₂}
    length(levels) == numind(tsrc) ||
        throw(ArgumentError("incorrect levels $levels for tensor map $(codomain(tsrc)) ← $(domain(tsrc))"))

    levels1 = TupleTools.getindices(levels, codomainind(tsrc))
    levels2 = TupleTools.getindices(levels, domainind(tsrc))
    return _add!(α, tsrc, β, tdst, p1, p2,
                 (f₁, f₂) -> braid(f₁, f₂, levels1, levels2, p1, p2))
end
@propagate_inbounds function add_transpose!(α, tsrc::AbstractTensorMap{S},
                                            β, tdst::AbstractTensorMap{S,N₁,N₂},
                                            p1::IndexTuple{N₁},
                                            p2::IndexTuple{N₂}) where {S,N₁,N₂}
    return _add!(α, tsrc, β, tdst, p1, p2, (f₁, f₂) -> transpose(f₁, f₂, p1, p2))
end

function _add!(α, tsrc::AbstractTensorMap{S}, β, tdst::AbstractTensorMap{S,N₁,N₂},
               p1::IndexTuple{N₁}, p2::IndexTuple{N₂}, fusiontreemap) where {S,N₁,N₂}
    @boundscheck begin
        all(i -> space(tsrc, p1[i]) == space(tdst, i), 1:N₁) ||
            throw(SpaceMismatch("tsrc = $(codomain(tsrc))←$(domain(tsrc)),
            tdst = $(codomain(tdst))←$(domain(tdst)), p1 = $(p1), p2 = $(p2)"))
        all(i -> space(tsrc, p2[i]) == space(tdst, N₁ + i), 1:N₂) ||
            throw(SpaceMismatch("tsrc = $(codomain(tsrc))←$(domain(tsrc)),
            tdst = $(codomain(tdst))←$(domain(tdst)), p1 = $(p1), p2 = $(p2)"))
    end

    # do some kind of dispatch which is compiled away if S is known at compile time,
    # and makes the compiler give up quickly if S is unknown
    I = sectortype(S)
    i = I === Trivial ? 1 : (FusionStyle(I) isa UniqueFusion ? 2 : 3)
    if p1 == codomainind(tsrc) && p2 == domainind(tsrc)
        axpby!(α, tsrc, β, tdst)
    else
        _add_kernel! = _add_kernels[i]
        _add_kernel!(α, tsrc, β, tdst, p1, p2, fusiontreemap)
    end
    return tdst
end

function _add_trivial_kernel!(α, tsrc::AbstractTensorMap, β, tdst::AbstractTensorMap,
                              p1::IndexTuple, p2::IndexTuple, fusiontreemap)
    cod = codomain(tsrc)
    dom = domain(tsrc)
    n = length(cod)
    pdata = (p1..., p2...)
    axpby!(α, permutedims(tsrc[], pdata), β, tdst[])
    return nothing
end

function _add_abelian_kernel!(α, tsrc::AbstractTensorMap, β, tdst::AbstractTensorMap,
                              p1::IndexTuple, p2::IndexTuple, fusiontreemap)
    if Threads.nthreads() > 1
        nstridedthreads = Strided.get_num_threads()
        Strided.set_num_threads(1)
        Threads.@sync for (f₁, f₂) in fusiontrees(tsrc)
            Threads.@spawn _addabelianblock!(α, tsrc, β, tdst, p1, p2, f₁, f₂,
                                             fusiontreemap)
        end
        Strided.set_num_threads(nstridedthreads)
    else # debugging is easier this way
        for (f₁, f₂) in fusiontrees(tsrc)
            _addabelianblock!(α, tsrc, β, tdst, p1, p2, f₁, f₂, fusiontreemap)
        end
    end
    return nothing
end

function _addabelianblock!(α, tsrc::AbstractTensorMap,
                           β, tdst::AbstractTensorMap,
                           p1::IndexTuple, p2::IndexTuple,
                           f₁::FusionTree, f₂::FusionTree,
                           fusiontreemap)
    cod = codomain(tsrc)
    dom = domain(tsrc)
    (f₁′, f₂′), coeff = first(fusiontreemap(f₁, f₂))
    pdata = (p1..., p2...)
    @inbounds axpby!(α * coeff, permutedims(tsrc[f₁, f₂], pdata), β, tdst[f₁′, f₂′])
end

function _add_general_kernel!(α, tsrc::AbstractTensorMap, β, tdst::AbstractTensorMap,
                              p1::IndexTuple, p2::IndexTuple, fusiontreemap)
    cod = codomain(tsrc)
    dom = domain(tsrc)
    n = length(cod)
    pdata = (p1..., p2...)
    if iszero(β)
        fill!(tdst, β)
    elseif β != 1
        mul!(tdst, β, tdst)
    end
    for (f₁, f₂) in fusiontrees(tsrc)
        for ((f₁′, f₂′), coeff) in fusiontreemap(f₁, f₂)
            @inbounds axpy!(α * coeff, permutedims(tsrc[f₁, f₂], pdata), tdst[f₁′, f₂′])
        end
    end
    return nothing
end

const _add_kernels = (_add_trivial_kernel!, _add_abelian_kernel!, _add_general_kernel!)

function trace!(α, tsrc::AbstractTensorMap{S}, β, tdst::AbstractTensorMap{S,N₁,N₂},
                p1::IndexTuple{N₁}, p2::IndexTuple{N₂},
                q1::IndexTuple{N₃}, q2::IndexTuple{N₃}) where {S,N₁,N₂,N₃}
    if !(BraidingStyle(sectortype(S)) isa SymmetricBraiding)
        throw(SectorMismatch("only tensors with symmetric braiding rules can be contracted; try `@planar` instead"))
    end
    @boundscheck begin
        all(i -> space(tsrc, p1[i]) == space(tdst, i), 1:N₁) ||
            throw(SpaceMismatch("trace: tsrc = $(codomain(tsrc))←$(domain(tsrc)),
                    tdst = $(codomain(tdst))←$(domain(tdst)), p1 = $(p1), p2 = $(p2)"))
        all(i -> space(tsrc, p2[i]) == space(tdst, N₁ + i), 1:N₂) ||
            throw(SpaceMismatch("trace: tsrc = $(codomain(tsrc))←$(domain(tsrc)),
                    tdst = $(codomain(tdst))←$(domain(tdst)), p1 = $(p1), p2 = $(p2)"))
        all(i -> space(tsrc, q1[i]) == dual(space(tsrc, q2[i])), 1:N₃) ||
            throw(SpaceMismatch("trace: tsrc = $(codomain(tsrc))←$(domain(tsrc)),
                    q1 = $(q1), q2 = $(q2)"))
    end

    I = sectortype(S)
    if I === Trivial
        cod = codomain(tsrc)
        dom = domain(tsrc)
        n = length(cod)
        TO.tensortrace!(tdst[], (p1, p2), tsrc[], (q1, q2), :N, α, β)
        # elseif FusionStyle(I) isa UniqueFusion
        # TODO: is it worth multithreading UniqueFusion case for traces?
    else
        cod = codomain(tsrc)
        dom = domain(tsrc)
        n = length(cod)
        if iszero(β)
            fill!(tdst, β)
        elseif β != 1
            mul!(tdst, β, tdst)
        end
        r1 = (p1..., q1...)
        r2 = (p2..., q2...)
        for (f₁, f₂) in fusiontrees(tsrc)
            for ((f₁′, f₂′), coeff) in permute(f₁, f₂, r1, r2)
                f₁′′, g1 = split(f₁′, N₁)
                f₂′′, g2 = split(f₂′, N₂)
                if g1 == g2
                    coeff *= dim(g1.coupled) / dim(g1.uncoupled[1])
                    for i in 2:length(g1.uncoupled)
                        if !(g1.isdual[i])
                            coeff *= twist(g1.uncoupled[i])
                        end
                    end
                    TO.tensortrace!(tdst[f₁′′, f₂′′], (p1, p2), tsrc[f₁, f₂], (q1, q2), :N,
                                    α * coeff, true)
                end
            end
        end
    end
    return tdst
end

# TODO: contraction with either A or B a rank (1, 1) tensor does not require to
# permute the fusion tree and should therefore be special cased. This will speed
# up MPS algorithms
function contract!(α, A::AbstractTensorMap{S}, B::AbstractTensorMap{S},
                   β, C::AbstractTensorMap{S},
                   oindA::IndexTuple{N₁}, cindA::IndexTuple,
                   oindB::IndexTuple{N₂}, cindB::IndexTuple,
                   p1::IndexTuple, p2::IndexTuple,
                   syms::Union{Nothing,NTuple{3,Symbol}}=nothing) where {S,N₁,N₂}
    # find optimal contraction scheme
    hsp = has_shared_permute
    ipC = TupleTools.invperm((p1..., p2...))
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
    memcost1 = memcost2 = dC * (!hsp(C, oindAinC, oindBinC))
    memcost1 += dA * (!hsp(A, oindA, cindA′)) +
                dB * (!hsp(B, cindB′, oindB))
    memcost2 += dA * (!hsp(A, oindA, cindA′′)) +
                dB * (!hsp(B, cindB′′, oindB))

    # reverse order A en B, check possibilities for cind
    memcost3 = memcost4 = dC * (!hsp(C, oindBinC, oindAinC))
    memcost3 += dB * (!hsp(B, oindB, cindB′)) +
                dA * (!hsp(A, cindA′, oindA))
    memcost4 += dB * (!hsp(B, oindB, cindB′′)) +
                dA * (!hsp(A, cindA′′, oindA))

    if min(memcost1, memcost2) <= min(memcost3, memcost4)
        if memcost1 <= memcost2
            return _contract!(α, A, B, β, C, oindA, cindA′, oindB, cindB′, p1, p2, syms)
        else
            return _contract!(α, A, B, β, C, oindA, cindA′′, oindB, cindB′′, p1, p2, syms)
        end
    else
        p1′ = map(n -> ifelse(n > N₁, n - N₁, n + N₂), p1)
        p2′ = map(n -> ifelse(n > N₁, n - N₁, n + N₂), p2)
        if memcost3 <= memcost4
            return _contract!(α, B, A, β, C, oindB, cindB′, oindA, cindA′, p1′, p2′, syms)
        else
            return _contract!(α, B, A, β, C, oindB, cindB′′, oindA, cindA′′, p1′, p2′, syms)
        end
    end
end

function _contract!(α, A::AbstractTensorMap{S}, B::AbstractTensorMap{S},
                    β, C::AbstractTensorMap{S},
                    oindA::IndexTuple{N₁}, cindA::IndexTuple,
                    oindB::IndexTuple{N₂}, cindB::IndexTuple,
                    p1::IndexTuple, p2::IndexTuple,
                    syms::Union{Nothing,NTuple{3,Symbol}}=nothing) where {S,N₁,N₂}
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
    if syms === nothing
        A′ = permute(A, oindA, cindA; copy=copyA)
        B′ = permute(B, cindB, oindB)
    else
        A′ = cached_permute(syms[1], A, oindA, cindA; copy=copyA)
        B′ = cached_permute(syms[2], B, cindB, oindB)
    end
    if BraidingStyle(sectortype(S)) isa Fermionic
        for i in domainind(A′)
            if !isdual(space(A′, i))
                A′ = twist!(A′, i)
            end
        end
    end
    ipC = TupleTools.invperm((p1..., p2...))
    oindAinC = TupleTools.getindices(ipC, ntuple(n -> n, N₁))
    oindBinC = TupleTools.getindices(ipC, ntuple(n -> n + N₁, N₂))
    if has_shared_permute(C, oindAinC, oindBinC)
        C′ = permute(C, oindAinC, oindBinC)
        mul!(C′, A′, B′, α, β)
    else
        if syms === nothing
            C′ = A′ * B′
        else
            p1′ = ntuple(identity, N₁)
            p2′ = N₁ .+ ntuple(identity, N₂)
            TC = scalartype(C)
            C′ = TO.cached_similar_from_indices(syms[3], TC, oindA, oindB, p1′, p2′, A, B,
                                                :N, :N)
            mul!(C′, A′, B′)
        end
        add!(α, C′, β, C, p1, p2)
    end
    return C
end

function scalar(t::AbstractTensorMap{S}) where {S<:IndexSpace}
    return dim(codomain(t)) == dim(domain(t)) == 1 ?
           first(blocks(t))[2][1, 1] : throw(DimensionMismatch())
end
