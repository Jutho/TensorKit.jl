function cached_permute(sym::Symbol, t::TensorMap{S},
                            p1::IndexTuple{N₁},  p2::IndexTuple{N₂}=()) where {S, N₁, N₂}
    cod = ProductSpace{S, N₁}(map(n->space(t, n), p1))
    dom = ProductSpace{S, N₂}(map(n->dual(space(t, n)), p2))

    # share data if possible
    if p1 === codomainind(t) && p2 === domainind(t)
        return t
    elseif isa(t, TensorMap) && sectortype(S) === Trivial
        stridet = i->stride(t[], i)
        sizet = i->size(t[], i)
        canfuse1, d1, s1 = TensorOperations._canfuse(sizet.(p1), stridet.(p1))
        canfuse2, d2, s2 = TensorOperations._canfuse(sizet.(p2), stridet.(p2))
        if canfuse1 && canfuse2 && s1 == 1 && (d2 == 1 || s2 == d1)
            return TensorMap(reshape(t.data, dim(cod), dim(dom)), cod, dom)
        end
    end
    # general case
    @inbounds begin
        tp = TO.cached_similar_from_indices(sym, eltype(t), p1, p2, t, :N)
        return add!(true, t, false, tp, p1, p2)
    end
end

function cached_permute(sym::Symbol, t::AdjointTensorMap{S},
                            p1::IndexTuple{N₁},  p2::IndexTuple{N₂}=()) where {S, N₁, N₂}

    p1′ = adjointtensorindices(t, p2)
    p2′ = adjointtensorindices(t, p1)
    adjoint(cached_permute(sym, adjoint(t), p1′, p2′))
end

scalar(t::AbstractTensorMap{S}) where {S<:IndexSpace} =
    dim(codomain(t)) == dim(domain(t)) == 1 ?
        first(blocks(t))[2][1, 1] : throw(SpaceMismatch())

@propagate_inbounds function add!(α, tsrc::AbstractTensorMap{S},
                                    β, tdst::AbstractTensorMap{S},
                                    p1::IndexTuple, p2::IndexTuple) where {S}
    I = sectortype(S)
    if BraidingStyle(I) isa SymmetricBraiding
        add!(α, tsrc, β, tdst, p1, p2, (codomainind(tsrc)..., domainind(tsrc)...))
    else
        # If BraidingStyle is non-trivial, then check if the given permutation can be done
        # without braiding.
        N1 = numout(tsrc)
        N2 = numin(tsrc)
        p = (p1..., reverse(p2)...)
        N = N1 + N2
        nop = (1:N1..., reverse(N1+1:N)...)  # The noop permutation.
        firstind_p = findfirst(x -> x == 1, p)
        firstind_nop = findfirst(x -> x == 1, nop)
        # Check that p and nop are related to each other rotations by, i.e. taking indices
        # from the end putting them in the beginning, or vice versa.
        isplanar = all(p[mod1(firstind_p + i - 1, N)] == nop[mod1(firstind_nop + i - 1, N)]
                       for i in 1:N)
        if !isplanar
            msg = "add! without levels only if `BraidingStyle(sectortype(...)) isa SymmetricBraiding`, or the permutation does not require braiding. Here codomainind = $(codomainind(tsrc)), domainind = $(domainind(tsrc)), but p1 = $p1, p2 = $p2."
            throw(ArgumentError(msg))
        end

        # If no actual braiding is required, do the permutation in two steps: first bend
        # legs between the domain and codomain at the left end of the tensor, as necessary.
        # This needs to be done one index at a time, to avoid spurious braids.
        p_first = p[1]
        # TODO This copy if wasteful, but necessary with the current implementation because
        # bend_first_up and bend_first_down are inplace. Rethink once the structure is more
        # settled, i.e. maybe this whole bit is going to be moved to a different function.
        tsrc = deepcopy(tsrc)
        while p_first != 1
            if p_first <= N1
                tsrc = bend_first_up(tsrc)
                p1 = bend_first_up(p1, N1)
                p2 = bend_first_up(p2, N1)
                N1 -= 1
            else
                tsrc = bend_first_down(tsrc)
                p1 = bend_first_down(p1, N1)
                p2 = bend_first_down(p2, N1)
                N1 += 1
            end
            p_first = length(p1) > 0 ? p1[1] : p2[end]
        end

        # Second, bend legs over the right end.
        levels = (1:N...,)
        return add!(α, tsrc, β, tdst, p1, p2, levels)
    end
end

"""
    bend_first_up(t::AbstractTensorMap; copy::Bool=false)

Take the first index of the codomain of `t`, and bend it up over the "left" end of the
tensor, i.e. permute it to become the first index of the domain without any braiding or
twists.
"""
function bend_first_up(t::AbstractTensorMap; copy::Bool=false)
    N1 = numout(t)
    N = numind(t)
    p1 = (2:N1...,)
    p2 = (1, N1+1:N...)
    levels = (1:N...,)
    # The `braid` function braids bends the leg over the right end of the tensor, which
    # introduces a twist. We cancel that twist.
    t = braid(t, levels, p1, p2; copy = copy)
    return twist!(t, N1; inv = false)
end

"""
    bend_first_down(t::AbstractTensorMap; copy::Bool=false)

Take the first index of the domain of `t`, and bend it down over the "left" end of the
tensor, i.e. permute it to become the first index of the codomain without any braiding or
twists.
"""
function bend_first_down(t::AbstractTensorMap; copy::Bool=false)
    N1 = numout(t)
    N2 = numin(t)
    N = numind(t)
    p1 = (N1+1, 1:N1...)
    p2 = (N1+2:N...,)
    levels = (N2+1:N..., 1:N2...)
    # The `braid` function braids bends the leg over the right end of the tensor, which
    # introduces a twist. We cancel that twist.
    t = braid(t, levels, p1, p2; copy = copy)
    return twist!(t, 1; inv = true)
end

function bend_first_up(p::IndexTuple, N1::Int)
    return map(i -> ifelse(i == 1, N1, ifelse(i <= N1, i-1, i)), p)
end

function bend_first_down(p::IndexTuple, N1::Int)
    return map(i -> ifelse(i == N1+1, 1, ifelse(i <= N1, i+1, i)), p)
end

function add!(α, tsrc::AbstractTensorMap{S}, β, tdst::AbstractTensorMap{S, N₁, N₂},
                p1::IndexTuple{N₁}, p2::IndexTuple{N₂}, levels::IndexTuple) where {S, N₁, N₂}
    @boundscheck begin
        all(i->space(tsrc, p1[i]) == space(tdst, i), 1:N₁) ||
            throw(SpaceMismatch("tsrc = $(codomain(tsrc))←$(domain(tsrc)),
            tdst = $(codomain(tdst))←$(domain(tdst)), p1 = $(p1), p2 = $(p2)"))
        all(i->space(tsrc, p2[i]) == space(tdst, N₁+i), 1:N₂) ||
            throw(SpaceMismatch("tsrc = $(codomain(tsrc))←$(domain(tsrc)),
            tdst = $(codomain(tdst))←$(domain(tdst)), p1 = $(p1), p2 = $(p2)"))
        length(levels) == numind(tsrc) ||
            throw(ArgumentError("incorrect levels $levels for tensor map $(codomain(t)) ← $(domain(t))"))
    end

    # do some kind of dispatch which is compiled away if S is known at compile time,
    # and makes the compiler give up quickly if S is unknown
    I = sectortype(S)
    i = I === Trivial ? 1 : (FusionStyle(I) isa Abelian ? 2 : 3)
    _add_kernel! = _add_kernels[i]
    _add_kernel!(α, tsrc, β, tdst, p1, p2, levels)

    return tdst
end

function _add_trivial_kernel!(α, tsrc::AbstractTensorMap, β, tdst::AbstractTensorMap,
                                p1::IndexTuple, p2::IndexTuple, levels::IndexTuple)
    cod = codomain(tsrc)
    dom = domain(tsrc)
    n = length(cod)
    pdata = (p1..., p2...)
    axpby!(α, permutedims(tsrc[], pdata), β, tdst[])
    return nothing
end

function _add_abelian_kernel!(α, tsrc::AbstractTensorMap, β, tdst::AbstractTensorMap,
                                p1::IndexTuple, p2::IndexTuple, levels::IndexTuple)
    if Threads.nthreads() > 1
        nstridedthreads = Strided.get_num_threads()
        Strided.set_num_threads(1)
        Threads.@sync for (f1, f2) in fusiontrees(tsrc)
            Threads.@spawn _addabelianblock!(α, tsrc, β, tdst, p1, p2, f1, f2)
        end
        Strided.set_num_threads(nstridedthreads)
    else # debugging is easier this way
        for (f1, f2) in fusiontrees(tsrc)
            _addabelianblock!(α, tsrc, β, tdst, p1, p2, f1, f2)
        end
    end
    return nothing
end

function _addabelianblock!(α, tsrc::AbstractTensorMap,
                            β, tdst::AbstractTensorMap,
                            p1::IndexTuple, p2::IndexTuple,
                            f1::FusionTree, f2::FusionTree)
    cod = codomain(tsrc)
    dom = domain(tsrc)
    (f1′, f2′), coeff = first(permute(f1, f2, p1, p2))
    pdata = (p1..., p2...)
    @inbounds axpby!(α*coeff, permutedims(tsrc[f1, f2], pdata), β, tdst[f1′, f2′])
end

function _add_general_kernel!(α, tsrc::AbstractTensorMap, β, tdst::AbstractTensorMap,
                                p1::IndexTuple, p2::IndexTuple, levels::IndexTuple)
    cod = codomain(tsrc)
    dom = domain(tsrc)
    n = length(cod)
    pdata = (p1..., p2...)
    if iszero(β)
        fill!(tdst, β)
    elseif β != 1
        mul!(tdst, β, tdst)
    end
    levels1 = TupleTools.getindices(levels, codomainind(tsrc))
    levels2 = TupleTools.getindices(levels, domainind(tsrc))
    for (f1, f2) in fusiontrees(tsrc)
        for ((f1′, f2′), coeff) in braid(f1, f2, levels1, levels2, p1, p2)
            @inbounds axpy!(α*coeff, permutedims(tsrc[f1, f2], pdata), tdst[f1′, f2′])
        end
    end
    return nothing
end

const _add_kernels = (_add_trivial_kernel!, _add_abelian_kernel!, _add_general_kernel!)

function trace!(α, tsrc::AbstractTensorMap{S}, β, tdst::AbstractTensorMap{S, N₁, N₂},
                p1::IndexTuple{N₁}, p2::IndexTuple{N₂},
                q1::IndexTuple{N₃}, q2::IndexTuple{N₃}) where {S, N₁, N₂, N₃}
    # TODO: check Frobenius-Schur indicators!, and  add fermions!
    @boundscheck begin
        all(i->space(tsrc, p1[i]) == space(tdst, i), 1:N₁) ||
            throw(SpaceMismatch("trace: tsrc = $(codomain(tsrc))←$(domain(tsrc)),
                    tdst = $(codomain(tdst))←$(domain(tdst)), p1 = $(p1), p2 = $(p2)"))
        all(i->space(tsrc, p2[i]) == space(tdst, N₁+i), 1:N₂) ||
            throw(SpaceMismatch("trace: tsrc = $(codomain(tsrc))←$(domain(tsrc)),
                    tdst = $(codomain(tdst))←$(domain(tdst)), p1 = $(p1), p2 = $(p2)"))
        all(i->space(tsrc, q1[i]) == dual(space(tsrc, q2[i])), 1:N₃) ||
            throw(SpaceMismatch("trace: tsrc = $(codomain(tsrc))←$(domain(tsrc)),
                    q1 = $(p1), q2 = $(q2)"))
    end

    I = sectortype(S)
    if I === Trivial
        cod = codomain(tsrc)
        dom = domain(tsrc)
        n = length(cod)
        pdata = (p1..., p2...)
        TO._trace!(α, tsrc[], β, tdst[], pdata, q1, q2)
    # elseif FusionStyle(I) isa Abelian
    # TODO: is it worth multithreading Abelian case for traces?
    else
        cod = codomain(tsrc)
        dom = domain(tsrc)
        n = length(cod)
        pdata = (p1..., p2...)
        if iszero(β)
            fill!(tdst, β)
        elseif β != 1
            mul!(tdst, β, tdst)
        end
        r1 = (p1..., q1...)
        r2 = (p2..., q2...)
        for (f1, f2) in fusiontrees(tsrc)
            for ((f1′, f2′), coeff) in permute(f1, f2, r1, r2)
                f1′′, g1 = split(f1′, N₁)
                f2′′, g2 = split(f2′, N₂)
                if g1 == g2
                    coeff *= dim(g1.coupled)/dim(g1.uncoupled[1])
                    TO._trace!(α*coeff, tsrc[f1, f2], true, tdst[f1′′, f2′′], pdata, q1, q2)
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
                    syms::Union{Nothing, NTuple{3, Symbol}} = nothing) where {S, N₁, N₂}
    if BraidingStyle(sectortype(S)) == Anyonic()
        # We shouldn't touch the ordering of the tensors or indices, since braiding is
        # non-trivial.
        return _contract!(α, A, B, β, C, oindA, cindA, oindB, cindB, p1, p2, syms)
    end
    # find optimal contraction scheme
    hsp = has_shared_permute
    ipC = TupleTools.invperm((p1..., p2...))
    oindAinC = TupleTools.getindices(ipC, ntuple(n->n, N₁))
    oindBinC = TupleTools.getindices(ipC, ntuple(n->n+N₁, N₂))

    qA = TupleTools.sortperm(cindA)
    cindA′ = TupleTools.getindices(cindA, qA)
    cindB′ = TupleTools.getindices(cindB, qA)

    qB = TupleTools.sortperm(cindB)
    cindA′′ = TupleTools.getindices(cindA, qB)
    cindB′′ = TupleTools.getindices(cindB, qB)

    dA, dB, dC = dim(A), dim(B), dim(C)

    # keep order A en B, check possibilities for cind
    memcost1 = memcost2 = dC*(!hsp(C, oindAinC, oindBinC))
    memcost1 += dA*(!hsp(A, oindA, cindA′)) +
                dB*(!hsp(B, cindB′, oindB))
    memcost2 += dA*(!hsp(A, oindA, cindA′′)) +
                dB*(!hsp(B, cindB′′, oindB))

    # reverse order A en B, check possibilities for cind
    memcost3 = memcost4 = dC*(!hsp(C, oindBinC, oindAinC))
    memcost3 += dB*(!hsp(B, oindB, cindB′)) +
                dA*(!hsp(A, cindA′, oindA))
    memcost4 += dB*(!hsp(B, oindB, cindB′′)) +
                dA*(!hsp(A, cindA′′, oindA))

    if min(memcost1, memcost2) <= min(memcost3, memcost4)
        if memcost1 <= memcost2
            return _contract!(α, A, B, β, C, oindA, cindA′, oindB, cindB′, p1, p2, syms)
        else
            return _contract!(α, A, B, β, C, oindA, cindA′′, oindB, cindB′′, p1, p2, syms)
        end
    else
        p1′ = map(n->ifelse(n>N₁, n-N₁, n+N₂), p1)
        p2′ = map(n->ifelse(n>N₁, n-N₁, n+N₂), p2)
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
                    syms::Union{Nothing, NTuple{3, Symbol}} = nothing) where {S, N₁, N₂}

    if syms === nothing
        A′ = permute(A, oindA, cindA)
        B′ = permute(B, cindB, oindB)
    else
        A′ = cached_permute(syms[1], A, oindA, cindA)
        B′ = cached_permute(syms[2], B, cindB, oindB)
    end
    ipC = TupleTools.invperm((p1..., p2...))
    oindAinC = TupleTools.getindices(ipC, ntuple(n->n, N₁))
    oindBinC = TupleTools.getindices(ipC, ntuple(n->n+N₁, N₂))
    if has_shared_permute(C, oindAinC, oindBinC)
        C′ = permute(C, oindAinC, oindBinC)
        mul!(C′, A′, B′, α, β)
    else
        if syms === nothing
            C′ = A′*B′
        else
            p1′ = ntuple(identity, N₁)
            p2′ = N₁ .+ ntuple(identity, N₂)
            TC = eltype(C)
            C′ = TO.cached_similar_from_indices(syms[3], TC, oindA, oindB, p1′, p2′, A, B, :N, :N)
            mul!(C′, A′, B′)
        end
        add!(α, C′, β, C, p1, p2)
    end
    return C
end

# Add support for cache and API (`@tensor` macro & friends) from TensorOperations.jl:
# compatibility layer
function TensorOperations.memsize(t::TensorMap)
    s = 0
    for (c, b) in blocks(t)
        s += sizeof(b)
    end
    return s
end
TensorOperations.memsize(t::AdjointTensorMap) = TensorOperations.memsize(t')

function TO.similarstructure_from_indices(T::Type, p1::IndexTuple, p2::IndexTuple,
        A::AbstractTensorMap, CA::Symbol = :N)
    if CA == :N
        _similarstructure_from_indices(T, p1, p2, A)
    else
        p1 = adjointtensorindices(A, p1)
        p2 = adjointtensorindices(A, p2)
        _similarstructure_from_indices(T, p1, p2, adjoint(A))
    end
end

function TO.similarstructure_from_indices(T::Type, poA::IndexTuple, poB::IndexTuple,
        p1::IndexTuple, p2::IndexTuple,
        A::AbstractTensorMap, B::AbstractTensorMap,
        CA::Symbol = :N, CB::Symbol = :N)

    if CA == :N && CB == :N
        _similarstructure_from_indices(T, poA, poB, p1, p2, A, B)
    elseif CA == :C && CB == :N
        poA = adjointtensorindices(A, poA)
        _similarstructure_from_indices(T, poA, poB, p1, p2, adjoint(A), B)
    elseif CA == :N && CB == :C
        poB = adjointtensorindices(B, poB)
        _similarstructure_from_indices(T, poA, poB, p1, p2, A, adjoint(B))
    else
        poA = adjointtensorindices(A, poA)
        poB = adjointtensorindices(B, poB)
        _similarstructure_from_indices(T, poA, poB, p1, p2, adjoint(A), adjoint(B))
    end
end

function _similarstructure_from_indices(::Type{T}, p1::IndexTuple{N₁}, p2::IndexTuple{N₂},
        t::AbstractTensorMap{S}) where {T, S<:IndexSpace, N₁, N₂}

    cod = ProductSpace{S, N₁}(space.(Ref(t), p1))
    dom = ProductSpace{S, N₂}(dual.(space.(Ref(t), p2)))
    return dom→cod
end
function _similarstructure_from_indices(::Type{T}, oindA::IndexTuple, oindB::IndexTuple,
        p1::IndexTuple{N₁}, p2::IndexTuple{N₂},
        tA::AbstractTensorMap{S}, tB::AbstractTensorMap{S}) where {T, S<:IndexSpace, N₁, N₂}

    spaces = (space.(Ref(tA), oindA)..., space.(Ref(tB), oindB)...)
    cod = ProductSpace{S, N₁}(getindex.(Ref(spaces), p1))
    dom = ProductSpace{S, N₂}(dual.(getindex.(Ref(spaces), p2)))
    return dom→cod
end

TO.scalar(t::AbstractTensorMap) = scalar(t)

function TO.add!(α, tsrc::AbstractTensorMap{S}, CA::Symbol, β,
    tdst::AbstractTensorMap{S, N₁, N₂}, p1::IndexTuple, p2::IndexTuple) where {S, N₁, N₂}

    if CA == :N
        p = (p1..., p2...)
        pl = TupleTools.getindices(p, codomainind(tdst))
        pr = TupleTools.getindices(p, domainind(tdst))
        add!(α, tsrc, β, tdst, pl, pr)
    else
        p = adjointtensorindices(tsrc, (p1..., p2...))
        pl = TupleTools.getindices(p, codomainind(tdst))
        pr = TupleTools.getindices(p, domainind(tdst))
        add!(α, adjoint(tsrc), β, tdst, pl, pr)
    end
    return tdst
end

function TO.trace!(α, tsrc::AbstractTensorMap{S}, CA::Symbol, β,
    tdst::AbstractTensorMap{S, N₁, N₂}, p1::IndexTuple, p2::IndexTuple,
    q1::IndexTuple, q2::IndexTuple) where {S, N₁, N₂}

    if CA == :N
        p = (p1..., p2...)
        pl = TupleTools.getindices(p, codomainind(tdst))
        pr = TupleTools.getindices(p, domainind(tdst))
        trace!(α, tsrc, β, tdst, pl, pr, q1, q2)
    else
        p = adjointtensorindices(tsrc, (p1..., p2...))
        pl = TupleTools.getindices(p, codomainind(tdst))
        pr = TupleTools.getindices(p, domainind(tdst))
        q1 = adjointtensorindices(tsrc, q1)
        q2 = adjointtensorindices(tsrc, q2)
        trace!(α, adjoint(tsrc), β, tdst, pl, pr, q1, q2)
    end
    return tdst
end

function TO.contract!(α,
    tA::AbstractTensorMap{S}, CA::Symbol,
    tB::AbstractTensorMap{S}, CB::Symbol,
    β, tC::AbstractTensorMap{S, N₁, N₂},
    oindA::IndexTuple, cindA::IndexTuple,
    oindB::IndexTuple, cindB::IndexTuple,
    p1::IndexTuple, p2::IndexTuple,
    syms::Union{Nothing, NTuple{3, Symbol}} = nothing) where {S, N₁, N₂}

    p = (p1..., p2...)
    pl = ntuple(n->p[n], N₁)
    pr = ntuple(n->p[N₁+n], N₂)
    if CA == :N && CB == :N
        contract!(α, tA, tB, β, tC, oindA, cindA, oindB, cindB, pl, pr, syms)
    elseif CA == :N && CB == :C
        oindB = adjointtensorindices(tB, oindB)
        cindB = adjointtensorindices(tB, cindB)
        contract!(α, tA, tB', β, tC, oindA, cindA, oindB, cindB, pl, pr, syms)
    elseif CA == :C && CB == :N
        oindA = adjointtensorindices(tA, oindA)
        cindA = adjointtensorindices(tA, cindA)
        contract!(α, tA', tB, β, tC, oindA, cindA, oindB, cindB, pl, pr, syms)
    elseif CA == :C && CB == :C
        oindA = adjointtensorindices(tA, oindA)
        cindA = adjointtensorindices(tA, cindA)
        oindB = adjointtensorindices(tB, oindB)
        cindB = adjointtensorindices(tB, cindB)
        contract!(α, tA', tB', β, tC, oindA, cindA, oindB, cindB, pl, pr, syms)
    else
        error("unknown conjugation flags: $CA and $CB")
    end
    return tC
end
