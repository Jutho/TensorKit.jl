# BraidingTensor:
# special (2,2) tensor that implements a standard braiding operation
#====================================================================#
"""
    struct BraidingTensor{S<:IndexSpace} <: AbstractTensorMap{S, 2, 2}

Specific subtype of [`AbstractTensorMap`](@ref) for representing the braiding tensor that
braids the first input over the second input; its inverse can be obtained as the adjoint.
"""
struct BraidingTensor{S<:IndexSpace, A} <: AbstractTensorMap{S, 2, 2}
    V1::S
    V2::S
    adjoint::Bool
    function BraidingTensor(V1::S, V2::S, adjoint::Bool = false, ::Type{A} = Matrix{ComplexF64}) where
                {S<:IndexSpace, A<:DenseMatrix}
        for a in sectors(V1)
            for b in sectors(V2)
                for c in (a ⊗ b)
                    Nsymbol(a, b, c) == Nsymbol(b, a, c) ||
                        throw(ArgumentError("Cannot define a braiding between $a and $b"))
                end
            end
        end
        return new{S,A}(V1, V2, adjoint)
        # partial construction: only construct rowr and colr when needed
    end
end

Base.adjoint(b::BraidingTensor{S,A}) where {S<:IndexSpace, A<:DenseMatrix} =
    BraidingTensor(b.V1, b.V2, !b.adjoint, A)

domain(b::BraidingTensor) = b.adjoint ? b.V2 ⊗ b.V1 : b.V1 ⊗ b.V2
codomain(b::BraidingTensor) = b.adjoint ? b.V1 ⊗ b.V2 : b.V2 ⊗ b.V1

storagetype(::Type{BraidingTensor{S,A}}) where {S<:IndexSpace, A<:DenseMatrix} = A

blocksectors(b::BraidingTensor) = blocksectors(b.V1 ⊗ b.V2)
hasblock(b::BraidingTensor, s::Sector) = s ∈ blocksectors(b)

function fusiontrees(b::BraidingTensor)
    codom = codomain(b)
    dom = domain(b)
    I = sectortype(b)
    F = fusiontreetype(I, 2)
    rowr = SectorDict{I, FusionTreeDict{F, UnitRange{Int}}}()
    colr = SectorDict{I, FusionTreeDict{F, UnitRange{Int}}}()
    for c in blocksectors(codom)
        rowrc = FusionTreeDict{F, UnitRange{Int}}()
        colrc = FusionTreeDict{F, UnitRange{Int}}()
        offset1 = 0
        for s1 in sectors(codom)
            for f1 in fusiontrees(s1, c, map(isdual, codom.spaces))
                r = (offset1 + 1):(offset1 + dim(codom, s1))
                push!(rowrc, f1 => r)
                offset1 = last(r)
            end
        end
        dim1 = offset1
        offset2 = 0
        for s2 in sectors(dom)
            for f2 in fusiontrees(s2, c, map(isdual, dom.spaces))
                r = (offset2 + 1):(offset2 + dim(dom, s2))
                push!(colrc, f2 => r)
                offset2 = last(r)
            end
        end
        dim2 = offset2
        push!(rowr, c=>rowrc)
        push!(colr, c=>colrc)
    end
    return TensorKeyIterator(rowr, colr)
end

function Base.getindex(b::BraidingTensor{S}) where S
    sectortype(S) == Trivial || throw(SectorMismatch())
    (V1, V2) = domain(b)
    d = (dim(V2), dim(V1), dim(V1), dim(V2))
    return sreshape(StridedView(block(b, Trivial())), d)
end

@inline function Base.getindex(b::BraidingTensor, f1::FusionTree{I,2}, f2::FusionTree{I,2}) where {I<:Sector}
    I == sectortype(b) || throw(SectorMismatch())
    c = f1.coupled
    V1, V2 = domain(b)
    @boundscheck begin
        c == f2.coupled || throw(SectorMismatch())
        ((f1.uncoupled[1] ∈ sectors(V2)) && (f2.uncoupled[1] ∈ sectors(V1))) || throw(SectorMismatch())
        ((f1.uncoupled[2] ∈ sectors(V1)) && (f2.uncoupled[2] ∈ sectors(V2))) || throw(SectorMismatch())
    end
    @inbounds begin
        d = (dims(V2 ⊗ V1, f1.uncoupled)..., dims(V1 ⊗ V2, f2.uncoupled)...)
        n1 = d[1]*d[2]
        n2 = d[3]*d[4]
        data = fill!(storagetype(b)(undef, (n1, n2)), zero(eltype(b)))
        a1, a2 = f2.uncoupled
        if f1.uncoupled == (a2, a1)
            braiddict = artin_braid(f2, 1; inv = b.adjoint)
            r = get(braiddict, f1, zero(valtype(braiddict)))
            si = 1 + d[1]*d[2]*d[3]
            sj = d[1] + d[1]*d[2]
            @inbounds for i = 1:d[1], j = 1:d[2]
                data[(i-1)*si + (j-1)*sj + 1] = r
            end
        end
        return sreshape(StridedView(data), d)
    end
end

Base.copy(b::BraidingTensor) = copy!(similar(b), b)
function Base.copy!(t::TensorMap, b::BraidingTensor)
    space(t) == space(b) || throw(SectorMismatch())
    fill!(t, zero(eltype(t)))
    for (f1, f2) in fusiontrees(t)
        data = t[f1, f2]
        if sectortype(t) == Trivial
            r = one(eltype(t))
        else
            a1, a2 = f2.uncoupled
            c = f2.coupled
            f1.uncoupled == (a2, a1) || continue
            braiddict = artin_braid(f2, 1; inv = b.adjoint)
            r = convert(eltype(t), get(braiddict, f1, zero(valtype(braiddict))))
        end
        for i = 1:size(data, 1), j = 1:size(data, 2)
            data[i, j, j, i] = r
        end
    end
    return t
end
TensorMap(b::BraidingTensor) = copy(b)
Base.convert(::Type{TensorMap}, b::BraidingTensor) = copy(b)

function block(b::BraidingTensor, s::Sector)
    sectortype(b) == typeof(s) || throw(SectorMismatch())
    (V1, V2) = domain(b)
    if sectortype(b) == Trivial
        d1, d2 = dim(V1), dim(V2)
        n = d1*d2
        data = fill!(storagetype(b)(undef, (n, n)), zero(eltype(b)))
        si = 1 + d2*d1*d1
        sj = d2 + d2*d1
        @inbounds for i = 1:d2, j = 1:d1
            data[(i-1)*si + (j-1)*sj + 1] = one(eltype(b))
        end
        return data
    end
    n = blockdim(domain(b), s)
    data = fill!(storagetype(b)(undef, (n, n)), zero(eltype(b)))
    iter = fusiontrees(b) # actually contains information about ranges as well
    for (f2, r2) in iter.colr[s]
        for (f1, r1) in iter.rowr[s]
            a1, a2 = f2.uncoupled
            d1 = dim(V1, a1)
            d2 = dim(V2, a2)
            f1.uncoupled == (a2, a1) || continue
            braiddict = artin_braid(f2, 1; inv = b.adjoint)
            r = convert(eltype(b), get(braiddict, f1, zero(valtype(braiddict))))
            si = 1 + n*d1
            sj = d2 + n
            start = first(r1) + (first(r2)-1) * n
            @inbounds for i = 1:d2, j=1:d1
                data[(i-1)*si + (j-1)*sj + start] = r
            end
        end
    end
    return data
end

blocks(b::BraidingTensor) = blocks(TensorMap(b))

function planar_contract!(α, A::BraidingTensor, B::AbstractTensorMap{S},
                            β, C::AbstractTensorMap{S},
                            oindA::IndexTuple{2}, cindA::IndexTuple{2},
                            oindB::IndexTuple, cindB::IndexTuple{2},
                            p1::IndexTuple, p2::IndexTuple,
                            syms::Union{Nothing, NTuple{3, Symbol}}) where {S}

    codA, domA = codomainind(A), domainind(A)
    codB, domB = codomainind(B), domainind(B)
    oindA, cindA, oindB, cindB =
        reorder_indices(codA, domA, codB, domB, oindA, cindA, oindB, cindB, p1, p2)

    @assert space(B, cindB[1]) == space(A, cindA[1])' &&
                space(B, cindB[2]) == space(A, cindA[2])'

    if BraidingStyle(sectortype(B)) isa Bosonic
        return add!(α, B, β, C, reverse(cindB), oindB)
    end

    if iszero(β)
        fill!(C, β)
    elseif β != 1
        rmul!(C, β)
    end
    braidingtensor_levels = A.adjoint ? (1,2,2,1) : (2,1,1,2)
    inv_braid = braidingtensor_levels[cindA[1]] > braidingtensor_levels[cindA[2]]
    for (f1, f2) in fusiontrees(B)
        local newtrees
        for ((f1′, f2′), coeff′) in transpose(f1, f2, cindB, oindB)
            for (f1′′, coeff′′) in artin_braid(f1′, 1, inv = inv_braid)

                f12 = (f1′′, f2′)
                coeff = coeff′*coeff′′
                if @isdefined newtrees
                    newtrees[f12] = get(newtrees, f12, zero(coeff)) + coeff
                else
                    newtrees = Dict(f12 => coeff)
                end
            end
        end
        for ((f1′,f2′), coeff) in newtrees
            TO._add!(coeff*α, B[f1, f2], true, C[f1′,f2′], (reverse(cindB)...,oindB...))
        end
    end
    return C
end
function planar_contract!(α, A::AbstractTensorMap{S}, B::BraidingTensor,
                            β, C::AbstractTensorMap{S},
                            oindA::IndexTuple, cindA::IndexTuple{2},
                            oindB::IndexTuple{2}, cindB::IndexTuple{2},
                            p1::IndexTuple, p2::IndexTuple,
                            syms::Union{Nothing, NTuple{3, Symbol}}) where {S}
    codA, domA = codomainind(A), domainind(A)
    codB, domB = codomainind(B), domainind(B)
    oindA, cindA, oindB, cindB =
        reorder_indices(codA, domA, codB, domB, oindA, cindA, oindB, cindB, p1, p2)

    @assert space(B, cindB[1]) == space(A, cindA[1])' &&
                space(B, cindB[2]) == space(A, cindA[2])'

    if BraidingStyle(sectortype(A)) isa Bosonic
        return add!(α, A, β, C, oindA, reverse(cindA))
    end

    if iszero(β)
        fill!(C, β)
    elseif β != 1
        rmul!(C, β)
    end
    braidingtensor_levels = B.adjoint ? (1,2,2,1) : (2,1,1,2)
    inv_braid = braidingtensor_levels[cindB[1]] > braidingtensor_levels[cindB[2]]
    for (f1, f2) in fusiontrees(A)
        local newtrees
        for ((f1′,f2′), coeff′) in transpose(f1, f2, oindA, cindA)
            for (f2′′, coeff′′) in artin_braid(f2′, 1, inv = inv_braid)
                f12 = (f1′,f2′′)
                coeff = coeff′*conj(coeff′′)
                if @isdefined newtrees
                    newtrees[f12] = get(newtrees, f12, zero(coeff)) + coeff
                else
                    newtrees = Dict(f12 => coeff)
                end
            end
        end
        for ((f1′,f2′), coeff) in newtrees
            TO._add!(coeff*α, A[f1, f2], true, C[f1′,f2′], (oindA...,reverse(cindA)...))
        end
    end
    C
end

function planar_contract!(α, A::BraidingTensor, B::AbstractTensorMap{S},
                            β, C::AbstractTensorMap{S},
                            oindA::IndexTuple{0}, cindA::IndexTuple{4},
                            oindB::IndexTuple, cindB::IndexTuple{4},
                            p1::IndexTuple, p2::IndexTuple,
                            syms::Union{Nothing, NTuple{3, Symbol}}) where {S}

    codA, domA = codomainind(A), domainind(A)
    codB, domB = codomainind(B), domainind(B)
    oindA, cindA, oindB, cindB =
        reorder_indices(codA, domA, codB, domB, oindA, cindA, oindB, cindB, p1, p2)

    @assert space(B, cindB[1]) == space(A, cindA[1])' &&
                space(B, cindB[2]) == space(A, cindA[2])' &&
                space(B, cindB[3]) == space(A, cindA[3])' &&
                space(B, cindB[4]) == space(A, cindA[4])'

    if BraidingStyle(sectortype(B)) isa Bosonic
        return trace!(α, B, β, C, (), oindB, (cindB[1], cindB[2]), (cindB[3], cindB[4]))
    end

    if iszero(β)
        fill!(C, β)
    elseif β != 1
        rmul!(C, β)
    end
    I = sectortype(B)
    u = one(I)
    f₀ = FusionTree{I}((), u, (), (), ())
    braidingtensor_levels = A.adjoint ? (1,2,2,1) : (2,1,1,2)
    inv_braid = braidingtensor_levels[cindA[2]] > braidingtensor_levels[cindA[3]]
    for (f1, f2) in fusiontrees(B)
        local newtrees
        for ((f1′,f2′), coeff′) in transpose(f1, f2, cindB, oindB)
            f1′.coupled == u || continue
            a = f1′.uncoupled[1]
            b = f1′.uncoupled[2]
            f1′.uncoupled[3] == dual(a) || continue
            f1′.uncoupled[4] == dual(b) || continue
            # should be automatic by matching spaces:
            # f1′.isdual[1] != f1′.isdual[3] || continue
            # f1′.isdual[2] != f1′.isdual[4] || continue
            for (f1′′, coeff′′) in artin_braid(f1′, 2, inv = inv_braid)
                f1′′.innerlines[1] == u || continue
                coeff = coeff′ * coeff′′ * sqrtdim(a) * sqrtdim(b)
                if f1′′.isdual[1]
                    coeff *= frobeniusschur(a)
                end
                if f1′′.isdual[3]
                    coeff *= frobeniusschur(b)
                end
                f12 = (f₀, f2′)
                if @isdefined newtrees
                    newtrees[f12] = get(newtrees, f12, zero(coeff)) + coeff
                else
                    newtrees = Dict(f12 => coeff)
                end
            end
        end
        @isdefined(newtrees) || continue
        for ((f1′, f2′), coeff) in newtrees
            TO._trace!(coeff*α, B[f1, f2], true, C[f1′,f2′], oindB,
                                            (cindB[1], cindB[2]), (cindB[3], cindB[4]))
        end
    end
    return C
end

function planar_contract!(α, A::AbstractTensorMap{S}, B::BraidingTensor,
                            β, C::AbstractTensorMap{S},
                            oindA::IndexTuple, cindA::IndexTuple{4},
                            oindB::IndexTuple{0}, cindB::IndexTuple{4},
                            p1::IndexTuple, p2::IndexTuple,
                            syms::Union{Nothing, NTuple{3, Symbol}}) where {S}

    codA, domA = codomainind(A), domainind(A)
    codB, domB = codomainind(B), domainind(B)
    oindA, cindA, oindB, cindB =
        reorder_indices(codA, domA, codB, domB, oindA, cindA, oindB, cindB, p1, p2)

    @assert space(B, cindB[1]) == space(A, cindA[1])' &&
                space(B, cindB[2]) == space(A, cindA[2])' &&
                space(B, cindB[3]) == space(A, cindA[3])' &&
                space(B, cindB[4]) == space(A, cindA[4])'

    if BraidingStyle(sectortype(B)) isa Bosonic
        return trace!(α, A, β, C, oindA, (), (cindA[1], cindA[2]), (cindA[3], cindA[4]))
    end

    if iszero(β)
        fill!(C, β)
    elseif β != 1
        rmul!(C, β)
    end
    I = sectortype(B)
    u = one(I)
    f₀ = FusionTree{I}((), u, (), (), ())
    braidingtensor_levels = B.adjoint ? (1,2,2,1) : (2,1,1,2)
    inv_braid = braidingtensor_levels[cindB[2]] > braidingtensor_levels[cindB[3]]
    for (f1, f2) in fusiontrees(A)
        local newtrees
        for ((f1′, f2′), coeff′) in transpose(f1, f2, oindA, cindA)
            f2′.coupled == u || continue
            a = f2′.uncoupled[1]
            b = f2′.uncoupled[2]
            f2′.uncoupled[3] == dual(a) || continue
            f2′.uncoupled[4] == dual(b) || continue
            # should be automatic by matching spaces:
            # f2′.isdual[1] != f2′.isdual[3] || continue
            # f2′.isdual[3] != f2′.isdual[4] || continue
            for (f2′′, coeff′′) in artin_braid(f2′, 2, inv = inv_braid)
                f2′′.innerlines[1] == u || continue
                coeff = coeff′ * conj(coeff′′ * sqrtdim(a) * sqrtdim(b))
                if f2′′.isdual[1]
                    coeff *= conj(frobeniusschur(a))
                end
                if f2′′.isdual[3]
                    coeff *= conj(frobeniusschur(b))
                end
                f12 = (f1′, f₀)
                if @isdefined newtrees
                    newtrees[f12] = get(newtrees, f12, zero(coeff)) + coeff
                else
                    newtrees = Dict(f12 => coeff)
                end
            end
        end
        @isdefined(newtrees) || continue
        for ((f1′, f2′), coeff) in newtrees
            TO._trace!(coeff*α, A[f1, f2], true, C[f1′,f2′], oindA,
                                            (cindA[1], cindA[2]), (cindA[3], cindA[4]))
        end
    end
    return C
end

function planar_contract!(α, A::BraidingTensor, B::AbstractTensorMap{S},
                            β, C::AbstractTensorMap{S},
                            oindA::IndexTuple{1}, cindA::IndexTuple{3},
                            oindB::IndexTuple, cindB::IndexTuple{3},
                            p1::IndexTuple, p2::IndexTuple,
                            syms::Union{Nothing, NTuple{3, Symbol}}) where {S}

    codA, domA = codomainind(A), domainind(A)
    codB, domB = codomainind(B), domainind(B)
    oindA, cindA, oindB, cindB =
        reorder_indices(codA, domA, codB, domB, oindA, cindA, oindB, cindB, p1, p2)

    @assert space(B, cindB[1]) == space(A, cindA[1])' &&
                space(B, cindB[2]) == space(A, cindA[2])' &&
                space(B, cindB[3]) == space(A, cindA[3])'

    if BraidingStyle(sectortype(B)) isa Bosonic
        return trace!(α, B, β, C, (cindB[2],), oindB, (cindB[1],), (cindB[3],))
    end

    if iszero(β)
        fill!(C, β)
    elseif β != 1
        rmul!(C, β)
    end
    I = sectortype(B)
    u = one(I)
    braidingtensor_levels = A.adjoint ? (1,2,2,1) : (2,1,1,2)
    inv_braid = braidingtensor_levels[cindA[2]] > braidingtensor_levels[cindA[3]]
    for (f1, f2) in fusiontrees(B)
        local newtrees
        for ((f1′,f2′), coeff′) in transpose(f1, f2, cindB, oindB)
            a = f1′.uncoupled[1]
            b = f1′.uncoupled[2]
            b == f1′.coupled || continue
            a == dual(f1′.uncoupled[3]) || continue
            # should be automatic by matching spaces:
            # f1′.isdual[1] != f1.isdual[3] || continue
            for (f1′′, coeff′′) in artin_braid(f1′, 2, inv = inv_braid)
                f1′′.innerlines[1] == u || continue
                coeff = coeff′ * coeff′′ * sqrtdim(a)
                if f1′′.isdual[1]
                    coeff *= frobeniusschur(a)
                end
                f1′′′ = FusionTree{I}((b,), b, (f1′′.isdual[3],), (), ())
                f12 = (f1′′′, f2′)
                if @isdefined newtrees
                    newtrees[f12] = get(newtrees, f12, zero(coeff)) + coeff
                else
                    newtrees = Dict(f12 => coeff)
                end
            end
        end
        @isdefined(newtrees) || continue
        for ((f1′,f2′), coeff) in newtrees
            TO._trace!(coeff*α, B[f1, f2], true, C[f1′,f2′],
                                        (cindB[2], oindB...) , (cindB[1],), (cindB[3],))
        end
    end
    return C
end

function planar_contract!(α, A::AbstractTensorMap{S}, B::BraidingTensor,
                            β, C::AbstractTensorMap{S},
                            oindA::IndexTuple, cindA::IndexTuple{3},
                            oindB::IndexTuple{1}, cindB::IndexTuple{3},
                            p1::IndexTuple, p2::IndexTuple,
                            syms::Union{Nothing, NTuple{3, Symbol}}) where {S}

    codA, domA = codomainind(A), domainind(A)
    codB, domB = codomainind(B), domainind(B)
    oindA, cindA, oindB, cindB =
        reorder_indices(codA, domA, codB, domB, oindA, cindA, oindB, cindB, p1, p2)

    @assert space(B, cindB[1]) == space(A, cindA[1])' &&
                space(B, cindB[2]) == space(A, cindA[2])' &&
                space(B, cindB[3]) == space(A, cindA[3])'

    if BraidingStyle(sectortype(A)) isa Bosonic
        return trace!(α, A, β, C, oindA, (cindA[2],), (cindA[1],), (cindA[3],))
    end

    if iszero(β)
        fill!(C, β)
    elseif β != 1
        rmul!(C, β)
    end
    I = sectortype(B)
    u = one(I)
    braidingtensor_levels = B.adjoint ? (1,2,2,1) : (2,1,1,2)
    inv_braid = braidingtensor_levels[cindB[2]] > braidingtensor_levels[cindB[3]]
    for (f1, f2) in fusiontrees(A)
        local newtrees
        for ((f1′,f2′), coeff′) in transpose(f1, f2, oindA, cindA)
            a = f2′.uncoupled[1]
            b = f2′.uncoupled[2]
            b == f2′.coupled || continue
            a == dual(f2′.uncoupled[3]) || continue
            # should be automatic by matching spaces:
            # f2′.isdual[1] != f2.isdual[3] || continue
            for (f2′′, coeff′′) in artin_braid(f2′, 2, inv = inv_braid)
                f2′′.innerlines[1] == u || continue
                coeff = coeff′ * conj(coeff′′ * sqrtdim(a))
                if f2′′.isdual[1]
                    coeff *= conj(frobeniusschur(a))
                end
                f2′′′ = FusionTree{I}((b,), b, (f2′′.isdual[3],), (), ())
                f12 = (f1′, f2′′′)
                if @isdefined newtrees
                    newtrees[f12] = get(newtrees, f12, zero(coeff)) + coeff
                else
                    newtrees = Dict(f12 => coeff)
                end
            end
        end
        @isdefined(newtrees) || continue
        for ((f1′,f2′), coeff) in newtrees
            TO._trace!(coeff*α, A[f1, f2], true, C[f1′,f2′],
                                            (oindA...,cindA[2]) , (cindA[1],), (cindA[3],))
        end
    end
    return C
end

has_shared_permute(t::BraidingTensor, args...) = false
function cached_permute(sym::Symbol, t::BraidingTensor, p1, p2; copy=false)
    tp = TO.cached_similar_from_indices(sym, eltype(t), p1, p2, t, :N)
    return add!(true, t, false, tp, p1, p2)
end
