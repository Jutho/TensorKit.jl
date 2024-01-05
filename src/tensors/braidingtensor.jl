# BraidingTensor:
# special (2,2) tensor that implements a standard braiding operation
#====================================================================#
"""
    struct BraidingTensor{S<:IndexSpace} <: AbstractTensorMap{S, 2, 2}
    BraidingTensor(V1::S, V2::S, adjoint::Bool=false) where {S<:IndexSpace}

Specific subtype of [`AbstractTensorMap`](@ref) for representing the braiding tensor that
braids the first input over the second input; its inverse can be obtained as the adjoint.

It holds that `domain(BraidingTensor(V1, V2)) == V1 ⊗ V2` and
`codomain(BraidingTensor(V1, V2)) == V2 ⊗ V1`.
"""
struct BraidingTensor{S<:IndexSpace,A} <: AbstractTensorMap{S,2,2}
    V1::S
    V2::S
    adjoint::Bool
    function BraidingTensor{S,A}(V1::S, V2::S,
                                 adjoint::Bool=false) where {S<:IndexSpace,A<:DenseMatrix}
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
function BraidingTensor(V1::S, V2::S, adjoint::Bool=false) where {S<:IndexSpace}
    if BraidingStyle(sectortype(S)) isa SymmetricBraiding
        return BraidingTensor{S,Matrix{Float64}}(V1, V2, adjoint)
    else
        return BraidingTensor{S,Matrix{ComplexF64}}(V1, V2, adjoint)
    end
end
function BraidingTensor(V::HomSpace{S}, adjoint::Bool=false) where {S<:IndexSpace}
    domain(V) == reverse(codomain(V)) ||
        throw(SpaceMismatch("Cannot define a braiding on $V"))
    return BraidingTensor(V[1], V[2], adjoint)
end
function Base.adjoint(b::BraidingTensor{S,A}) where {S<:IndexSpace,A<:DenseMatrix}
    return BraidingTensor{S,A}(b.V1, b.V2, !b.adjoint)
end

domain(b::BraidingTensor) = b.adjoint ? b.V2 ⊗ b.V1 : b.V1 ⊗ b.V2
codomain(b::BraidingTensor) = b.adjoint ? b.V1 ⊗ b.V2 : b.V2 ⊗ b.V1

storagetype(::Type{BraidingTensor{S,A}}) where {S<:IndexSpace,A<:DenseMatrix} = A

blocksectors(b::BraidingTensor) = blocksectors(b.V1 ⊗ b.V2)
hasblock(b::BraidingTensor, s::Sector) = s ∈ blocksectors(b)

function fusiontrees(b::BraidingTensor)
    codom = codomain(b)
    dom = domain(b)
    I = sectortype(b)
    F = fusiontreetype(I, 2)
    rowr = SectorDict{I,FusionTreeDict{F,UnitRange{Int}}}()
    colr = SectorDict{I,FusionTreeDict{F,UnitRange{Int}}}()
    for c in blocksectors(codom)
        rowrc = FusionTreeDict{F,UnitRange{Int}}()
        colrc = FusionTreeDict{F,UnitRange{Int}}()
        offset1 = 0
        for s1 in sectors(codom)
            for f₁ in fusiontrees(s1, c, map(isdual, codom.spaces))
                r = (offset1 + 1):(offset1 + dim(codom, s1))
                push!(rowrc, f₁ => r)
                offset1 = last(r)
            end
        end
        dim1 = offset1
        offset2 = 0
        for s2 in sectors(dom)
            for f₂ in fusiontrees(s2, c, map(isdual, dom.spaces))
                r = (offset2 + 1):(offset2 + dim(dom, s2))
                push!(colrc, f₂ => r)
                offset2 = last(r)
            end
        end
        dim2 = offset2
        push!(rowr, c => rowrc)
        push!(colr, c => colrc)
    end
    return TensorKeyIterator(rowr, colr)
end

function Base.getindex(b::BraidingTensor{S}) where {S}
    sectortype(S) == Trivial || throw(SectorMismatch())
    (V1, V2) = domain(b)
    d = (dim(V2), dim(V1), dim(V1), dim(V2))
    return sreshape(StridedView(block(b, Trivial())), d)
end

@inline function Base.getindex(b::BraidingTensor, f₁::FusionTree{I,2},
                               f₂::FusionTree{I,2}) where {I<:Sector}
    I == sectortype(b) || throw(SectorMismatch())
    c = f₁.coupled
    V1, V2 = domain(b)
    @boundscheck begin
        c == f₂.coupled || throw(SectorMismatch())
        ((f₁.uncoupled[1] ∈ sectors(V2)) && (f₂.uncoupled[1] ∈ sectors(V1))) ||
            throw(SectorMismatch())
        ((f₁.uncoupled[2] ∈ sectors(V1)) && (f₂.uncoupled[2] ∈ sectors(V2))) ||
            throw(SectorMismatch())
    end
    @inbounds begin
        d = (dims(V2 ⊗ V1, f₁.uncoupled)..., dims(V1 ⊗ V2, f₂.uncoupled)...)
        n1 = d[1] * d[2]
        n2 = d[3] * d[4]
        data = fill!(storagetype(b)(undef, (n1, n2)), zero(scalartype(b)))
        a1, a2 = f₂.uncoupled
        if f₁.uncoupled == (a2, a1)
            braiddict = artin_braid(f₂, 1; inv=b.adjoint)
            r = get(braiddict, f₁, zero(valtype(braiddict)))
            si = 1 + d[1] * d[2] * d[3]
            sj = d[1] + d[1] * d[2]
            @inbounds for i in 1:d[1], j in 1:d[2]
                data[(i - 1) * si + (j - 1) * sj + 1] = r
            end
        end
        return sreshape(StridedView(data), d)
    end
end

Base.similar(::BraidingTensor, T::Type, P::TensorMapSpace) = TensorMap(undef, T, P)

Base.copy(b::BraidingTensor) = copy!(similar(b), b)
function Base.copy!(t::TensorMap, b::BraidingTensor)
    space(t) == space(b) || throw(SectorMismatch())
    fill!(t, zero(scalartype(t)))
    for (f₁, f₂) in fusiontrees(t)
        data = t[f₁, f₂]
        if sectortype(t) == Trivial
            r = one(scalartype(t))
        else
            a1, a2 = f₂.uncoupled
            c = f₂.coupled
            f₁.uncoupled == (a2, a1) || continue
            braiddict = artin_braid(f₂, 1; inv=b.adjoint)
            r = convert(scalartype(t), get(braiddict, f₁, zero(valtype(braiddict))))
        end
        for i in 1:size(data, 1), j in 1:size(data, 2)
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
        n = d1 * d2
        data = fill!(storagetype(b)(undef, (n, n)), zero(scalartype(b)))
        si = 1 + d2 * d1 * d1
        sj = d2 + d2 * d1
        @inbounds for i in 1:d2, j in 1:d1
            data[(i - 1) * si + (j - 1) * sj + 1] = one(scalartype(b))
        end
        return data
    end
    n = blockdim(domain(b), s)
    data = fill!(storagetype(b)(undef, (n, n)), zero(scalartype(b)))
    iter = fusiontrees(b) # actually contains information about ranges as well
    for (f₂, r2) in iter.colr[s]
        for (f₁, r1) in iter.rowr[s]
            a1, a2 = f₂.uncoupled
            d1 = dim(V1, a1)
            d2 = dim(V2, a2)
            f₁.uncoupled == (a2, a1) || continue
            braiddict = artin_braid(f₂, 1; inv=b.adjoint)
            r = convert(scalartype(b), get(braiddict, f₁, zero(valtype(braiddict))))
            si = 1 + n * d1
            sj = d2 + n
            start = first(r1) + (first(r2) - 1) * n
            @inbounds for i in 1:d2, j in 1:d1
                data[(i - 1) * si + (j - 1) * sj + start] = r
            end
        end
    end
    return data
end

blocks(b::BraidingTensor) = blocks(TensorMap(b))

# Index manipulations
# -------------------
has_shared_permute(t::BraidingTensor, args...) = false
function add_transform!(tdst::AbstractTensorMap{S,N₁,N₂},
                        tsrc::BraidingTensor{S},
                        (p₁, p₂)::Index2Tuple{N₁,N₂},
                        fusiontreetransform,
                        α::Number,
                        β::Number,
                        backend::Backend...) where {S,N₁,N₂}
    return add_transform!(tdst, copy(tsrc), (p₁, p₂), fusiontreetransform, α, β, backend...)
end

# VectorInterface
# ---------------
# TODO

# TensorOperations
# ----------------
# TODO: implement specialized methods
function TO.tensoradd!(C::AbstractTensorMap{S,N₁,N₂}, pC::Index2Tuple{N₁,N₂},
                       A::BraidingTensor{S}, conjA::Symbol, α::Number, β::Number,
                       backend::Backend...) where {S,N₁,N₂}
    return TO.tensoradd!(C, pC, copy(A), conjA, α, β, backend...)
end

# Planar operations
# -----------------
function planaradd!(C::AbstractTensorMap{S,N₁,N₂},
                    A::BraidingTensor{S},
                    p::Index2Tuple{N₁,N₂},
                    α::Number, β::Number,
                    backend::Backend...) where {S,N₁,N₂}
    return planaradd!(C, copy(A), p, α, β, backend...)
end

function planarcontract!(C::AbstractTensorMap{S,N₁,N₂},
                         A::BraidingTensor{S},
                         (oindA, cindA)::Index2Tuple{2,2},
                         B::AbstractTensorMap{S},
                         (cindB, oindB)::Index2Tuple{2,N₃},
                         (p1, p2)::Index2Tuple{N₁,N₂},
                         α::Number, β::Number,
                         backend::Backend...) where {S,N₁,N₂,N₃}
    codA, domA = codomainind(A), domainind(A)
    codB, domB = codomainind(B), domainind(B)
    oindA, cindA, oindB, cindB = reorder_indices(codA, domA, codB, domB, oindA, cindA,
                                                 oindB, cindB, p1, p2)

    if space(B, cindB[1]) != space(A, cindA[1])' ||
       space(B, cindB[2]) != space(A, cindA[2])'
        throw(SpaceMismatch("$(space(C)) ≠ permute($(space(A))[$oindA, $cindA] * $(space(B))[$cindB, $oindB], ($p1, $p2)"))
    end

    if BraidingStyle(sectortype(B)) isa Bosonic
        return add_permute!(C, B, (reverse(cindB), oindB), α, β, backend...)
    end

    τ_levels = A.adjoint ? (1, 2, 2, 1) : (2, 1, 1, 2)
    scale!(C, β)

    inv_braid = τ_levels[cindA[1]] > τ_levels[cindA[2]]
    for (f₁, f₂) in fusiontrees(B)
        local newtrees
        for ((f₁′, f₂′), coeff′) in transpose(f₁, f₂, cindB, oindB)
            for (f₁′′, coeff′′) in artin_braid(f₁′, 1; inv=inv_braid)
                f12 = (f₁′′, f₂′)
                coeff = coeff′ * coeff′′
                if @isdefined newtrees
                    newtrees[f12] = get(newtrees, f12, zero(coeff)) + coeff
                else
                    newtrees = Dict(f12 => coeff)
                end
            end
        end
        for ((f₁′, f₂′), coeff) in newtrees
            TO.tensoradd!(C[f₁′, f₂′], (reverse(cindB), oindB), B[f₁, f₂], :N, α * coeff,
                          One(), backend...)
        end
    end
    return C
end
function planarcontract!(C::AbstractTensorMap{S,N₁,N₂},
                         A::AbstractTensorMap{S},
                         (oindA, cindA)::Index2Tuple{N₃,2},
                         B::BraidingTensor{S},
                         (cindB, oindB)::Index2Tuple{2,2},
                         (p1, p2)::Index2Tuple{N₁,N₂},
                         α::Number, β::Number,
                         backend::Backend...) where {S,N₁,N₂,N₃}
    codA, domA = codomainind(A), domainind(A)
    codB, domB = codomainind(B), domainind(B)
    oindA, cindA, oindB, cindB = reorder_indices(codA, domA, codB, domB, oindA, cindA,
                                                 oindB, cindB, p1, p2)

    if space(B, cindB[1]) != space(A, cindA[1])' ||
       space(B, cindB[2]) != space(A, cindA[2])'
        throw(SpaceMismatch("$(space(C)) ≠ permute($(space(A))[$oindA, $cindA] * $(space(B))[$cindB, $oindB], ($p1, $p2)"))
    end

    if BraidingStyle(sectortype(A)) isa Bosonic
        return add_permute!(C, A, (oindA, reverse(cindA)), α, β, backend...)
    end

    scale!(C, β)
    τ_levels = B.adjoint ? (1, 2, 2, 1) : (2, 1, 1, 2)
    inv_braid = τ_levels[cindB[1]] > τ_levels[cindB[2]]

    for (f₁, f₂) in fusiontrees(A)
        local newtrees
        for ((f₁′, f₂′), coeff′) in transpose(f₁, f₂, oindA, cindA)
            for (f₂′′, coeff′′) in artin_braid(f₂′, 1; inv=inv_braid)
                f12 = (f₁′, f₂′′)
                coeff = coeff′ * conj(coeff′′)
                if @isdefined newtrees
                    newtrees[f12] = get(newtrees, f12, zero(coeff)) + coeff
                else
                    newtrees = Dict(f12 => coeff)
                end
            end
        end
        for ((f₁′, f₂′), coeff) in newtrees
            TO.tensoradd!(C[f₁′, f₂′], (oindA, reverse(cindA)), A[f₁, f₂], :N, α * coeff,
                          One(), backend...)
        end
    end
    return C
end
function planarcontract!(C::AbstractTensorMap{S,N₁,N₂},
                         A::BraidingTensor{S},
                         (oindA, cindA)::Index2Tuple{2,2},
                         B::BraidingTensor{S},
                         (cindB, oindB)::Index2Tuple{2,2},
                         (p1, p2)::Index2Tuple{N₁,N₂},
                         α::Number, β::Number,
                         backend::Backend...) where {S,N₁,N₂}
    return planarcontract!(C, copy(A), (oindA, cindA), B, (cindB, oindB), (p1, p2), α, β,
                           backend...)
end

# Fallback cases for planarcontract!
# TODO: implement specialised cases for contracting 0, 1, 3 and 4 indices
function planarcontract!(C::AbstractTensorMap{S}, A::BraidingTensor{S}, pA::Index2Tuple,
                         B::BraidingTensor{S}, pB::Index2Tuple, α::Number, β::Number,
                         backend::Backend...) where {S}
    return planarcontract!(C, copy(A), pA, copy(B), pB, α, β, backend...)
end
function planarcontract!(C::AbstractTensorMap{S}, A::BraidingTensor{S}, pA::Index2Tuple,
                         B::AbstractTensorMap{S}, pB::Index2Tuple, α::Number, β::Number,
                         backend::Backend...) where {S}
    return planarcontract!(C, copy(A), pA, B, pB, α, β, backend...)
end
function planarcontract!(C::AbstractTensorMap{S}, A::AbstractTensorMap{S}, pA::Index2Tuple,
                         B::BraidingTensor{S}, pB::Index2Tuple, α::Number, β::Number,
                         backend::Backend...) where {S}
    return planarcontract!(C, A, pA, copy(B), pB, α, β, backend...)
end

function planartrace!(C::AbstractTensorMap{S,N₁,N₂},
                      A::BraidingTensor{S},
                      p::Index2Tuple{N₁,N₂}, q::Index2Tuple{N₃,N₃},
                      α::Number, β::Number,
                      backend::Backend...) where {S,N₁,N₂,N₃}
    return planartrace!(C, copy(A), p, q, α, β, backend...)
end

# function planarcontract!(C::AbstractTensorMap{S,N₁,N₂},
#                          A::BraidingTensor{S},
#                          (oindA, cindA)::Index2Tuple{0,4},
#                          B::AbstractTensorMap{S},
#                          (cindB, oindB)::Index2Tuple{4,<:Any},
#                          (p1, p2)::Index2Tuple{N₁,N₂},
#                          α::Number, β::Number,
#                          backend::Backend...) where {S,N₁,N₂}
#     codA, domA = codomainind(A), domainind(A)
#     codB, domB = codomainind(B), domainind(B)
#     oindA, cindA, oindB, cindB = reorder_indices(codA, domA, codB, domB, oindA, cindA,
#                                                  oindB, cindB, p1, p2)

#     @assert space(B, cindB[1]) == space(A, cindA[1])' &&
#             space(B, cindB[2]) == space(A, cindA[2])' &&
#             space(B, cindB[3]) == space(A, cindA[3])' &&
#             space(B, cindB[4]) == space(A, cindA[4])'

#     if BraidingStyle(sectortype(B)) isa Bosonic
#         return trace!(α, B, β, C, (), oindB, (cindB[1], cindB[2]), (cindB[3], cindB[4]))
#     end

#     if iszero(β)
#         fill!(C, β)
#     elseif β != 1
#         rmul!(C, β)
#     end
#     I = sectortype(B)
#     u = one(I)
#     f₀ = FusionTree{I}((), u, (), (), ())
#     braidingtensor_levels = A.adjoint ? (1, 2, 2, 1) : (2, 1, 1, 2)
#     inv_braid = braidingtensor_levels[cindA[2]] > braidingtensor_levels[cindA[3]]
#     for (f₁, f₂) in fusiontrees(B)
#         local newtrees
#         for ((f₁′, f₂′), coeff′) in transpose(f₁, f₂, cindB, oindB)
#             f₁′.coupled == u || continue
#             a = f₁′.uncoupled[1]
#             b = f₁′.uncoupled[2]
#             f₁′.uncoupled[3] == dual(a) || continue
#             f₁′.uncoupled[4] == dual(b) || continue
#             # should be automatic by matching spaces:
#             # f₁′.isdual[1] != f₁′.isdual[3] || continue
#             # f₁′.isdual[2] != f₁′.isdual[4] || continue
#             for (f₁′′, coeff′′) in artin_braid(f₁′, 2; inv=inv_braid)
#                 f₁′′.innerlines[1] == u || continue
#                 coeff = coeff′ * coeff′′ * sqrtdim(a) * sqrtdim(b)
#                 if f₁′′.isdual[1]
#                     coeff *= frobeniusschur(a)
#                 end
#                 if f₁′′.isdual[3]
#                     coeff *= frobeniusschur(b)
#                 end
#                 f12 = (f₀, f₂′)
#                 if @isdefined newtrees
#                     newtrees[f12] = get(newtrees, f12, zero(coeff)) + coeff
#                 else
#                     newtrees = Dict(f12 => coeff)
#                 end
#             end
#         end
#         @isdefined(newtrees) || continue
#         for ((f₁′, f₂′), coeff) in newtrees
#             TO._trace!(coeff * α, B[f₁, f₂], true, C[f₁′, f₂′], oindB,
#                        (cindB[1], cindB[2]), (cindB[3], cindB[4]))
#         end
#     end
#     return C
# end
# function planarcontract!(C::AbstractTensorMap{S,N₁,N₂},
#                          A::AbstractTensorMap{S},
#                          (oindA, cindA)::Index2Tuple{0,4},
#                          B::BraidingTensor{S},
#                          (cindB, oindB)::Index2Tuple{4,<:Any},
#                          (p1, p2)::Index2Tuple{N₁,N₂},
#                          α::Number, β::Number,
#                          backends...) where {S,N₁,N₂}
#     codA, domA = codomainind(A), domainind(A)
#     codB, domB = codomainind(B), domainind(B)
#     oindA, cindA, oindB, cindB = reorder_indices(codA, domA, codB, domB, oindA, cindA,
#                                                  oindB, cindB, p1, p2)

#     @assert space(B, cindB[1]) == space(A, cindA[1])' &&
#             space(B, cindB[2]) == space(A, cindA[2])' &&
#             space(B, cindB[3]) == space(A, cindA[3])' &&
#             space(B, cindB[4]) == space(A, cindA[4])'

#     if BraidingStyle(sectortype(B)) isa Bosonic
#         return trace!(α, A, β, C, oindA, (), (cindA[1], cindA[2]), (cindA[3], cindA[4]))
#     end

#     if iszero(β)
#         fill!(C, β)
#     elseif β != 1
#         rmul!(C, β)
#     end
#     I = sectortype(B)
#     u = one(I)
#     f₀ = FusionTree{I}((), u, (), (), ())
#     braidingtensor_levels = B.adjoint ? (1, 2, 2, 1) : (2, 1, 1, 2)
#     inv_braid = braidingtensor_levels[cindB[2]] > braidingtensor_levels[cindB[3]]
#     for (f₁, f₂) in fusiontrees(A)
#         local newtrees
#         for ((f₁′, f₂′), coeff′) in transpose(f₁, f₂, oindA, cindA)
#             f₂′.coupled == u || continue
#             a = f₂′.uncoupled[1]
#             b = f₂′.uncoupled[2]
#             f₂′.uncoupled[3] == dual(a) || continue
#             f₂′.uncoupled[4] == dual(b) || continue
#             # should be automatic by matching spaces:
#             # f₂′.isdual[1] != f₂′.isdual[3] || continue
#             # f₂′.isdual[3] != f₂′.isdual[4] || continue
#             for (f₂′′, coeff′′) in artin_braid(f₂′, 2; inv=inv_braid)
#                 f₂′′.innerlines[1] == u || continue
#                 coeff = coeff′ * conj(coeff′′ * sqrtdim(a) * sqrtdim(b))
#                 if f₂′′.isdual[1]
#                     coeff *= conj(frobeniusschur(a))
#                 end
#                 if f₂′′.isdual[3]
#                     coeff *= conj(frobeniusschur(b))
#                 end
#                 f12 = (f₁′, f₀)
#                 if @isdefined newtrees
#                     newtrees[f12] = get(newtrees, f12, zero(coeff)) + coeff
#                 else
#                     newtrees = Dict(f12 => coeff)
#                 end
#             end
#         end
#         @isdefined(newtrees) || continue
#         for ((f₁′, f₂′), coeff) in newtrees
#             TO._trace!(coeff * α, A[f₁, f₂], true, C[f₁′, f₂′], oindA,
#                        (cindA[1], cindA[2]), (cindA[3], cindA[4]))
#         end
#     end
#     return C
# end
# function planarcontract!(C::AbstractTensorMap{S,N₁,N₂},
#                          A::BraidingTensor{S},
#                          (oindA, cindA)::Index2Tuple{1,3},
#                          B::AbstractTensorMap{S},
#                          (cindB, oindB)::Index2Tuple{1,<:Any},
#                          (p1, p2)::Index2Tuple{N₁,N₂},
#                          α::Number, β::Number,
#                          backend::Backend...) where {S,N₁,N₂}
#     codA, domA = codomainind(A), domainind(A)
#     codB, domB = codomainind(B), domainind(B)
#     oindA, cindA, oindB, cindB = reorder_indices(codA, domA, codB, domB, oindA, cindA,
#                                                  oindB, cindB, p1, p2)

#     @assert space(B, cindB[1]) == space(A, cindA[1])' &&
#             space(B, cindB[2]) == space(A, cindA[2])' &&
#             space(B, cindB[3]) == space(A, cindA[3])'

#     if BraidingStyle(sectortype(B)) isa Bosonic
#         return trace!(α, B, β, C, (cindB[2],), oindB, (cindB[1],), (cindB[3],))
#     end

#     if iszero(β)
#         fill!(C, β)
#     elseif β != 1
#         rmul!(C, β)
#     end
#     I = sectortype(B)
#     u = one(I)
#     braidingtensor_levels = A.adjoint ? (1, 2, 2, 1) : (2, 1, 1, 2)
#     inv_braid = braidingtensor_levels[cindA[2]] > braidingtensor_levels[cindA[3]]
#     for (f₁, f₂) in fusiontrees(B)
#         local newtrees
#         for ((f₁′, f₂′), coeff′) in transpose(f₁, f₂, cindB, oindB)
#             a = f₁′.uncoupled[1]
#             b = f₁′.uncoupled[2]
#             b == f₁′.coupled || continue
#             a == dual(f₁′.uncoupled[3]) || continue
#             # should be automatic by matching spaces:
#             # f₁′.isdual[1] != f₁.isdual[3] || continue
#             for (f₁′′, coeff′′) in artin_braid(f₁′, 2; inv=inv_braid)
#                 f₁′′.innerlines[1] == u || continue
#                 coeff = coeff′ * coeff′′ * sqrtdim(a)
#                 if f₁′′.isdual[1]
#                     coeff *= frobeniusschur(a)
#                 end
#                 f₁′′′ = FusionTree{I}((b,), b, (f₁′′.isdual[3],), (), ())
#                 f12 = (f₁′′′, f₂′)
#                 if @isdefined newtrees
#                     newtrees[f12] = get(newtrees, f12, zero(coeff)) + coeff
#                 else
#                     newtrees = Dict(f12 => coeff)
#                 end
#             end
#         end
#         @isdefined(newtrees) || continue
#         for ((f₁′, f₂′), coeff) in newtrees
#             TO._trace!(coeff * α, B[f₁, f₂], true, C[f₁′, f₂′],
#                        (cindB[2], oindB...), (cindB[1],), (cindB[3],))
#         end
#     end
#     return C
# end
# function planarcontract!(C::AbstractTensorMap{S,N₁,N₂},
#                          A::AbstractTensorMap{S},
#                          (oindA, cindA)::Index2Tuple{<:Any,3},
#                          B::BraidingTensor{S},
#                          (cindB, oindB)::Index2Tuple{3,1},
#                          (p1, p2)::Index2Tuple{N₁,N₂},
#                          α::Number, β::Number,
#                          backend::Backend...) where {S,N₁,N₂}
#     codA, domA = codomainind(A), domainind(A)
#     codB, domB = codomainind(B), domainind(B)
#     oindA, cindA, oindB, cindB = reorder_indices(codA, domA, codB, domB, oindA, cindA,
#                                                  oindB, cindB, p1, p2)

#     @assert space(B, cindB[1]) == space(A, cindA[1])' &&
#             space(B, cindB[2]) == space(A, cindA[2])' &&
#             space(B, cindB[3]) == space(A, cindA[3])'

#     if BraidingStyle(sectortype(A)) isa Bosonic
#         return trace!(α, A, β, C, oindA, (cindA[2],), (cindA[1],), (cindA[3],))
#     end

#     if iszero(β)
#         fill!(C, β)
#     elseif β != 1
#         rmul!(C, β)
#     end
#     I = sectortype(B)
#     u = one(I)
#     braidingtensor_levels = B.adjoint ? (1, 2, 2, 1) : (2, 1, 1, 2)
#     inv_braid = braidingtensor_levels[cindB[2]] > braidingtensor_levels[cindB[3]]
#     for (f₁, f₂) in fusiontrees(A)
#         local newtrees
#         for ((f₁′, f₂′), coeff′) in transpose(f₁, f₂, oindA, cindA)
#             a = f₂′.uncoupled[1]
#             b = f₂′.uncoupled[2]
#             b == f₂′.coupled || continue
#             a == dual(f₂′.uncoupled[3]) || continue
#             # should be automatic by matching spaces:
#             # f₂′.isdual[1] != f₂.isdual[3] || continue
#             for (f₂′′, coeff′′) in artin_braid(f₂′, 2; inv=inv_braid)
#                 f₂′′.innerlines[1] == u || continue
#                 coeff = coeff′ * conj(coeff′′ * sqrtdim(a))
#                 if f₂′′.isdual[1]
#                     coeff *= conj(frobeniusschur(a))
#                 end
#                 f₂′′′ = FusionTree{I}((b,), b, (f₂′′.isdual[3],), (), ())
#                 f12 = (f₁′, f₂′′′)
#                 if @isdefined newtrees
#                     newtrees[f12] = get(newtrees, f12, zero(coeff)) + coeff
#                 else
#                     newtrees = Dict(f12 => coeff)
#                 end
#             end
#         end
#         @isdefined(newtrees) || continue
#         for ((f₁′, f₂′), coeff) in newtrees
#             TO._trace!(coeff * α, A[f₁, f₂], true, C[f₁′, f₂′],
#                        (oindA..., cindA[2]), (cindA[1],), (cindA[3],))
#         end
#     end
#     return C
# end
