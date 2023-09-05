module TensorKitChainRulesCoreExt

using TensorOperations
using TensorKit
using ChainRulesCore
using LinearAlgebra
using TupleTools

# Utility
# -------

_conj(conjA::Symbol) = conjA == :C ? :N : :C
trivtuple(N) = ntuple(identity, N)

function _repartition(p::IndexTuple, N₁::Int)
    length(p) >= N₁ ||
        throw(ArgumentError("cannot repartition $(typeof(p)) to $N₁, $(length(p) - N₁)"))
    return p[1:N₁], p[(N₁ + 1):end]
end
_repartition(p::Index2Tuple, N₁::Int) = _repartition(linearize(p), N₁)
function _repartition(p::Union{IndexTuple,Index2Tuple}, ::Index2Tuple{N₁}) where {N₁}
    return _repartition(p, N₁)
end
function _repartition(p::Union{IndexTuple,Index2Tuple},
                      ::AbstractTensorMap{<:Any,N₁}) where {N₁}
    return _repartition(p, N₁)
end

# Constructors
# ------------

@non_differentiable TensorKit.TensorMap(f::Function, storagetype, cod, dom)
@non_differentiable TensorKit.isomorphism(args...)
@non_differentiable TensorKit.isometry(args...)
@non_differentiable TensorKit.unitary(args...)

function ChainRulesCore.rrule(::Type{<:TensorMap}, d::DenseArray, args...)
    function TensorMap_pullback(Δt)
        ∂d = @thunk(convert(Array, Δt))
        return NoTangent(), ∂d, fill(NoTangent(), length(args))...
    end
    return TensorMap(d, args...), TensorMap_pullback
end

function ChainRulesCore.rrule(::typeof(convert), ::Type{<:Array}, t::AbstractTensorMap)
    function convert_pullback(Δt)
        spacetype(t) <: ComplexSpace ||
            error("currently only implemented or ComplexSpace spacetypes")
        ∂d = TensorMap(Δt, codomain(t), domain(t))
        return NoTangent(), NoTangent(), ∂d
    end
    return convert(Array, t), convert_pullback
end

# Base Linear Algebra
# -------------------

function ChainRulesCore.rrule(::typeof(+), a::AbstractTensorMap, b::AbstractTensorMap)
    plus_pullback(Δc) = NoTangent(), Δc, Δc
    return a + b, plus_pullback
end

function ChainRulesCore.rrule(::typeof(-), a::AbstractTensorMap, b::AbstractTensorMap)
    minus_pullback(Δc) = NoTangent(), Δc, -Δc
    return a - b, minus_pullback
end

function ChainRulesCore.rrule(::typeof(*), a::AbstractTensorMap, b::AbstractTensorMap)
    times_pullback(Δc) = NoTangent(), @thunk(Δc * b'), @thunk(a' * Δc)
    return a * b, times_pullback
end

function ChainRulesCore.rrule(::typeof(*), a::AbstractTensorMap, b::Number)
    times_pullback(Δc) = NoTangent(), @thunk(Δc * b'), @thunk(dot(a, Δc))
    return a * b, times_pullback
end

function ChainRulesCore.rrule(::typeof(*), a::Number, b::AbstractTensorMap)
    times_pullback(Δc) = NoTangent(), @thunk(dot(b, Δc)), @thunk(a' * Δc)
    return a * b, times_pullback
end

function ChainRulesCore.rrule(::typeof(permute), t::AbstractTensorMap, p::Index2Tuple)
    function permute_pullback(c)
        invpt = _repartition(TupleTools.invperm(linearize(p)), t)
        return NoTangent(), permute(c, invpt), NoTangent()
    end
    return permute(t, p), permute_pullback
end

function ChainRulesCore.rrule(::typeof(scalar), t::AbstractTensorMap)
    scalar_pullback(Δc) = NoTangent(), fill!(similar(t), Δc)
    return scalar(t), scalar_pullback
end

# LinearAlgebra
# -------------

function ChainRulesCore.rrule(::typeof(tr), A::AbstractTensorMap)
    tr_pullback(Δtr) = NoTangent(), @thunk(Δtr * id(domain(A)))
    return tr(A), tr_pullback
end

function ChainRulesCore.rrule(::typeof(adjoint), A::AbstractTensorMap)
    adjoint_pullback(Δadjoint) = NoTangent(), adjoint(Δadjoint)
    return adjoint(A), adjoint_pullback
end

function ChainRulesCore.rrule(::typeof(dot), a::AbstractTensorMap, b::AbstractTensorMap)
    dot_pullback(Δd) = NoTangent(), @thunk(b * Δd'), @thunk(a * Δd)
    return dot(a, b), dot_pullback
end

function ChainRulesCore.rrule(::typeof(norm), a::AbstractTensorMap, p)
    p == 2 || error("currently only implemented for p = 2")
    n = norm(a, p)
    norm_pullback(Δn) = NoTangent(), a * (Δn' + Δn) / (n * 2), NoTangent()
    return n, norm_pullback
end

# Factorizations
# --------------

function ChainRulesCore.rrule(::typeof(TensorKit.tsvd), t::AbstractTensorMap; kwargs...)
    T = eltype(t)

    U, S, V = tsvd(t; kwargs...)

    F = similar(S)
    for (k, dst) in blocks(F)
        src = blocks(S)[k]
        @inbounds for i in axes(dst, 1), j in axes(dst, 2)
            if i == j
                dst[i, j] = zero(eltype(S))
            else
                sᵢ, sⱼ = src[i, i], src[j, j]
                dst[i, j] = 1 / (abs(sⱼ - sᵢ) < 1e-12 ? 1e-12 : sⱼ^2 - sᵢ^2)
            end
        end
    end

    function tsvd_pullback(ΔUSV)
        dU, dS, dV = ΔUSV

        ∂t = zero(t)
        #A_s bar term
        if dS != ChainRulesCore.ZeroTangent()
            ∂t += U * _elementwise_mult(dS, one(dS)) * V
        end
        #A_uo bar term
        if dU != ChainRulesCore.ZeroTangent()
            J = _elementwise_mult((U' * dU), F)
            ∂t += U * (J + J') * S * V
        end
        #A_vo bar term
        if dV != ChainRulesCore.ZeroTangent()
            VpdV = V * dV'
            K = _elementwise_mult(VpdV, F)
            ∂t += U * S * (K + K') * V
        end
        #A_d bar term, only relevant if matrix is complex
        if dV != ChainRulesCore.ZeroTangent() && T <: Complex
            L = _elementwise_mult(VpdV, one(F))
            ∂t += 1 / 2 * U * pinv(S) * (L' - L) * V
        end

        if codomain(t) != domain(t)
            pru = U * U'
            prv = V' * V
            ∂t += (one(pru) - pru) * dU * pinv(S) * V
            ∂t += U * pinv(S) * dV * (one(prv) - prv)
        end

        return NoTangent(), ∂t, fill(NoTangent(), length(kwargs))...
    end

    return (U, S, V), tsvd_pullback
end

function _elementwise_mult(a::AbstractTensorMap, b::AbstractTensorMap)
    dst = similar(a)
    for (k, block) in blocks(dst)
        copyto!(block, blocks(a)[k] .* blocks(b)[k])
    end
    return dst
end

function ChainRulesCore.rrule(::typeof(leftorth!), t; alg=QRpos())
    alg isa TensorKit.QR || alg isa TensorKit.QRpos || error("only QR and QRpos supported")
    Q, R = leftorth(t; alg)

    function leftorth_pullback((ΔQ, ΔR))
        ∂t = similar(t)
        ΔR = ΔR isa ZeroTangent ? zero(R) : ΔR
        ΔQ = ΔQ isa ZeroTangent ? zero(Q) : ΔQ

        if sectortype(t) === Trivial
            copyto!(∂t.data, qr_pullback(t.data, Q.data, R.data, ΔQ.data, ΔR.data))
        else
            for (c, b) in blocks(∂t)
                copyto!(b,
                        qr_pullback(block(t, c), block(Q, c), block(R, c),
                                    block(ΔQ, c), block(ΔR, c)))
            end
        end

        return NoTangent(), ∂t
    end

    return (Q, R), leftorth_pullback
end

function ChainRulesCore.rrule(::typeof(rightorth!), tensor; alg=LQpos())
    alg isa TensorKit.LQ || alg isa TensorKit.LQpos || error("only LQ and LQpos supported")
    L, Q = rightorth(tensor; alg)

    function rightorth_pullback((ΔL, ΔQ))
        ∂t = similar(t)
        ΔL = ΔL isa ZeroTangent ? zero(L) : ΔL
        ΔQ = ΔQ isa ZeroTangent ? zero(Q) : ΔQ

        if sectortype(t) === Trivial
            copyto!(∂t.data, lq_pullback(t.data, Q.data, L.data, ΔQ.data, ΔL.data))
        else
            for (c, b) in blocks(∂t)
                copyto!(b,
                        lq_pullback(block(t, c), block(Q, c), block(L, c),
                                    block(ΔQ, c), block(ΔL, c)))
            end
        end

        return NoTangent(), ∂t
    end

    return (L, Q), rightorth_pullback
end

"""
    copyltu!(A::AbstractMatrix)

Copy the conjugated lower triangular part of `A` to the upper triangular part.
"""
function copyltu!(A::AbstractMatrix)
    m, n = size(A)
    for i in axes(A, 1)
        A[i, i] = real(A[i, i])
        @inbounds for j in (i + 1):n
            A[i, j] = conj(A[j, i])
        end
    end
    return A
end

qr_pullback(A, Q, R, ::Nothing, ::Nothing) = nothing
function qr_pullback(A, Q, R, ΔQ, ΔR)
    M = qr_rank(R)
    N = size(R, 2)

    q = view(Q, :, 1:M)
    Δq = isnothing(ΔQ) ? nothing : view(ΔQ, :, 1:M)

    r = view(R, 1:M, :)
    Δr = isnothing(ΔR) ? nothing : view(ΔR, 1:M, :)

    N == M && return qr_pullback_fullrank(q, r, Δq, Δr)

    B = view(A, :, (M + 1):N)
    U = view(r, :, 1:M)

    if !isnothing(ΔR)
        ΔD = view(Δr, :, (M + 1):N)
        ΔA = qr_pullback_fullrank(q, U, !isnothing(Δq) ? Δq + B * ΔD' : B * ΔD',
                                  view(Δr, :, 1:M))
        ΔB = q * ΔD
    else
        ΔA = qr_pullback_fullrank(q, U, Δq, nothing)
        ΔB = zero(B)
    end

    return hcat(ΔA, ΔB)
end

lq_pullback(A, L, Q, ::Nothing, ::Nothing) = nothing
function lq_pullback(A, L, Q, ΔL, ΔQ)
    M = lq_rqnk(L)
    N = size(L, 1)

    l = view(L, :, 1:M)
    Δl = isnothing(ΔL) ? nothing : view(ΔL, :, 1:M)
    q = view(Q, 1:M, :)
    Δq = isnothing(ΔQ) ? nothing : view(ΔQ, 1:M, :)

    N == M && return lq_pullback_fullrank(l, q, Δl, Δq)

    B = view(A, (M + 1):N, :)
    U = view(l, 1:M, :)

    if !isnothing(ΔL)
        ΔD = view(Δl, (M + 1):N, :)
        ΔA = lq_pullback_fullrank(U, q, view(Δl, 1:M, :),
                                  !isnothing(Δq) ? Δq + ΔD' * B : ΔD' * B)
        ΔB = ΔD * q
    else
        ΔA = lq_pullback_fullrank(U, q, nothing, Δq)
        ΔB = zero(B)
    end

    return vcat(ΔA, ΔB)
end

qr_pullback_fullrank(Q, R, ::Nothing, ::Nothing) = nothing
function qr_pullback_fullrank(Q, R, ΔQ, ::Nothing)
    b = ΔQ + q * copyltu!(-ΔQ' * Q)
    return LinearAlgebra.LAPACK.trtrs!('U', 'N', 'N', r, copy(adjoint(b)))
end
function qr_pullback_fullrank(Q, R, ::Nothing, ΔR)
    b = q * copyltu!(R * ΔR)
    return LinearAlgebra.LAPACK.trtrs!('U', 'N', 'N', r, copy(adjoint(b)))
end
function qr_pullback_fullrank(Q, R, ΔQ, ΔR)
    b = ΔQ + q * copyltu(R * ΔR' - ΔQ' * Q)
    return LinearAlgebra.LAPACK.trtrs!('U', 'N', 'N', r, copy(adjoint(b)))
end

lq_pullback_fullrank(L, Q, ::Nothing, ::Nothing) = nothing
function lq_pullback_fullrank(L, Q, ΔL, ::Nothing)
    b = copyltu!(L' * ΔL) * Q
    return LinearAlgebra.LAPACK.trtrs!('L', 'N', 'N', L, b)
end
function lq_pullback_fullrank(L, Q, ::Nothing, ΔQ)
    b = copyltu!(-ΔQ * Q') + ΔQ
    return LinearAlgebra.LAPACK.trtrs!('L', 'N', 'N', L, b)
end
function lq_pullback_fullrank(L, Q, ΔL, ΔQ)
    b = copyltu!(L' * ΔL - ΔQ * Q') + ΔQ
    return LinearAlgebra.LAPACK.trtrs!('L', 'N', 'N', L, b)
end

function qr_rank(r::AbstractMatrix)
    Base.require_one_based_indexing(r)
    r₀ = r[1, 1]
    for i in axes(r, 1)
        abs(r[i, i] / r₀) < 1e-12 && return i - 1
    end
    return size(r, 1)
end

function lq_rank(l::AbstractMatrix)
    Base.require_one_based_indexing(r)
    l₀ = l[1, 1]
    for i in axes(l, 2)
        abs(l[i, i] / l₀) < 1e-12 && return i - 1
    end
    return size(l, 2)
end

function ChainRulesCore.rrule(::typeof(Base.convert), ::Type{Dict}, t::AbstractTensorMap)
    out = convert(Dict, t)
    function convert_pullback(c)
        if haskey(c, :data) # :data is the only thing for which this dual makes sense
            dual = copy(out)
            dual[:data] = c[:data]
            return (NoTangent(), NoTangent(), convert(TensorMap, dual))
        else
            # instead of zero(t) you can also return ZeroTangent(), which is type unstable
            return (NoTangent(), NoTangent(), zero(t))
        end
    end
    return out, convert_pullback
end
function ChainRulesCore.rrule(::typeof(Base.convert), ::Type{TensorMap},
                              t::Dict{Symbol,Any})
    return convert(TensorMap, t), v -> (NoTangent(), NoTangent(), convert(Dict, v))
end

end
