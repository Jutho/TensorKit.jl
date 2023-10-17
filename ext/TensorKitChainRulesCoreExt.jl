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

TensorKit.block(t::ZeroTangent, c::Sector) = t

# Constructors
# ------------

@non_differentiable TensorKit.TensorMap(f::Function, storagetype, cod, dom)
@non_differentiable TensorKit.isomorphism(args...)
@non_differentiable TensorKit.isometry(args...)
@non_differentiable TensorKit.unitary(args...)

function ChainRulesCore.rrule(::Type{<:TensorMap}, d::DenseArray, args...)
    function TensorMap_pullback(Δt)
        ∂d = convert(Array, Δt)
        return NoTangent(), ∂d, fill(NoTangent(), length(args))...
    end
    return TensorMap(d, args...), TensorMap_pullback
end

function ChainRulesCore.rrule(::typeof(convert), T::Type{<:Array}, t::AbstractTensorMap)
    A = convert(T, t)
    function convert_pullback(ΔA)
        ∂t = TensorMap(ΔA, codomain(t), domain(t))
        return NoTangent(), NoTangent(), ∂t
    end
    return A, convert_pullback
end

function ChainRulesCore.rrule(::typeof(Base.copy), t::AbstractTensorMap)
    copy_pullback(Δt) = NoTangent(), Δt
    return copy(t), copy_pullback
end

ChainRulesCore.ProjectTo(::T) where {T<:AbstractTensorMap} = ProjectTo{T}()
function (::ProjectTo{T1})(x::T2) where {S,N1,N2,T1<:AbstractTensorMap{S,N1,N2},
                                         T2<:AbstractTensorMap{S,N1,N2}}
    T1 === T2 && return x
    y = similar(x, scalartype(T1))
    for (c, b) in blocks(y)
        p = ProjectTo(b)
        b .= p(block(x, c))
    end
    return y
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

function ChainRulesCore.rrule(::typeof(permute), tsrc::AbstractTensorMap, p::Index2Tuple;
                              copy::Bool=false)
    function permute_pullback(Δtdst)
        invp = TensorKit._canonicalize(TupleTools.invperm(linearize(p)), tsrc)
        return NoTangent(), permute(unthunk(Δtdst), invp; copy=true), NoTangent()
    end
    return permute(tsrc, p; copy=true), permute_pullback
end

# LinearAlgebra
# -------------

function ChainRulesCore.rrule(::typeof(tr), A::AbstractTensorMap)
    tr_pullback(Δtr) = NoTangent(), Δtr * id(domain(A))
    return tr(A), tr_pullback
end

function ChainRulesCore.rrule(::typeof(adjoint), A::AbstractTensorMap)
    adjoint_pullback(Δadjoint) = NoTangent(), adjoint(unthunk(Δadjoint))
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

function ChainRulesCore.rrule(::typeof(TensorKit.tsvd!), t::AbstractTensorMap; kwargs...)
    U, S, V, ϵ = tsvd(t; kwargs...)

    function tsvd!_pullback((ΔU, ΔS, ΔV, Δϵ))
        ∂t = similar(t)
        for (c, b) in blocks(∂t)
            copyto!(b,
                    svd_rev(block(U, c), block(S, c), block(V, c),
                            block(ΔU, c), block(ΔS, c), block(ΔV, c)))
        end

        return NoTangent(), ∂t
    end

    return (U, S, V, ϵ), tsvd!_pullback
end

"""
    svd_rev(U, S, V, ΔU, ΔS, ΔV; tol=eps(real(scalartype(Σ)))^(4 / 5))

Implements the following back propagation formula for the SVD:

```math
ΔA = UΔSV' + U(J + J')SV' + US(K + K')V' + \\frac{1}{2}US^{-1}(L' - L)V'\\
J = F ∘ (U'ΔU), \\qquad K = F ∘ (V'ΔV), \\qquad L = I ∘ (V'ΔV)\\
F_{i ≠ j} = \\frac{1}{s_j^2 - s_i^2}\\
F_{ii} = 0
```

# References

Wan, Zhou-Quan, and Shi-Xin Zhang. 2019. “Automatic Differentiation for Complex Valued SVD.” https://doi.org/10.48550/ARXIV.1909.02659.
"""
function svd_rev(U::AbstractMatrix, S::AbstractMatrix, V::AbstractMatrix, ΔU, ΔS, ΔV;
                 atol::Real=0,
                 rtol::Real=atol > 0 ? 0 : eps(scalartype(S))^(3 / 4))
    # project out gauge invariance dependence?
    # ΔU * U + ΔV * V' = 0

    tol = atol > 0 ? atol : rtol * S[1, 1]
    F = _invert_S²(S, tol)
    S⁻¹ = pinv(S; atol=tol)

    term = ΔS isa ZeroTangent ? ΔS : Diagonal(diag(ΔS))

    J = F .* (U' * ΔU)
    term += (J + J') * S
    VΔV = (V * ΔV')
    K = F .* VΔV
    term += S * (K + K')

    if scalartype(U) <: Complex && !(ΔV isa ZeroTangent) && !(ΔU isa ZeroTangent)
        L = LinearAlgebra.Diagonal(diag(VΔV))
        term += 0.5 * S⁻¹ * (L' - L)
    end

    ΔA = U * term * V

    if size(U, 1) != size(V, 2)
        UUd = U * U'
        VdV = V' * V
        ΔA += (one(UUd) - UUd) * ΔU * S⁻¹ * V + U * S⁻¹ * ΔV * (one(VdV) - VdV)
    end

    return ΔA
end

function _invert_S²(S::AbstractMatrix{T}, tol::Real) where {T<:Real}
    F = similar(S)
    @inbounds for i in axes(F, 1), j in axes(F, 2)
        F[i, j] = if i == j
            zero(T)
        else
            sᵢ, sⱼ = S[i, i], S[j, j]
            1 / (abs(sⱼ - sᵢ) < tol ? tol : sⱼ^2 - sᵢ^2)
        end
    end
    return F
end

function ChainRulesCore.rrule(::typeof(leftorth!), t::AbstractTensorMap; alg=QRpos())
    alg isa TensorKit.QR || alg isa TensorKit.QRpos || error("only QR and QRpos supported")
    Q, R = leftorth(t; alg)
    leftorth!_pullback((ΔQ, ΔR)) = NoTangent(), qr_pullback!(similar(t), t, Q, R, ΔQ, ΔR)
    leftorth!_pullback(::Tuple{ZeroTangent,ZeroTangent}) = ZeroTangent()
    return (Q, R), leftorth!_pullback
end

function ChainRulesCore.rrule(::typeof(rightorth!), t::AbstractTensorMap; alg=LQpos())
    alg isa TensorKit.LQ || alg isa TensorKit.LQpos || error("only LQ and LQpos supported")
    L, Q = rightorth(t; alg)
    rightorth!_pullback((ΔL, ΔQ)) = NoTangent(), lq_pullback!(similar(t), t, L, Q, ΔL, ΔQ)
    rightorth!_pullback(::Tuple{ZeroTangent,ZeroTangent}) = ZeroTangent()
    return (L, Q), rightorth!_pullback
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

function qr_pullback!(ΔA::AbstractTensorMap{S}, t::AbstractTensorMap{S},
                      Q::AbstractTensorMap{S}, R::AbstractTensorMap{S}, ΔQ, ΔR) where {S}
    for (c, b) in blocks(ΔA)
        qr_pullback!(b, block(t, c), block(Q, c), block(R, c), block(ΔQ, c), block(ΔR, c))
    end
    return ΔA
end

function qr_pullback!(ΔA, A, Q::M, R::M, ΔQ, ΔR) where {M<:AbstractMatrix}
    m = qr_rank(R)
    n = size(R, 2)

    if n == m # full rank
        q = view(Q, :, 1:m)
        Δq = view(ΔQ, :, 1:m)
        r = view(R, 1:m, :)
        Δr = view(ΔR, 1:m, :)
        ΔA = qr_pullback_fullrank!(ΔA, q, r, Δq, Δr)
    else
        q = view(Q, :, 1:m)
        Δq = view(ΔQ, :, 1:m) + view(A, :, (m + 1):n) * view(ΔR, :, (m + 1):n)'
        r = view(R, 1:m, 1:m)
        Δr = view(ΔR, 1:m, 1:m)

        qr_pullback_fullrank!(view(ΔA, :, 1:m), q, r, Δq, Δr)
        ΔA[:, (m + 1):n] = q * view(ΔR, :, (m + 1):n)
    end

    return ΔA
end

function qr_pullback_fullrank!(ΔA, Q, R, ΔQ, ΔR)
    b = ΔQ + Q * copyltu!(R * ΔR' - ΔQ' * Q)
    return adjoint!(ΔA, LinearAlgebra.LAPACK.trtrs!('U', 'N', 'N', R, copy(adjoint(b))))
end

function lq_pullback!(ΔA::AbstractTensorMap{S}, t::AbstractTensorMap{S},
                      L::AbstractTensorMap{S}, Q::AbstractTensorMap{S}, ΔL, ΔQ) where {S}
    for (c, b) in blocks(ΔA)
        lq_pullback!(b, block(t, c), block(L, c), block(Q, c), block(ΔL, c), block(ΔQ, c))
    end
    return ΔA
end

function lq_pullback!(ΔA, A, L::M, Q::M, ΔL, ΔQ) where {M<:AbstractMatrix}
    m = qr_rank(L)
    n = size(L, 1)

    if n == m # full rank
        l = view(L, :, 1:m)
        Δl = view(ΔL, :, 1:m)
        q = view(Q, 1:m, :)
        Δq = view(ΔQ, 1:m, :)
        ΔA = lq_pullback_fullrank!(ΔA, l, q, Δl, Δq)
    else
        l = view(L, 1:m, 1:m)
        Δl = view(ΔL, 1:m, 1:m)
        q = view(Q, 1:m, :)
        Δq = view(ΔQ, 1:m, :) + view(ΔL, (m + 1):n, 1:m)' * view(A, (m + 1):n, :)

        lq_pullback_fullrank!(view(ΔA, 1:m, :), l, q, Δl, Δq)
        ΔA[(m + 1):n, :] = view(ΔL, (m + 1):n, :) * q
    end

    return ΔA
end

function lq_pullback_fullrank!(ΔA, L, Q, ΔL, ΔQ)
    mul!(ΔA, copyltu!(L' * ΔL - ΔQ * Q'), Q)
    axpy!(true, ΔQ, ΔA)
    return LinearAlgebra.LAPACK.trtrs!('L', 'C', 'N', L, ΔA)
end

function qr_rank(r::AbstractMatrix)
    Base.require_one_based_indexing(r)
    m, n = size(r)
    r₀ = r[1, 1]
    i = findfirst(x -> abs(x / r₀) < 1e-12, diag(r))
    return isnothing(i) ? min(m, n) : i - 1
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
