module TensorKitChainRulesCoreExt

using TensorKit
using ChainRulesCore
using LinearAlgebra

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
    norm_pullback(Δn) = NoTangent(), @thunk(a * (Δn' + Δn) / (n * 2)), NoTangent()
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
                dst[i, j] = 1 / (abs(sᵢ - sⱼ) < 1e-12) ? 1e-12 : sᵢ^2 - sⱼ^2
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

function ChainRulesCore.rrule(::typeof(leftorth!), t, 

function ChainRulesCore.rrule(::typeof(leftorth), t, leftind=codomainind(t),
                              rightind=domainind(t); alg=QRpos())
    alg isa TensorKit.QR || alg isa TensorKit.QRpos || error("only QR and QRpos supported")
    
    (permuted, permback) = ChainRulesCore.rrule(permute, t, leftind, rightind)
    Q, R = leftorth(permuted; alg=alg)

    if alg isa TensorKit.QR || alg isa TensorKit.QRpos
        pullback = v -> backwards_leftorth_qr(permuted, Q, R, v[1], v[2])
    else
        pullback = v -> @assert false
    end
    
    return (Q, R),
           v -> (NoTangent(), permback(pullback(v))[2], NoTangent(), NoTangent(),
                 NoTangent())
end

function backwards_leftorth_qr(A, q, r, dq, dr)
    out = similar(A)
    dr = dr isa ZeroTangent ? zero(r) : dr
    dq = dq isa ZeroTangent ? zero(q) : dq

    if sectortype(A) == Trivial
        copyto!(out.data, qr_back(A.data, q.data, r.data, dq.data, dr.data))
    else
        for b in keys(blocks(A))
            cA = A[b]
            cq = q[b]
            cr = r[b]
            cdq = dq[b]
            cdr = dr[b]

            copyto!(out[b], qr_back(cA, cq, cr, cdq, cdr))
        end
    end
    #@show norm(A),norm(dq),norm(dr),norm(out)
    return out
end

function ChainRulesCore.rrule(::typeof(rightorth), tensor, leftind=codomainind(tensor),
                              rightind=domainind(tensor); alg=LQpos())
    (permuted, permback) = ChainRulesCore.rrule(permute, tensor, leftind, rightind)
    (l, q) = rightorth(permuted; alg=alg)

    if alg isa TensorKit.LQ || alg isa TensorKit.LQpos
        pullback = v -> backwards_rightorth_lq(permuted, l, q, v[1], v[2])
    else
        pullback = v -> @assert false
    end

    return (l, q),
           v -> (NoTangent(), permback(pullback(v))[2], NoTangent(), NoTangent(),
                 NoTangent())
end

function backwards_rightorth_lq(A, l, q, dl, dq)
    out = similar(A)
    dl = dl isa ZeroTangent ? zero(l) : dl
    dq = dq isa ZeroTangent ? zero(q) : dq

    if sectortype(A) == Trivial
        copyto!(out.data, lq_back(A.data, l.data, q.data, dl.data, dq.data))
    else
        for b in keys(blocks(A))
            cA = A[b]
            cl = l[b]
            cq = q[b]
            cdl = dl[b]
            cdq = dq[b]

            copyto!(out[b], lq_back(cA, cl, cq, cdl, cdq))
        end
    end
    #@show norm(A),norm(l),norm(q),norm(out)
    return out
end

function ChainRulesCore.rrule(::typeof(Base.convert), ::Type{Dict}, t::AbstractTensorMap)
    out = convert(Dict, t)
    function pullback(c)
        if haskey(c, :data) # :data is the only thing for which this dual makes sense
            dual = copy(out)
            dual[:data] = c[:data]
            return (NoTangent(), NoTangent(), convert(TensorMap, dual))
        else
            # instead of zero(t) you can also return ZeroTangent(), which is type unstable
            return (NoTangent(), NoTangent(), zero(t))
        end
    end
    return out, pullback
end
function ChainRulesCore.rrule(::typeof(Base.convert), ::Type{TensorMap},
                              t::Dict{Symbol,Any})
    return convert(TensorMap, t), v -> (NoTangent(), NoTangent(), convert(Dict, v))
end


end