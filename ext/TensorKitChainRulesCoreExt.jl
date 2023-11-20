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

ChainRulesCore.rrule(::typeof(-), a::AbstractTensorMap) = -a, Δc -> (NoTangent(), -Δc)
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

function ChainRulesCore.rrule(::typeof(⊗), A::AbstractTensorMap, B::AbstractTensorMap)
    C = A ⊗ B
    projectA = ProjectTo(A)
    projectB = ProjectTo(B)
    function otimes_pullback(ΔC_)
        ΔC = unthunk(ΔC_)
        pΔC = ((codomainind(A)..., (domainind(A) .+ numout(B))...),
               ((codomainind(B) .+ numout(A))...,
                (domainind(B) .+ (numin(A) + numout(A)))...))
        dA_ = @thunk begin
            ipA = (codomainind(A), domainind(A))
            pB = (allind(B), ())
            dA = zerovector(A,
                            TensorOperations.promote_contract(scalartype(ΔC),
                                                              scalartype(B)))
            dA = tensorcontract!(dA, ipA, ΔC, pΔC, :N, B, pB, :C)
            return projectA(dA)
        end
        dB_ = @thunk begin
            ipB = (codomainind(B), domainind(B))
            pA = ((), allind(A))
            dB = zerovector(B,
                            TensorOperations.promote_contract(scalartype(ΔC),
                                                              scalartype(A)))
            dB = tensorcontract!(dB, ipB, A, pA, :C, ΔC, pΔC, :N)
            return projectB(dB)
        end
        return NoTangent(), dA_, dB_
    end
    return C, otimes_pullback
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
function ChainRulesCore.rrule(::typeof(TensorKit.tsvd!), t::AbstractTensorMap;
                              trunc::TensorKit.TruncationScheme=TensorKit.NoTruncation(),
                              p::Real=2,
                              alg::Union{TensorKit.SVD,TensorKit.SDD}=TensorKit.SDD())
    U, Σ, V, ϵ = tsvd(t; trunc=TensorKit.NoTruncation(), p=p, alg=alg)

    if !(trunc isa TensorKit.NoTruncation) && !isempty(blocksectors(t))
        Σddata = TensorKit.SectorDict(c => diag(b) for (c, b) in blocks(Σ))
        dims = TensorKit.SectorDict(c => length(b) for (c, b) in Σddata)
        Σddata, ϵ = TensorKit._truncate!(Σddata, trunc, p)
        Udata = TensorKit.SectorDict(c => b for (c, b) in blocks(U))
        Vdata = TensorKit.SectorDict(c => b for (c, b) in blocks(V))
        Udata′, Σddata′, Vdata′, dims′ = TensorKit._implement_svdtruncation!(t,
                                                                             Udata,
                                                                             Σddata,
                                                                             Vdata,
                                                                             dims)
        W = spacetype(t)(dims′)
        if W ≅ domain(Σ)
            W = domain(Σ)
        end
        U′, Σ′, V′ = TensorKit._create_svdtensors(t, Udata′, Σddata′, Vdata′, W)
    else
        U′, Σ′, V′ = U, Σ, V
    end

    function tsvd!_pullback((ΔU, ΔΣ, ΔV, Δϵ))
        Δt = similar(t)
        for (c, b) in blocks(Δt)
            Uc, Σc, Vc = block(U, c), block(Σ, c), block(V, c)
            ΔUc, ΔΣc, ΔVc = block(ΔU, c), block(ΔΣ, c), block(ΔV, c)
            Σdc = view(Σc, diagind(Σc))
            ΔΣdc = (ΔΣc isa AbstractZero) ? ΔΣc : view(ΔΣc, diagind(ΔΣc))
            copyto!(b, svd_pullback(Uc, Σdc, Vc, ΔUc, ΔΣdc, ΔVc))
        end
        return NoTangent(), Δt
    end

    return (U′, Σ′, V′, ϵ), tsvd!_pullback
end

# SVD_pullback: pullback implementation for general (possibly truncated) SVD
#
# Arguments are U, S and Vd of full (non-truncated, but still thin) SVD, as well as
# cotangent ΔU, ΔS, ΔVd variables of truncated SVD
# 
# Checks whether the cotangent variables are such that they would couple to gauge-dependent
# degrees of freedom (phases of singular vectors), and prints a warning if this is the case
#
# An implementation that only uses U, S, and Vd from truncated SVD is also possible, but
# requires solving a Sylvester equation, which does not seem to be supported on GPUs.
#
# Other implementation considerations for GPU compatibility:
# no scalar indexing, lots of broadcasting and views
#
safe_inv(a, tol) = abs(a) < tol ? zero(a) : inv(a)
function svd_pullback(U::AbstractMatrix, S::AbstractVector, Vd::AbstractMatrix, ΔU, ΔS, ΔVd;
                      atol::Real=0,
                      rtol::Real=atol > 0 ? 0 : eps(scalartype(S))^(3 / 4))

    # Basic size checks and determination
    m, n = size(U, 1), size(Vd, 2)
    size(U, 2) == size(Vd, 1) == length(S) == min(m, n) || throw(DimensionMismatch())
    ΔU isa AbstractZero && ΔVd isa AbstractZero && ΔS isa AbstractZero &&
        return ZeroTangent()
    p = -1
    if !(ΔU isa AbstractZero)
        m == size(ΔU, 1) || throw(DimensionMismatch())
        p = size(ΔU, 2)
    end
    if !(ΔVd isa AbstractZero)
        n == size(ΔVd, 2) || throw(DimensionMismatch())
        if p == -1
            p = size(ΔVd, 1)
        else
            p == size(ΔVd, 1) || throw(DimensionMismatch())
        end
    end
    if !(ΔS isa AbstractZero)
        if ΔS isa AbstractMatrix
            ΔSr = real(diag(ΔS))
        else # ΔS isa AbstractVector
            ΔSr = real(ΔS)
        end
        if p == -1
            p = length(ΔSr)
        else
            p == length(ΔSr) || throw(DimensionMismatch())
        end
    end
    Up = view(U, :, 1:p)
    Vp = view(Vd, 1:p, :)'
    Sp = view(S, 1:p)

    # tolerance and rank
    tol = atol > 0 ? atol : rtol * S[1, 1]
    r = findlast(>=(tol), S)

    # compute antihermitian part of projection of ΔU and ΔV onto U and V
    # also already subtract this projection from ΔU and ΔV
    if !(ΔU isa AbstractZero)
        UΔU = Up' * ΔU
        aUΔU = rmul!(UΔU - UΔU', 1 / 2)
        if m > p
            ΔU -= Up * UΔU
        end
    else
        aUΔU = fill!(similar(U, (p, p)), 0)
    end
    if !(ΔVd isa AbstractZero)
        VΔV = Vp' * ΔVd'
        aVΔV = rmul!(VΔV - VΔV', 1 / 2)
        if n > p
            ΔVd -= VΔV' * Vp'
        end
    else
        aVΔV = fill!(similar(Vd, (p, p)), 0)
    end

    # check whether cotangents arise from gauge-invariance objective function
    mask = abs.(Sp' .- Sp) .< tol
    gaugepart = view(aUΔU, mask) + view(aVΔV, mask)
    norm(gaugepart, Inf) < tol || @warn "cotangents sensitive to gauge choice"
    if p > r
        rprange = (r + 1):p
        norm(view(aUΔU, rprange, rprange), Inf) < tol ||
            @warn "cotangents sensitive to gauge choice"
        norm(view(aVΔV, rprange, rprange), Inf) < tol ||
            @warn "cotangents sensitive to gauge choice"
    end

    UdΔAV = (aUΔU .+ aVΔV) .* safe_inv.(Sp' .- Sp, tol) .+
            (aUΔU .- aVΔV) .* safe_inv.(Sp' .+ Sp, tol)
    if !(ΔS isa ZeroTangent)
        UdΔAV[diagind(UdΔAV)] .+= ΔSr
    end
    ΔA = Up * UdΔAV * Vp'

    if r > p # contribution from truncation
        Ur = view(U, :, (p + 1):r)
        Vr = view(Vd, (p + 1):r, :)'
        Sr = view(S, (p + 1):r)

        if !(ΔU isa AbstractZero)
            UrΔU = Ur' * ΔU
            if m > r
                ΔU -= Ur * UrΔU # subtract this part from ΔU
            end
        else
            UrΔU = fill!(similar(U, (r - p, p)), 0)
        end
        if !(ΔVd isa AbstractZero)
            VrΔV = Vr' * ΔVd'
            if n > r
                ΔVd -= VrΔV' * Vr' # subtract this part from ΔV
            end
        else
            VrΔV = fill!(similar(Vd, (r - p, p)), 0)
        end

        X = (1 // 2) .* ((UrΔU .+ VrΔV) .* safe_inv.(Sp' .- Sr, tol) .+
                         (UrΔU .- VrΔV) .* safe_inv.(Sp' .+ Sr, tol))
        Y = (1 // 2) .* ((UrΔU .+ VrΔV) .* safe_inv.(Sp' .- Sr, tol) .-
                         (UrΔU .- VrΔV) .* safe_inv.(Sp' .+ Sr, tol))

        # ΔA += Ur * X * Vp' + Up * Y' * Vr'
        mul!(ΔA, Ur, X * Vp', 1, 1)
        mul!(ΔA, Up * Y', Vr', 1, 1)
    end

    if m > max(r, p) && !(ΔU isa AbstractZero) # remaining ΔU is already orthogonal to U[:,1:max(p,r)]
        # ΔA += (ΔU .* safe_inv.(Sp', tol)) * Vp'
        mul!(ΔA, ΔU .* safe_inv.(Sp', tol), Vp', 1, 1)
    end
    if n > max(r, p) && !(ΔVd isa AbstractZero) # remaining ΔV is already orthogonal to V[:,1:max(p,r)]
        # ΔA += U * (safe_inv.(Sp, tol) .* ΔVd)
        mul!(ΔA, Up, safe_inv.(Sp, tol) .* ΔVd, 1, 1)
    end
    return ΔA
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
