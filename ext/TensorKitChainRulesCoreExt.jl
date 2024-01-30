module TensorKitChainRulesCoreExt

using TensorOperations
using TensorOperations: Backend, promote_contract
using TensorKit
using TensorKit: planaradd!, planarcontract!, planarcontract, _canonicalize
using VectorInterface
using ChainRulesCore
using LinearAlgebra
using TupleTools
using TupleTools: getindices

# Utility
# -------

_conj(conjA::Symbol) = conjA == :C ? :N : :C
trivtuple(N) = ntuple(identity, N)
trivtuple(::Index2Tuple{N₁,N₂}) where {N₁,N₂} = trivtuple(N₁ + N₂)

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
                            promote_contract(scalartype(ΔC), scalartype(B)))
            dA = tensorcontract!(dA, ipA, ΔC, pΔC, :N, B, pB, :C)
            return projectA(dA)
        end
        dB_ = @thunk begin
            ipB = (codomainind(B), domainind(B))
            pA = ((), allind(A))
            dB = zerovector(B,
                            promote_contract(scalartype(ΔC), scalartype(A)))
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
        invp = _canonicalize(TupleTools.invperm(linearize(p)), tsrc)
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

function ChainRulesCore.rrule(::typeof(norm), a::AbstractTensorMap, p::Real=2)
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
            svd_pullback!(b, Uc, Σdc, Vc, ΔUc, ΔΣdc, ΔVc)
        end
        return NoTangent(), Δt
    end
    function tsvd!_pullback(::Tuple{ZeroTangent,ZeroTangent,ZeroTangent})
        return NoTangent(), ZeroTangent()
    end

    return (U′, Σ′, V′, ϵ), tsvd!_pullback
end

function ChainRulesCore.rrule(::typeof(TensorKit.eig!), t::AbstractTensorMap; kwargs...)
    D, V = eig(t; kwargs...)

    function eig!_pullback((ΔD, ΔV))
        Δt = similar(t)
        for (c, b) in blocks(Δt)
            Dc, Vc = block(D, c), block(V, c)
            ΔDc, ΔVc = block(ΔD, c), block(ΔV, c)
            Ddc = view(Dc, diagind(Dc))
            ΔDdc = (ΔDc isa AbstractZero) ? ΔDc : view(ΔDc, diagind(ΔDc))
            eig_pullback!(b, Ddc, Vc, ΔDdc, ΔVc)
        end
        return NoTangent(), Δt
    end
    function eig!_pullback(::Tuple{ZeroTangent,ZeroTangent})
        return NoTangent(), ZeroTangent()
    end

    return (D, V), eig!_pullback
end

function ChainRulesCore.rrule(::typeof(TensorKit.eigh!), t::AbstractTensorMap; kwargs...)
    D, V = eigh(t; kwargs...)

    function eigh!_pullback((ΔD, ΔV))
        Δt = similar(t)
        for (c, b) in blocks(Δt)
            Dc, Vc = block(D, c), block(V, c)
            ΔDc, ΔVc = block(ΔD, c), block(ΔV, c)
            Ddc = view(Dc, diagind(Dc))
            ΔDdc = (ΔDc isa AbstractZero) ? ΔDc : view(ΔDc, diagind(ΔDc))
            eigh_pullback!(b, Ddc, Vc, ΔDdc, ΔVc)
        end
        return NoTangent(), Δt
    end
    function eigh!_pullback(::Tuple{ZeroTangent,ZeroTangent})
        return NoTangent(), ZeroTangent()
    end

    return (D, V), eigh!_pullback
end

function ChainRulesCore.rrule(::typeof(leftorth!), t::AbstractTensorMap; alg=QRpos())
    alg isa TensorKit.QR || alg isa TensorKit.QRpos ||
        error("only `alg=QR()` and `alg=QRpos()` are supported")
    Q, R = leftorth(t; alg)
    function leftorth!_pullback((ΔQ, ΔR))
        Δt = similar(t)
        for (c, b) in blocks(Δt)
            qr_pullback!(b, block(Q, c), block(R, c), block(ΔQ, c), block(ΔR, c))
        end
        return NoTangent(), Δt
    end
    leftorth!_pullback(::Tuple{ZeroTangent,ZeroTangent}) = NoTangent(), ZeroTangent()
    return (Q, R), leftorth!_pullback
end

function ChainRulesCore.rrule(::typeof(rightorth!), t::AbstractTensorMap; alg=LQpos())
    alg isa TensorKit.LQ || alg isa TensorKit.LQpos ||
        error("only `alg=LQ()` and `alg=LQpos()` are supported")
    L, Q = rightorth(t; alg)
    function rightorth!_pullback((ΔL, ΔQ))
        Δt = similar(t)
        for (c, b) in blocks(Δt)
            lq_pullback!(b, block(L, c), block(Q, c), block(ΔL, c), block(ΔQ, c))
        end
        return NoTangent(), Δt
    end
    rightorth!_pullback(::Tuple{ZeroTangent,ZeroTangent}) = NoTangent(), ZeroTangent()
    return (L, Q), rightorth!_pullback
end

# Corresponding matrix factorisations: implemented as mutating methods
# ---------------------------------------------------------------------
# helper routines
safe_inv(a, tol) = abs(a) < tol ? zero(a) : inv(a)

function lowertriangularind(A::AbstractMatrix)
    m, n = size(A)
    I = Vector{Int}(undef, div(m * (m - 1), 2) + m * (n - m))
    offset = 0
    for j in 1:n
        r = (j + 1):m
        I[offset .- j .+ r] = (j - 1) * m .+ r
        offset += length(r)
    end
    return I
end

function uppertriangularind(A::AbstractMatrix)
    m, n = size(A)
    I = Vector{Int}(undef, div(m * (m - 1), 2) + m * (n - m))
    offset = 0
    for i in 1:m
        r = (i + 1):n
        I[offset .- i .+ r] = i .+ m .* (r .- 1)
        offset += length(r)
    end
    return I
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
function svd_pullback!(ΔA::AbstractMatrix, U::AbstractMatrix, S::AbstractVector,
                       Vd::AbstractMatrix, ΔU, ΔS, ΔVd;
                       atol::Real=0,
                       rtol::Real=atol > 0 ? 0 : eps(eltype(S))^(3 / 4))

    # Basic size checks and determination
    m, n = size(U, 1), size(Vd, 2)
    size(U, 2) == size(Vd, 1) == length(S) == min(m, n) || throw(DimensionMismatch())
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
        if p == -1
            p = length(ΔS)
        else
            p == length(ΔS) || throw(DimensionMismatch())
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
        UdΔAV[diagind(UdΔAV)] .+= real.(ΔS)
        # in principle, ΔS is real, but maybe not if coming from an anyonic tensor
    end
    mul!(ΔA, Up, UdΔAV * Vp')

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

function eig_pullback!(ΔA::AbstractMatrix, D::AbstractVector, V::AbstractMatrix, ΔD, ΔV;
                       atol::Real=0,
                       rtol::Real=atol > 0 ? 0 : eps(real(eltype(D)))^(3 / 4))

    # Basic size checks and determination
    n = LinearAlgebra.checksquare(V)
    n == length(D) || throw(DimensionMismatch())

    # tolerance and rank
    tol = atol > 0 ? atol : rtol * maximum(abs, D)

    if !(ΔV isa AbstractZero)
        VdΔV = V' * ΔV

        mask = abs.(transpose(D) .- D) .< tol
        gaugepart = view(VdΔV, mask)
        norm(gaugepart, Inf) < tol || @warn "cotangents sensitive to gauge choice"

        VdΔV .*= conj.(safe_inv.(transpose(D) .- D, tol))

        if !(ΔD isa AbstractZero)
            view(VdΔV, diagind(VdΔV)) .+= ΔD
        end
        PΔV = V' \ VdΔV
        if eltype(ΔA) <: Real
            ΔAc = mul!(VdΔV, PΔV, V') # recycle VdΔV memory
            ΔA .= real.(ΔAc)
        else
            mul!(ΔA, PΔV, V')
        end
    else
        PΔV = V' \ Diagonal(ΔD)
        if eltype(ΔA) <: Real
            ΔAc = PΔV * V'
            ΔA .= real.(ΔAc)
        else
            mul!(ΔA, PΔV, V')
        end
    end
    return ΔA
end

function eigh_pullback!(ΔA::AbstractMatrix, D::AbstractVector, V::AbstractMatrix, ΔD, ΔV;
                        atol::Real=0,
                        rtol::Real=atol > 0 ? 0 : eps(real(eltype(D)))^(3 / 4))

    # Basic size checks and determination
    n = LinearAlgebra.checksquare(V)
    n == length(D) || throw(DimensionMismatch())

    # tolerance and rank
    tol = atol > 0 ? atol : rtol * maximum(abs, D)

    if !(ΔV isa AbstractZero)
        VdΔV = V' * ΔV
        aVdΔV = rmul!(VdΔV - VdΔV', 1 / 2)

        mask = abs.(D' .- D) .< tol
        gaugepart = view(aVdΔV, mask)
        norm(gaugepart, Inf) < tol || @warn "cotangents sensitive to gauge choice"

        aVdΔV .*= safe_inv.(D' .- D, tol)

        if !(ΔD isa AbstractZero)
            view(aVdΔV, diagind(aVdΔV)) .+= real.(ΔD)
            # in principle, ΔD is real, but maybe not if coming from an anyonic tensor
        end
        # recylce VdΔV space
        mul!(ΔA, mul!(VdΔV, V, aVdΔV), V')
    else
        mul!(ΔA, V * Diagonal(ΔD), V')
    end
    return ΔA
end

function qr_pullback!(ΔA::AbstractMatrix, Q::AbstractMatrix, R::AbstractMatrix, ΔQ, ΔR;
                      atol::Real=0,
                      rtol::Real=atol > 0 ? 0 : eps(real(eltype(R)))^(3 / 4))
    Rd = view(R, diagind(R))
    p = let tol = atol > 0 ? atol : rtol * maximum(abs, Rd)
        findlast(x -> abs(x) >= tol, Rd)
    end
    m, n = size(R)

    Q1 = view(Q, :, 1:p)
    R1 = view(R, 1:p, :)
    R11 = view(R, 1:p, 1:p)

    ΔA1 = view(ΔA, :, 1:p)
    ΔQ1 = view(ΔQ, :, 1:p)
    ΔR1 = view(ΔR, 1:p, :)
    ΔR11 = view(ΔR, 1:p, 1:p)

    M = similar(R, (p, p))
    ΔR isa AbstractZero || mul!(M, ΔR1, R1')
    ΔQ isa AbstractZero || mul!(M, Q1', ΔQ1, -1, +1)
    view(M, lowertriangularind(M)) .= conj.(view(M, uppertriangularind(M)))
    if eltype(M) <: Complex
        Md = view(M, diagind(M))
        Md .= real.(Md)
    end

    ΔA1 .= ΔQ1
    mul!(ΔA1, Q1, M, +1, 1)

    if n > p
        R12 = view(R, 1:p, (p + 1):n)
        ΔA2 = view(ΔA, :, (p + 1):n)
        ΔR12 = view(ΔR, 1:p, (p + 1):n)

        if ΔR isa AbstractZero
            ΔA2 .= zero(eltype(ΔA))
        else
            mul!(ΔA2, Q1, ΔR12)
            mul!(ΔA1, ΔA2, R12', -1, 1)
        end
    end
    if m > p && !(ΔQ isa AbstractZero) # case where R is not full rank
        Q2 = view(Q, :, (p + 1):m)
        ΔQ2 = view(ΔQ, :, (p + 1):m)
        Q1dΔQ2 = Q1' * ΔQ2
        gaugepart = mul!(copy(ΔQ2), Q1, Q1dΔQ2, -1, 1)
        norm(gaugepart, Inf) < tol || @warn "cotangents sensitive to gauge choice"
        mul!(ΔA1, Q2, Q1dΔQ2', -1, 1)
    end
    rdiv!(ΔA1, UpperTriangular(R11)')
    return ΔA
end

function lq_pullback!(ΔA::AbstractMatrix, L::AbstractMatrix, Q::AbstractMatrix, ΔL, ΔQ;
                      atol::Real=0,
                      rtol::Real=atol > 0 ? 0 : eps(real(eltype(L)))^(3 / 4))
    Ld = view(L, diagind(L))
    p = let tol = atol > 0 ? atol : rtol * maximum(abs, Ld)
        findlast(x -> abs(x) >= tol, Ld)
    end
    m, n = size(L)

    L1 = view(L, :, 1:p)
    L11 = view(L, 1:p, 1:p)
    Q1 = view(Q, 1:p, :)

    ΔA1 = view(ΔA, 1:p, :)
    ΔQ1 = view(ΔQ, 1:p, :)
    ΔL1 = view(ΔL, :, 1:p)
    ΔR11 = view(ΔL, 1:p, 1:p)

    M = similar(L, (p, p))
    ΔL isa AbstractZero || mul!(M, L1', ΔL1)
    ΔQ isa AbstractZero || mul!(M, ΔQ1, Q1', -1, +1)
    view(M, uppertriangularind(M)) .= conj.(view(M, lowertriangularind(M)))
    if eltype(M) <: Complex
        Md = view(M, diagind(M))
        Md .= real.(Md)
    end

    ΔA1 .= ΔQ1
    mul!(ΔA1, M, Q1, +1, 1)

    if m > p
        L21 = view(L, (p + 1):m, 1:p)
        ΔA2 = view(ΔA, (p + 1):m, :)
        ΔL21 = view(ΔL, (p + 1):m, 1:p)

        if ΔL isa AbstractZero
            ΔA2 .= zero(eltype(ΔA))
        else
            mul!(ΔA2, ΔL21, Q1)
            mul!(ΔA1, L21', ΔA2, -1, 1)
        end
    end
    if n > p && !(ΔQ isa AbstractZero) # case where R is not full rank
        Q2 = view(Q, (p + 1):n, :)
        ΔQ2 = view(ΔQ, (p + 1):n, :)
        ΔQ2Q1d = ΔQ2 * Q1'
        gaugepart = mul!(copy(ΔQ2), ΔQ2Q1d, Q1, -1, 1)
        norm(gaugepart, Inf) < tol || @warn "cotangents sensitive to gauge choice"
        mul!(ΔA1, ΔQ2Q1d', Q2, -1, 1)
    end
    ldiv!(LowerTriangular(L11)', ΔA1)
    return ΔA
end

# Planar rrules
# --------------
function ChainRulesCore.rrule(::typeof(TensorKit.planaradd!), C::AbstractTensorMap{S,N₁,N₂},
                              A::AbstractTensorMap{S}, p::Index2Tuple{N₁,N₂},
                              α::Number, β::Number,
                              backend::Backend...) where {S,N₁,N₂}
    C′ = planaradd!(copy(C), A, p, α, β, backend...)

    projectA = ProjectTo(A)
    projectC = ProjectTo(C)
    projectα = ProjectTo(α)
    projectβ = ProjectTo(β)

    function planaradd_pullback(ΔC′)
        ΔC = unthunk(ΔC′)

        dC = @thunk projectC(scale(ΔC, conj(β)))
        dA = @thunk begin
            ip = _canonicalize(invperm(linearize(p)), A)
            _dA = zerovector(A, VectorInterface.promote_add(ΔC, α))
            _dA = planaradd!(_dA, ΔC, ip, conj(α), Zero(), backend...)
            return projectA(_dA)
        end
        dα = @thunk begin
            p′ = TensorKit.adjointtensorindices(A, p)
            _dα = tensorscalar(planarcontract(A', ((), linearize(p′)),
                                              ΔC, (trivtuple(p), ()),
                                              ((), ()), One(), backend...))
            return projectα(_dα)
        end
        dβ = @thunk begin
            p′ = TensorKit.adjointtensorindices(C, trivtuple(p))
            _dβ = tensorscalar(planarcontract(C', ((), p′),
                                              ΔC, (trivtuple(p), ()),
                                              ((), ()), One(), backend...))
            return projectβ(_dβ)
        end
        dbackend = map(x -> NoTangent(), backend)
        return NoTangent(), dC, dA, NoTangent(), dα, dβ, dbackend...
    end

    return C′, planaradd_pullback
end

function ChainRulesCore.rrule(::typeof(TensorKit.planarcontract!),
                              C::AbstractTensorMap{S,N₁,N₂},
                              A::AbstractTensorMap{S}, pA::Index2Tuple,
                              B::AbstractTensorMap{S}, pB::Index2Tuple,
                              pAB::Index2Tuple{N₁,N₂},
                              α::Number, β::Number, backend::Backend...) where {S,N₁,N₂}
    indA = (codomainind(A), reverse(domainind(A)))
    indB = (codomainind(B), reverse(domainind(B)))
    pA, pB, pAB = TensorKit.reorder_planar_indices(indA, pA, indB, pB, pAB)
    C′ = planarcontract!(copy(C), A, pA, B, pB, pAB, α, β, backend...)

    projectA = ProjectTo(A)
    projectB = ProjectTo(B)
    projectC = ProjectTo(C)
    projectα = ProjectTo(α)
    projectβ = ProjectTo(β)

    function planarcontract_pullback(ΔC′)
        ΔC = unthunk(ΔC′)
        ipAB = invperm(linearize(pAB))
        pΔC = (getindices(ipAB, trivtuple(length(pA[1]))),
               getindices(ipAB, length(pA[1]) .+ trivtuple(length(pB[2]))))
        dC = @thunk projectC(scale(ΔC, conj(β)))
        dA = @thunk begin
            ipA = _canonicalize(invperm(linearize(pA)), A)
            _dA = zerovector(A, promote_contract(scalartype(ΔC), scalartype(B), typeof(α)))
            pB′ = TensorKit.adjointtensorindices(B, reverse(pB))
            _dA = planarcontract!(_dA, ΔC, pΔC, adjoint(B), pB′, ipA,
                                  conj(α), Zero(), backend...)
            return projectA(_dA)
        end
        dB = @thunk begin
            ipB = _canonicalize((invperm(linearize(pB)), ()), B)
            _dB = zerovector(B, promote_contract(scalartype(ΔC), scalartype(A), typeof(α)))
            pA′ = TensorKit.adjointtensorindices(A, reverse(pA))
            _dB = planarcontract!(_dB, adjoint(A), pA′, ΔC, pΔC, ipB,
                                  conj(α), Zero(), backend...)
            return projectB(_dB)
        end
        dα = @thunk begin
            AB = planarcontract!(similar(C), A, pA, B, pB, pAB, One(), Zero(), backend...)
            p′ = TensorKit.adjointtensorindices(AB, trivtuple(pAB))
            _dα = tensorscalar(planarcontract(AB', ((), p′),
                                              ΔC, (trivtuple(pAB), ()), ((), ()),
                                              One(), backend...))
            return projectα(_dα)
        end
        dβ = @thunk begin
            p′ = TensorKit.adjointtensorindices(C, trivtuple(pAB))
            _dβ = tensorscalar(planarcontract(C', ((), p′),
                                              ΔC, (trivtuple(pAB), ()), ((), ()),
                                              One(), backend...))
            return projectβ(_dβ)
        end
        dbackend = map(x -> NoTangent(), backend)
        return NoTangent(), dC, dA, NoTangent(), dB, NoTangent(), NoTangent(),
               dα, dβ, dbackend...
    end

    return C′, planarcontract_pullback
end

# Convert rrules
#----------------
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
