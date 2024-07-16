# Factorizations rules
# --------------------
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

function ChainRulesCore.rrule(::typeof(LinearAlgebra.svdvals!), t::AbstractTensorMap)
    U, S, V = tsvd(t)
    s = diag(S)
    project_t = ProjectTo(t)

    function svdvals_pullback(Δs′)
        Δs = unthunk(Δs′)
        ΔS = diagm(codomain(S), domain(S), Δs)
        return NoTangent(), project_t(U * ΔS * V)
    end

    return s, svdvals_pullback
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

function ChainRulesCore.rrule(::typeof(LinearAlgebra.eigvals!), t::AbstractTensorMap;
                              sortby=nothing, kwargs...)
    @assert sortby === nothing "only `sortby=nothing` is supported"
    (D, _), eig_pullback = rrule(TensorKit.eig!, t; kwargs...)
    d = diag(D)
    project_t = ProjectTo(t)
    function eigvals_pullback(Δd′)
        Δd = unthunk(Δd′)
        ΔD = diagm(codomain(D), domain(D), Δd)
        return NoTangent(), project_t(eig_pullback((ΔD, ZeroTangent()))[2])
    end

    return d, eigvals_pullback
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
    Δgauge = norm(view(aUΔU, mask) + view(aVΔV, mask), Inf)
    if p > r
        rprange = (r + 1):p
        Δgauge = max(Δgauge, norm(view(aUΔU, rprange, rprange), Inf))
        Δgauge = max(Δgauge, norm(view(aVΔV, rprange, rprange), Inf))
    end
    Δgauge < tol ||
        @warn "`svd` cotangents sensitive to gauge choice: (|Δgauge| = $Δgauge)"

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
        Δgauge = norm(view(VdΔV, mask), Inf)
        Δgauge < tol ||
            @warn "`eig` cotangents sensitive to gauge choice: (|Δgauge| = $Δgauge)"

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
        Δgauge = norm(view(aVdΔV, mask))
        Δgauge < tol ||
            @warn "`eigh` cotangents sensitive to gauge choice: (|Δgauge| = $Δgauge)"

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
    tol = atol > 0 ? atol : rtol * maximum(abs, Rd)
    p = findlast(>=(tol) ∘ abs, Rd)
    m, n = size(R)

    Q1 = view(Q, :, 1:p)
    R1 = view(R, 1:p, :)
    R11 = view(R, 1:p, 1:p)

    ΔA1 = view(ΔA, :, 1:p)
    ΔQ1 = view(ΔQ, :, 1:p)
    ΔR1 = view(ΔR, 1:p, :)

    M = similar(R, (p, p))
    ΔR isa AbstractZero || mul!(M, ΔR1, R1')
    ΔQ isa AbstractZero || mul!(M, Q1', ΔQ1, -1, !(ΔR isa AbstractZero))
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
        Δgauge = norm(mul!(copy(ΔQ2), Q1, Q1dΔQ2, -1, 1), Inf)
        Δgauge < tol ||
            @warn "`qr` cotangents sensitive to gauge choice: (|Δgauge| = $Δgauge)"
        mul!(ΔA1, Q2, Q1dΔQ2', -1, 1)
    end
    rdiv!(ΔA1, UpperTriangular(R11)')
    return ΔA
end

function lq_pullback!(ΔA::AbstractMatrix, L::AbstractMatrix, Q::AbstractMatrix, ΔL, ΔQ;
                      atol::Real=0,
                      rtol::Real=atol > 0 ? 0 : eps(real(eltype(L)))^(3 / 4))
    Ld = view(L, diagind(L))
    tol = atol > 0 ? atol : rtol * maximum(abs, Ld)
    p = findlast(>=(tol) ∘ abs, Ld)
    m, n = size(L)

    L1 = view(L, :, 1:p)
    L11 = view(L, 1:p, 1:p)
    Q1 = view(Q, 1:p, :)

    ΔA1 = view(ΔA, 1:p, :)
    ΔQ1 = view(ΔQ, 1:p, :)
    ΔL1 = view(ΔL, :, 1:p)

    M = similar(L, (p, p))
    ΔL isa AbstractZero || mul!(M, L1', ΔL1)
    ΔQ isa AbstractZero || mul!(M, ΔQ1, Q1', -1, !(ΔL isa AbstractZero))
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
        Δgauge = norm(mul!(copy(ΔQ2), ΔQ2Q1d, Q1, -1, 1))
        Δgauge < tol ||
            @warn "`lq` cotangents sensitive to gauge choice: (|Δgauge| = $Δgauge)"
        mul!(ΔA1, ΔQ2Q1d', Q2, -1, 1)
    end
    ldiv!(LowerTriangular(L11)', ΔA1)
    return ΔA
end
