using ChainRulesCore
using ChainRulesTestUtils
using Random
using FiniteDifferences
using LinearAlgebra

const _repartition = @static if isdefined(Base, :get_extension)
    Base.get_extension(TensorKit, :TensorKitChainRulesCoreExt)._repartition
else
    TensorKit.TensorKitChainRulesCoreExt._repartition
end

# Test utility
# -------------
function ChainRulesTestUtils.rand_tangent(rng::AbstractRNG, x::AbstractTensorMap)
    return TensorMap(randn, scalartype(x), space(x))
end
function ChainRulesTestUtils.test_approx(actual::AbstractTensorMap,
                                         expected::AbstractTensorMap, msg=""; kwargs...)
    for (c, b) in blocks(actual)
        ChainRulesTestUtils.@test_msg msg isapprox(b, block(expected, c); kwargs...)
    end
end
function FiniteDifferences.to_vec(t::T) where {T<:TensorKit.TrivialTensorMap}
    vec, from_vec = to_vec(t.data)
    return vec, x -> T(from_vec(x), codomain(t), domain(t))
end
function FiniteDifferences.to_vec(t::AbstractTensorMap)
    vec = mapreduce(vcat, blocks(t); init=scalartype(t)[]) do (c, b)
        return reshape(b, :) .* sqrt(dim(c))
    end
    vec_real = scalartype(t) <: Real ? vec : collect(reinterpret(real(scalartype(t)), vec))

    function from_vec(x_real)
        x = scalartype(t) <: Real ? x_real : reinterpret(scalartype(t), x_real)
        t′ = similar(t)
        ctr = 0
        for (c, b) in blocks(t′)
            n = length(b)
            copyto!(b, reshape(view(x, ctr .+ (1:n)), size(b)) ./ sqrt(dim(c)))
            ctr += n
        end
        return t′
    end
    return vec_real, from_vec
end
FiniteDifferences.to_vec(t::TensorKit.AdjointTensorMap) = to_vec(copy(t))

# make sure that norms are computed correctly:
function FiniteDifferences.to_vec(t::TensorKit.SectorDict)
    T = scalartype(valtype(t))
    vec = mapreduce(vcat, t; init=T[]) do (c, b)
        return reshape(b, :) .* sqrt(dim(c))
    end
    vec_real = T <: Real ? vec : collect(reinterpret(real(T), vec))

    function from_vec(x_real)
        x = T <: Real ? x_real : reinterpret(T, x_real)
        ctr = 0
        return TensorKit.SectorDict(c => (n = length(b); b′ = reshape(view(x, ctr .+ (1:n)), size(b)) ./ sqrt(dim(c)); ctr += n; b′)
                                    for (c, b) in t)
    end
    return vec_real, from_vec
end

function _randomize!(a::TensorMap)
    for b in values(blocks(a))
        copyto!(b, randn(size(b)))
    end
    return a
end

# Float32 and finite differences don't mix well
precision(::Type{<:Union{Float32,Complex{Float32}}}) = 1e-2
precision(::Type{<:Union{Float64,Complex{Float64}}}) = 1e-6

function randindextuple(N::Int, k::Int=rand(0:N))
    @assert 0 ≤ k ≤ N
    _p = randperm(N)
    return (tuple(_p[1:k]...), tuple(_p[(k + 1):end]...))
end

# rrules for functions that destroy inputs
# ----------------------------------------
function ChainRulesCore.rrule(::typeof(TensorKit.tsvd), args...; kwargs...)
    return ChainRulesCore.rrule(tsvd!, args...; kwargs...)
end
function ChainRulesCore.rrule(::typeof(LinearAlgebra.svdvals), args...; kwargs...)
    return ChainRulesCore.rrule(svdvals!, args...; kwargs...)
end
function ChainRulesCore.rrule(::typeof(TensorKit.eig), args...; kwargs...)
    return ChainRulesCore.rrule(eig!, args...; kwargs...)
end
function ChainRulesCore.rrule(::typeof(TensorKit.eigh), args...; kwargs...)
    return ChainRulesCore.rrule(eigh!, args...; kwargs...)
end
function ChainRulesCore.rrule(::typeof(LinearAlgebra.eigvals), args...; kwargs...)
    return ChainRulesCore.rrule(eigvals!, args...; kwargs...)
end
function ChainRulesCore.rrule(::typeof(TensorKit.leftorth), args...; kwargs...)
    return ChainRulesCore.rrule(leftorth!, args...; kwargs...)
end
function ChainRulesCore.rrule(::typeof(TensorKit.rightorth), args...; kwargs...)
    return ChainRulesCore.rrule(rightorth!, args...; kwargs...)
end

# eigh′: make argument of eigh explicitly Hermitian
#---------------------------------------------------
eigh′(t::AbstractTensorMap) = eigh(scale!(t + t', 1 / 2))

function ChainRulesCore.rrule(::typeof(eigh′), args...; kwargs...)
    return ChainRulesCore.rrule(eigh!, args...; kwargs...)
end

# complex-valued svd?
# -------------------
function remove_svdgauge_depence!(ΔU, ΔV, U, S, V)
    # simple implementation, assumes no degeneracies or zeros in singular values
    gaugepart = U' * ΔU + V * ΔV'
    for (c, b) in blocks(gaugepart)
        mul!(block(ΔU, c), block(U, c), Diagonal(imag(diag(b))), -im, 1)
    end
    return ΔU, ΔV
end

# Tests
# -----

ChainRulesTestUtils.test_method_tables()

Vlist = ((ℂ^2, (ℂ^3)', ℂ^3, ℂ^2, (ℂ^2)'),
         (ℂ[Z2Irrep](0 => 1, 1 => 1),
          ℂ[Z2Irrep](0 => 1, 1 => 2)',
          ℂ[Z2Irrep](0 => 3, 1 => 2)',
          ℂ[Z2Irrep](0 => 2, 1 => 3),
          ℂ[Z2Irrep](0 => 2, 1 => 2)),
         (ℂ[FermionParity](0 => 1, 1 => 1),
          ℂ[FermionParity](0 => 1, 1 => 2)',
          ℂ[FermionParity](0 => 2, 1 => 2)',
          ℂ[FermionParity](0 => 2, 1 => 3),
          ℂ[FermionParity](0 => 2, 1 => 2)),
         (ℂ[U1Irrep](0 => 2, 1 => 2, -1 => 2),
          ℂ[U1Irrep](0 => 3, 1 => 1, -1 => 1),
          ℂ[U1Irrep](0 => 2, 1 => 2, -1 => 1)',
          ℂ[U1Irrep](0 => 1, 1 => 2, -1 => 2),
          ℂ[U1Irrep](0 => 1, 1 => 3, -1 => 2)'),
         (ℂ[SU2Irrep](0 => 2, 1 // 2 => 1),
          ℂ[SU2Irrep](0 => 1, 1 => 1),
          ℂ[SU2Irrep](1 // 2 => 1, 1 => 1)',
          ℂ[SU2Irrep](1 // 2 => 2),
          ℂ[SU2Irrep](0 => 1, 1 // 2 => 1, 3 // 2 => 1)'))

@timedtestset "Automatic Differentiation with spacetype $(TensorKit.type_repr(eltype(V)))" verbose = true for V in
                                                                                                              Vlist
    # @timedtestset "Basic Linear Algebra with scalartype $T" for T in (Float64, ComplexF64)
    #     A = TensorMap(randn, T, V[1] ⊗ V[2] ← V[3] ⊗ V[4] ⊗ V[5])
    #     B = TensorMap(randn, T, space(A))
    #
    #     test_rrule(+, A, B)
    #     test_rrule(-, A)
    #     test_rrule(-, A, B)
    #
    #     α = randn(T)
    #     test_rrule(*, α, A)
    #     test_rrule(*, A, α)
    #
    #     C = TensorMap(randn, T, domain(A), codomain(A))
    #     test_rrule(*, A, C)
    #
    #     test_rrule(permute, A, ((1, 3, 2), (5, 4)))
    #
    #     D = TensorMap(randn, T, V[1] ⊗ V[2] ← V[3])
    #     E = TensorMap(randn, T, V[4] ← V[5])
    #     test_rrule(⊗, D, E)
    # end

    # @timedtestset "Linear Algebra part II with scalartype $T" for T in (Float64, ComplexF64)
    #     for i in 1:3
    #         E = TensorMap(randn, T, ⊗(V[1:i]...) ← ⊗(V[1:i]...))
    #         test_rrule(LinearAlgebra.tr, E)
    #     end
    #
    #     A = TensorMap(randn, T, V[1] ⊗ V[2] ← V[3] ⊗ V[4] ⊗ V[5])
    #     test_rrule(LinearAlgebra.adjoint, A)
    #     test_rrule(LinearAlgebra.norm, A, 2)
    # end

    # @timedtestset "TensorOperations with scalartype $T" for T in (Float64, ComplexF64)
    #     atol = precision(T)
    #     rtol = precision(T)
    #
    #     @timedtestset "tensortrace!" begin
    #         for _ in 1:5
    #             k1 = rand(0:3)
    #             k2 = k1 == 3 ? 1 : rand(1:2)
    #             V1 = map(v -> rand(Bool) ? v' : v, rand(V, k1))
    #             V2 = map(v -> rand(Bool) ? v' : v, rand(V, k2))
    #
    #             (_p, _q) = randindextuple(k1 + 2 * k2, k1)
    #             p = _repartition(_p, rand(0:k1))
    #             q = _repartition(_q, k2)
    #             ip = _repartition(invperm(linearize((_p, _q))), rand(0:(k1 + 2 * k2)))
    #             A = TensorMap(randn, T, permute(prod(V1) ⊗ prod(V2) ← prod(V2), ip))
    #
    #             α = randn(T)
    #             β = randn(T)
    #             for conjA in (:N, :C)
    #                 C = _randomize!(TensorOperations.tensoralloc_add(T, p, A, conjA, false))
    #                 test_rrule(tensortrace!, C, p, A, q, conjA, α, β; atol, rtol)
    #             end
    #         end
    #     end
    #
    #     @timedtestset "tensoradd!" begin
    #         A = TensorMap(randn, T, V[1] ⊗ V[2] ⊗ V[3] ← V[4] ⊗ V[5])
    #         α = randn(T)
    #         β = randn(T)
    #
    #         # repeat a couple times to get some distribution of arrows
    #         for _ in 1:5
    #             p = randindextuple(length(V))
    #
    #             C1 = _randomize!(TensorOperations.tensoralloc_add(T, p, A, :N, false))
    #             test_rrule(tensoradd!, C1, p, A, :N, α, β; atol, rtol)
    #
    #             C2 = _randomize!(TensorOperations.tensoralloc_add(T, p, A, :C, false))
    #             test_rrule(tensoradd!, C2, p, A, :C, α, β; atol, rtol)
    #
    #             A = rand(Bool) ? C1 : C2
    #         end
    #     end
    #
    #     @timedtestset "tensorcontract!" begin
    #         for _ in 1:5
    #             d = 0
    #             local V1, V2, V3
    #             # retry a couple times to make sure there are at least some nonzero elements
    #             for _ in 1:10
    #                 k1 = rand(0:3)
    #                 k2 = rand(0:2)
    #                 k3 = rand(0:2)
    #                 V1 = prod(v -> rand(Bool) ? v' : v, rand(V, k1); init=one(V[1]))
    #                 V2 = prod(v -> rand(Bool) ? v' : v, rand(V, k2); init=one(V[1]))
    #                 V3 = prod(v -> rand(Bool) ? v' : v, rand(V, k3); init=one(V[1]))
    #                 d = min(dim(V1 ← V2), dim(V1' ← V2), dim(V2 ← V3), dim(V2' ← V3))
    #                 d > 0 && break
    #             end
    #             ipA = randindextuple(length(V1) + length(V2))
    #             pA = _repartition(invperm(linearize(ipA)), length(V1))
    #             ipB = randindextuple(length(V2) + length(V3))
    #             pB = _repartition(invperm(linearize(ipB)), length(V2))
    #             pAB = randindextuple(length(V1) + length(V3))
    #
    #             α = randn(T)
    #             β = randn(T)
    #             V2_conj = prod(conj, V2; init=one(V[1]))
    #
    #             for conjA in (:N, :C), conjB in (:N, :C)
    #                 A = TensorMap(randn, T,
    #                               permute(V1 ← (conjA === :C ? V2_conj : V2), ipA))
    #                 B = TensorMap(randn, T,
    #                               permute((conjB === :C ? V2_conj : V2) ← V3, ipB))
    #                 C = _randomize!(TensorOperations.tensoralloc_contract(T, pAB, A, pA,
    #                                                                       conjA,
    #                                                                       B, pB, conjB,
    #                                                                       false))
    #                 test_rrule(tensorcontract!, C, pAB,
    #                            A, pA, conjA, B, pB, conjB,
    #                            α, β; atol, rtol)
    #             end
    #         end
    #     end
    #
    #     @timedtestset "tensorscalar" begin
    #         A = Tensor(randn, T, ProductSpace{typeof(V[1]),0}())
    #         test_rrule(tensorscalar, A)
    #     end
    # end

    @timedtestset "Factorizations with scalartype $T" for T in (Float64, ComplexF64)
        A = TensorMap(randn, T, V[1] ⊗ V[2] ← V[3] ⊗ V[4] ⊗ V[5])
        B = TensorMap(randn, T, space(A)')
        C = TensorMap(randn, T, V[1] ⊗ V[2] ← V[1] ⊗ V[2])
        H = TensorMap(randn, T, V[3] ⊗ V[4] ← V[3] ⊗ V[4])
        H = (H + H') / 2
        atol = precision(T)
        #
        # for alg in (TensorKit.QR(), TensorKit.QRpos())
        #     test_rrule(leftorth, A; fkwargs=(; alg=alg), atol)
        #     test_rrule(leftorth, B; fkwargs=(; alg=alg), atol)
        #     test_rrule(leftorth, C; fkwargs=(; alg=alg), atol)
        # end
        #
        # for alg in (TensorKit.LQ(), TensorKit.LQpos())
        #     test_rrule(rightorth, A; fkwargs=(; alg=alg), atol)
        #     test_rrule(rightorth, B; fkwargs=(; alg=alg), atol)
        #     test_rrule(rightorth, C; fkwargs=(; alg=alg), atol)
        # end
        #
        # let (D, V) = eig(C)
        #     ΔD = TensorMap(randn, scalartype(D), space(D))
        #     ΔV = TensorMap(randn, scalartype(V), space(V))
        #     gaugepart = V' * ΔV
        #     for (c, b) in blocks(gaugepart)
        #         mul!(block(ΔV, c), inv(block(V, c))', Diagonal(diag(b)), -1, 1)
        #     end
        #     test_rrule(eig, C; atol, output_tangent=(ΔD, ΔV))
        # end
        #
        # let (D, U) = eigh′(H)
        #     ΔD = TensorMap(randn, scalartype(D), space(D))
        #     ΔU = TensorMap(randn, scalartype(U), space(U))
        #     if T <: Complex
        #         gaugepart = U' * ΔU
        #         for (c, b) in blocks(gaugepart)
        #             mul!(block(ΔU, c), block(U, c), Diagonal(imag(diag(b))), -im, 1)
        #         end
        #     end
        #     test_rrule(eigh′, H; atol, output_tangent=(ΔD, ΔU))
        # end
        #
        # let (U, S, V, ϵ) = tsvd(A)
        #     ΔU = TensorMap(randn, scalartype(U), space(U))
        #     ΔS = TensorMap(randn, scalartype(S), space(S))
        #     ΔV = TensorMap(randn, scalartype(V), space(V))
        #     if T <: Complex # remove gauge dependent components
        #         gaugepart = U' * ΔU + V * ΔV'
        #         for (c, b) in blocks(gaugepart)
        #             mul!(block(ΔU, c), block(U, c), Diagonal(imag(diag(b))), -im, 1)
        #         end
        #     end
        #     test_rrule(tsvd, A; atol, output_tangent=(ΔU, ΔS, ΔV, 0.0))
        #
        #     allS = mapreduce(x -> diag(x[2]), vcat, blocks(S))
        #     truncval = (maximum(allS) + minimum(allS)) / 2
        #     U, S, V, ϵ = tsvd(A; trunc=truncerr(truncval))
        #     ΔU = TensorMap(randn, scalartype(U), space(U))
        #     ΔS = TensorMap(randn, scalartype(S), space(S))
        #     ΔV = TensorMap(randn, scalartype(V), space(V))
        #     T <: Complex && remove_svdgauge_depence!(ΔU, ΔV, U, S, V)
        #     test_rrule(tsvd, A; atol, output_tangent=(ΔU, ΔS, ΔV, 0.0),
        #                fkwargs=(; trunc=truncerr(truncval)))
        # end
        #
        # let (U, S, V, ϵ) = tsvd(B)
        #     ΔU = TensorMap(randn, scalartype(U), space(U))
        #     ΔS = TensorMap(randn, scalartype(S), space(S))
        #     ΔV = TensorMap(randn, scalartype(V), space(V))
        #     T <: Complex && remove_svdgauge_depence!(ΔU, ΔV, U, S, V)
        #     test_rrule(tsvd, B; atol, output_tangent=(ΔU, ΔS, ΔV, 0.0))
        #
        #     Vtrunc = spacetype(S)(TensorKit.SectorDict(c => ceil(Int, size(b, 1) / 2)
        #                                                for (c, b) in blocks(S)))
        #
        #     U, S, V, ϵ = tsvd(B; trunc=truncspace(Vtrunc))
        #     ΔU = TensorMap(randn, scalartype(U), space(U))
        #     ΔS = TensorMap(randn, scalartype(S), space(S))
        #     ΔV = TensorMap(randn, scalartype(V), space(V))
        #     T <: Complex && remove_svdgauge_depence!(ΔU, ΔV, U, S, V)
        #     test_rrule(tsvd, B; atol, output_tangent=(ΔU, ΔS, ΔV, 0.0),
        #                fkwargs=(; trunc=truncspace(Vtrunc)))
        # end
        #
        # let (U, S, V, ϵ) = tsvd(C)
        #     ΔU = TensorMap(randn, scalartype(U), space(U))
        #     ΔS = TensorMap(randn, scalartype(S), space(S))
        #     ΔV = TensorMap(randn, scalartype(V), space(V))
        #     T <: Complex && remove_svdgauge_depence!(ΔU, ΔV, U, S, V)
        #     test_rrule(tsvd, C; atol, output_tangent=(ΔU, ΔS, ΔV, 0.0))
        #
        #     c, = TensorKit.MatrixAlgebra._argmax(x -> sqrt(dim(x[1])) * maximum(diag(x[2])),
        #                                          blocks(S))
        #     U, S, V, ϵ = tsvd(C; trunc=truncdim(2 * dim(c)))
        #     ΔU = TensorMap(randn, scalartype(U), space(U))
        #     ΔS = TensorMap(randn, scalartype(S), space(S))
        #     ΔV = TensorMap(randn, scalartype(V), space(V))
        #     T <: Complex && remove_svdgauge_depence!(ΔU, ΔV, U, S, V)
        #     test_rrule(tsvd, C; atol, output_tangent=(ΔU, ΔS, ΔV, 0.0),
        #                fkwargs=(; trunc=truncdim(2 * dim(c))))
        # end

        let D = LinearAlgebra.eigvals(C)
            ΔD = diag(TensorMap(randn, complex(scalartype(C)), space(C)))
            test_rrule(LinearAlgebra.eigvals, C; atol, output_tangent=ΔD,
                       fkwargs=(; sortby=nothing))
        end

        let S = LinearAlgebra.svdvals(C)
            ΔS = diag(TensorMap(randn, real(scalartype(C)), space(C)))
            test_rrule(LinearAlgebra.svdvals, C; atol, output_tangent=ΔS)
        end
    end
end
