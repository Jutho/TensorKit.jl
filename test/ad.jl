using ChainRulesCore
using ChainRulesTestUtils
using Random
using FiniteDifferences
using LinearAlgebra

## Test utility
# -------------
function ChainRulesTestUtils.rand_tangent(rng::AbstractRNG, x::AbstractTensorMap)
    return TensorMap(randn, scalartype(x), space(x))
end
function ChainRulesTestUtils.test_approx(actual::AbstractTensorMap,
                                         expected::AbstractTensorMap, msg=""; kwargs...)
    for (c, b) in blocks(actual)
        ChainRulesTestUtils.@test_msg msg isapprox(block(expected, c), b; kwargs...)
    end
end
function FiniteDifferences.to_vec(t::T) where {T<:TensorKit.TrivialTensorMap}
    vec, from_vec = to_vec(t.data)
    return vec, x -> T(from_vec(x), codomain(t), domain(t))
end
function FiniteDifferences.to_vec(t::AbstractTensorMap)
    vec = mapreduce(vcat, blocks(t)) do (c, b)
        if scalartype(t) <: Real
            return reshape(b, :) .* sqrt(dim(c))
        else
            v = reshape(b, :) .* sqrt(dim(c))
            return vcat(real(v), imag(v))
        end
    end

    function from_vec(x)
        t′ = similar(t)
        T = scalartype(t)
        ctr = 0
        for (c, b) in blocks(t′)
            n = length(b)
            if T <: Real
                copyto!(b, reshape(x[(ctr + 1):(ctr + n)], size(b)) ./ sqrt(dim(c)))
            else
                v = x[(ctr + 1):(ctr + 2n)]
                copyto!(b,
                        complex.(x[(ctr + 1):(ctr + n)], x[(ctr + n + 1):(ctr + 2n)]) ./
                        sqrt(dim(c)))
            end
            ctr += T <: Real ? n : 2n
        end
        return t′
    end

    return vec, from_vec
end
FiniteDifferences.to_vec(t::TensorKit.AdjointTensorMap) = to_vec(copy(t))

# rrules for functions that destroy inputs
# ----------------------------------------
function ChainRulesCore.rrule(::typeof(TensorKit.tsvd), args...)
    return ChainRulesCore.rrule(tsvd!, args...)
end
function ChainRulesCore.rrule(::typeof(TensorKit.leftorth), args...; kwargs...)
    return ChainRulesCore.rrule(leftorth!, args...; kwargs...)
end
function ChainRulesCore.rrule(::typeof(TensorKit.rightorth), args...; kwargs...)
    return ChainRulesCore.rrule(rightorth!, args...; kwargs...)
end

# complex-valued svd?
# -------------------

# function _gaugefix!(U, V)
#     s = LinearAlgebra.Diagonal(TensorKit._safesign.(diag(U)))
#     rmul!(U, s)
#     lmul!(s', V)
#     return U, V
# end

# function _tsvd(t::AbstractTensorMap)
#     U, S, V, ϵ = tsvd(t)
#     for (c, b) in blocks(U)
#         _gaugefix!(b, block(V, c))
#     end
#     return U, S, V, ϵ
# end

# svd_rev = Base.get_extension(TensorKit, :TensorKitChainRulesCoreExt).svd_rev

# function ChainRulesCore.rrule(::typeof(_tsvd), t::AbstractTensorMap)
#     U, S, V, ϵ = _tsvd(t)
#     function _tsvd_pullback((ΔU, ΔS, ΔV, Δϵ))
#         ∂t = similar(t)
#         for (c, b) in blocks(∂t)
#             copyto!(b,
#                     svd_rev(block(U, c), block(S, c), block(V, c),
#                             block(ΔU, c), block(ΔS, c), block(ΔV, c)))
#         end
#         return NoTangent(), ∂t
#     end
#     return (U, S, V, ϵ), _tsvd_pullback
# end

# Tests
# -----

ChainRulesTestUtils.test_method_tables()

Vlist = ((ℂ^2, (ℂ^3)', ℂ^3, ℂ^2, (ℂ^2)'),
         (ℂ[Z2Irrep](0 => 1, 1 => 1),
          ℂ[Z2Irrep](0 => 1, 1 => 2)',
          ℂ[Z2Irrep](0 => 3, 1 => 2)',
          ℂ[Z2Irrep](0 => 2, 1 => 3),
          ℂ[Z2Irrep](0 => 2, 1 => 2)),
         (ℂ[U1Irrep](0 => 1, 1 => 2, -1 => 2),
          ℂ[U1Irrep](0 => 3, 1 => 1, -1 => 1),
          ℂ[U1Irrep](0 => 2, 1 => 2, -1 => 1)',
          ℂ[U1Irrep](0 => 1, 1 => 2, -1 => 2),
          ℂ[U1Irrep](0 => 1, 1 => 3, -1 => 2)'),
         (ℂ[SU2Irrep](0 => 3, 1 // 2 => 1),
          ℂ[SU2Irrep](0 => 2, 1 => 1),
          ℂ[SU2Irrep](1 // 2 => 1, 1 => 1)',
          ℂ[SU2Irrep](0 => 2, 1 // 2 => 2),
          ℂ[SU2Irrep](0 => 1, 1 // 2 => 1, 3 // 2 => 1)'))

@testset "Automatic Differentiation ($(eltype(V)))" verbose = true for V in Vlist
    @testset "Basic Linear Algebra ($T)" for T in (Float64, ComplexF64)
        A = TensorMap(randn, T, V[1] ⊗ V[2] ← V[3] ⊗ V[4] ⊗ V[5])
        B = TensorMap(randn, T, space(A))

        test_rrule(+, A, B)
        test_rrule(-, A, B)

        α = randn(T)
        test_rrule(*, α, A)
        test_rrule(*, A, α)

        C = TensorMap(randn, T, domain(A), codomain(A))
        test_rrule(*, A, C)

        test_rrule(permute, A, ((1, 3, 2), (5, 4)))

        D = Tensor(randn, T, ProductSpace{ComplexSpace,0}())
        test_rrule(TensorKit.scalar, D)
    end

    @testset "Linear Algebra part II ($T)" for T in (Float64, ComplexF64)
        for i in 1:3
            E = TensorMap(randn, T, ⊗(V[1:i]...) ← ⊗(V[1:i]...))
            test_rrule(LinearAlgebra.tr, E)
        end

        A = TensorMap(randn, T, V[1] ⊗ V[2] ← V[3] ⊗ V[4] ⊗ V[5])
        test_rrule(LinearAlgebra.adjoint, A)
        test_rrule(LinearAlgebra.norm, A, 2)
    end

    @testset "Factorizations ($T)" for T in (Float64, ComplexF64)
        A = TensorMap(randn, T, V[1] ⊗ V[2] ← V[3] ⊗ V[4] ⊗ V[5])
        B = TensorMap(randn, T, space(A)')
        C = TensorMap(randn, T, V[1] ⊗ V[2] ← V[1] ⊗ V[2])
        atol = 1e-6

        for alg in (TensorKit.QR(), TensorKit.QRpos())
            test_rrule(leftorth, A; fkwargs=(; alg=alg), atol)
            test_rrule(leftorth, B; fkwargs=(; alg=alg), atol)
            test_rrule(leftorth, C; fkwargs=(; alg=alg), atol)
        end

        for alg in (TensorKit.LQ(), TensorKit.LQpos())
            test_rrule(rightorth, A; fkwargs=(; alg=alg), atol)
            test_rrule(rightorth, B; fkwargs=(; alg=alg), atol)
            test_rrule(rightorth, C; fkwargs=(; alg=alg), atol)
        end

        # Complex-valued SVD tests are incompatible with finite differencing,
        # because U and V are not unique.
        if T <: Real
            test_rrule(tsvd, A; atol)
            test_rrule(tsvd, B; atol)
            test_rrule(tsvd, C; atol)
        end
    end
end
