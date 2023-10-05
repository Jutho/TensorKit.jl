using ChainRulesCore
using ChainRulesTestUtils
using Random
using FiniteDifferences
using TensorOperations
using TensorOperations: tensoralloc_add, tensoralloc_contract

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

function _randomize!(a::TensorMap)
    for b in values(blocks(a))
        copyto!(b, randn(size(b)))
    end
    return a
end

# Float32 and finite differences don't mix well
precision(::Type{<:Union{Float32,Complex{Float32}}}) = 1e-2
precision(::Type{<:Union{Float64,Complex{Float64}}}) = 1e-8

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
         (ℂ[U1Irrep](0 => 1, 1 => 2, -1 => 1),
          ℂ[U1Irrep](0 => 1, 1 => 1, -1 => 1),
          ℂ[U1Irrep](0 => 2, 1 => 2, -1 => 1)',
          ℂ[U1Irrep](0 => 1, 1 => 2, -1 => 1),
          ℂ[U1Irrep](0 => 1, 1 => 2, -1 => 2)'),
         (ℂ[SU2Irrep](0 => 1, 1 // 2 => 1),
          ℂ[SU2Irrep](0 => 2, 1 => 1),
          ℂ[SU2Irrep](1 // 2 => 1, 1 => 1)',
          ℂ[SU2Irrep](0 => 2, 1 // 2 => 2),
          ℂ[SU2Irrep](0 => 1, 1 // 2 => 2)'))

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

    @testset "TensorOperations ($T)" for T in (Float64, ComplexF64)
        atol = precision(T)
        rtol = precision(T)

        @testset "tensortrace!" begin
            A = TensorMap(randn, T, V[1] ⊗ V[2] ← V[3] ⊗ V[1] ⊗ V[5])
            pC = ((3, 5), (2,))
            pA = ((1,), (4,))
            α = randn(T)
            β = randn(T)

            C = _randomize!(tensoralloc_add(T, pC, A, :N, false))
            test_rrule(tensortrace!, C, pC, A, pA, :N, α, β; atol, rtol)

            C = _randomize!(tensoralloc_add(T, pC, A, :C, false))
            test_rrule(tensortrace!, C, pC, A, pA, :C, α, β; atol, rtol)
        end

        @testset "tensoradd!" begin
            p = ((1, 3, 2), (5, 4))
            A = TensorMap(randn, T, V[1] ⊗ V[2] ← V[3] ⊗ V[4] ⊗ V[5])
            C = _randomize!(tensoralloc_add(T, p, A, :N, false))
            α = randn(T)
            β = randn(T)
            test_rrule(tensoradd!, C, p, A, :N, α, β; atol, rtol)

            C = _randomize!(tensoralloc_add(T, p, A, :C, false))
            test_rrule(tensoradd!, C, p, A, :C, α, β; atol, rtol)
        end

        @testset "tensorcontract!" begin
            A = TensorMap(randn, T, V[1] ⊗ V[2] ← V[3] ⊗ V[4] ⊗ V[5])
            B = TensorMap(randn, T, V[3] ⊗ V[1]' ← V[2])
            pC = ((3, 2), (4, 1))
            pA = ((2, 4, 5), (1, 3))
            pB = ((2, 1), (3,))
            α = randn(T)
            β = randn(T)

            C = _randomize!(tensoralloc_contract(T, pC, A, pA, :N,
                                                 B, pB, :N, false))
            test_rrule(tensorcontract!, C, pC, A, pA, :N, B, pB, :N, α, β; atol, rtol)

            A2 = TensorMap(randn, T, V[1]' ⊗ V[2]' ← V[3]' ⊗ V[4]' ⊗ V[5]')
            C = _randomize!(tensoralloc_contract(T, pC, A2, pA, :C,
                                                 B, pB, :N, false))
            test_rrule(tensorcontract!, C, pC, A2, pA, :C, B, pB, :N, α, β; atol, rtol)

            B2 = TensorMap(randn, T, V[3]' ⊗ V[1] ← V[2]')
            C = _randomize!(tensoralloc_contract(T, pC, A, pA, :N,
                                                 B2, pB, :C, false))
            test_rrule(tensorcontract!, C, pC, A, pA, :N, B2, pB, :C, α, β; atol, rtol)

            C = _randomize!(tensoralloc_contract(T, pC, A2, pA, :C,
                                                 B2, pB, :C, false))
            test_rrule(tensorcontract!, C, pC, A2, pA, :C, B2, pB, :C, α, β; atol, rtol)
        end

        @testset "tensorscalar" begin
            A = Tensor(randn, T, ProductSpace{typeof(V[1]),0}())
            test_rrule(tensorscalar, A)
        end
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
