using TestEnv: TestEnv;
TestEnv.activate("TensorKit");
using TensorKit
using TensorOperations
using ChainRulesCore
using ChainRulesTestUtils
using Random
using FiniteDifferences
using Test

## Test utility
# -------------
function ChainRulesTestUtils.rand_tangent(rng::AbstractRNG, x::AbstractTensorMap)
    return TensorMap(randn, scalartype(x), space(x))
end
function ChainRulesTestUtils.test_approx(actual::AbstractTensorMap, expected::AbstractTensorMap, msg=""; kwargs...)
    ChainRulesTestUtils.@test_msg msg isapprox(actual, expected; kwargs...)
end
# function ChainRulesTestUtils.test_approx(actual::NTuple{N}, expected::NTuple{N}, msg="";
#                                          kwargs...) where {N}
#     @test all(isapprox.(actual, expected; Ref(kwargs)...))
# end
function FiniteDifferences.to_vec(t::T) where {T<:TensorKit.TrivialTensorMap}
    vec, from_vec = to_vec(t.data)
    return vec, x -> T(from_vec(x), codomain(t), domain(t))
end
function FiniteDifferences.to_vec(t::AbstractTensorMap)
    vec, from_vec′ = to_vec(blocks(t))
    function from_vec(x)
        blocks′ = from_vec′(x)
        t′ = similar(t)
        for (c, b) in blocks(t′)
            b .= blocks′[c]
        end
        return t′
    end
    
    return vec, from_vec
end
FiniteDifferences.to_vec(t::TensorKit.AdjointTensorMap) = to_vec(copy(t))

ChainRulesCore.rrule(::typeof(TensorKit.tsvd), args...) = ChainRulesCore.rrule(tsvd!, args...)
function ChainRulesCore.rrule(::typeof(TensorKit.leftorth), args...; kwargs...)
    return ChainRulesCore.rrule(leftorth!, args...; kwargs...)
end
function ChainRulesCore.rrule(::typeof(TensorKit.rightorth), args...; kwargs...)
    return ChainRulesCore.rrule(rightorth!, args...; kwargs...)
end
##

ChainRulesTestUtils.test_method_tables()

Vtr = (ℂ^3, (ℂ^4)', ℂ^5, ℂ^6, (ℂ^7)')
T = Float64

A = TensorMap(randn, T, Vtr[1] ⊗ Vtr[2] ← Vtr[3] ⊗ Vtr[4] ⊗ Vtr[5])
B = TensorMap(randn, T, space(A))
test_rrule(+, A, B)
test_rrule(-, A, B)
C = TensorMap(randn, T, domain(A), codomain(A))
test_rrule(*, A, C)
α = randn(T)
test_rrule(*, α, A)
test_rrule(*, A, α)

test_rrule(permute, A, ((1, 3, 2), (5, 4)))

D = Tensor(randn, T, ProductSpace{ComplexSpace,0}())
test_rrule(TensorKit.scalar, D)

# LinearAlgebra
# -------------
using LinearAlgebra
for i in 1:3
    E = TensorMap(randn, T, ⊗(Vtr[1:i]...) ← ⊗(Vtr[1:i]...))
    test_rrule(tr, E)
end

test_rrule(adjoint, A)
test_rrule(norm, A, 2)



test_rrule(tsvd, A; atol=1e-6)
test_rrule(leftorth, A; fkwargs=(;alg = TensorKit.QR()))
test_rrule(leftorth, A; fkwargs=(;alg = TensorKit.QRpos()))
test_rrule(rightorth, A; fkwargs=(;alg = TensorKit.LQ()))
test_rrule(rightorth, A; fkwargs=(;alg = TensorKit.LQpos()))