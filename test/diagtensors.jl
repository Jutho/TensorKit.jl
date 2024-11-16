diagspacelist = ((ℂ^4)', ℂ[Z2Irrep](0 => 2, 1 => 3),
                 ℂ[FermionNumber](0 => 2, 1 => 2, -1 => 1)',
                 ℂ[SU2Irrep](0 => 2, 1 => 1), ℂ[FibonacciAnyon](:I => 2, :τ => 2))

@testset "DiagonalTensor with domain $V" for V in diagspacelist
    @timedtestset "Basic properties and algebra" begin
        for T in (Float32, Float64, ComplexF32, ComplexF64, BigFloat)
            t = @constinferred DiagonalTensorMap{T}(undef, V)
            t = @constinferred DiagonalTensorMap(rand(T, reduceddim(V)), V)
            @test @constinferred(hash(t)) == hash(deepcopy(t))
            @test scalartype(t) == T
            @test codomain(t) == ProductSpace(V)
            @test domain(t) == ProductSpace(V)
            @test space(t) == (V ← V)
            @test space(t') == (V ← V)
            @test dim(t) == dim(space(t))
            @test isa(@constinferred(norm(t)), real(T))
            @test norm(t)^2 ≈ dot(t, t)
            α = rand(T)
            @test norm(α * t) ≈ abs(α) * norm(t)
            @test norm(t + t, 2) ≈ 2 * norm(t, 2)
            @test norm(t + t, 1) ≈ 2 * norm(t, 1)
            @test norm(t + t, Inf) ≈ 2 * norm(t, Inf)
            p = 3 * rand(Float64)
            @test norm(t + t, p) ≈ 2 * norm(t, p)
            @test norm(t) ≈ norm(t')

            @test t == @constinferred(TensorMap(t))
            @test norm(t + TensorMap(t)) ≈ 2 * norm(t)

            @test norm(zerovector!(t)) == 0
            @test norm(one!(t)) ≈ sqrt(dim(V))
            @test one!(t) == id(V)
            @test norm(one!(t) - id(V)) == 0

            t1 = DiagonalTensorMap(rand(T, reduceddim(V)), V)
            t2 = DiagonalTensorMap(rand(T, reduceddim(V)), V)
            t3 = DiagonalTensorMap(rand(T, reduceddim(V)), V)
            α = rand(T)
            β = rand(T)
            @test @constinferred(dot(t1, t2)) ≈ conj(dot(t2, t1))
            @test dot(t2, t1) ≈ conj(dot(t2', t1'))
            @test dot(t3, α * t1 + β * t2) ≈ α * dot(t3, t1) + β * dot(t3, t2)
        end
    end
    I = sectortype(V)
    @timedtestset "Basic linear algebra: test via conversion" begin
        for T in (Float32, ComplexF64)
            t1 = DiagonalTensorMap(rand(T, reduceddim(V)), V)
            t2 = DiagonalTensorMap(rand(T, reduceddim(V)), V)
            @test norm(t1, 2) ≈ norm(convert(TensorMap, t1), 2)
            @test dot(t2, t1) ≈ dot(convert(TensorMap, t2), convert(TensorMap, t1))
            α = rand(T)
            @test convert(TensorMap, α * t1) ≈ α * convert(TensorMap, t1)
            @test convert(TensorMap, t1') ≈ convert(TensorMap, t1)'
            @test convert(TensorMap, t1 + t2) ≈
                  convert(TensorMap, t1) + convert(TensorMap, t2)
        end
    end
    @timedtestset "Real and imaginary parts" begin
        for T in (Float64, ComplexF64, ComplexF32)
            t = DiagonalTensorMap(rand(T, reduceddim(V)), V)

            tr = @constinferred real(t)
            @test scalartype(tr) <: Real
            @test real(convert(TensorMap, t)) == convert(TensorMap, tr)

            ti = @constinferred imag(t)
            @test scalartype(ti) <: Real
            @test imag(convert(TensorMap, t)) == convert(TensorMap, ti)

            tc = @inferred complex(t)
            @test scalartype(tc) <: Complex
            @test complex(convert(TensorMap, t)) == convert(TensorMap, tc)

            tc2 = @inferred complex(tr, ti)
            @test tc2 ≈ tc
        end
    end
    @timedtestset "Tensor conversion" begin
        t = @constinferred DiagonalTensorMap(undef, V)
        rand!(t.data)
        tc = complex(t)
        @test convert(typeof(tc), t) == tc
        @test typeof(convert(typeof(tc), t)) == typeof(tc)
    end
    I = sectortype(V)
    if BraidingStyle(I) isa SymmetricBraiding
        @timedtestset "Permutations" begin
            t = DiagonalTensorMap(randn(ComplexF64, reduceddim(V)), V)
            t1 = @constinferred permute(t, $(((2,), (1,))))
            if BraidingStyle(sectortype(V)) isa Bosonic
                @test t1 == transpose(t)
            end
            @test convert(TensorMap, t1) == permute(convert(TensorMap, t), (((2,), (1,))))
            t2 = @constinferred permute(t, $(((1, 2), ())))
            @test convert(TensorMap, t2) == permute(convert(TensorMap, t), (((1, 2), ())))
            t3 = @constinferred permute(t, $(((2, 1), ())))
            @test convert(TensorMap, t3) == permute(convert(TensorMap, t), (((2, 1), ())))
            t4 = @constinferred permute(t, $(((), (1, 2))))
            @test convert(TensorMap, t4) == permute(convert(TensorMap, t), (((), (1, 2))))
            t5 = @constinferred permute(t, $(((), (2, 1))))
            @test convert(TensorMap, t5) == permute(convert(TensorMap, t), (((), (2, 1))))
        end
    end
    @timedtestset "Trace, Multiplication and inverse" begin
        t1 = DiagonalTensorMap(rand(Float64, reduceddim(V)), V)
        t2 = DiagonalTensorMap(rand(ComplexF64, reduceddim(V)), V)
        @test tr(TensorMap(t1)) == @constinferred tr(t1)
        @test tr(TensorMap(t2)) == @constinferred tr(t2)
        @test TensorMap(@constinferred t1 * t2) ≈ TensorMap(t1) * TensorMap(t2)
        @test TensorMap(@constinferred t1 \ t2) ≈ TensorMap(t1) \ TensorMap(t2)
        @test TensorMap(@constinferred t1 / t2) ≈ TensorMap(t1) / TensorMap(t2)
        @test TensorMap(@constinferred inv(t1)) ≈ inv(TensorMap(t1))
        @test TensorMap(@constinferred pinv(t1)) ≈ pinv(TensorMap(t1))
        @test all(Base.Fix2(isa, DiagonalTensorMap),
                  (t1 * t2, t1 \ t2, t1 / t2, inv(t1), pinv(t1)))

        u = randn(Float64, V * V' * V, V)
        @test u * t1 ≈ u * TensorMap(t1)
        @test u / t1 ≈ u / TensorMap(t1)
        @test t1 * u' ≈ TensorMap(t1) * u'
        @test t1 \ u' ≈ TensorMap(t1) \ u'
    end
end
