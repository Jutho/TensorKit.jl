using Base.Test
include("../src/TensorKit.jl")

@testset "Testing groups and fusion trees" begin # mostly type inference tests
for G in (ℤ₂, U₁, SU₂)
    @show G
    @testset "Basic group properties" begin
        s = map(G, ntuple(n->n,Val(3)))
        @inferred one(s[1])
        @inferred one(G)
        @test one(s[1]) == one(G)
        @inferred dual(s[1])
        @inferred Nsymbol(s...)
        @inferred Rsymbol(s...)
        @inferred Bsymbol(s...)
        @inferred Fsymbol(s..., s...)
        @inferred s[1] ⊗ s[2]
        @inferred ⊗(s..., s...)

        out = (s..., dual(first(⊗(s...))))
    end

    @testset "Fusion trees" begin
        N = 4
        s = map(G, ntuple(n->n,Val(N-1)))
        out = (s..., dual(first(⊗(s...))))

        it = @inferred fusiontrees(out)
        state = @inferred start(it)
        f, state = @inferred next(it, state)
        @inferred done(it, state)

        @inferred braid(f, 2)

        p = tuple(randperm(N)...)
        ip = invperm(p)

        d = @inferred permute(f, p)
        d2 = Dict{typeof(f), valtype(d)}()
        for (f2, coeff) in d
            for (f1,coeff2) in permute(f2, ip)
                d2[f1] = get(d2, f1, zero(coeff)) + coeff2*coeff
            end
        end
        for (f1, coeff2) in d2
            if f1 == f
                @test coeff2 ≈ 1
            else
                @test isapprox(coeff2, 0; atol = eps())
            end
        end

        f1 = f
        f2, = first(d)
        for n = 0:2*N
            d = @inferred repartition(f1, f2, Val(n))
            d2 = Dict{typeof((f1,f2)), valtype(d)}()
            for ((f1′,f2′),coeff) in d
                for ((f1′′,f2′′),coeff2) in repartition(f1′,f2′, Val(N))
                    d2[(f1′′,f2′′)] = get(d2, (f1′′,f2′′), zero(coeff)) + coeff2*coeff
                end
            end
            for ((f1′,f2′), coeff2) in d2
                if f1 == f1′ && f2 == f2′
                    @test coeff2 ≈ 1
                else
                    @test isapprox(coeff2, 0; atol = eps())
                end
            end
        end

        p = (randperm(2*N)...)
        p1, p2 = p[1:3], p[4:2N]
        ip = invperm(p)
        ip1, ip2 = ip[1:N], ip[N+1:2N]

        d = @inferred permute(f1, f2, p1, p2)
        d2 = Dict{typeof((f1,f2)), valtype(d)}()
        for ((f1′,f2′),coeff) in d
            for ((f1′′,f2′′),coeff2) in permute(f1′,f2′, ip1, ip2)
                d2[(f1′′,f2′′)] = get(d2, (f1′′,f2′′), zero(coeff)) + coeff2*coeff
            end
        end
        for ((f1′,f2′), coeff2) in d2
            if f1 == f1′ && f2 == f2′
                @test coeff2 ≈ 1
            else
                @test isapprox(coeff2, 0; atol = eps())
            end
        end

    end
end
end
