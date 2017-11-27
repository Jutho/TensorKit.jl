@testset "Sectors and fusion trees for sector $G" for G in (ℤ₂, ℤ₃, ℤ₄, U₁, SU₂, ℤ₃ × ℤ₄, U₁ × SU₂, SU₂ × SU₂)
    @testset "Sector $G: Basic properties" begin
        s = (randsector(G), randsector(G), randsector(G))
        @test @inferred(one(s[1])) == @inferred(one(G))
        @inferred dual(s[1])
        @inferred dim(s[1])
        @inferred frobeniusschur(s[1])
        @inferred Nsymbol(s...)
        @inferred Rsymbol(s...)
        @inferred Bsymbol(s...)
        @inferred Fsymbol(s..., s...)
        @inferred s[1] ⊗ s[2]
        @inferred ⊗(s..., s...)
    end
    @testset "Sector $G: Pentagon equation" begin
        (a,b,c,d) = (randsector(G), randsector(G), randsector(G), randsector(G))
        for f in ⊗(a,b), j in ⊗(c,d)
            for g in ⊗(f,c), i in ⊗(b,j)
                for e in intersect(⊗(g,d), ⊗(a,i))
                    p1 = Fsymbol(f,c,d,e,g,j)*Fsymbol(a,b,j,e,f,i)
                    p2 = zero(p1)
                    for h in ⊗(b,c)
                        p2 += Fsymbol(a,b,c,g,f,h)*Fsymbol(a,h,d,e,g,i)*Fsymbol(b,c,d,i,h,j)
                    end
                    @test isapprox(p1, p2; atol=10*eps())
                end
            end
        end
    end
    @testset "Sector $G: Hexagon equation" begin
        (a,b,c) = (randsector(G), randsector(G), randsector(G))
        for e in ⊗(a,b), f in ⊗(b,c)
            for d in intersect(⊗(e,c), ⊗(a,f))
                p1 = Rsymbol(a,b,e)*Fsymbol(b,a,c,d,e,f)*Rsymbol(a,c,f)
                p2 = zero(p1)
                for h in ⊗(b,c)
                    p2 += Fsymbol(a,b,c,d,e,h)*Rsymbol(a,h,d)*Fsymbol(b,c,a,d,h,f)
                end
                @test isapprox(p1, p2; atol=10*eps())
            end
        end
    end

    @testset "Sector $G: Fusion trees" begin
        N = 3
        s = ntuple(n->randsector(G), StaticLength(N-1))
        out = (s..., dual(first(⊗(s...))))

        it = @inferred fusiontrees(out)
        state = @inferred start(it)
        f, state = @inferred next(it, state)
        @inferred done(it, state)

        @inferred braid(f, 2)

        p = tuple(randperm(N)...,)
        ip = invperm(p)

        d = @inferred TensorKit.permute(f, p)
        d2 = Dict{typeof(f), valtype(d)}()
        for (f2, coeff) in d
            for (f1,coeff2) in TensorKit.permute(f2, ip)
                d2[f1] = get(d2, f1, zero(coeff)) + coeff2*coeff
            end
        end
        for (f1, coeff2) in d2
            if f1 == f
                @test coeff2 ≈ 1
            else
                @test isapprox(coeff2, 0; atol = 10*eps())
            end
        end

        f1 = f
        f2, = first(d)
        for n = 0:2*N
            d = @inferred repartition(f1, f2, StaticLength(n))
            d2 = Dict{typeof((f1,f2)), valtype(d)}()
            for ((f1′,f2′),coeff) in d
                for ((f1′′,f2′′),coeff2) in repartition(f1′,f2′, StaticLength(N))
                    d2[(f1′′,f2′′)] = get(d2, (f1′′,f2′′), zero(coeff)) + coeff2*coeff
                end
            end
            for ((f1′,f2′), coeff2) in d2
                if f1 == f1′ && f2 == f2′
                    @test coeff2 ≈ 1
                else
                    @test isapprox(coeff2, 0; atol = 10*eps())
                end
            end
        end

        p = (randperm(2*N)...,)
        p1, p2 = p[1:3], p[4:2N]
        ip = invperm(p)
        ip1, ip2 = ip[1:N], ip[N+1:2N]

        d = @inferred TensorKit.permute(f1, f2, p1, p2)
        d2 = Dict{typeof((f1,f2)), valtype(d)}()
        for ((f1′,f2′),coeff) in d
            for ((f1′′,f2′′),coeff2) in TensorKit.permute(f1′,f2′, ip1, ip2)
                d2[(f1′′,f2′′)] = get(d2, (f1′′,f2′′), zero(coeff)) + coeff2*coeff
            end
        end
        for ((f1′,f2′), coeff2) in d2
            if f1 == f1′ && f2 == f2′
                @test coeff2 ≈ 1
            else
                @test isapprox(coeff2, 0; atol = 10*eps())
            end
        end
    end
end
