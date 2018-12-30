@testset "Sectors and fusion trees for sector $G" for G in (ℤ₂, ℤ₃, ℤ₄, U₁, CU₁, SU₂, ℤ₃ × ℤ₄, U₁ × SU₂, SU₂ × SU₂)
    @testset "Sector $G: Basic properties" begin
        s = (randsector(G), randsector(G), randsector(G))
        @test eval(Meta.parse(sprint(show,G))) == G
        @test eval(Meta.parse(sprint(show,s[1]))) == s[1]
        @test @inferred(one(s[1])) == @inferred(one(G))
        @inferred dual(s[1])
        @inferred dim(s[1])
        @inferred frobeniusschur(s[1])
        @inferred Nsymbol(s...)
        @inferred Rsymbol(s...)
        @inferred Bsymbol(s...)
        @inferred Fsymbol(s..., s...)
        it = @inferred s[1] ⊗ s[2]
        @inferred ⊗(s..., s...)
    end
    if hasmethod(fusiontensor, Tuple{G,G,G})
        @testset "Sector $G: fusion tensor and F-move and R-move" begin
            using TensorKit: fusiontensor
            for a in smallset(G), b in smallset(G)
                for c in ⊗(a,b)
                    @test permutedims(fusiontensor(a,b,c),(2,1,3)) ≈ Rsymbol(a,b,c)*fusiontensor(b,a,c)
                end
            end
            for a in smallset(G), b in smallset(G), c in smallset(G)
                for e in ⊗(a,b), f in ⊗(b,c)
                    for d in intersect(⊗(e,c), ⊗(a,f))
                        X1 = fusiontensor(a,b,e)
                        X2 = fusiontensor(e,c,d)
                        Y1 = fusiontensor(b,c,f)
                        Y2 = fusiontensor(a,f,d)
                        @tensor f1 = conj(Y2[a,f,d])*conj(Y1[b,c,f])*X1[a,b,e]*X2[e,c,d]
                        f2 = Fsymbol(a,b,c,d,e,f)*dim(d)
                        @test f1≈f2 atol=1e-12
                    end
                end
            end
        end
    end
    @testset "Sector $G: Unitarity of F-move" begin
        for a in smallset(G), b in smallset(G), c in smallset(G)
            for d in ⊗(a,b,c)
                es = collect(intersect(⊗(a,b), map(dual, ⊗(c,dual(d)))))
                fs = collect(intersect(⊗(b,c), map(dual, ⊗(dual(d),a))))
                @test length(es) == length(fs)
                F = [Fsymbol(a,b,c,d,e,f) for e in es, f in fs]
                @test F'*F ≈ one(F)
            end
        end
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
        N = 5
        out = ntuple(n->randsector(G), StaticLength(N))
        in = rand(collect(⊗(out...)))
        numtrees = count(n->true, fusiontrees(out, in))
        while !(0 < numtrees < 30)
            out = ntuple(n->randsector(G), StaticLength(N))
            in = rand(collect(⊗(out...)))
            numtrees = count(n->true, fusiontrees(out, in))
        end

        it = @inferred fusiontrees(out, in)
        f, state = iterate(it)

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

        if hasmethod(fusiontensor, Tuple{G,G,G})
            Af = convert(Array, f)
            Afp = permutedims(Af, (p..., N+1))
            Afp2 = zero(Afp)
            for (f1, coeff) in d
                Afp2 .+= coeff .* convert(Array, f1)
            end
            @test Afp ≈ Afp2
        end
    end
    @testset "Sector $G: Double fusion trees" begin
        if G <: ProductSector
            N = 3
        else
            N = 4
        end
        out = ntuple(n->randsector(G), StaticLength(N))
        numtrees = count(n->true, fusiontrees((out..., map(dual, out)...)))
        while !(0 < numtrees < 100)
            out = ntuple(n->randsector(G), StaticLength(N))
            numtrees = count(n->true, fusiontrees((out..., map(dual, out)...)))
        end
        in = rand(collect(⊗(out...)))
        f1 = rand(collect(fusiontrees(out, in)))
        f2 = rand(collect(fusiontrees(out[randperm(N)], in)))

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
                    if !(coeff2 ≈ 1)
                        @show f1, f2, n
                    end
                else
                    @test isapprox(coeff2, 0; atol = 10*eps())
                end
            end
        end

        p = (randperm(2*N)...,)
        p1, p2 = p[1:2], p[3:2N]
        ip = invperm(p)
        ip1, ip2 = ip[1:N], ip[N+1:2N]

        d = @inferred TensorKit.permute(f1, f2, p1, p2)
        d2 = Dict{typeof((f1,f2)), valtype(d)}()
        for ((f1′,f2′), coeff) in d
            d′ = TensorKit.permute(f1′,f2′, ip1, ip2)
            for ((f1′′,f2′′), coeff2) in d′
                d2[(f1′′,f2′′)] = get(d2, (f1′′,f2′′), zero(coeff)) + coeff2*coeff
            end
        end
        for ((f1′,f2′), coeff2) in d2
            if f1 == f1′ && f2 == f2′
                @test coeff2 ≈ 1
                if !(coeff2 ≈ 1)
                    @show f1, f2, p
                end
            else
                @test abs(coeff2) < 10*eps()
            end
        end
    end
end
