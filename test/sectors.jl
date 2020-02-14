println("------------------------------------")
println("Sectors")
println("------------------------------------")
ti = time()
@testset TimedTestSet "Properties of sector $G" for G in (ℤ₂, ℤ₃, ℤ₄, U₁, CU₁, SU₂, FibonacciAnyon, ℤ₃ × ℤ₄, U₁ × SU₂, SU₂ × SU₂, ℤ₂ × FibonacciAnyon × FibonacciAnyon)
    @testset "Sector $G: Basic properties" begin
        s = (randsector(G), randsector(G), randsector(G))
        @test eval(Meta.parse(sprint(show,G))) == G
        @test eval(Meta.parse(sprint(show,s[1]))) == s[1]
        @test @inferred(hash(s[1])) == hash(deepcopy(s[1]))
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
    @testset "Sector $G: Value iterator" begin
        @test eltype(values(G)) == G
        sprev = one(G)
        for (i, s) in enumerate(values(G))
            @test !isless(s, sprev) # confirm compatibility with sort order
            if Base.IteratorSize(values(G)) == Base.IsInfinite() && G <: ProductSector
                @test_throws ArgumentError values(G)[i]
                @test_throws ArgumentError TensorKit.findindex(values(G), s)
            else
                @test s == @inferred (values(G)[i])
                @test TensorKit.findindex(values(G), s) == i
            end
            sprev = s
            i >= 10 && break
        end
        @test one(G) == first(values(G))
        if Base.IteratorSize(values(G)) == Base.IsInfinite() && G <: ProductSector
            @test_throws ArgumentError TensorKit.findindex(values(G), one(G))
        else
            @test (@inferred TensorKit.findindex(values(G), one(G))) == 1
            for s in smallset(G)
                @test (@inferred values(G)[TensorKit.findindex(values(G), s)]) == s
            end
        end
    end
    if hasfusiontensor(G)
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
        # (a,b,c,d) = (randsector(G), randsector(G), randsector(G), randsector(G))
        for a in smallset(G), b in smallset(G), c in smallset(G), d in smallset(G)
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
        (a,b,c,d) = (randsector(G), randsector(G), randsector(G), randsector(G))
    end
    @testset "Sector $G: Hexagon equation" begin
        for a in smallset(G), b in smallset(G), c in smallset(G)
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
    end
end
tf = time()
printstyled("Finished sector tests in ",
            string(round(tf-ti; sigdigits=3)),
            " seconds."; bold = true, color = Base.info_color())
println()
