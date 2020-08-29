println("------------------------------------")
println("Sectors")
println("------------------------------------")
ti = time()
@testset TimedTestSet "Properties of sector $I" for I in sectorlist
    @testset "Sector $I: Basic properties" begin
        s = (randsector(I), randsector(I), randsector(I))
        @test eval(Meta.parse(sprint(show,I))) == I
        @test eval(Meta.parse(sprint(show,s[1]))) == s[1]
        @test @constinferred(hash(s[1])) == hash(deepcopy(s[1]))
        @test @constinferred(one(s[1])) == @constinferred(one(I))
        @constinferred dual(s[1])
        @constinferred dim(s[1])
        @constinferred frobeniusschur(s[1])
        @constinferred Nsymbol(s...)
        @constinferred Rsymbol(s...)
        @constinferred Bsymbol(s...)
        @constinferred Fsymbol(s..., s...)
        it = @constinferred s[1] ⊗ s[2]
        @constinferred ⊗(s..., s...)
    end
    @testset "Sector $I: Value iterator" begin
        @test eltype(values(I)) == I
        sprev = one(I)
        for (i, s) in enumerate(values(I))
            @test !isless(s, sprev) # confirm compatibility with sort order
            if Base.IteratorSize(values(I)) == Base.IsInfinite() && I <: ProductSector
                @test_throws ArgumentError values(I)[i]
                @test_throws ArgumentError TensorKit.findindex(values(I), s)
            else
                @test s == @constinferred (values(I)[i])
                @test TensorKit.findindex(values(I), s) == i
            end
            sprev = s
            i >= 10 && break
        end
        @test one(I) == first(values(I))
        if Base.IteratorSize(values(I)) == Base.IsInfinite() && I <: ProductSector
            @test_throws ArgumentError TensorKit.findindex(values(I), one(I))
        else
            @test (@constinferred TensorKit.findindex(values(I), one(I))) == 1
            for s in smallset(I)
                @test (@constinferred values(I)[TensorKit.findindex(values(I), s)]) == s
            end
        end
    end
    if hasfusiontensor(I)
        @testset "Sector $I: fusion tensor and F-move and R-move" begin
            using TensorKit: fusiontensor
            for a in smallset(I), b in smallset(I)
                for c in ⊗(a,b)
                    @test permutedims(fusiontensor(a,b,c),(2,1,3)) ≈ Rsymbol(a,b,c)*fusiontensor(b,a,c)
                end
            end
            for a in smallset(I), b in smallset(I), c in smallset(I)
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
    @testset "Sector $I: Unitarity of F-move" begin
        for a in smallset(I), b in smallset(I), c in smallset(I)
            for d in ⊗(a,b,c)
                es = collect(intersect(⊗(a,b), map(dual, ⊗(c,dual(d)))))
                fs = collect(intersect(⊗(b,c), map(dual, ⊗(dual(d),a))))
                @test length(es) == length(fs)
                F = [Fsymbol(a,b,c,d,e,f) for e in es, f in fs]
                @test F'*F ≈ one(F)
            end
        end
    end
    @testset "Sector $I: Pentagon equation" begin
        for a in smallset(I), b in smallset(I), c in smallset(I), d in smallset(I)
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
    end
    @testset "Sector $I: Hexagon equation" begin
        for a in smallset(I), b in smallset(I), c in smallset(I)
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
