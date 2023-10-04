println("------------------------------------")
println("Sectors")
println("------------------------------------")

module SectorTests

include("utility.jl")

@testset "$(TensorKit.type_repr(I))" verbose = true for I in sectorlist
    @testset "Basic properties" begin
        s = (randsector(I), randsector(I), randsector(I))
        @test eval(Meta.parse(sprint(show, I))) == I
        @test eval(Meta.parse(TensorKit.type_repr(I))) == I
        @test eval(Meta.parse(sprint(show, s[1]))) == s[1]
        @test @constinferred(hash(s[1])) == hash(deepcopy(s[1]))
        @test @constinferred(one(s[1])) == @constinferred(one(I))
        @constinferred dual(s[1])
        @constinferred dim(s[1])
        @constinferred frobeniusschur(s[1])
        @constinferred Nsymbol(s...)
        BraidingStyle(I) isa NoBraiding || @constinferred Rsymbol(s...)
        @constinferred Bsymbol(s...)
        @constinferred Fsymbol(s..., s...)
        it = @constinferred s[1] ⊗ s[2]
        @constinferred ⊗(s..., s...)
    end

    @testset "Value iterator" begin
        @test eltype(values(I)) == I
        sprev = one(I)
        for (i, s) in enumerate(values(I))
            @test !isless(s, sprev) # confirm compatibility with sort order
            if Base.IteratorSize(values(I)) == Base.IsInfinite() && I <: ProductSector
                @test_throws ArgumentError values(I)[i]
                @test_throws ArgumentError TensorKit.findindex(values(I), s)
            elseif hasmethod(Base.getindex, Tuple{typeof(values(I)),Int})
                @test s == @constinferred (values(I)[i])
                @test TensorKit.findindex(values(I), s) == i
            end
            sprev = s
            i >= 10 && break
        end
        @test one(I) == first(values(I))
        if Base.IteratorSize(values(I)) == Base.IsInfinite() && I <: ProductSector
            @test_throws ArgumentError TensorKit.findindex(values(I), one(I))
        elseif hasmethod(Base.getindex, Tuple{typeof(values(I)),Int})
            @test (@constinferred TensorKit.findindex(values(I), one(I))) == 1
            for s in smallset(I)
                @test (@constinferred values(I)[TensorKit.findindex(values(I), s)]) == s
            end
        end
    end

    @testset "Unitarity of F-move" begin
        for a in smallset(I), b in smallset(I), c in smallset(I)
            for d in ⊗(a, b, c)
                es = collect(intersect(⊗(a, b), map(dual, ⊗(c, dual(d)))))
                fs = collect(intersect(⊗(b, c), map(dual, ⊗(dual(d), a))))
                if FusionStyle(I) isa MultiplicityFreeFusion
                    @test length(es) == length(fs)
                    F = [Fsymbol(a, b, c, d, e, f) for e in es, f in fs]
                else
                    Fblocks = Vector{Any}()
                    for e in es
                        for f in fs
                            Fs = Fsymbol(a, b, c, d, e, f)
                            push!(Fblocks,
                                  reshape(Fs,
                                          (size(Fs, 1) * size(Fs, 2),
                                           size(Fs, 3) * size(Fs, 4))))
                        end
                    end
                    F = hvcat(length(fs), Fblocks...)
                end
                @test isapprox(F' * F, one(F); atol=1e-12, rtol=1e-12)
            end
        end
    end

    @testset "Pentagon equation" begin
        for a in smallset(I), b in smallset(I), c in smallset(I), d in smallset(I)
            @test pentagon_equation(a, b, c, d; atol=1e-12, rtol=1e-12)
        end
    end
    if !(BraidingStyle(I) isa NoBraiding)
        @testset "Hexagon equation" begin
            for a in smallset(I), b in smallset(I), c in smallset(I)
                @test hexagon_equation(a, b, c; atol=1e-12, rtol=1e-12)
            end
        end
    end

    if hasfusiontensor(I)
        @testset "Fusion tensor and F-move" begin
            for a in smallset(I), b in smallset(I), c in smallset(I)
                for e in ⊗(a, b), f in ⊗(b, c)
                    for d in intersect(⊗(e, c), ⊗(a, f))
                        X1 = fusiontensor(a, b, e)
                        X2 = fusiontensor(e, c, d)
                        Y1 = fusiontensor(b, c, f)
                        Y2 = fusiontensor(a, f, d)
                        @tensor f1[-1, -2, -3, -4] := conj(Y2[a, f, d, -4]) *
                                                      conj(Y1[b, c, f, -3]) *
                                                      X1[a, b, e, -1] * X2[e, c, d, -2]
                        if FusionStyle(I) isa MultiplicityFreeFusion
                            f2 = fill(Fsymbol(a, b, c, d, e, f) * dim(d), (1, 1, 1, 1))
                        else
                            f2 = Fsymbol(a, b, c, d, e, f) * dim(d)
                        end
                        @test isapprox(f1, f2; atol=1e-12, rtol=1e-12)
                    end
                end
            end
        end
        if !(BraidingStyle(I) isa NoBraiding)
            @testset "Fusion tensor and R-move" begin
                for a in smallset(I), b in smallset(I)
                    for c in ⊗(a, b)
                        X1 = permutedims(fusiontensor(a, b, c), (2, 1, 3, 4))
                        X2 = fusiontensor(b, a, c)
                        l = dim(a) * dim(b) * dim(c)
                        R = LinearAlgebra.transpose(Rsymbol(a, b, c))
                        sz = (l, convert(Int, Nsymbol(a, b, c)))
                        @test reshape(X1, sz) ≈ reshape(X2, sz) * R
                    end
                end
            end
        end
    end
end

end
