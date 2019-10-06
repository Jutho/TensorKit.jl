@testset "Fusion trees for sector $G" for G in (ℤ₂, ℤ₃, ℤ₄, U₁, CU₁, SU₂, FibonacciAnyon, ℤ₃ × ℤ₄, U₁ × SU₂, SU₂ × SU₂, ℤ₂ × FibonacciAnyon × FibonacciAnyon)
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
    f = @inferred first(it)
    @testset "Fusion tree $G: printing" begin
        @test eval(Meta.parse(sprint(show,f))) == f
    end
    @testset "Fusion tree $G: braiding" begin
        for in = ⊗(out...)
            for f in fusiontrees(out, in)
                d1 = @inferred artin_braid(f, 2)
                d2 = empty(d1)
                for (f1, coeff1) in d1
                    for (f2,coeff2) in artin_braid(f1, 2; inv = true)
                        d2[f2] = get(d2, f2, zero(coeff1)) + coeff2*coeff1
                    end
                end
                for (f2, coeff2) in d2
                    if f2 == f
                        @test coeff2 ≈ 1
                    else
                        @test isapprox(coeff2, 0; atol = 10*eps())
                    end
                end
            end
        end

        f = rand(collect(it))
        d1 = artin_braid(f, 2)
        d2 = empty(d1)
        for (f1, coeff1) in d1
            for (f2,coeff2) in artin_braid(f1, 3)
                d2[f2] = get(d2, f2, zero(coeff1)) + coeff2*coeff1
            end
        end
        d1 = d2
        d2 = empty(d1)
        for (f1, coeff1) in d1
            for (f2,coeff2) in artin_braid(f1, 3; inv = true)
                d2[f2] = get(d2, f2, zero(coeff1)) + coeff2*coeff1
            end
        end
        d1 = d2
        d2 = empty(d1)
        for (f1, coeff1) in d1
            for (f2,coeff2) in artin_braid(f1, 2; inv = true)
                d2[f2] = get(d2, f2, zero(coeff1)) + coeff2*coeff1
            end
        end
        d1 = d2
        for (f1, coeff1) in d1
            if f1 == f
                @test coeff1 ≈ 1
            else
                @test isapprox(coeff1, 0; atol = 10*eps())
            end
        end
    end
    @testset "Fusion tree $G: braiding and permuting" begin
        p = tuple(randperm(N)...,)
        ip = invperm(p)

        levels = ntuple(identity, N)
        d = @inferred braid(f, levels, p)
        d2 = Dict{typeof(f), valtype(d)}()
        levels2 = p
        for (f2, coeff) in d
            for (f1,coeff2) in braid(f2, levels2, ip)
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

        if (BraidingStyle(G) isa Bosonic) && hasfusiontensor(G)
            Af = convert(Array, f)
            Afp = permutedims(Af, (p..., N+1))
            Afp2 = zero(Afp)
            for (f1, coeff) in d
                Afp2 .+= coeff .* convert(Array, f1)
            end
            @test Afp ≈ Afp2
        end
    end
    @testset "Fusion tree $G: insertat" begin
        N = 4
        out2 = ntuple(n->randsector(G), StaticLength(N))
        in2 = rand(collect(⊗(out2...)))
        f2 = rand(collect(fusiontrees(out2, in2)))
        for i = 1:N
            out1 = ntuple(n->randsector(G), StaticLength(N))
            out1 = Base.setindex(out1, in2, i)
            in1 = rand(collect(⊗(out1...)))
            f1 = rand(collect(fusiontrees(out1, in1)))

            trees = @inferred insertat(f1, i, f2)
            @test norm(values(trees)) ≈ 1

            f1a, f1b = split(f1, StaticLength(i))
            @test length(insertat(f1b, 1, f1a)) == 1
            @test first(insertat(f1b, 1, f1a)) == (f1 => 1)

            levels = ntuple(identity, N)
            gen = Base.Generator(braid(f1, levels, (i, (1:i-1)..., (i+1:N)...))) do (t, c)
                (t′, c′) = first(insertat(t, 1, f2))
                @test c′ == one(c′)
                return t′ => c
            end
            trees2 = Dict(gen)
            trees3 = empty(trees2)
            p = ((N+1:N+i-1)..., (1:N)..., (N+i:2N-1)...)
            levels = ((i:N+i-1)..., (1:i-1)..., (i+N:2N-1)...)
            for (t,coeff) in trees2
                for (t′,coeff′) in braid(t, levels, p)
                    trees3[t′] = get(trees3, t′, zero(coeff′)) + coeff*coeff′
                end
            end
            for (t, coeff) in trees3
                @test get(trees, t, zero(coeff)) ≈ coeff atol = 1e-12
            end

            if (BraidingStyle(G) isa Bosonic) && hasfusiontensor(G)
                Af1 = convert(Array, f1)
                Af2 = convert(Array, f2)
                Af = TensorOperations.tensorcontract(Af1, [1:i-1; -1; N-1 .+ (i+1:N+1)],
                                                     Af2, [i-1 .+ (1:N); -1], 1:2N)
                Af′ = zero(Af)
                for (f, coeff) in trees
                    Af′ .+= coeff .* convert(Array, f)
                end
                @test Af ≈ Af′
            end
        end
    end
    @testset "Fusion tree $G: merging" begin
        N = 3
        out1 = ntuple(n->randsector(G), StaticLength(N-1))
        in1 = rand(collect(⊗(out1...)))
        f1 = rand(collect(fusiontrees((out1..., dual(in1)), one(in1))))
        out2 = ntuple(n->randsector(G), StaticLength(N))
        in2 = rand(collect(⊗(out2...)))
        f2 = rand(collect(fusiontrees(out2, in2)))
        trees1 = @inferred merge(f1, f2)
        @test sum(abs2(c)*dim(f.coupled) for (f,c) in trees1) ≈
                dim(f1.coupled)*dim(f2.coupled)

        # test merge and braid interplay
        trees2 = merge(f2, f1)
        perm = ( (N.+(1:N))... , (1:N)...)
        levels = ntuple(identity, 2*N)
        trees3 = Dict{keytype(trees1), complex(valtype(trees1))}()
        for (t, coeff) in trees1
            for (t′, coeff′) in braid(t, levels, perm)
                trees3[t′] = get(trees3, t′, zero(coeff′)) + coeff*coeff′
            end
        end
        for (t, coeff) in trees3
            @test isapprox(coeff, get(trees2, t, zero(coeff)); atol = 10*eps())
        end

        if (BraidingStyle(G) isa Bosonic) && hasfusiontensor(G)
            Af1 = convert(Array, f1)
            Af2 = convert(Array, f2)
            for c in f1.coupled ⊗ f2.coupled
                Af0 = convert(Array, FusionTree((f1.coupled, f2.coupled), c, ()))
                _Af = TensorOperations.tensorcontract(Af1, [1:N;-1],
                                                        Af0, [-1;N+1;N+2], 1:N+2)
                Af = TensorOperations.tensorcontract(Af2, [N .+ (1:N); -1],
                                                        _Af, [1:N; -1; 2N+1], 1:2N+1)
                Af′ = zero(Af)
                for (f, coeff) in trees1
                    if f.coupled == c
                        Af′ .+= coeff .* convert(Array, f)
                    end
                end
                @test Af ≈ Af′
            end
        end
    end

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

    @testset "Double fusion tree $G: repartioning" begin
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
    end
    @testset "Double fusion tree $G: permutation" begin
        if BraidingStyle(G) isa SymmetricBraiding
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
end
