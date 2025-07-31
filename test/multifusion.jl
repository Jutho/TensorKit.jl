I = IsingBimod
Istr = TensorKit.type_repr(I)

println("------------------------------------")
println("Multifusion tests for $Istr")
println("------------------------------------")
ti = time()

@timedtestset "Multifusion spaces " verbose = true begin
    @timedtestset "GradedSpace: $(TensorKit.type_repr(Vect[I]))" begin
        gen = (values(I)[k] => (k + 1) for k in 1:length(values(I)))

        V = GradedSpace(gen)
        @test eval(Meta.parse(TensorKit.type_repr(typeof(V)))) == typeof(V)
        @test eval(Meta.parse(sprint(show, V))) == V
        @test eval(Meta.parse(sprint(show, V'))) == V'
        @test V' == GradedSpace(gen; dual = true)
        @test V == @constinferred GradedSpace(gen...)
        @test V' == @constinferred GradedSpace(gen...; dual = true)
        @test V == @constinferred GradedSpace(tuple(gen...))
        @test V' == @constinferred GradedSpace(tuple(gen...); dual = true)
        @test V == @constinferred GradedSpace(Dict(gen))
        @test V' == @constinferred GradedSpace(Dict(gen); dual = true)
        @test V == @inferred Vect[I](gen)
        @test V' == @constinferred Vect[I](gen; dual = true)
        @test V == @constinferred Vect[I](gen...)
        @test V' == @constinferred Vect[I](gen...; dual = true)
        @test V == @constinferred Vect[I](Dict(gen))
        @test V' == @constinferred Vect[I](Dict(gen); dual = true)
        @test V == @constinferred typeof(V)(c => dim(V, c) for c in sectors(V))
        @test @constinferred(hash(V)) == hash(deepcopy(V)) != hash(V')
        @test V == GradedSpace(reverse(collect(gen))...)
        @test eval(Meta.parse(sprint(show, V))) == V
        @test eval(Meta.parse(sprint(show, typeof(V)))) == typeof(V)

        # space with a single sector
        Wleft = @constinferred Vect[I](I(1, 1, 0) => 1, I(1, 1, 1) => 1)
        Wright = @constinferred Vect[I](I(2, 2, 0) => 1, I(2, 2, 1) => 1)
        WM = @constinferred Vect[I](I(1, 2, 0) => 1)
        WMop = @constinferred Vect[I](I(2, 1, 0) => 1)

        @test @constinferred(oneunit(Wleft)) == leftoneunit(Wleft) == rightoneunit(Wleft)
        @test @constinferred(oneunit(Wright)) == leftoneunit(Wright) == rightoneunit(Wright)
        @test @constinferred(leftoneunit(⊕(Wleft, WM))) == oneunit(Wleft)
        @test @constinferred(leftoneunit(⊕(Wright, WMop))) == oneunit(Wright)
        @test @constinferred(rightoneunit(⊕(Wright, WM))) == oneunit(Wright)
        @test @constinferred(rightoneunit(⊕(Wleft, WMop))) == oneunit(Wleft)

        @test_throws ArgumentError oneunit(I)
        @test_throws ArgumentError oneunit(WM)
        @test_throws ArgumentError oneunit(WMop)

        @test isa(V, VectorSpace)
        @test isa(V, ElementarySpace)
        @test isa(InnerProductStyle(V), HasInnerProduct)
        @test isa(InnerProductStyle(V), EuclideanInnerProduct)
        @test isa(V, GradedSpace)
        @test isa(V, GradedSpace{I})
        @test @constinferred(dual(V)) == @constinferred(conj(V)) ==
            @constinferred(adjoint(V)) != V
        @test @constinferred(field(V)) == ℂ
        @test @constinferred(sectortype(V)) == I
        slist = @constinferred sectors(V)
        @test @constinferred(hassector(V, first(slist)))
        @test @constinferred(dim(V)) == sum(dim(s) * dim(V, s) for s in slist)
        @test @constinferred(reduceddim(V)) == sum(dim(V, s) for s in slist)
        @constinferred dim(V, first(slist))

        @test @constinferred(⊕(V, zero(V))) == V
        @test @constinferred(⊕(V, V)) == Vect[I](c => 2dim(V, c) for c in sectors(V))
        @test @constinferred(⊕(V, V, V, V)) == Vect[I](c => 4dim(V, c) for c in sectors(V))

        for W in [Wleft, Wright]
            @test @constinferred(⊕(W, oneunit(W))) ==
                Vect[I](c => isone(c) + dim(W, c) for c in sectors(W))
            @test @constinferred(fuse(W, oneunit(W))) == W
        end

        # sensible direct sums and fuses
        @test @constinferred(⊕(Wleft, WM)) ==
            Vect[I](c => 1 for c in sectors(V) if leftone(c) == I(1, 1, 0))
        @test @constinferred(⊕(Wright, WMop)) ==
            Vect[I](c => 1 for c in sectors(V) if leftone(c) == I(2, 2, 0))
        @test @constinferred(⊕(Wright, WM)) ==
            Vect[I](c => 1 for c in sectors(V) if rightone(c) == I(2, 2, 0))
        @test @constinferred(⊕(Wleft, WMop)) ==
            Vect[I](c => 1 for c in sectors(V) if rightone(c) == I(1, 1, 0))
        @test @constinferred(fuse(Wleft, WM)) == Vect[I](I(1, 2, 0) => 2)
        @test @constinferred(fuse(Wright, WMop)) == Vect[I](I(2, 1, 0) => 2)

        # less sensible direct sums and fuses
        @test @constinferred(⊕(Wleft, Wright)) ==
            Vect[I](c => 1 for c in sectors(V) if leftone(c) == rightone(c))
        @test @constinferred(fuse(Wleft, WMop)) == fuse(Wright, WM) ==
            Vect[I](c => 0 for c in sectors(V))

        d = Dict{I, Int}()
        for a in sectors(V), b in sectors(V)
            for c in a ⊗ b
                d[c] = get(d, c, 0) + dim(V, a) * dim(V, b) * Nsymbol(a, b, c)
            end
        end
        @test @constinferred(fuse(V, V)) == GradedSpace(d)
        @test @constinferred(flip(V)) ==
            Vect[I](conj(c) => dim(V, c) for c in sectors(V))'
        @test flip(V) ≅ V
        @test flip(V) ≾ V
        @test flip(V) ≿ V
        @test @constinferred(⊕(V, V)) == @constinferred supremum(V, ⊕(V, V))
        @test V == @constinferred infimum(V, ⊕(V, V))
        @test V ≺ ⊕(V, V)
        @test !(V ≻ ⊕(V, V))
        @test infimum(V, GradedSpace(I(1, 1, 0) => 3)) ==
            GradedSpace(I(1, 1, 0) => 2)
        @test infimum(V, GradedSpace(I(1, 2, 0) => 6)) ==
            GradedSpace(I(1, 2, 0) => 5)
        for W in [WM, WMop, Wright]
            @test infimum(Wleft, W) == Vect[I](c => 0 for c in sectors(V))
        end
        @test_throws SpaceMismatch (⊕(V, V'))
    end

    VIB1 = (
        Vect[I](I(1, 1, 0) => 1, I(1, 1, 1) => 1),
        Vect[I](I(1, 1, 0) => 1, I(1, 1, 1) => 2),
        Vect[I](I(1, 1, 0) => 3, I(1, 1, 1) => 2),
        Vect[I](I(1, 1, 0) => 2, I(1, 1, 1) => 3),
        Vect[I](I(1, 1, 0) => 2, I(1, 1, 1) => 5),
    )

    VIB2 = (
        Vect[I](I(2, 2, 0) => 1, I(2, 2, 1) => 1),
        Vect[I](I(2, 2, 0) => 1, I(2, 2, 1) => 2),
        Vect[I](I(2, 2, 0) => 3, I(2, 2, 1) => 2),
        Vect[I](I(2, 2, 0) => 2, I(2, 2, 1) => 3),
        Vect[I](I(2, 2, 0) => 2, I(2, 2, 1) => 5),
    )

    @timedtestset "HomSpace with $(TensorKit.type_repr(Vect[I])) " begin
        for (V1, V2, V3, V4, V5) in (VIB1, VIB2) #TODO: examples with module spaces
            W = HomSpace(V1 ⊗ V2, V3 ⊗ V4 ⊗ V5)
            @test W == (V3 ⊗ V4 ⊗ V5 → V1 ⊗ V2)
            @test W == (V1 ⊗ V2 ← V3 ⊗ V4 ⊗ V5)
            @test W' == (V1 ⊗ V2 → V3 ⊗ V4 ⊗ V5)
            @test eval(Meta.parse(sprint(show, W))) == W
            @test eval(Meta.parse(sprint(show, typeof(W)))) == typeof(W)
            @test spacetype(W) == typeof(V1)
            @test sectortype(W) == sectortype(V1)
            @test W[1] == V1
            @test W[2] == V2
            @test W[3] == V3'
            @test W[4] == V4'
            @test W[5] == V5'

            @test @constinferred(hash(W)) == hash(deepcopy(W)) != hash(W')
            @test W == deepcopy(W)
            @test W == @constinferred permute(W, ((1, 2), (3, 4, 5)))
            @test permute(W, ((2, 4, 5), (3, 1))) == (V2 ⊗ V4' ⊗ V5' ← V3 ⊗ V1')
            @test (V1 ⊗ V2 ← V1 ⊗ V2) == @constinferred TensorKit.compose(W, W')

            @test_throws ErrorException insertleftunit(W)
            @test insertrightunit(W) == (V1 ⊗ V2 ← V3 ⊗ V4 ⊗ V5 ⊗ oneunit(V1))
            @test_throws ErrorException insertrightunit(W, 6)
            @test_throws ErrorException insertleftunit(W, 6)

            @test (V1 ⊗ V2 ⊗ oneunit(V1) ← V3 ⊗ V4 ⊗ V5) ==
                @constinferred(insertrightunit(W, 2))
            @test (V1 ⊗ V2 ← oneunit(V1) ⊗ V3 ⊗ V4 ⊗ V5) ==
                @constinferred(insertleftunit(W, 3))
            @test @constinferred(removeunit(insertleftunit(W, 3), 3)) == W
            @test_throws ErrorException @constinferred(insertrightunit(one(V1) ← V1, 0)) # should I specify it's the other error?
            @test_throws ErrorException insertleftunit(one(V1) ← V1, 0)
        end
    end
end

@timedtestset "Fusion trees for $(TensorKit.type_repr(I))" verbose = true begin
    N = 6
    C0, C1, D0, D1, M, Mop = I(1, 1, 0), I(1, 1, 1), I(2, 2, 0), I(2, 2, 1), I(1, 2, 0), I(2, 1, 0)
    out = (Mop, C0, C1, M, D0, D1) # should I try to make a non-hardcoded example?
    isdual = ntuple(n -> rand(Bool), N)
    in = rand(collect(⊗(out...))) # will be D0 or D1 in this choice of out
    numtrees = length(fusiontrees(out, in, isdual)) # will be 1
    @test numtrees == count(n -> true, fusiontrees(out, in, isdual))

    it = @constinferred fusiontrees(out, in, isdual)
    @constinferred Nothing iterate(it)
    f, s = iterate(it)
    @constinferred Nothing iterate(it, s)
    @test f == @constinferred first(it)
    @testset "Fusion tree $Istr: printing" begin
        @test eval(Meta.parse(sprint(show, f))) == f
    end
    @testset "Fusion tree $Istr: constructor properties" for u in (C0, D0)
        @constinferred FusionTree((), u, (), (), ())
        @constinferred FusionTree((u,), u, (false,), (), ())
        @constinferred FusionTree((u, u), u, (false, false), (), (1,))
        @constinferred FusionTree((u, u, u), u, (false, false, false), (u,), (1, 1))
        @constinferred FusionTree(
            (u, u, u, u), u, (false, false, false, false), (u, u),
            (1, 1, 1)
        )
        @test_throws MethodError FusionTree((u, u, u), u, (false, false), (u,), (1, 1))
        @test_throws MethodError FusionTree(
            (u, u, u), u, (false, false, false), (u, u),
            (1, 1)
        )
        @test_throws MethodError FusionTree(
            (u, u, u), u, (false, false, false), (u,),
            (1, 1, 1)
        )
        @test_throws MethodError FusionTree((u, u, u), u, (false, false, false), (), (1,))

        f = FusionTree((u, u, u), u, (false, false, false), (u,), (1, 1))
        @test sectortype(f) == I
        @test length(f) == 3
        @test FusionStyle(f) == FusionStyle(I)
        @test BraidingStyle(f) == BraidingStyle(I)

        # SimpleFusion
        errstr = "fusion tree requires inner lines if `FusionStyle(I) <: MultipleFusion`"
        @test_throws errstr FusionTree((), u, ())
        @test_throws errstr FusionTree((u,), u, (false,))
        @test_throws errstr FusionTree((u, u), u, (false, false))
        @test_throws errstr FusionTree((u, u, u), u)
        @test_throws errstr FusionTree((u, u, u, u)) # custom FusionTree constructor required here
    end

    # CONTINUE HERE

    @testset "Fusion tree $Istr: insertat" begin
        N = 4
        in2 = nothing # attempt at not hard-coding
        out2 = nothing
        while in2 === nothing
            out2 = ntuple(n -> randsector(I), N)
            try
                in2 = rand(collect(⊗(out2...)))
            catch e
                if isa(e, ArgumentError)
                    in2 = nothing
                else
                    rethrow(e)
                end
            end
        end
        isdual2 = ntuple(n -> rand(Bool), N)
        f2 = rand(collect(fusiontrees(out2, in2, isdual2)))
        for i in 1:N
            in1 = nothing
            out1 = nothing
            while in1 === nothing
                try
                    out1 = ntuple(n -> randsector(I), N)
                    out1 = Base.setindex(out1, in2, i)
                    in1 = rand(collect(⊗(out1...)))
                catch e
                    if isa(e, ArgumentError)
                        in1 = nothing
                    else
                        rethrow(e)
                    end
                end
            end
            isdual1 = ntuple(n -> rand(Bool), N)
            isdual1 = Base.setindex(isdual1, false, i)
            f1 = rand(collect(fusiontrees(out1, in1, isdual1)))

            trees = @constinferred TK.insertat(f1, i, f2)
            @test norm(values(trees)) ≈ 1

            f1a, f1b = @constinferred TK.split(f1, $i)
            @test length(TK.insertat(f1b, 1, f1a)) == 1
            @test first(TK.insertat(f1b, 1, f1a)) == (f1 => 1)

            # no braid tests for non-hardcoded example
        end
    end
    # no planar trace tests
    @testset "Fusion tree $Istr: elementary artin braid" begin
        N = length(out)
        isdual = ntuple(n -> rand(Bool), N)
        # no general artin braid test

        # not sure how useful this test is, it does the trivial braiding
        f = rand(collect(it)) # in this case the 1 tree
        d1 = TK.artin_braid(f, 2) # takes a unit C0
        d2 = empty(d1)
        for (f1, coeff1) in d1
            for (f2, coeff2) in TK.artin_braid(f1, 3)
                d2[f2] = get(d2, f2, zero(coeff1)) + coeff2 * coeff1
            end
        end
        d1 = d2
        d2 = empty(d1)
        for (f1, coeff1) in d1
            for (f2, coeff2) in TK.artin_braid(f1, 3; inv = true)
                d2[f2] = get(d2, f2, zero(coeff1)) + coeff2 * coeff1
            end
        end
        d1 = d2
        d2 = empty(d1)
        for (f1, coeff1) in d1
            for (f2, coeff2) in TK.artin_braid(f1, 2; inv = true)
                d2[f2] = get(d2, f2, zero(coeff1)) + coeff2 * coeff1
            end
        end
        d1 = d2
        for (f1, coeff1) in d1
            if f1 == f
                @test coeff1 ≈ 1
            else
                @test isapprox(coeff1, 0; atol = 1.0e-12, rtol = 1.0e-12)
            end
        end
    end
    # no braiding and permuting test
    @testset "Fusion tree $Istr: merging" begin
        N = 3
        out1, in1, out2, in2 = nothing, nothing, nothing, nothing
        while (in1 === nothing && in2 === nothing) || isempty(in1 ⊗ in2)
            try
                out1 = ntuple(n -> randsector(I), N)
                in1 = rand(collect(⊗(out1...)))
                out2 = ntuple(n -> randsector(I), N)
                in2 = rand(collect(⊗(out2...)))
            catch e
                if isa(e, ArgumentError)
                    in1, in2 = nothing, nothing
                else
                    rethrow(e)
                end
            end
        end

        f1 = rand(collect(fusiontrees(out1, in1)))
        f2 = rand(collect(fusiontrees(out2, in2)))

        @constinferred TK.merge(f1, f2, first(in1 ⊗ in2), 1)
        @constinferred TK.merge(f1, f2, first(in1 ⊗ in2))

        @test dim(in1) * dim(in2) ≈ sum(
            abs2(coeff) * dim(c) for c in in1 ⊗ in2
                for μ in 1:Nsymbol(in1, in2, c)
                for (f, coeff) in TK.merge(f1, f2, c, μ)
        )
        # no merge and braid interplay tests
    end

    # hardcoded double fusion tree tests
    N = 6
    out = (Mop, C0, C1, M, D0, D1) # same as above
    out2 = (D0, D1, Mop, C0, C1, M) # different order that still fuses to D0 or D1
    incoming = rand(collect(⊗(out...))) # will be D0 or D1
    f1 = rand(collect(fusiontrees(out, incoming, ntuple(n -> rand(Bool), N))))
    f2 = rand(collect(fusiontrees(out2, incoming, ntuple(n -> rand(Bool), N))))

    @testset "Double fusion tree $Istr: repartioning" begin
        for n in 0:(2 * N)
            d = @constinferred TK.repartition(f1, f2, $n)
            @test dim(incoming) ≈
                sum(abs2(coef) * dim(f1.coupled) for ((f1, f2), coef) in d)
            d2 = Dict{typeof((f1, f2)), valtype(d)}()
            for ((f1′, f2′), coeff) in d
                for ((f1′′, f2′′), coeff2) in TK.repartition(f1′, f2′, N)
                    d2[(f1′′, f2′′)] = get(d2, (f1′′, f2′′), zero(coeff)) + coeff2 * coeff
                end
            end
            for ((f1′, f2′), coeff2) in d2
                if f1 == f1′ && f2 == f2′
                    @test coeff2 ≈ 1
                else
                    @test isapprox(coeff2, 0; atol = 1.0e-12, rtol = 1.0e-12)
                end
            end
        end
    end
    # no double fusion tree permutation tests
    @testset "Double fusion tree $Istr: transposition" begin
        for n in 0:(2N)
            i0 = rand(1:(2N))
            p = mod1.(i0 .+ (1:(2N)), 2N)
            ip = mod1.(-i0 .+ (1:(2N)), 2N)
            p′ = tuple(getindex.(Ref(vcat(1:N, (2N):-1:(N + 1))), p)...)
            p1, p2 = p′[1:n], p′[(2N):-1:(n + 1)]
            ip′ = tuple(getindex.(Ref(vcat(1:n, (2N):-1:(n + 1))), ip)...)
            ip1, ip2 = ip′[1:N], ip′[(2N):-1:(N + 1)]

            d = @constinferred transpose(f1, f2, p1, p2)
            @test dim(incoming) ≈
                sum(abs2(coef) * dim(f1.coupled) for ((f1, f2), coef) in d)
            d2 = Dict{typeof((f1, f2)), valtype(d)}()
            for ((f1′, f2′), coeff) in d
                d′ = transpose(f1′, f2′, ip1, ip2)
                for ((f1′′, f2′′), coeff2) in d′
                    d2[(f1′′, f2′′)] = get(d2, (f1′′, f2′′), zero(coeff)) + coeff2 * coeff
                end
            end
            for ((f1′, f2′), coeff2) in d2
                if f1 == f1′ && f2 == f2′
                    @test coeff2 ≈ 1
                else
                    @test abs(coeff2) < 1.0e-12
                end
            end
        end
    end
    @testset "Double fusion tree $Istr: planar trace" begin
        d1 = transpose(f1, f1, (N + 1, 1:N..., ((2N):-1:(N + 3))...), (N + 2,))
        f1front, = TK.split(f1, N - 1)
        T = sectorscalartype(I)
        d2 = Dict{typeof((f1front, f1front)), T}()
        for ((f1′, f2′), coeff′) in d1
            for ((f1′′, f2′′), coeff′′) in
                TK.planar_trace(
                    f1′, f2′, (2:N...,), (1, ((2N):-1:(N + 3))...), (N + 1,),
                    (N + 2,)
                )
                coeff = coeff′ * coeff′′
                d2[(f1′′, f2′′)] = get(d2, (f1′′, f2′′), zero(coeff)) + coeff
            end
        end
        for ((f1_, f2_), coeff) in d2
            if (f1_, f2_) == (f1front, f1front)
                @test coeff ≈ dim(f1.coupled) / dim(f1front.coupled)
            else
                @test abs(coeff) < 1.0e-12
            end
        end
    end
end

V = Vect[I](values(I)[k] => 1 for k in 1:length(values(I)))

@timedtestset "DiagonalTensor with domain $V" begin
    @timedtestset "Basic properties and algebra" begin
        for T in (Float32, Float64, ComplexF32, ComplexF64, BigFloat)
            # constructors
            t = @constinferred DiagonalTensorMap{T}(undef, V)
            t = @constinferred DiagonalTensorMap(rand(T, reduceddim(V)), V)
            t2 = @constinferred DiagonalTensorMap{T}(undef, space(t))
            @test space(t2) == space(t)
            @test_throws ArgumentError DiagonalTensorMap{T}(undef, V^2 ← V)
            t2 = @constinferred DiagonalTensorMap{T}(undef, domain(t))
            @test space(t2) == space(t)
            @test_throws ArgumentError DiagonalTensorMap{T}(undef, V^2)
            # properties
            @test @constinferred(hash(t)) == hash(deepcopy(t))
            @test scalartype(t) == T
            @test codomain(t) == ProductSpace(V)
            @test domain(t) == ProductSpace(V)
            @test space(t) == (V ← V)
            @test space(t') == (V ← V)
            @test dim(t) == dim(space(t))
            # blocks
            bs = @constinferred blocks(t)
            (c, b1), state = @constinferred Nothing iterate(bs)
            @test c == first(blocksectors(V ← V))
            next = @constinferred Nothing iterate(bs, state)
            b2 = @constinferred block(t, first(blocksectors(t)))
            @test b1 == b2
            @test eltype(bs) === Pair{typeof(c), typeof(b1)}
            @test typeof(b1) === TensorKit.blocktype(t)
            # basic linear algebra
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
        # element type conversion
        tc = complex(t)
        @test convert(typeof(tc), t) == tc
        @test typeof(convert(typeof(tc), t)) == typeof(tc)
        # to and from generic TensorMap
        td = DiagonalTensorMap(TensorMap(t))
        @test t == td
        @test typeof(td) == typeof(t)
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
        @test all(
            Base.Fix2(isa, DiagonalTensorMap),
            (t1 * t2, t1 \ t2, t1 / t2, inv(t1), pinv(t1))
        )
        # no V * V' * V ← V or V^2 ← V tests due to Nsymbol erroring where fusion is forbidden
    end
    @timedtestset "Tensor contraction" begin
        for W in (Vect[I](I(1, 1, 0) => 2, I(1, 1, 1) => 3), Vect[I](I(2, 2, 0) => 2, I(2, 2, 1) => 3))
            d = DiagonalTensorMap(rand(ComplexF64, reduceddim(W)), W)
            t = TensorMap(d)
            A = randn(ComplexF64, W ⊗ W' ⊗ W, W)
            B = randn(ComplexF64, W ⊗ W' ⊗ W, W ⊗ W') # empty for modules so untested

            @planar E1[-1 -2 -3; -4 -5] := B[-1 -2 -3; 1 -5] * d[1; -4]
            @planar E2[-1 -2 -3; -4 -5] := B[-1 -2 -3; 1 -5] * t[1; -4]
            @test E1 ≈ E2
            @planar E1[-1 -2 -3; -4 -5] = B[-1 -2 -3; -4 1] * d'[-5; 1]
            @planar E2[-1 -2 -3; -4 -5] = B[-1 -2 -3; -4 1] * t'[-5; 1]
            @test E1 ≈ E2
            @planar E1[-1 -2 -3; -4 -5] = B[1 -2 -3; -4 -5] * d[-1; 1]
            @planar E2[-1 -2 -3; -4 -5] = B[1 -2 -3; -4 -5] * t[-1; 1]
            @test E1 ≈ E2
            @planar E1[-1 -2 -3; -4 -5] = B[-1 1 -3; -4 -5] * d[1; -2]
            @planar E2[-1 -2 -3; -4 -5] = B[-1 1 -3; -4 -5] * t[1; -2]
            @test E1 ≈ E2
            @planar E1[-1 -2 -3; -4 -5] = B[-1 -2 1; -4 -5] * d'[-3; 1]
            @planar E2[-1 -2 -3; -4 -5] = B[-1 -2 1; -4 -5] * t'[-3; 1]
            @test E1 ≈ E2
        end
    end
    @timedtestset "Factorization" begin
        for T in (Float32, ComplexF64)
            t = DiagonalTensorMap(rand(T, reduceddim(V)), V)
            @testset "eig" begin
                D, W = @constinferred eig(t)
                @test t * W ≈ W * D
                t2 = t + t'
                D2, V2 = @constinferred eigh(t2)
                VdV2 = V2' * V2
                @test VdV2 ≈ one(VdV2)
                @test t2 * V2 ≈ V2 * D2

                @test rank(D) ≈ rank(t)
                @test cond(D) ≈ cond(t)
                @test all(
                    ((s, t),) -> isapprox(s, t),
                    zip(
                        values(LinearAlgebra.eigvals(D)),
                        values(LinearAlgebra.eigvals(t))
                    )
                )
            end
            @testset "leftorth with $alg" for alg in (TensorKit.QR(), TensorKit.QL())
                Q, R = @constinferred leftorth(t; alg = alg)
                QdQ = Q' * Q
                @test QdQ ≈ one(QdQ)
                @test Q * R ≈ t
                if alg isa Polar
                    @test isposdef(R)
                end
            end
            @testset "rightorth with $alg" for alg in (TensorKit.RQ(), TensorKit.LQ())
                L, Q = @constinferred rightorth(t; alg = alg)
                QQd = Q * Q'
                @test QQd ≈ one(QQd)
                @test L * Q ≈ t
                if alg isa Polar
                    @test isposdef(L)
                end
            end
            @testset "tsvd with $alg" for alg in (TensorKit.SVD(), TensorKit.SDD())
                U, S, Vᴴ = @constinferred tsvd(t; alg = alg)
                UdU = U' * U
                @test UdU ≈ one(UdU)
                VdV = Vᴴ * Vᴴ'
                @test VdV ≈ one(VdV)
                @test U * S * Vᴴ ≈ t

                @test rank(S) ≈ rank(t)
                @test cond(S) ≈ cond(t)
                @test all(
                    ((s, t),) -> isapprox(s, t),
                    zip(
                        values(LinearAlgebra.svdvals(S)),
                        values(LinearAlgebra.svdvals(t))
                    )
                )
            end
        end
    end
    @timedtestset "Tensor functions" begin
        for T in (Float64, ComplexF64)
            d = DiagonalTensorMap(rand(T, reduceddim(V)), V)
            # rand is important for positive numbers in the real case, for log and sqrt
            t = TensorMap(d)
            @test @constinferred exp(d) ≈ exp(t)
            @test @constinferred log(d) ≈ log(t)
            @test @constinferred sqrt(d) ≈ sqrt(t)
            @test @constinferred sin(d) ≈ sin(t)
            @test @constinferred cos(d) ≈ cos(t)
            @test @constinferred tan(d) ≈ tan(t)
            @test @constinferred cot(d) ≈ cot(t)
            @test @constinferred sinh(d) ≈ sinh(t)
            @test @constinferred cosh(d) ≈ cosh(t)
            @test @constinferred tanh(d) ≈ tanh(t)
            @test @constinferred coth(d) ≈ coth(t)
            @test @constinferred asin(d) ≈ asin(t)
            @test @constinferred acos(d) ≈ acos(t)
            @test @constinferred atan(d) ≈ atan(t)
            @test @constinferred acot(d) ≈ acot(t)
            @test @constinferred asinh(d) ≈ asinh(t)
            @test @constinferred acosh(one(d) + d) ≈ acosh(one(t) + t)
            @test @constinferred atanh(d) ≈ atanh(t)
            @test @constinferred acoth(one(t) + d) ≈ acoth(one(d) + t)
        end
    end
end

# whatever V will be, for now VIB1
#TODO: test with non-diagonal sectors
# needs to be 1 dimensional stuff for the isomorphism test in removeunit
V1, V2, V3, V4, V5 = V
@assert V3 * V4 * V2 ≿ V1' * V5' # necessary for leftorth tests
@assert V3 * V4 ≾ V1' * V2' * V5' # necessary for rightorth tests

@timedtestset "Tensors with symmetry: $Istr" verbose = true begin
    V1, V2, V3, V4, V5 = V
    @timedtestset "Basic tensor properties" begin # passes for diagonal sectors
        W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
        for T in (Int, Float32, Float64, ComplexF32, ComplexF64, BigFloat)
            t = @constinferred zeros(T, W)
            @test @constinferred(hash(t)) == hash(deepcopy(t))
            @test scalartype(t) == T
            @test norm(t) == 0
            @test codomain(t) == W
            @test space(t) == (W ← one(W))
            @test domain(t) == one(W)
            @test typeof(t) == TensorMap{T,spacetype(t),5,0,Vector{T}}
            # blocks
            bs = @constinferred blocks(t)
            (c, b1), state = @constinferred Nothing iterate(bs)
            @test c == first(blocksectors(W))
            next = @constinferred Nothing iterate(bs, state)
            b2 = @constinferred block(t, first(blocksectors(t)))
            @test b1 == b2
            @test eltype(bs) === Pair{typeof(c),typeof(b1)}
            @test typeof(b1) === TensorKit.blocktype(t)
            @test typeof(c) === sectortype(t)
        end
    end
    @timedtestset "Tensor Dict conversion" begin
        W = V1 ⊗ V2 ⊗ V3 ← V4 ⊗ V5
        for T in (Int, Float32, ComplexF64)
            t = @constinferred rand(T, W)
            d = convert(Dict, t)
            @test t == convert(TensorMap, d)
        end
    end
    # no tensor array conversion tests: no fusion tensor
    @timedtestset "Basic linear algebra" begin
        W = V1 ⊗ V2 ⊗ V3 ← V4 ⊗ V5
        for T in (Float32, ComplexF64)
            t = @constinferred rand(T, W)
            @test scalartype(t) == T
            @test space(t) == W
            @test space(t') == W'
            @test dim(t) == dim(space(t))
            @test codomain(t) == codomain(W)
            @test domain(t) == domain(W)
            # blocks for adjoint
            bs = @constinferred blocks(t')
            (c, b1), state = @constinferred Nothing iterate(bs)
            @test c == first(blocksectors(W'))
            next = @constinferred Nothing iterate(bs, state)
            b2 = @constinferred block(t', first(blocksectors(t')))
            @test b1 == b2
            @test eltype(bs) === Pair{typeof(c),typeof(b1)}
            @test typeof(b1) === TensorKit.blocktype(t')
            @test typeof(c) === sectortype(t)
            # linear algebra
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

            t2 = @constinferred rand!(similar(t))
            β = rand(T)
            @test @constinferred(dot(β * t2, α * t)) ≈ conj(β) * α * conj(dot(t, t2))
            @test dot(t2, t) ≈ conj(dot(t, t2))
            @test dot(t2, t) ≈ conj(dot(t2', t'))
            @test dot(t2, t) ≈ dot(t', t2')

            i1 = @constinferred(isomorphism(T, V1 ⊗ V2, V2 ⊗ V1))
            i2 = @constinferred(isomorphism(Vector{T}, V2 ⊗ V1, V1 ⊗ V2))
            @test i1 * i2 == @constinferred(id(T, V1 ⊗ V2))
            @test i2 * i1 == @constinferred(id(Vector{T}, V2 ⊗ V1))

            w = @constinferred(isometry(T, V1 ⊗ (oneunit(V1) ⊕ oneunit(V1)), # works for diagonal
                                        V1))
            @test dim(w) == 2 * dim(V1 ← V1)
            @test w' * w == id(Vector{T}, V1)
            @test w * w' == (w * w')^2
        end
    end
    @timedtestset "Trivial space insertion and removal" begin
        W = V1 ⊗ V2 ⊗ V3 ← V4 ⊗ V5 # passes for diagonal sectors
        for T in (Float32, ComplexF64)
            t = @constinferred rand(T, W)
            t2 = @constinferred insertleftunit(t, 5) # default errors

            @test t2 == @constinferred insertrightunit(t, 4) # default doesn't error bc i==N then
            @test numind(t2) == numind(t) + 1
            @test space(t2) == insertleftunit(space(t), 5)
            @test scalartype(t2) === T
            @test t.data === t2.data
            @test @constinferred(removeunit(t2, $(numind(t2)) - 1)) == t # -1 required

            t3 = @constinferred insertleftunit(t, 5; copy=true) # same here
            @test t3 == @constinferred insertrightunit(t, 4; copy=true)
            @test t.data !== t3.data
            for (c, b) in blocks(t)
                @test b == block(t3, c)
            end
            @test @constinferred(removeunit(t3, $(numind(t3)) - 1)) == t
            t4 = @constinferred insertrightunit(t, 3; dual=true)
            @test numin(t4) == numin(t) && numout(t4) == numout(t) + 1
            for (c, b) in blocks(t)
                @test b == block(t4, c)
            end
            @test @constinferred(removeunit(t4, 4)) == t
            t5 = @constinferred insertleftunit(t, 4; dual=true)
            @test numin(t5) == numin(t) + 1 && numout(t5) == numout(t)
            for (c, b) in blocks(t)
                @test b == block(t5, c)
            end
            @test @constinferred(removeunit(t5, 4)) == t
        end
    end
    # no basic linear algebra tests via conversion: no fusion tensor
    @timedtestset "Tensor conversion" begin
        W = V1 ⊗ V2
        t = @constinferred randn(W ← W)
        @test typeof(convert(TensorMap, t')) == typeof(t)
        tc = complex(t)
        @test convert(typeof(tc), t) == tc
        @test typeof(convert(typeof(tc), t)) == typeof(tc)
        @test typeof(convert(typeof(tc), t')) == typeof(tc)
        @test Base.promote_typeof(t, tc) == typeof(tc)
        @test Base.promote_typeof(tc, t) == typeof(tc + t)
    end
    # no permutations test via inner product invariance: NoBraiding
    # no permutations test via conversion: NoBraiding and no fusion tensor
    @timedtestset "Full trace: test self-consistency" begin
            t = rand(ComplexF64, V1 ⊗ V2' ⊗ V2 ⊗ V1')
            t2 = permute(t, ((1, 2), (4, 3))) #TODO: rewrite to not permute
            s = @constinferred tr(t2)
            @test conj(s) ≈ tr(t2')
            if !isdual(V1)
                t2 = twist!(t2, 1)
            end
            if isdual(V2)
                t2 = twist!(t2, 2)
            end
            ss = tr(t2)
            @tensor s2 = t[a, b, b, a]
            @tensor t3[a, b] := t[a, c, c, b]
            @tensor s3 = t3[a, a]
            @test ss ≈ s2
            @test ss ≈ s3
        end
    @timedtestset "Partial trace: test self-consistency" begin
        t = rand(ComplexF64, V1 ⊗ V2' ⊗ V3 ⊗ V2 ⊗ V1' ⊗ V3') #TODO: fix by removing need to permute
        @tensor t2[a, b] := t[c, d, b, d, c, a] # change @tensor to @planar
        @tensor t4[a, b, c, d] := t[d, e, b, e, c, a]
        @tensor t5[a, b] := t4[a, b, c, c]
        @test t2 ≈ t5
    end
    # no trace test via conversion: NoBraiding and no fusion tensor
    @timedtestset "Trace and contraction" begin
        t1 = rand(ComplexF64, V1 ⊗ V2 ⊗ V3)
        t2 = rand(ComplexF64, V2' ⊗ V4 ⊗ V1')
        t3 = t1 ⊗ t2
        @tensor ta[a, b] := t1[x, y, a] * t2[y, b, x] #TODO: fix by making planar
        @tensor tb[a, b] := t3[x, y, a, y, b, x]
        @test ta ≈ tb
    end
    # no tensor contraction test via conversion: NoBraiding and no fusion tensor
    # no index flipping tests: NoBraiding
    @timedtestset "Multiplication of isometries: test properties" begin
        W2 = V4 ⊗ V5 # works for diagonal sectors
        W1 = W2 ⊗ (oneunit(V1) ⊕ oneunit(V1))
        for T in (Float64, ComplexF64)
            t1 = randisometry(T, W1, W2)
            t2 = randisometry(T, W2 ← W2)
            @test t1' * t1 ≈ one(t2)
            @test t2' * t2 ≈ one(t2)
            @test t2 * t2' ≈ one(t2)
            P = t1 * t1'
            @test P * P ≈ P
        end
    end
    @timedtestset "Multiplication and inverse: test compatibility" begin
        W1 = V1 ⊗ V2 ⊗ V3 # works for diagonal sectors
        W2 = V4 ⊗ V5
        for T in (Float64, ComplexF64)
            t1 = rand(T, W1, W1)
            t2 = rand(T, W2 ← W2)
            t = rand(T, W1, W2)
            @test t1 * (t1 \ t) ≈ t
            @test (t / t2) * t2 ≈ t
            @test t1 \ one(t1) ≈ inv(t1)
            @test one(t1) / t1 ≈ pinv(t1)
            @test_throws SpaceMismatch inv(t)
            @test_throws SpaceMismatch t2 \ t
            @test_throws SpaceMismatch t / t1
            tp = pinv(t) * t
            @test tp ≈ tp * tp
        end
    end
    # no multiplication and inverse test via conversion: NoBraiding and no fusion tensor
    @timedtestset "diag/diagm" begin
        W = V1 ⊗ V2 ⊗ V3 ← V4 ⊗ V5 # works for diagonal sectors
        t = randn(ComplexF64, W)
        d = LinearAlgebra.diag(t)
        D = LinearAlgebra.diagm(codomain(t), domain(t), d)
        @test LinearAlgebra.isdiag(D)
        @test LinearAlgebra.diag(D) == d
    end
    @timedtestset "Factorization" begin
        W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
        for T in (Float32, ComplexF64)
            # no left/rightorth, left/rightnull, tsvd, cond and rank tests for filled tensor
            #TODO: rewrite these tests without permuting

            # Test both a normal tensor and an adjoint one.
            ts = (rand(T, W), rand(T, W)')
            for t in ts
                @testset "leftorth with $alg" for alg in
                                                    (TensorKit.QR(), TensorKit.QRpos(),
                                                    TensorKit.QL(), TensorKit.QLpos(),
                                                    TensorKit.Polar(), TensorKit.SVD(),
                                                    TensorKit.SDD())
                    Q, R = @constinferred leftorth(t, ((3, 4, 2), (1, 5)); alg=alg)
                    QdQ = Q' * Q
                    @test QdQ ≈ one(QdQ)
                    @test Q * R ≈ permute(t, ((3, 4, 2), (1, 5)))
                    if alg isa Polar
                        @test isposdef(R)
                        @test domain(R) == codomain(R) == space(t, 1)' ⊗ space(t, 5)'
                    end
                end
                @testset "leftnull with $alg" for alg in
                                                    (TensorKit.QR(), TensorKit.SVD(),
                                                    TensorKit.SDD())
                    N = @constinferred leftnull(t, ((3, 4, 2), (1, 5)); alg=alg)
                    NdN = N' * N
                    @test NdN ≈ one(NdN)
                    @test norm(N' * permute(t, ((3, 4, 2), (1, 5)))) <
                            100 * eps(norm(t))
                end
                @testset "rightorth with $alg" for alg in
                                                    (TensorKit.RQ(), TensorKit.RQpos(),
                                                    TensorKit.LQ(), TensorKit.LQpos(),
                                                    TensorKit.Polar(), TensorKit.SVD(),
                                                    TensorKit.SDD())
                    L, Q = @constinferred rightorth(t, ((3, 4), (2, 1, 5)); alg=alg)
                    QQd = Q * Q'
                    @test QQd ≈ one(QQd)
                    @test L * Q ≈ permute(t, ((3, 4), (2, 1, 5)))
                    if alg isa Polar
                        @test isposdef(L)
                        @test domain(L) == codomain(L) == space(t, 3) ⊗ space(t, 4)
                    end
                end
                @testset "rightnull with $alg" for alg in
                                                    (TensorKit.LQ(), TensorKit.SVD(),
                                                    TensorKit.SDD())
                    M = @constinferred rightnull(t, ((3, 4), (2, 1, 5)); alg=alg)
                    MMd = M * M'
                    @test MMd ≈ one(MMd)
                    @test norm(permute(t, ((3, 4), (2, 1, 5))) * M') <
                            100 * eps(norm(t))
                end
                @testset "tsvd with $alg" for alg in (TensorKit.SVD(), TensorKit.SDD())
                    U, S, V = @constinferred tsvd(t, ((3, 4, 2), (1, 5)); alg=alg)
                    UdU = U' * U
                    @test UdU ≈ one(UdU)
                    VVd = V * V'
                    @test VVd ≈ one(VVd)
                    t2 = permute(t, ((3, 4, 2), (1, 5)))
                    @test U * S * V ≈ t2

                    s = LinearAlgebra.svdvals(t2)
                    s′ = LinearAlgebra.diag(S)
                    for (c, b) in s
                        @test b ≈ s′[c]
                    end
                end
                @testset "cond and rank" begin
                    t2 = permute(t, ((3, 4, 2), (1, 5)))
                    d1 = dim(codomain(t2))
                    d2 = dim(domain(t2))
                    @test rank(t2) == min(d1, d2)
                    M = leftnull(t2)
                    @test rank(M) == max(d1, d2) - min(d1, d2)
                    t3 = unitary(T, V1 ⊗ V2, V1 ⊗ V2)
                    @test cond(t3) ≈ one(real(T))
                    @test rank(t3) == dim(V1 ⊗ V2)
                    t4 = randn(T, V1 ⊗ V2, V1 ⊗ V2)
                    t4 = (t4 + t4') / 2
                    vals = LinearAlgebra.eigvals(t4)
                    λmax = maximum(s -> maximum(abs, s), values(vals))
                    λmin = minimum(s -> minimum(abs, s), values(vals))
                    @test cond(t4) ≈ λmax / λmin
                end
            end

            # how useful is this test?
            @testset "empty tensor" begin # passes for diagonal sectors
                t = randn(T, V1 ⊗ V2, zero(V1))
                @testset "leftorth with $alg" for alg in
                                                    (TensorKit.QR(), TensorKit.QRpos(),
                                                    TensorKit.QL(), TensorKit.QLpos(),
                                                    TensorKit.Polar(), TensorKit.SVD(),
                                                    TensorKit.SDD())
                    Q, R = @constinferred leftorth(t; alg=alg)
                    @test Q == t
                    @test dim(Q) == dim(R) == 0
                end
                @testset "leftnull with $alg" for alg in
                                                    (TensorKit.QR(), TensorKit.SVD(),
                                                    TensorKit.SDD())
                    N = @constinferred leftnull(t; alg=alg)
                    @test N' * N ≈ id(domain(N))
                    @test N * N' ≈ id(codomain(N))
                end
                @testset "rightorth with $alg" for alg in
                                                    (TensorKit.RQ(), TensorKit.RQpos(),
                                                    TensorKit.LQ(), TensorKit.LQpos(),
                                                    TensorKit.Polar(), TensorKit.SVD(),
                                                    TensorKit.SDD())
                    L, Q = @constinferred rightorth(copy(t'); alg=alg)
                    @test Q == t'
                    @test dim(Q) == dim(L) == 0
                end
                @testset "rightnull with $alg" for alg in
                                                    (TensorKit.LQ(), TensorKit.SVD(),
                                                    TensorKit.SDD())
                    M = @constinferred rightnull(copy(t'); alg=alg)
                    @test M * M' ≈ id(codomain(M))
                    @test M' * M ≈ id(domain(M))
                end
                @testset "tsvd with $alg" for alg in (TensorKit.SVD(), TensorKit.SDD())
                    U, S, V = @constinferred tsvd(t; alg=alg)
                    @test U == t
                    @test dim(U) == dim(S) == dim(V)
                end
                @testset "cond and rank" begin
                    @test rank(t) == 0
                    W2 = zero(V1) * zero(V2)
                    t2 = rand(W2, W2)
                    @test rank(t2) == 0
                    @test cond(t2) == 0.0
                end
            end
            t = rand(T, V1 ⊗ V1' ⊗ V2 ⊗ V2')
            @testset "eig and isposdef" begin
                D, V = eigen(t, ((1, 3), (2, 4)))
                t2 = permute(t, ((1, 3), (2, 4)))
                @test t2 * V ≈ V * D

                d = LinearAlgebra.eigvals(t2; sortby=nothing)
                d′ = LinearAlgebra.diag(D)
                for (c, b) in d
                    @test b ≈ d′[c]
                end

                # Somehow moving these test before the previous one gives rise to errors
                # with T=Float32 on x86 platforms. Is this an OpenBLAS issue? 
                VdV = V' * V
                VdV = (VdV + VdV') / 2
                @test isposdef(VdV)

                @test !isposdef(t2) # unlikely for non-hermitian map
                t2 = (t2 + t2')
                D, V = eigen(t2)
                VdV = V' * V
                @test VdV ≈ one(VdV)
                D̃, Ṽ = @constinferred eigh(t2)
                @test D ≈ D̃
                @test V ≈ Ṽ
                λ = minimum(minimum(real(LinearAlgebra.diag(b)))
                            for (c, b) in blocks(D))
                @test cond(Ṽ) ≈ one(real(T))
                @test isposdef(t2) == isposdef(λ)
                @test isposdef(t2 - λ * one(t2) + 0.1 * one(t2))
                @test !isposdef(t2 - λ * one(t2) - 0.1 * one(t2))
            end
        end
    end
    @timedtestset "Tensor truncation" begin # works for diagonal case
        for T in (Float32, ComplexF64)
            for p in (1, 2, 3, Inf)
                # Test both a normal tensor and an adjoint one.
                ts = (randn(T, V1 ⊗ V2 ⊗ V3, V4 ⊗ V5),
                        randn(T, V4 ⊗ V5, V1 ⊗ V2 ⊗ V3)')
                for t in ts
                    U₀, S₀, V₀, = tsvd(t)
                    t = rmul!(t, 1 / norm(S₀, p))
                    U, S, V, ϵ = @constinferred tsvd(t; trunc=truncerr(5e-1), p=p)
                    # @show p, ϵ
                    # @show domain(S)
                    # @test min(space(S,1), space(S₀,1)) != space(S₀,1)
                    U′, S′, V′, ϵ′ = tsvd(t; trunc=truncerr(nextfloat(ϵ)), p=p)
                    @test (U, S, V, ϵ) == (U′, S′, V′, ϵ′)
                    U′, S′, V′, ϵ′ = tsvd(t; trunc=truncdim(ceil(Int, dim(domain(S)))),
                                            p=p)
                    @test (U, S, V, ϵ) == (U′, S′, V′, ϵ′)
                    U′, S′, V′, ϵ′ = tsvd(t; trunc=truncspace(space(S, 1)), p=p)
                    @test (U, S, V, ϵ) == (U′, S′, V′, ϵ′)
                    # results with truncationcutoff cannot be compared because they don't take degeneracy into account, and thus truncate differently
                    U, S, V, ϵ = tsvd(t; trunc=truncbelow(1 / dim(domain(S₀))), p=p)
                    # @show p, ϵ
                    # @show domain(S)
                    # @test min(space(S,1), space(S₀,1)) != space(S₀,1)
                    U′, S′, V′, ϵ′ = tsvd(t; trunc=truncspace(space(S, 1)), p=p)
                    @test (U, S, V, ϵ) == (U′, S′, V′, ϵ′)
                end
            end
        end
    end
    # no tensor functions tests: NoBraiding and no fusion tensor
    @timedtestset "Sylvester equation" begin # works for diagonal case
        for T in (Float32, ComplexF64)
            tA = rand(T, V1 ⊗ V3, V1 ⊗ V3)
            tB = rand(T, V2 ⊗ V4, V2 ⊗ V4)
            tA = 3 // 2 * leftorth(tA; alg=Polar())[1]
            tB = 1 // 5 * leftorth(tB; alg=Polar())[1]
            tC = rand(T, V1 ⊗ V3, V2 ⊗ V4)
            t = @constinferred sylvester(tA, tB, tC)
            @test codomain(t) == V1 ⊗ V3
            @test domain(t) == V2 ⊗ V4
            @test norm(tA * t + t * tB + tC) <
                    (norm(tA) + norm(tB) + norm(tC)) * eps(real(T))^(2 / 3)
            # no reshape test
        end
    end
    @timedtestset "Tensor product: test via norm preservation" begin # works for diagonal case
        for T in (Float32, ComplexF64)
            t1 = rand(T, V2 ⊗ V3 ⊗ V1, V1 ⊗ V2)
            t2 = rand(T, V2 ⊗ V1 ⊗ V3, V1 ⊗ V1)
            t = @constinferred (t1 ⊗ t2)
            @test norm(t) ≈ norm(t1) * norm(t2)
        end
    end
    # no tensor product test via conversion: NoBraiding and no fusion tensor
    @timedtestset "Tensor product: test via tensor contraction" begin # works for diagonal case
        for T in (Float32, ComplexF64)
            t1 = rand(T, V2 ⊗ V3 ⊗ V1)
            t2 = rand(T, V2 ⊗ V1 ⊗ V3)
            t = @constinferred (t1 ⊗ t2)
            @tensor t′[1, 2, 3, 4, 5, 6] := t1[1, 2, 3] * t2[4, 5, 6]
            @test t ≈ t′
        end
    end
end

# TODO: add AD tests?

TensorKit.empty_globalcaches!()
##########
tf = time()
printstyled(
    "Finished multifusion tests in ",
    string(round(tf - ti; sigdigits = 3)),
    " seconds."; bold = true, color = Base.info_color()
)
println()
