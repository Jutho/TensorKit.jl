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
        @test V' == GradedSpace(gen; dual=true)
        @test V == @constinferred GradedSpace(gen...)
        @test V' == @constinferred GradedSpace(gen...; dual=true)
        @test V == @constinferred GradedSpace(tuple(gen...))
        @test V' == @constinferred GradedSpace(tuple(gen...); dual=true)
        @test V == @constinferred GradedSpace(Dict(gen))
        @test V' == @constinferred GradedSpace(Dict(gen); dual=true)
        @test V == @inferred Vect[I](gen)
        @test V' == @constinferred Vect[I](gen; dual=true)
        @test V == @constinferred Vect[I](gen...)
        @test V' == @constinferred Vect[I](gen...; dual=true)
        @test V == @constinferred Vect[I](Dict(gen))
        @test V' == @constinferred Vect[I](Dict(gen); dual=true)
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
        @test @constinferred(rightoneunit(⊕(Wright, WMop))) == oneunit(Wright)

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
        @test @constinferred(TensorKit.hassector(V, first(slist)))
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

        d = Dict{I,Int}()
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

    @timedtestset "HomSpace" begin
        # for (V1, V2, V3, V4, V5) in (Vtr, Vℤ₃, VSU₂)
        #     W = TensorKit.HomSpace(V1 ⊗ V2, V3 ⊗ V4 ⊗ V5)
        #     @test W == (V3 ⊗ V4 ⊗ V5 → V1 ⊗ V2)
        #     @test W == (V1 ⊗ V2 ← V3 ⊗ V4 ⊗ V5)
        #     @test W' == (V1 ⊗ V2 → V3 ⊗ V4 ⊗ V5)
        #     @test eval(Meta.parse(sprint(show, W))) == W
        #     @test eval(Meta.parse(sprint(show, typeof(W)))) == typeof(W)
        #     @test spacetype(W) == typeof(V1)
        #     @test sectortype(W) == sectortype(V1)
        #     @test W[1] == V1
        #     @test W[2] == V2
        #     @test W[3] == V3'
        #     @test W[4] == V4'
        #     @test W[5] == V5'
        #     @test @constinferred(hash(W)) == hash(deepcopy(W)) != hash(W')
        #     @test W == deepcopy(W)
        #     @test W == @constinferred permute(W, ((1, 2), (3, 4, 5)))
        #     @test permute(W, ((2, 4, 5), (3, 1))) == (V2 ⊗ V4' ⊗ V5' ← V3 ⊗ V1')
        #     @test (V1 ⊗ V2 ← V1 ⊗ V2) == @constinferred TensorKit.compose(W, W')
        #     @test (V1 ⊗ V2 ← V3 ⊗ V4 ⊗ V5 ⊗ oneunit(V5)) ==
        #             @constinferred(insertleftunit(W)) ==
        #             @constinferred(insertrightunit(W))
        #     @test @constinferred(removeunit(insertleftunit(W), $(numind(W) + 1))) == W
        #     @test (V1 ⊗ V2 ← V3 ⊗ V4 ⊗ V5 ⊗ oneunit(V5)') ==
        #             @constinferred(insertleftunit(W; conj=true)) ==
        #             @constinferred(insertrightunit(W; conj=true))
        #     @test (oneunit(V1) ⊗ V1 ⊗ V2 ← V3 ⊗ V4 ⊗ V5) ==
        #             @constinferred(insertleftunit(W, 1)) ==
        #             @constinferred(insertrightunit(W, 0))
        #     @test (V1 ⊗ V2 ⊗ oneunit(V1) ← V3 ⊗ V4 ⊗ V5) ==
        #             @constinferred(insertrightunit(W, 2))
        #     @test (V1 ⊗ V2 ← oneunit(V1) ⊗ V3 ⊗ V4 ⊗ V5) ==
        #             @constinferred(insertleftunit(W, 3))
        #     @test @constinferred(removeunit(insertleftunit(W, 3), 3)) == W
        #     @test @constinferred(insertrightunit(one(V1) ← V1, 0)) == (oneunit(V1) ← V1)
        #     @test_throws BoundsError insertleftunit(one(V1) ← V1, 0)
        # end
    end
end

@timedtestset "Fusion trees for $(TensorKit.type_repr(I))" verbose = true begin
    
end

multifusion_diagspacelist = (Vect[I](values(I)[k] => 1 for k in 1:length(values(I))))
# TODO: more examples necessary?

@testset "DiagonalTensor with domain $V" for V in multifusion_diagspacelist
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
            @test eltype(bs) === Pair{typeof(c),typeof(b1)}
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
        @test all(Base.Fix2(isa, DiagonalTensorMap),
                  (t1 * t2, t1 \ t2, t1 / t2, inv(t1), pinv(t1)))

        u = randn(Float64, V * V' * V, V)
        @test u * t1 ≈ u * TensorMap(t1)
        @test u / t1 ≈ u / TensorMap(t1)
        @test t1 * u' ≈ TensorMap(t1) * u'
        @test t1 \ u' ≈ TensorMap(t1) \ u'

        t3 = rand(Float64, V ← V^2)
        t4 = rand(ComplexF64, V ← V^2)
        @test t1 * t3 ≈ lmul!(t1, copy(t3))
        @test t2 * t4 ≈ lmul!(t2, copy(t4))

        t3 = rand(Float64, V^2 ← V)
        t4 = rand(ComplexF64, V^2 ← V)
        @test t3 * t1 ≈ rmul!(copy(t3), t1)
        @test t4 * t2 ≈ rmul!(copy(t4), t2)
    end
    @timedtestset "Tensor contraction" begin
        d = DiagonalTensorMap(rand(ComplexF64, reduceddim(v)), v)
        # d = DiagonalTensorMap(rand(ComplexF64, reduceddim(V)), V)
        t = TensorMap(d)
        v = Vect[I](I(2,1,0)=>1)
        A = randn(ComplexF64, v ⊗ v' ⊗ v, v)
        B = randn(ComplexF64, v ⊗ v' ⊗ v, v ⊗ v')
        # A = randn(ComplexF64, V ⊗ V' ⊗ V, V)
        # B = randn(ComplexF64, V ⊗ V' ⊗ V, V ⊗ V')

        @planar E1[-1 -2 -3; -4 -5] := B[-1 -2 -3; 1 -5] * d[1; -4]
        @planar E2[-1 -2 -3; -4 -5] := B[-1 -2 -3; 1 -5] * t[1; -4]
        @test E1 ≈ E2
        @planar E1[-1 -2 -3; -4 -5] = B[-1 -2 -3; -4 1] * d'[-5; 1]
        @planar E2[-1 -2 -3; -4 -5] = B[-1 -2 -3; -4 1] * t'[-5; 1]
        @test E1 ≈ E2
        @planar E1[-1 -2 -3; -4 -5] = B[1 -2 -3; -4 -5] * d[-1; 1] # don't work for modules
        @planar E2[-1 -2 -3; -4 -5] = B[1 -2 -3; -4 -5] * t[-1; 1]
        @test E1 ≈ E2
        @planar E1[-1 -2 -3; -4 -5] = B[-1 1 -3; -4 -5] * d[1; -2]
        @planar E2[-1 -2 -3; -4 -5] = B[-1 1 -3; -4 -5] * t[1; -2]
        @test E1 ≈ E2
        @planar E1[-1 -2 -3; -4 -5] = B[-1 -2 1; -4 -5] * d'[-3; 1]
        @planar E2[-1 -2 -3; -4 -5] = B[-1 -2 1; -4 -5] * t'[-3; 1]
        @test E1 ≈ E2
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
                @test all(((s, t),) -> isapprox(s, t),
                          zip(values(LinearAlgebra.eigvals(D)),
                              values(LinearAlgebra.eigvals(t))))
            end
            @testset "leftorth with $alg" for alg in (TensorKit.QR(), TensorKit.QL())
                Q, R = @constinferred leftorth(t; alg=alg)
                QdQ = Q' * Q
                @test QdQ ≈ one(QdQ)
                @test Q * R ≈ t
                if alg isa Polar
                    @test isposdef(R)
                end
            end
            @testset "rightorth with $alg" for alg in (TensorKit.RQ(), TensorKit.LQ())
                L, Q = @constinferred rightorth(t; alg=alg)
                QQd = Q * Q'
                @test QQd ≈ one(QQd)
                @test L * Q ≈ t
                if alg isa Polar
                    @test isposdef(L)
                end
            end
            @testset "tsvd with $alg" for alg in (TensorKit.SVD(), TensorKit.SDD())
                U, S, Vᴴ = @constinferred tsvd(t; alg=alg)
                UdU = U' * U
                @test UdU ≈ one(UdU)
                VdV = Vᴴ * Vᴴ'
                @test VdV ≈ one(VdV)
                @test U * S * Vᴴ ≈ t

                @test rank(S) ≈ rank(t)
                @test cond(S) ≈ cond(t)
                @test all(((s, t),) -> isapprox(s, t),
                          zip(values(LinearAlgebra.svdvals(S)),
                              values(LinearAlgebra.svdvals(t))))
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






##########
tf = time()
printstyled("Finished multifusion tests in ",
            string(round(tf - ti; sigdigits=3)),
            " seconds."; bold=true, color=Base.info_color())
println()
