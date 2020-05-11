println("------------------------------------")
println("|     Fields and vector spaces     |")
println("------------------------------------")
ti = time()
@testset TimedTestSet "Fields" begin
    @test isa(ℝ, Field)
    @test isa(ℂ, Field)
    @test eval(Meta.parse(sprint(show, ℝ))) == ℝ
    @test eval(Meta.parse(sprint(show, ℂ))) == ℂ
    @test ℝ ⊆ ℝ
    @test ℝ ⊆ ℂ
    @test ℂ ⊆ ℂ
    @test !(ℂ ⊆ ℝ)

    for T in (Int8, Int16, Int32, Int64, BigInt)
        @test one(T) ∈ ℝ
        @test one(Rational{T}) ∈ ℝ
        @test !(one(Complex{T}) ∈ ℝ)
        @test !(one(Complex{Rational{T}}) ∈ ℝ)
        @test one(T) ∈ ℂ
        @test one(Rational{T}) ∈ ℂ
        @test one(Complex{T}) ∈ ℂ
        @test one(Complex{Rational{T}} ∈ ℂ)

        @test T ⊆ ℝ
        @test Rational{T} ⊆ ℝ
        @test !(Complex{T} ⊆ ℝ)
        @test !(Complex{Rational{T}} ⊆ ℝ)
        @test T ⊆ ℂ
        @test Rational{T} ⊆ ℂ
        @test Complex{T} ⊆ ℂ
        @test Complex{Rational{T}} ⊆ ℂ
    end
    for T in (Float32, Float64, BigFloat)
        @test one(T) ∈ ℝ
        @test !(one(Complex{T}) ∈ ℝ)
        @test one(T) ∈ ℂ
        @test one(Complex{T} ∈ ℂ)

        @test T ⊆ ℝ
        @test !(Complex{T} ⊆ ℝ)
        @test T ⊆ ℂ
        @test Complex{T} ⊆ ℂ
    end
end

@testset TimedTestSet "ElementarySpace: CartesianSpace" begin
    d = 2
    V = ℝ^d
    @test eval(Meta.parse(sprint(show, V))) == V
    @test eval(Meta.parse(sprint(show, typeof(V)))) == typeof(V)
    @test isa(V, VectorSpace)
    @test isa(V, ElementarySpace)
    @test isa(V, InnerProductSpace)
    @test isa(V, EuclideanSpace)
    @test isa(V, CartesianSpace)
    @test !isdual(V)
    @test !isdual(V')
    @test V == CartesianSpace(Trivial() => d) == CartesianSpace(Dict(Trivial() => d))
    @test @inferred(hash(V)) == hash(deepcopy(V))
    @test V == @inferred(dual(V)) == @inferred(conj(V)) == @inferred(adjoint(V))
    @test field(V) == ℝ
    @test @inferred(sectortype(V)) == Trivial
    @test ((@inferred sectors(V))...,) == (Trivial(),)
    @test length(sectors(V)) == 1
    @test @inferred(TensorKit.hassector(V, Trivial()))
    @test @inferred(dim(V)) == d == @inferred(dim(V, Trivial()))
    @test dim(@inferred(typeof(V)())) == 0
    @test (sectors(typeof(V)())...,) == ()
    @test @inferred(TensorKit.axes(V)) == Base.OneTo(d)
    @test V == ℝ[d] == ℝ[](d) == typeof(V)(d)
    W = @inferred ℝ[1]
    @test @inferred(oneunit(V)) == W == oneunit(typeof(V))
    @test @inferred(⊕(V,V)) == ℝ^(2d)
    @test @inferred(⊕(V,oneunit(V))) == ℝ^(d+1)
    @test @inferred(⊕(V,V,V,V)) == ℝ^(4d)
    @test @inferred(fuse(V,V)) == ℝ^(d^2)
    @test @inferred(fuse(V,V',V,V')) == ℝ^(d^4)
    @test @inferred(flip(V)) == V'
    @test flip(V) ≅ V
    @test flip(V) ≾ V
    @test flip(V) ≿ V
    @test V ≺ ⊕(V,V)
    @test !(V ≻ ⊕(V,V))
    @test @inferred(infinum(V, ℝ^3)) == V
    @test @inferred(supremum(V', ℝ^3)) == ℝ^3
end

@testset TimedTestSet "ElementarySpace: ComplexSpace" begin
    d = 2
    V = ℂ^d
    @test eval(Meta.parse(sprint(show, V))) == V
    @test eval(Meta.parse(sprint(show, V'))) == V'
    @test eval(Meta.parse(sprint(show, typeof(V)))) == typeof(V)
    @test isa(V, VectorSpace)
    @test isa(V, ElementarySpace)
    @test isa(V, InnerProductSpace)
    @test isa(V, EuclideanSpace)
    @test isa(V, ComplexSpace)
    @test !isdual(V)
    @test isdual(V')
    @test V == ComplexSpace(Trivial() => d) == ComplexSpace(Dict(Trivial() => d))
    @test @inferred(hash(V)) == hash(deepcopy(V)) != hash(V')
    @test @inferred(dual(V)) == @inferred(conj(V)) == @inferred(adjoint(V)) != V
    @test @inferred(field(V)) == ℂ
    @test @inferred(sectortype(V)) == Trivial
    @test @inferred(sectortype(V)) == Trivial
    @test ((@inferred sectors(V))...,) == (Trivial(),)
    @test length(sectors(V)) == 1
    @test @inferred(TensorKit.hassector(V, Trivial()))
    @test @inferred(dim(V)) == d == @inferred(dim(V, Trivial()))
    @test dim(@inferred(typeof(V)())) == 0
    @test (sectors(typeof(V)())...,) == ()
    @test @inferred(TensorKit.axes(V)) == Base.OneTo(d)
    @test V == ℂ[d] == ℂ[](d) == typeof(V)(d)
    W = @inferred ℂ[1]
    @test @inferred(oneunit(V)) == W == oneunit(typeof(V))
    @test @inferred(⊕(V, V)) == ℂ^(2d)
    @test_throws SpaceMismatch (⊕(V, V'))
    @test_throws MethodError (⊕(ℝ^d, ℂ^d))
    @test_throws MethodError (⊗(ℝ^d, ℂ^d))
    @test @inferred(⊕(V,V)) == ℂ^(2d)
    @test @inferred(⊕(V,oneunit(V))) == ℂ^(d+1)
    @test @inferred(⊕(V,V,V,V)) == ℂ^(4d)
    @test @inferred(fuse(V,V)) == ℂ^(d^2)
    @test @inferred(fuse(V,V',V,V')) == ℂ^(d^4)
    @test @inferred(flip(V)) == V'
    @test flip(V) ≅ V
    @test flip(V) ≾ V
    @test flip(V) ≿ V
    @test V ≺ ⊕(V,V)
    @test !(V ≻ ⊕(V,V))
    @test @inferred(infinum(V, ℂ^3)) == V
    @test @inferred(supremum(V', ℂ[3]')) == ℂ[3]'
end

@testset TimedTestSet "ElementarySpace: GeneralSpace" begin
    d = 2
    V = GeneralSpace{ℂ}(d)
    @test eval(Meta.parse(sprint(show, V))) == V
    @test eval(Meta.parse(sprint(show, dual(V)))) == dual(V)
    @test eval(Meta.parse(sprint(show, conj(V)))) == conj(V)
    @test eval(Meta.parse(sprint(show, typeof(V)))) == typeof(V)
    @test !isdual(V)
    @test isdual(V')
    @test !isdual(conj(V))
    @test isdual(conj(V'))
    @test !TensorKit.isconj(V)
    @test !TensorKit.isconj(V')
    @test TensorKit.isconj(conj(V))
    @test TensorKit.isconj(conj(V'))
    @test isa(V, VectorSpace)
    @test isa(V, ElementarySpace)
    @test !isa(V, InnerProductSpace)
    @test !isa(V, EuclideanSpace)
    @test @inferred(hash(V)) == hash(deepcopy(V)) != hash(V')
    @test @inferred(dual(V)) != @inferred(conj(V)) != V
    @test @inferred(field(V)) == ℂ
    @test @inferred(sectortype(V)) == Trivial
    @test @inferred(TensorKit.hassector(V, Trivial()))
    @test @inferred(dim(V)) == d == @inferred(dim(V, Trivial()))
    @test @inferred(TensorKit.axes(V)) == Base.OneTo(d)
end

@testset TimedTestSet "ElementarySpace: RepresentationSpace{$G}" for G in (ℤ₂, ℤ₃, ℤ₄, U₁, CU₁, SU₂, FibonacciAnyon, ℤ₃ × ℤ₄, U₁ × SU₂, SU₂ × SU₂, ℤ₂ × FibonacciAnyon × FibonacciAnyon)
    if Base.IteratorSize(values(G)) === Base.IsInfinite()
        set = unique(vcat(one(G), [randsector(G) for k = 1:10]))
        gen = (c=>2 for c in set)
    else
        gen = (values(G)[k]=>(k+1) for k in 1:length(values(G)))
    end
    V = RepresentationSpace(gen)
    @test eval(Meta.parse(sprint(show, V))) == V
    @test eval(Meta.parse(sprint(show, V'))) == V'
    @test V' == RepresentationSpace(gen; dual = true)
    @test V == @inferred RepresentationSpace(gen...)
    @test V' == @inferred RepresentationSpace(gen...; dual = true)
    @test V == @inferred RepresentationSpace(tuple(gen...))
    @test V' == @inferred RepresentationSpace(tuple(gen...); dual = true)
    @test V == @inferred RepresentationSpace(Dict(gen))
    @test V' == @inferred RepresentationSpace(Dict(gen); dual = true)
    @test V == @inferred RepresentationSpace{G}(gen)
    @test V' == @inferred RepresentationSpace{G}(gen; dual = true)
    @test V == @inferred RepresentationSpace{G}(gen...)
    @test V' == @inferred RepresentationSpace{G}(gen...; dual = true)
    @test V == @inferred RepresentationSpace{G}(Dict(gen))
    @test V' == @inferred RepresentationSpace{G}(Dict(gen); dual = true)
    @test V == @inferred typeof(V)(c=>dim(V,c) for c in sectors(V))
    if G isa ZNIrrep
        @test V == @inferred typeof(V)(V.dims)
        @test V' == @inferred typeof(V)(V.dims; dual = true)
    end
    @test @inferred(hash(V)) == hash(deepcopy(V)) != hash(V')
    @test V == RepresentationSpace(reverse(collect(gen))...)
    @test eval(Meta.parse(sprint(show, V))) == V
    @test eval(Meta.parse(sprint(show, typeof(V)))) == typeof(V)
    # space with no sectors
    @test dim(@inferred(typeof(V)())) == 0
    # space with a single sector
    W = @inferred ℂ[one(G)=>1]
    @test W == RepresentationSpace(one(G)=>1, randsector(G) => 0)
    @test @inferred(oneunit(V)) == W == oneunit(typeof(V))
    # randsector never returns trivial sector, so this cannot error
    @test_throws ArgumentError RepresentationSpace(one(G)=>1, randsector(G) => 0, one(G)=>3)
    @test eval(Meta.parse(sprint(show, W))) == W
    @test isa(V, VectorSpace)
    @test isa(V, ElementarySpace)
    @test isa(V, InnerProductSpace)
    @test isa(V, EuclideanSpace)
    @test isa(V, RepresentationSpace)
    @test isa(V, RepresentationSpace{G})
    @test isa(V, Base.IteratorSize(values(G)) == Base.IsInfinite() ?
                    TensorKit.GenericRepresentationSpace{G} :
                    TensorKit.FiniteRepresentationSpace{G})
    @test @inferred(dual(V)) == @inferred(conj(V)) == @inferred(adjoint(V)) != V
    @test @inferred(field(V)) == ℂ
    @test @inferred(sectortype(V)) == G
    slist = @inferred sectors(V)
    @test @inferred(TensorKit.hassector(V, first(slist)))
    @test @inferred(dim(V)) == sum((@inferred(dim(s)*dim(V, s))) for s in slist)
    if hasfusiontensor(G)
        @test @inferred(TensorKit.axes(V)) == Base.OneTo(dim(V))
    end
    @test @inferred(⊕(V,V)) == RepresentationSpace{G}(c=>2dim(V,c) for c in sectors(V))
    @test @inferred(⊕(V,V,V,V)) == RepresentationSpace{G}(c=>4dim(V,c) for c in sectors(V))
    @test @inferred(⊕(V,oneunit(V))) ==
            RepresentationSpace{G}(c=>isone(c)+dim(V,c) for c in sectors(V))
    @test @inferred(fuse(V,oneunit(V))) == V
    d = Dict{G,Int}()
    for a in sectors(V), b in sectors(V)
        for c in a ⊗ b
            d[c] = get(d, c, 0) + dim(V, a)*dim(V, b)*Nsymbol(a,b,c)
        end
    end
    @test @inferred(fuse(V,V)) == RepresentationSpace(d)
    @test @inferred(flip(V)) ==
            RepresentationSpace{G}(conj(c)=>dim(V,c) for c in sectors(V))'
    @test flip(V) ≅ V
    @test flip(V) ≾ V
    @test flip(V) ≿ V
    @test @inferred(⊕(V, V)) == @inferred supremum(V, ⊕(V, V))
    @test V == @inferred infinum(V, ⊕(V, V))
    @test V ≺ ⊕(V,V)
    @test !(V ≻ ⊕(V,V))
    @test infinum(V, RepresentationSpace(one(G)=>3)) == RepresentationSpace(one(G)=>2)
    @test_throws SpaceMismatch (⊕(V, V'))
end

@testset TimedTestSet "ProductSpace{ℂ}" begin
    V1, V2, V3, V4 = ℂ[1], ℂ[2], ℂ[3], ℂ[4]
    P = @inferred ProductSpace(V1, V2, V3, V4)
    @test eval(Meta.parse(sprint(show, P))) == P
    @test eval(Meta.parse(sprint(show, typeof(P)))) == typeof(P)
    @test isa(P, VectorSpace)
    @test isa(P, CompositeSpace)
    @test spacetype(P) == ComplexSpace
    @test sectortype(P) == Trivial
    @test @inferred(hash(P)) == hash(deepcopy(P)) != hash(P')
    @test P == deepcopy(P)
    @test P == typeof(P)(P...)
    @test @inferred(dual(P)) == P'
    @test @inferred(field(P)) == ℂ
    @test @inferred(*(V1, V2, V3, V4)) == P
    @test @inferred(⊗(V1, V2, V3, V4)) == P
    @test @inferred(⊗(V1, V2⊗V3⊗V4)) == P
    @test @inferred(⊗(V1⊗V2, V3⊗V4)) == P
    @test @inferred(⊗(V1, V2, V3⊗V4)) == P
    @test @inferred(⊗(V1, V2⊗V3, V4)) == P
    @test fuse(V1, V2', V3) ≅ V1 ⊗ V2' ⊗ V3
    @test fuse(V1, V2', V3) ≾ V1 ⊗ V2' ⊗ V3
    @test fuse(V1, V2', V3) ≿ V1 ⊗ V2' ⊗ V3
    @test fuse(flip(V1), V2, flip(V3)) ≅ V1 ⊗ V2 ⊗ V3
    @test @inferred(⊗(P)) == P
    @test @inferred(⊗(V1)) == ProductSpace(V1)
    @test eval(Meta.parse(sprint(show, ⊗(V1)))) == ⊗(V1)
    @test @inferred(one(V1)) == @inferred(one(typeof(V1))) ==
                @inferred(one(P)) == @inferred(one(typeof(P))) ==
                ProductSpace{ComplexSpace}(())
    @test eval(Meta.parse(sprint(show, one(P)))) == one(P)
    @test @inferred(⊗(one(P), P)) == P
    @test @inferred(⊗(P, one(P))) == P
    @test @inferred(⊗(one(P), one(P))) == one(P)
    @test @inferred(adjoint(P)) == dual(P) == V4' ⊗ V3' ⊗ V2' ⊗ V1'
    @test @inferred(dims(P)) == map(dim, (V1, V2, V3, V4))
    @test @inferred(dim(P)) == prod(dim, (V1, V2, V3, V4))
    @test @inferred(dim(P, 2)) == dim(V2)
    @test @inferred(sectors(P)) ==
            (mapreduce(sectors, (a, b)->tuple(a..., b...), (V1, V2, V3, V4)),)
    cube(x) = x^3
    @test @inferred(cube(V1)) == V1 ⊗ V1 ⊗ V1
    N = 3
    @test V1^N == V1 ⊗ V1 ⊗ V1
    @test P^2 == P ⊗ P
    @test @inferred(dims(P, first(sectors(P)))) == dims(P)
    @test ((@inferred blocksectors(P))...,) == (Trivial(),)
    @test (blocksectors(P ⊗ ℂ[0])...,) == ()
    @test @inferred(blockdim(P, first(blocksectors(P)))) == dim(P)
    @test Base.IteratorEltype(P) == Base.IteratorEltype(typeof(P)) ==
                                    Base.IteratorEltype(P.spaces)
    @test Base.IteratorSize(P) == Base.IteratorSize(typeof(P)) ==
                                    Base.IteratorSize(P.spaces)
    @test Base.eltype(P) == Base.eltype(typeof(P)) == typeof(V1)
    @test eltype(collect(P)) == typeof(V1)
    @test collect(P) == [V1, V2, V3, V4]
end

@testset TimedTestSet "ProductSpace{SU₂Space}" begin
    V1, V2, V3, V4, V5 = SU₂Space(0=>3, 1//2=>1), SU₂Space(0=>2, 1=>1),
                            SU₂Space(1//2=>1, 1=>1)', SU₂Space(0=>2, 1//2=>2),
                            SU₂Space(0=>1, 1//2=>1, 3//2=>1)'
    W = TensorKit.HomSpace(V1 ⊗ V2, V3 ⊗ V4 ⊗ V5)
    @test W == (V3 ⊗ V4 ⊗ V5 → V1 ⊗ V2)
    @test W == (V1 ⊗ V2 ← V3 ⊗ V4 ⊗ V5)
    @test W' == (V1 ⊗ V2 → V3 ⊗ V4 ⊗ V5)
    @test eval(Meta.parse(sprint(show, W))) == W
    @test eval(Meta.parse(sprint(show, typeof(W)))) == typeof(W)
    @test spacetype(W) == SU₂Space
    @test sectortype(W) == SU₂
    @test fuse(V1, V2', V3) ≅ V1 ⊗ V2' ⊗ V3
    @test fuse(V1, V2', V3) ≾ V1 ⊗ V2' ⊗ V3 ≾ fuse(V1 ⊗ V2' ⊗ V3)
    @test fuse(V1, V2') ⊗ V3 ≾ V1 ⊗ V2' ⊗ V3
    @test fuse(V1, V2', V3) ≿ V1 ⊗ V2' ⊗ V3 ≿ fuse(V1 ⊗ V2' ⊗ V3)
    @test V1 ⊗ fuse(V2', V3) ≿ V1 ⊗ V2' ⊗ V3
    @test fuse(flip(V1) ⊗ V2) ⊗ flip(V3) ≅ V1 ⊗ V2 ⊗ V3
    @test W[1] == V1
    @test W[2] == V2
    @test W[3] == V3'
    @test W[4] == V4'
    @test W[5] == V5'
    @test @inferred(hash(W)) == hash(deepcopy(W)) != hash(W')
    @test W == deepcopy(W)
end

@testset TimedTestSet "HomSpace" begin
    V1, V2, V3 = SU₂Space(0=>3, 1//2=>1), SU₂Space(0=>2, 1=>1), SU₂Space(1//2=>1, 1=>1)'
    P = @inferred ProductSpace(V1, V2, V3)
    @test eval(Meta.parse(sprint(show, P))) == P
    @test eval(Meta.parse(sprint(show, typeof(P)))) == typeof(P)
    @test isa(P, VectorSpace)
    @test isa(P, CompositeSpace)
    @test spacetype(P) == SU₂Space
    @test sectortype(P) == SU₂
    @test @inferred(hash(P)) == hash(deepcopy(P)) != hash(P')
    @test @inferred(dual(P)) == P'
    @test @inferred(field(P)) == ℂ
    @test @inferred(*(V1, V2, V3)) == P
    @test @inferred(⊗(V1, V2, V3)) == P
    @test @inferred(adjoint(P)) == dual(P) == V3' ⊗ V2' ⊗ V1'
    @test @inferred(⊗(V1)) == ProductSpace(V1)
    @test @inferred(one(V1)) == @inferred(one(typeof(V1))) ==
                @inferred(one(P)) == @inferred(one(typeof(P))) ==
                ProductSpace{ComplexSpace}(())
    @test @inferred(dims(P)) == map(dim, (V1, V2, V3))
    @test @inferred(dim(P)) == prod(dim, (V1, V2, V3))
    for s in @inferred(sectors(P))
        @test hassector(P, s)
        @test @inferred(dims(P, s)) == dim.((V1, V2, V3), s)
    end
    @test sum(dim(c)*blockdim(P, c) for c in @inferred(blocksectors(P))) == dim(P)
end
tf = time()
printstyled("Finished vector space tests in ",
            string(round(tf-ti; sigdigits=3)),
            " seconds."; bold = true, color = Base.info_color())
println()
