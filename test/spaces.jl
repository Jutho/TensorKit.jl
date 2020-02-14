println("------------------------------------")
println("Fields and vector spaces")
println("------------------------------------")
ti = time()
@testset TimedTestSet "Fields" begin
    @test isa(ℝ, Field)
    @test isa(ℂ, Field)

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
    @test eval(Meta.parse(sprint(show,V))) == V
    @test eval(Meta.parse(sprint(show,typeof(V)))) == typeof(V)
    @test isa(V, VectorSpace)
    @test isa(V, ElementarySpace)
    @test isa(V, InnerProductSpace)
    @test isa(V, EuclideanSpace)
    @test isa(V, CartesianSpace)
    @test @inferred(hash(V)) == hash(deepcopy(V))
    @test V == @inferred(dual(V)) == @inferred(conj(V)) == @inferred(adjoint(V))
    @test field(V) == ℝ
    @test @inferred(sectortype(V)) == Trivial
    @test @inferred(TensorKit.hassector(V, Trivial()))
    @test @inferred(dim(V)) == d == @inferred(dim(V, Trivial()))
    @test dim(@inferred(typeof(V)())) == 0
    @test @inferred(TensorKit.axes(V)) == Base.OneTo(d)
    @test V == ℝ[d] == ℝ[](d) == typeof(V)(d)
    @test ⊕(V,V) == ℝ^(2d)
    @test @inferred min(V, ℝ^3) == V
    @test @inferred max(V', ℝ^3) == ℝ^3
end

@testset TimedTestSet "ElementarySpace: ComplexSpace" begin
    d = 2
    V = ℂ^d
    @test eval(Meta.parse(sprint(show,V))) == V
    @test eval(Meta.parse(sprint(show,typeof(V)))) == typeof(V)
    @test isa(V, VectorSpace)
    @test isa(V, ElementarySpace)
    @test isa(V, InnerProductSpace)
    @test isa(V, EuclideanSpace)
    @test isa(V, ComplexSpace)
    @test @inferred(hash(V)) == hash(deepcopy(V)) != hash(V')
    @test @inferred(dual(V)) == @inferred(conj(V)) == @inferred(adjoint(V)) != V
    @test @inferred(field(V)) == ℂ
    @test @inferred(sectortype(V)) == Trivial
    @test @inferred(TensorKit.hassector(V, Trivial()))
    @test @inferred(dim(V)) == d == @inferred(dim(V, Trivial()))
    @test @inferred(TensorKit.axes(V)) == Base.OneTo(d)
    @test V == ℂ[d] == ℂ[](d) == typeof(V)(d)
    @test dim(@inferred(typeof(V)())) == 0
    @test @inferred(⊕(V,V)) == ℂ^(2d)
    @test_throws SpaceMismatch (⊕(V, V'))
    @test_throws MethodError (⊕(ℝ^d, ℂ^d))
    @test @inferred min(V, ℂ^3) == V
    @test @inferred max(V', ℂ[3]') == ℂ[3]'
end

@testset TimedTestSet "ElementarySpace: GeneralSpace" begin
    d = 2
    V = GeneralSpace{ℂ}(d)
    @test eval(Meta.parse(sprint(show,V))) == V
    @test eval(Meta.parse(sprint(show,typeof(V)))) == typeof(V)
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
    @test eval(Meta.parse(sprint(show,V))) == V
    @test V' == RepresentationSpace(gen; dual = true)
    @test V == @inferred RepresentationSpace(gen...)
    @test V' == @inferred RepresentationSpace(gen...; dual = true)
    @test V == @inferred RepresentationSpace(tuple(gen...))
    @test V' == @inferred RepresentationSpace(tuple(gen...); dual = true)
    @test V == @inferred RepresentationSpace{G}(gen)
    @test V' == @inferred RepresentationSpace{G}(gen; dual = true)
    @test V == @inferred RepresentationSpace{G}(gen...)
    @test V' == @inferred RepresentationSpace{G}(gen...; dual = true)
    @test @inferred(hash(V)) == hash(deepcopy(V)) != hash(V')
    @test eval(Meta.parse(sprint(show,V))) == V
    @test eval(Meta.parse(sprint(show,typeof(V)))) == typeof(V)
    W = RepresentationSpace(one(G)=>1) # space with a single sector
    @test W == RepresentationSpace(one(G)=>1, randsector(G) => 0)
    # space with no sectors
    @test dim(@inferred(typeof(V)())) == 0
    # randsector never returns trivial sector, so this cannot error
    @test_throws ArgumentError RepresentationSpace(one(G)=>1, randsector(G) => 0, one(G)=>3)
    @test eval(Meta.parse(sprint(show,W))) == W
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
    @test @inferred(dim(V)) == sum((@inferred(dim(s)*dim(V,s))) for s in slist)
    if hasfusiontensor(G)
        @test @inferred(TensorKit.axes(V)) == Base.OneTo(dim(V))
    end
    @test @inferred(⊕(V,V)) == @inferred max(V, ⊕(V,V))
    @test V == @inferred min(V, ⊕(V,V))
    @test min(V, RepresentationSpace(one(G)=>3)) == RepresentationSpace(one(G)=>2)
    @test_throws SpaceMismatch (⊕(V, V'))
end
tf = time()
printstyled("Finished vector space tests in ",
            string(round(tf-ti; sigdigits=3)),
            " seconds."; bold = true, color = Base.info_color())
println()
