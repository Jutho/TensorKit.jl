@testset "Fields" begin
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

@testset "ElementarySpaces" begin
    @testset "CartesianSpace" begin
        d = 2
        V = ℝ^d
        @test isa(V, VectorSpace)
        @test isa(V, ElementarySpace)
        @test isa(V, InnerProductSpace)
        @test isa(V, EuclideanSpace)
        @test isa(V, CartesianSpace)
        @test V == @inferred(dual(V)) == @inferred(conj(V)) == @inferred(adjoint(V))
        @test TensorKit.fieldtype(V) == ℝ
        @test @inferred(sectortype(V)) == Trivial
        @test @inferred(TensorKit.checksectors(V, Trivial()))
        @test @inferred(dim(V)) == d == @inferred(dim(V, Trivial()))
        @test @inferred(indices(V)) == Base.OneTo(d)
        @test V == ℝ[d] == ℝ[](d) == typeof(V)(d)
        @test ⊕(V,V) == ℝ^(2d)
    end
    @testset "ComplexSpace" begin
        d = 2
        V = ℂ^d
        @test isa(V, VectorSpace)
        @test isa(V, ElementarySpace)
        @test isa(V, InnerProductSpace)
        @test isa(V, EuclideanSpace)
        @test isa(V, ComplexSpace)
        @test @inferred(dual(V)) == @inferred(conj(V)) == @inferred(adjoint(V)) != V
        @test @inferred(TensorKit.fieldtype(V)) == ℂ
        @test @inferred(sectortype(V)) == Trivial
        @test @inferred(TensorKit.checksectors(V, Trivial()))
        @test @inferred(dim(V)) == d == @inferred(dim(V, Trivial()))
        @test @inferred(indices(V)) == Base.OneTo(d)
        @test V == ℂ[d] == ℂ[](d) == typeof(V)(d)
        @test ⊕(V,V) == ℂ^(2d)
        # @test_throws SpaceMismatch ⊕(V, V')
        # @test_throws MethodError ⊕(ℝ^d, ℂ^d)
    end
    @testset "GeneralSpace" begin
        d = 2
        V = GeneralSpace{ℂ}(d)
        @test isa(V, VectorSpace)
        @test isa(V, ElementarySpace)
        @test !isa(V, InnerProductSpace)
        @test !isa(V, EuclideanSpace)
        @test @inferred(dual(V)) != @inferred(conj(V)) != V
        @test @inferred(TensorKit.fieldtype(V)) == ℂ
        @test @inferred(sectortype(V)) == Trivial
        @test @inferred(TensorKit.checksectors(V, Trivial()))
        @test @inferred(dim(V)) == d == @inferred(dim(V, Trivial()))
        @test @inferred(indices(V)) == Base.OneTo(d)
    end
end
