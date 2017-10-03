using Test
include("../src/TensorKit.jl")

@testset "Testing groups and fusion trees" begin # mostly type inference tests
for G in (ℤ₂, U₁, SU₂)
    @show G
    s = map(G, ntuple(n->n,Val(3)))
    @inferred one(s[1])
    @inferred one(G)
    @test one(s[1]) == one(G)
    @inferred dual(s[1])
    @inferred Nsymbol(s...)
    @inferred Rsymbol(s...)
    @inferred Bsymbol(s...)
    @inferred Fsymbol(s..., s...)
    @inferred s[1] ⊗ s[2]
    @inferred ⊗(s..., s...)

    out = (s..., dual(first(⊗(s...))))

    it = @inferred fusiontrees(out)
    state = @inferred start(it)
    f, state = @inferred next(it, state)
    @inferred done(it, state)

    @inferred braid(f, 2)
    d = @inferred permute(f, (3,4,1,2))
    f2,coeff = first(d)

    @inferred repartition(f, f2, Val(3))

    @inferred permute(f, f2, (3,1,5,2), (4,6,8,7))
end
end
