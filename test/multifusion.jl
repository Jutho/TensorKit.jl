I = IsingBimod
Istr = TensorKit.type_repr(I)

println("------------------------------------")
println("Multifusion tests for $Istr")
println("------------------------------------")
ti = time()

@timedtestset "ElementarySpace: $(TensorKit.type_repr(Vect[I]))" begin
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
    # W = @constinferred GradedSpace(one(I) => 1)
    Wleft = @constinferred Vect[I](I(1, 1, 0) => 1, I(1, 1, 1) => 1)
    Wright = @constinferred Vect[I](I(2, 2, 0) => 1, I(2, 2, 1) => 1)
    WM = @constinferred Vect[I](I(1, 2, 0) => 1)
    WMop = @constinferred Vect[I](I(2, 1, 0) => 1)
    @test W == GradedSpace(one(I) => 1, randsector(I) => 0)
    @test @constinferred(oneunit(V)) == W == oneunit(typeof(V))

    # randsector never returns trivial sector, so this cannot error
    @test_throws ArgumentError GradedSpace(one(I) => 1, randsector(I) => 0, one(I) => 3)
    @test eval(Meta.parse(sprint(show, W))) == W

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
    # @test @constinferred(⊕(V, oneunit(V))) ==
    #         Vect[I](c => isone(c) + dim(V, c) for c in sectors(V))
    # @test @constinferred(fuse(V, oneunit(V))) == V
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
    # @test infimum(V, GradedSpace(one(I) => 3)) == GradedSpace(one(I) => 2)
    @test_throws SpaceMismatch (⊕(V, V'))
end


tf = time()
printstyled("Finished multifusion tests in ",
            string(round(tf - ti; sigdigits=3)),
            " seconds."; bold=true, color=Base.info_color())
println()