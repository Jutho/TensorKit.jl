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

tf = time()
printstyled("Finished multifusion tests in ",
            string(round(tf - ti; sigdigits=3)),
            " seconds."; bold=true, color=Base.info_color())
println()
