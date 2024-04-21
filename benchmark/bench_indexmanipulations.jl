# --------------------------------------------------------------------------------------- #
# permute!
# --------------------------------------------------------------------------------------- #
function benchmark_permute!(benchgroup, allparams::Dict)
    haskey(benchgroup, "permute") || addgroup!(benchgroup, "permute")
    bench = benchgroup["permute"]

    Ts = if haskey(allparams, "T")
        Tparam = allparams["T"]
        if Tparam isa String
            [eval(Meta.parse(Tparam))]
        else
            eval.(Meta.parse.(Tparam))
        end
    else
        [Float64]
    end

    @assert haskey(allparams, "spaces")
    Vparam = allparams["spaces"]
    @assert Vparam isa Vector

    for spaces in Vparam
        @assert haskey(spaces, "I")
        I = eval(Meta.parse(spaces["I"]))

        @assert haskey(spaces, "p")
        p = (tuple(spaces["p"][1]...), tuple(spaces["p"][2]...))

        @assert haskey(spaces, "dims")
        dims = spaces["dims"]

        I == Trivial || haskey(spaces, "sigmas")
        sigmas = I === Trivial ? fill(1, length(p[1]) + length(p[2])) :
                 spaces["sigmas"]

        for T in Ts, ds in dims
            benchmark_permute!(bench, T, I, ds, sigmas, p)
        end
    end

    return nothing
end
function benchmark_permute!(bench, T::Type{<:Number}, I::Type{<:Sector}, dims, sigmas, p)
    codomain = mapreduce(⊗, dims[1], sigmas[1:length(dims[1])]) do d, s
        return generate_space(I, d, s)
    end
    domain = mapreduce(⊗, dims[2], sigmas[(length(dims[1]) + 1):end]) do d, s
        return generate_space(I, d, s)
    end

    W = codomain ← domain
    A = TensorMap(randn, T, W)
    C = TensorMap(randn, T, permute(W, p))
    bench[T, W, p] = @benchmarkable permute!(C, $A, $p) setup = (C = copy($C))
    return nothing
end

# --------------------------------------------------------------------------------------- #
# transpose!
# --------------------------------------------------------------------------------------- #
function benchmark_transpose!(benchgroup, allparams::Dict)
    haskey(benchgroup, "transpose") || addgroup!(benchgroup, "transpose")
    bench = benchgroup["transpose"]

    Ts = if haskey(allparams, "T")
        Tparam = allparams["T"]
        if Tparam isa String
            [eval(Meta.parse(Tparam))]
        else
            eval.(Meta.parse.(Tparam))
        end
    else
        [Float64]
    end

    @assert haskey(allparams, "spaces")
    Vparam = allparams["spaces"]
    @assert Vparam isa Vector

    for spaces in Vparam
        @assert haskey(spaces, "I")
        I = eval(Meta.parse(spaces["I"]))

        @assert haskey(spaces, "p")
        p = (tuple(spaces["p"][1]...), tuple(spaces["p"][2]...))

        @assert haskey(spaces, "dims")
        dims = spaces["dims"]

        I == Trivial || haskey(spaces, "sigmas")
        sigmas = I === Trivial ? fill(1, length(p[1]) + length(p[2])) :
                 spaces["sigmas"]

        for T in Ts, ds in dims
            benchmark_transpose!(bench, T, I, ds, sigmas, p)
        end
    end

    return nothing
end
function benchmark_transpose!(bench, T, I::Type{<:Sector}, dims, sigmas, k, p)
    # TODO: assert permutation is cyclic 
    Vs = generate_space.(I, dims, sigmas)
    W = HomSpace(⊗(Vs[1:k]...), ⊗(Vs[(k + 1):end]...))
    A = TensorMap(randn, T, W)
    C = TensorMap(randn, T, permute(W, p))
    bench[T, W, p] = @benchmarkable transpose!(C, $A, $p) setup = (C = copy($C))
    return nothing
end
