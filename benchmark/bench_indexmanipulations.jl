# --------------------------------------------------------------------------------------- #
# permute!
# --------------------------------------------------------------------------------------- #
function init_permute_tensors(T, W, p)
    C = randn(T, permute(W, p))
    A = randn(T, W)
    return C, A
end
function benchmark_permute!(benchgroup, params::Dict)
    haskey(benchgroup, "permute") || addgroup!(benchgroup, "permute")
    bench = benchgroup["permute"]
    for kwargs in expand_kwargs(params)
        benchmark_permute!(bench; kwargs...)
    end
    return nothing
end
function benchmark_permute!(bench; sigmas=nothing, T="Float64", I="Trivial", dims, p)
    T_ = parse_type(T)
    I_ = parse_type(I)

    p_ = (Tuple(p[1]), Tuple(p[2]))
    Vs = generate_space.(I_, dims, sigmas)

    codomain = mapreduce(Base.Fix1(getindex, Vs), ⊗, p_[1]; init=one(eltype(Vs)))
    domain = mapreduce(Base.Fix1(getindex, Vs), ⊗, p_[2]; init=one(eltype(Vs)))
    init() = init_permute_tensors(T_, codomain ← domain, p_)

    bench[T, I, dims, sigmas, p] = @benchmarkable permute!(C, A, $p_) setup = ((C, A) = $init())
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
