# ---------------------------------------------------------------------------------------- #
# mul!
# ---------------------------------------------------------------------------------------- #
function benchmark_mul!(benchgroup, allparams::Dict)
    haskey(benchgroup, "mul") || addgroup!(benchgroup, "mul")
    bench = benchgroup["mul"]

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

        @assert haskey(spaces, "dims")
        dims = spaces["dims"]

        I == Trivial || haskey(spaces, "sigmas")
        sigmas = I === Trivial ? [1, 1, 1] : spaces["sigmas"]

        for T in Ts, ds in dims
            benchmark_mul!(bench, T, I, ds, sigmas)
        end
    end
    return nothing
end
function benchmark_mul!(bench, T::Type{<:Number}, I::Type{<:Sector}, dims,
                        sigmas)
    V1, V2, V3 = generate_space.(I, dims, sigmas)
    A = TensorMap(randn, T, V1 ← V2)
    B = TensorMap(randn, T, V2 ← V3)
    C = TensorMap(randn, T, V1 ← V3)

    bench[T, I, dims, sigmas] = @benchmarkable mul!(C, $A, $B) setup = (C = copy($C))
    return nothing
end

# ---------------------------------------------------------------------------------------- #
# svd!
# ---------------------------------------------------------------------------------------- #
function benchmark_svd!(benchgroup, allparams::Dict)
    haskey(benchgroup, "svd") || addgroup!(benchgroup, "svd")
    bench = benchgroup["svd"]

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

        @assert haskey(spaces, "dims")
        dims = spaces["dims"]

        I == Trivial || haskey(spaces, "sigmas")
        sigmas = I === Trivial ? fill(1, 2) : spaces["sigmas"]

        for T in Ts, ds in dims
            benchmark_svd!(bench, T, I, ds, sigmas)
        end
    end
    return nothing
end
function benchmark_svd(bench, T, I::Type{<:Sector}, dims, sigmas)
    haskey(SUITE["linalg"], "svd") || addgroup!(SUITE["linalg"], "mul")
    V1, V2 = generate_space.(I, dims, sigmas)
    A = TensorMap(randn, T, V1 ← V2)
    bench[T, I, dims, sigmas] = @benchmarkable tsvd!(A) setup = (A = copy($A))
    return nothing
end

# ---------------------------------------------------------------------------------------- #
# qr!
# ---------------------------------------------------------------------------------------- #
function benchmark_qr!(benchgroup, allparams::Dict)
    haskey(benchgroup, "qr") || addgroup!(benchgroup, "qr")
    bench = benchgroup["qr"]

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

        @assert haskey(spaces, "dims")
        dims = spaces["dims"]

        I == Trivial || haskey(spaces, "sigmas")
        sigmas = I === Trivial ? fill(1, 2) : spaces["sigmas"]

        for T in Ts, ds in dims
            benchmark_qr!(bench, T, I, ds, sigmas)
        end
    end
    return nothing
end
function benchmark_qr!(bench, T, I::Type{<:Sector}, dims, sigmas)
    haskey(SUITE["linalg"], "qr") || addgroup!(SUITE["linalg"], "qr")
    V1, V2 = generate_space.(I, dims, sigmas)
    A = TensorMap(randn, T, V1 ← V2)
    bench[T, I, dims, sigmas] = @benchmarkable qr!(A) setup = (A = copy($A))
    return nothing
end
