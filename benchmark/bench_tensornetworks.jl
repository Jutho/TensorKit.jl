# --------------------------------------------------------------------------------------- #
# mpo contraction
# --------------------------------------------------------------------------------------- #
function benchmark_mpo!(benchgroup, allparams::Dict)
    haskey(benchgroup, "mpo") || addgroup!(benchgroup, "mpo")
    bench = benchgroup["mpo"]

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
        sigmas = I === Trivial ? fill(1, 3) : spaces["sigmas"]

        for T in Ts, d in dims
            benchmark_mpo!(bench, T, I, d, sigmas)
        end
    end

    return nothing
end
function benchmark_mpo!(bench, T::Type{<:Number}, I::Type{<:Sector}, dims, sigmas)
    Vmps, Vmpo, Vphys = generate_space.(I, dims, sigmas)
    A = Tensor(randn, T, Vmps ⊗ Vphys ⊗ Vmps')
    M = Tensor(randn, T, Vmpo ⊗ Vphys ⊗ Vphys' ⊗ Vmpo')
    FL = Tensor(randn, T, Vmps ⊗ Vmpo' ⊗ Vmps')
    FR = Tensor(randn, T, Vmps ⊗ Vmpo ⊗ Vmps')
    bench[T, Vmpo, Vmps, Vphys] = @benchmarkable benchmark_mpo($A, $M, $FL, $FR)
    return nothing
end
function benchmark_mpo(A, M, FL, FR)
    return @tensor C = FL[4, 2, 1] * A[1, 3, 6] * M[2, 5, 3, 7] * conj(A[4, 5, 8]) *
                       FR[6, 7, 8]
end

# --------------------------------------------------------------------------------------- #
# pepo contraction
# --------------------------------------------------------------------------------------- #
function benchmark_pepo!(benchgroup, allparams::Dict)
    haskey(benchgroup, "pepo") || addgroup!(benchgroup, "pepo")
    bench = benchgroup["pepo"]

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
        sigmas = I === Trivial ? fill(1, 4) : spaces["sigmas"]

        for T in Ts, d in dims
            benchmark_pepo!(bench, T, I, d, sigmas)
        end
    end

    return nothing
end
function benchmark_pepo!(bench, T::Type{<:Number}, I::Type{<:Sector}, dims, sigmas)
    Vpepo, Vpeps, Vphys, Venv = generate_space.(I, dims, sigmas)
    A = Tensor(randn, T, Vpeps ⊗ Vpeps ⊗ Vphys ⊗ Vpeps' ⊗ Vpeps')
    P = Tensor(randn, T, Vpepo ⊗ Vpepo ⊗ Vphys ⊗ Vphys' ⊗ Vpepo' ⊗ Vpepo')
    FL = Tensor(randn, T, Venv ⊗ Vpeps ⊗ Vpepo' ⊗ Vpeps' ⊗ Venv')
    FD = Tensor(randn, T, Venv ⊗ Vpeps ⊗ Vpepo' ⊗ Vpeps' ⊗ Venv')
    FR = Tensor(randn, T, Venv ⊗ Vpeps ⊗ Vpepo ⊗ Vpeps' ⊗ Venv')
    FU = Tensor(randn, T, Venv ⊗ Vpeps ⊗ Vpepo ⊗ Vpeps' ⊗ Venv')
    bench[T, Vpepo, Vpeps, Venv, Vphys] = @benchmarkable benchmark_pepo($A, $P, $FL, $FD,
                                                                        $FR, $FU)
    return nothing
end
function benchmark_pepo(A, P, FL, FD, FR, FU)
    @tensor C = FL[18, 7, 4, 2, 1] * FU[1, 3, 6, 9, 10] * A[2, 17, 5, 3, 11] *
                P[4, 16, 8, 5, 6, 12] * conj(A[7, 15, 8, 9, 13]) * FR[10, 11, 12, 13, 14] *
                FD[14, 15, 16, 17, 18]
end

# --------------------------------------------------------------------------------------- #
# mera contraction
# --------------------------------------------------------------------------------------- #
function benchmark_mera!(benchgroup, allparams::Dict)
    haskey(benchgroup, "mera") || addgroup!(benchgroup, "mera")
    bench = benchgroup["mera"]
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
        sigmas = I === Trivial ? 1 : spaces["sigmas"]

        for T in Ts, d in dims
            benchmark_mera!(bench, T, I, d, sigmas)
        end
    end

    return nothing
end
function benchmark_mera!(bench, T::Type{<:Number}, I::Type{<:Sector}, dim, sigma)
    V = generate_space(I, dim, sigma)
    u = Tensor(randn, T, V ⊗ V ⊗ V' ⊗ V')
    w = Tensor(randn, T, V ⊗ V ⊗ V')
    ρ = Tensor(randn, T, V ⊗ V ⊗ V ⊗ V' ⊗ V' ⊗ V')
    h = Tensor(randn, T, V ⊗ V ⊗ V ⊗ V' ⊗ V' ⊗ V')
    bench[T, V] = @benchmarkable benchmark_mera($u, $w, $ρ, $h)
    return nothing
end
function benchmark_mera(u, w, ρ, h)
    @tensor C = (((((((h[9, 3, 4, 5, 1, 2] * u[1, 2, 7, 12]) * conj(u[3, 4, 11, 13])) *
                     (u[8, 5, 15, 6] * w[6, 7, 19])) *
                    (conj(u[8, 9, 17, 10]) * conj(w[10, 11, 22]))) *
                   ((w[12, 14, 20] * conj(w[13, 14, 23])) * ρ[18, 19, 20, 21, 22, 23])) *
                  w[16, 15, 18]) * conj(w[16, 17, 21]))
end
