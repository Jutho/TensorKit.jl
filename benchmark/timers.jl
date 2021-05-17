module Timers
    struct Timer{F,D<:Tuple}
        f::F
        argsref::Base.RefValue{D}
        Timer(f, args...) = new{typeof(f), typeof(args)}(f, Ref(args))
    end

    @noinline donothing(arg) = arg
    function (t::Timer)(; inner = 1, outer = 1)
        args = t.argsref[]
        f = t.f
        f(args...) # run once to compile
        times = zeros(Float64, (outer,))
        @inbounds for i = 1:outer
            if inner == 1
                start = Base.time_ns()
                donothing(f(args...))
                stop = Base.time_ns()
            else
                start = Base.time_ns()
                for _ = 1:inner
                    donothing(f(args...))
                end
                stop = Base.time_ns()
            end
            times[i] = (stop - start)/(1e9)/inner
        end
        return times
    end
end

module TensorKitTimers
    using TensorKit
    import ..Timers

    function mpo_timer(f = randn, T = Float64; Vmpo, Vmps, Vphys)
        A = Tensor(f, T, Vmps ⊗ Vphys ⊗ Vmps')
        M = Tensor(f, T, Vmpo ⊗ Vphys ⊗ Vphys' ⊗ Vmpo')
        FL = Tensor(f, T, Vmps ⊗ Vmpo' ⊗ Vmps')
        FR = Tensor(f, T, Vmps ⊗ Vmpo ⊗ Vmps')

        return Timers.Timer(A, M, FL, FR) do A, M, FL, FR
            @tensor C = FL[4,2,1]*A[1,3,6]*M[2,5,3,7]*conj(A[4,5,8])*FR[6,7,8]
            return C
        end
    end

    function pepo_timer(f = randn, T = Float64; Vpepo, Vpeps, Venv, Vphys)
        Vpepsl = Vpepsd = Vpepsu = Vpepsr = Vpeps
        Vpepol = Vpepod = Vpepou = Vpepor = Vpepo
        Vmpsld = Vmpslu = Vmpsrd = Vmpsru = Venv
        A = Tensor(f, T, Vpepsl ⊗ Vpepsd ⊗ Vphys ⊗ Vpepsu' ⊗ Vpepsr')
        P = Tensor(f, T, Vpepol ⊗ Vpepod ⊗ Vphys ⊗ Vphys' ⊗ Vpepou' ⊗ Vpepor')
        FL = Tensor(f, T, Vmpsld ⊗ Vpepsl ⊗ Vpepol' ⊗ Vpepsl' ⊗ Vmpslu')
        FD = Tensor(f, T, Vmpsrd ⊗ Vpepsd ⊗ Vpepod' ⊗ Vpepsd' ⊗ Vmpsld')
        FR = Tensor(f, T, Vmpsru ⊗ Vpepsr ⊗ Vpepor ⊗ Vpepsr' ⊗ Vmpsrd')
        FU = Tensor(f, T, Vmpslu ⊗ Vpepsu ⊗ Vpepou ⊗ Vpepsu' ⊗ Vmpsru')
        return Timers.Timer(A, P, FL, FD, FR, FU) do A, P, FL, FD, FR, FU
            @tensor C = FL[18,7,4,2,1]*FU[1,3,6,9,10]*
                        A[2,17,5,3,11]*P[4,16,8,5,6,12]*conj(A[7,15,8,9,13])*
                        FR[10,11,12,13,14]*FD[14,15,16,17,18]
            return C
        end
    end

    function mera_timer(f = randn, T = Float64; Vmera)
        u = Tensor(f, T, Vmera ⊗ Vmera ⊗ Vmera' ⊗ Vmera')
        w = Tensor(f, T, Vmera ⊗ Vmera ⊗ Vmera')
        ρ = Tensor(f, T, Vmera ⊗ Vmera ⊗ Vmera ⊗ Vmera' ⊗ Vmera' ⊗ Vmera')
        h = Tensor(f, T, Vmera ⊗ Vmera ⊗ Vmera ⊗ Vmera' ⊗ Vmera' ⊗ Vmera')
        return Timers.Timer(u, w, ρ, h) do u, w, ρ, h
            @tensor C = h[9,3,4,5,1,2]*u[1,2,7,12]*conj(u[3,4,11,13])*u[8,5,15,6]*w[6,7,19]*
                        conj(u[8,9,17,10])*conj(w[10,11,22])*w[12,14,20]*conj(w[13,14,23])*
                        w[16,15,18]*conj(w[16,17,21])*ρ[18,19,20,21,22,23]
            return C
        end
    end
end

module ITensorsTimers
    using ITensors
    import ..Timers

    function mpo_timer(T = Float64; Vmpo, Vmps, Vphys)
        A = randomITensor(T, Vmps, Vphys, dag(Vmps''))
        M = randomITensor(T, Vmpo, Vphys', dag(Vphys), dag(Vmpo'))
        FL = randomITensor(T, Vmps', dag(Vmpo), dag(Vmps))
        FR = randomITensor(T, Vmps'', Vmpo', dag(Vmps'''))

        return Timers.Timer(A, M, FL, FR) do A, M, FL, FR
            C = FL*A*M*dag(A')*FR
            return C[]
        end
    end

    function pepo_timer(T = Float64; Vpepo, Vpeps, Venv, Vphys)
        Vpepsl = addtags(Vpeps, "l")
        Vpepsd = addtags(Vpeps, "d")
        Vpepsu = addtags(Vpeps, "u")
        Vpepsr = addtags(Vpeps, "r")
        Vpepol = addtags(Vpepo, "l")
        Vpepod = addtags(Vpepo, "d")
        Vpepou = addtags(Vpepo, "u")
        Vpepor = addtags(Vpepo, "r")
        Vmpslu = addtags(Venv, "lu")
        Vmpsld = addtags(Venv, "ld")
        Vmpsru = addtags(Venv, "ru")
        Vmpsrd = addtags(Venv, "rd")
        A = randomITensor(T, Vpepsl, Vpepsd, Vphys, dag(Vpepsu), dag(Vpepsr))
        P = randomITensor(T, Vpepol, Vpepod, Vphys', dag(Vphys), dag(Vpepou), dag(Vpepor))
        FL = randomITensor(T, Vmpsld, Vpepsl', dag(Vpepol), dag(Vpepsl), dag(Vmpslu))
        FD = randomITensor(T, Vmpsrd, Vpepsd', dag(Vpepod), dag(Vpepsd), dag(Vmpsld))
        FR = randomITensor(T, Vmpsru, Vpepsr, Vpepor, dag(Vpepsr'), dag(Vmpsrd))
        FU = randomITensor(T, Vmpslu, Vpepsu, Vpepou, dag(Vpepsu'), dag(Vmpsru))

        return Timers.Timer(A, P, FL, FD, FR, FU) do A, P, FL, FD, FR, FU
            C = (((((FL*FU)*A)*P)*dag(A'))*FR)*FD
            return C[]
        end
    end

    # function mera_timer(f = randn, T = Float64; Vmera)
    #     u = Tensor(f, T, Vmera ⊗ Vmera ⊗ Vmera' ⊗ Vmera')
    #     w = Tensor(f, T, Vmera ⊗ Vmera ⊗ Vmera')
    #     ρ = Tensor(f, T, Vmera ⊗ Vmera ⊗ Vmera ⊗ Vmera' ⊗ Vmera' ⊗ Vmera')
    #     h = Tensor(f, T, Vmera ⊗ Vmera ⊗ Vmera ⊗ Vmera' ⊗ Vmera' ⊗ Vmera')
    #     return Timers.Timer(u, w, ρ, h) do u, w, ρ, h
    #         @tensor C = h[9,3,4,5,1,2]*u[1,2,7,12]*conj(u[3,4,11,13])*u[8,5,15,6]*w[6,7,19]*
    #                     conj(u[8,9,17,10])*conj(w[10,11,22])*w[12,14,20]*conj(w[13,14,23])*
    #                     w[16,15,18]*conj(w[16,17,21])*ρ[18,19,20,21,22,23]
    #         return C
    #     end
    # end
end
