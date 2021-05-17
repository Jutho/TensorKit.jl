include("bonds.txt")
include("timers.jl")
include("symmetries.jl")

import LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)

import TensorOperations
TensorOperations.disable_cache()
import TensorKit
using TensorKit: ℂ, Z2Space, U1Space
import ITensors
using ITensors: Index, QN
using JLD2

K = 50 # number of repititions; scale up for more reliable testing.

# MPO Contractions

# TRIVIAL = no symmetry
N = length(mpodims_triv[1]) # number of different cases
mpo_triv_times = zeros(Float64, (K, 2, N))
for i = 1:N
    Dmps = mpodims_triv[1][i]
    Dmpo = mpodims_triv[2][i]
    Dphys = mpodims_triv[3][i]

    timer1 = TensorKitTimers.mpo_timer(;    Vmps = ℂ^Dmps,
                                            Vmpo = ℂ^Dmpo,
                                            Vphys = ℂ^Dphys)
    timer2 = ITensorsTimers.mpo_timer(;     Vmps = Index(Dmps),
                                            Vmpo = Index(Dmpo),
                                            Vphys = Index(Dphys))

    mpo_triv_times[:, 1, i] = timer1(; outer = K)
    empty!(TensorOperations.cache)
    ITensors.disable_combine_contract()
    mpo_triv_times[:, 2, i] = timer2(; outer = K)

    t1 = sum(mpo_triv_times[:, 1, i])/K
    t2 = sum(mpo_triv_times[:, 2, i])/K

    println("MPO Contraction, trivial symmetry: case $i out of $N finished at average time $t1, $t2")
end

# Z2 Symmetry
N = length(mpodims_z2[1]) # number of different cases
K = 2 # number of times
mpo_z2_times = zeros(Float64, (K, 3, N))
for i = 1:N
    Dmps = distributeZ2(mpodims_z2[1][i])
    Dmpo = distributeZ2(mpodims_z2[2][i])
    Dphys = distributeZ2(mpodims_z2[3][i])

    timer1 = TensorKitTimers.mpo_timer(; Vmps = Z2Space((a=>b for (a,b) in Dmps)...),
                                         Vmpo = Z2Space((a=>b for (a,b) in Dmpo)...),
                                         Vphys = Z2Space((a=>b for (a,b) in Dphys)...))
    timer2 = ITensorsTimers.mpo_timer(;  Vmps = Index((QN(a,2)=>b for (a,b) in Dmps)...),
                                         Vmpo = Index((QN(a,2)=>b for (a,b) in Dmpo)...),
                                         Vphys = Index((QN(a,2)=>b for (a,b) in Dphys)...))

    mpo_z2_times[:, 1, i] = timer1(; outer = K)
    empty!(TensorOperations.cache)
    ITensors.disable_combine_contract()
    mpo_z2_times[:, 2, i] = timer2(; outer = K)
    # ITensors.enable_combine_contract()
    # mpo_z2_times[:, 3, i] = timer2(; outer = K)

    t1 = sum(mpo_z2_times[:, 1, i])/K
    t2 = sum(mpo_z2_times[:, 2, i])/K
    # t3 = sum(mpo_z2_times[:, 3, i])/K

    println("MPO Contraction, Z2 symmetry: case $i out of $N finished at average time $t1, $t2")

    # println("MPO Contraction, Z2 symmetry: case $i out of $N finished at average time $t1, $t2, $t3")
end

# U1 Symmetry
N = length(mpodims_u1[1]) # number of different cases
mpo_u1_times = zeros(Float64, (K, 3, N))
for i = 1:N
    Dmps = distributeU1(mpodims_u1[1][i])
    Dmpo = distributeU1(mpodims_u1[2][i])
    Dphys = distributeU1(mpodims_u1[3][i])

    timer1 = TensorKitTimers.mpo_timer(; Vmps = U1Space((a=>b for (a,b) in Dmps)...),
                                         Vmpo = U1Space((a=>b for (a,b) in Dmpo)...),
                                         Vphys = U1Space((a=>b for (a,b) in Dphys)...))
    timer2 = ITensorsTimers.mpo_timer(;  Vmps = Index((QN(a)=>b for (a,b) in Dmps)...),
                                         Vmpo = Index((QN(a)=>b for (a,b) in Dmpo)...),
                                         Vphys = Index((QN(a)=>b for (a,b) in Dphys)...))

    mpo_u1_times[:, 1, i] = timer1(; outer = K)
    empty!(TensorOperations.cache)
    ITensors.disable_combine_contract()
    mpo_u1_times[:, 2, i] = timer2(; outer = K)
    # ITensors.enable_combine_contract()
    # mpo_u1_times[:, 3, i] = timer2(; outer = K)

    t1 = sum(mpo_u1_times[:, 1, i])/K
    t2 = sum(mpo_u1_times[:, 2, i])/K
    # t3 = sum(mpo_u1_times[:, 3, i])/K

    println("MPO Contraction, U1 symmetry: case $i out of $N finished at average time $t1, $t2")

    # println("MPO Contraction, U1 symmetry: case $i out of $N finished at average time $t1, $t2, $t3")
end

@save "mpo_times.jld" mpo_triv_times, mpo_z2_times, mpo_u1_times

## PEPO Contractions

# TRIVIAL = no symmetry
N = length(pepodims_triv[1]) # number of different cases
pepo_triv_times = zeros(Float64, (K, 2, N))
for i = 1:N
    Dpeps = pepodims_triv[1][i]
    Dpepo = pepodims_triv[2][i]
    Dphys = pepodims_triv[3][i]
    Denv = pepodims_triv[4][i]

    timer1 = TensorKitTimers.pepo_timer(;   Vpeps = ℂ^Dpeps,
                                            Vpepo = ℂ^Dpepo,
                                            Vphys = ℂ^Dphys,
                                            Venv = ℂ^Denv)
    timer2 = ITensorsTimers.pepo_timer(;    Vpeps = Index(Dpeps),
                                            Vpepo = Index(Dpepo),
                                            Vphys = Index(Dphys),
                                            Venv = Index(Denv))

    pepo_triv_times[:, 1, i] = timer1(; outer = K)
    empty!(TensorOperations.cache)
    ITensors.disable_combine_contract()
    pepo_triv_times[:, 2, i] = timer2(; outer = K)

    t1 = sum(pepo_triv_times[:, 1, i])/K
    t2 = sum(pepo_triv_times[:, 2, i])/K

    println("PEPO Contraction, trivial symmetry: case $i out of $N finished at average time $t1, $t2")
end

# Z2 Symmetry
N = length(pepodims_z2[1]) # number of different cases
pepo_z2_times = zeros(Float64, (K, 3, N))
for i = 1:N
    Dpeps = distributeZ2(pepodims_z2[1][i])
    Dpepo = distributeZ2(pepodims_z2[2][i])
    Dphys = distributeZ2(pepodims_z2[3][i])
    Denv = distributeZ2(pepodims_z2[4][i])

    timer1 = TensorKitTimers.pepo_timer(;   Vpeps = Z2Space((a=>b for (a,b) in Dpeps)...),
                                            Vpepo = Z2Space((a=>b for (a,b) in Dpepo)...),
                                            Vphys = Z2Space((a=>b for (a,b) in Dphys)...),
                                            Venv = Z2Space((a=>b for (a,b) in Denv)...))

    timer2 = ITensorsTimers.pepo_timer(;Vpeps = Index((QN(a, 2)=>b for (a,b) in Dpeps)...),
                                        Vpepo = Index((QN(a, 2)=>b for (a,b) in Dpepo)...),
                                        Vphys = Index((QN(a, 2)=>b for (a,b) in Dphys)...),
                                        Venv = Index((QN(a, 2)=>b for (a,b) in Denv)...))

    pepo_z2_times[:, 1, i] = timer1(; outer = K)
    empty!(TensorOperations.cache)
    ITensors.disable_combine_contract()
    pepo_z2_times[:, 2, i] = timer2(; outer = K)
    # ITensors.enable_combine_contract()
    # pepo_z2_times[:, 3, i] = timer2(; outer = K)

    t1 = sum(pepo_z2_times[:, 1, i])/K
    t2 = sum(pepo_z2_times[:, 2, i])/K
    # t3 = sum(pepo_z2_times[:, 3, i])/K

    println("PEPO Contraction, Z2 symmetry: case $i out of $N finished at average time $t1, $t2, $t3")
end

# U1 Symmetry
N = length(pepodims_u1[1]) # number of different cases
pepo_u1_times = zeros(Float64, (K, 3, N))
for i = 1:N
    Dpeps = distributeU1(pepodims_u1[1][i])
    Dpepo = distributeU1(pepodims_u1[2][i])
    Dphys = distributeU1(pepodims_u1[3][i])
    Denv = distributeU1(pepodims_u1[4][i])

    timer1 = TensorKitTimers.pepo_timer(;   Vpeps = U1Space((a=>b for (a,b) in Dpeps)...),
                                            Vpepo = U1Space((a=>b for (a,b) in Dpepo)...),
                                            Vphys = U1Space((a=>b for (a,b) in Dphys)...),
                                            Venv = U1Space((a=>b for (a,b) in Denv)...))
    timer2 = ITensorsTimers.pepo_timer(;Vpeps = Index((QN(a)=>b for (a,b) in Dpeps)...),
                                        Vpepo = Index((QN(a)=>b for (a,b) in Dpepo)...),
                                        Vphys = Index((QN(a)=>b for (a,b) in Dphys)...),
                                        Venv = Index((QN(a)=>b for (a,b) in Denv)...))

    pepo_u1_times[:, 1, i] = timer1(; outer = K)
    empty!(TensorOperations.cache)
    ITensors.disable_combine_contract()
    pepo_u1_times[:, 2, i] = timer2(; outer = K)
    # ITensors.enable_combine_contract()
    # pepo_u1_times[:, 3, i] = timer2(; outer = K)

    t1 = sum(pepo_z2_times[:, 1, i])/K
    t2 = sum(pepo_z2_times[:, 2, i])/K
    # t3 = sum(pepo_z3_times[:, 2, i])/K

    println("PEPO Contraction, U1 symmetry: case $i out of $N finished at average time $t1, $t2")

    # println("PEPO Contraction, U1 symmetry: case $i out of $N finished at average time $t1, $t2, $t3")
end

@save "pepo_times.jld" pepo_triv_times, pepo_z2_times, pepo_u1_times

# ## MERA Contractions

# TRIVIAL = no symmetry
N = length(meradims_triv[1]) # number of different cases
mera_triv_times = zeros(Float64, (K, 2, N))
for i = 1:N
    Dmera = meradims_triv[1][i]

    timer1 = TensorKitTimers.mera_timer(;   Vmera = ℂ^Dmera)
    timer2 = ITensorsTimers.mera_timer(;    Vmera = Index(Dmera))

    mera_triv_times[:, 1, i] = timer1(; outer = K)
    empty!(TensorOperations.cache)
    ITensors.disable_combine_contract()
    mera_triv_times[:, 2, i] = timer2(; outer = K)

    t1 = sum(mera_triv_times[:, 1, i])/K
    t2 = sum(mera_triv_times[:, 2, i])/K

    println("MERA Contraction, trivial symmetry: case $i out of $N finished at average time $t1, $t2")
end

# Z2 Symmetry
N = length(meradims_z2[1]) # number of different cases
mera_z2_times = zeros(Float64, (K, 2, N))
for i = 1:N
    Dmera = distributeZ2(meradims_z2[1][i])

    timer1 = TensorKitTimers.mera_timer(; Vmera = Z2Space((a=>b for (a,b) in Dmera)...))
    timer2 = ITensorsTimers.mera_timer(;  Vmera = Index((QN(a, 2)=>b for (a,b) in Dmera)...))

    mera_z2_times[:, 1, i] = timer1(; outer = K)
    empty!(TensorOperations.cache)
    ITensors.disable_combine_contract()
    mera_z2_times[:, 2, i] = timer2(; outer = K)

    t1 = sum(mera_z2_times[:, 1, i])/K
    t2 = sum(mera_z2_times[:, 2, i])/K

    println("MERA Contraction, Z2 symmetry: case $i out of $N finished at average time $t1, $t2")
end

# U1 Symmetry
N = length(meradims_u1[1]) # number of different cases
mera_u1_times = zeros(Float64, (K, 2, N))
for i = 1:N
    Dmera = distributeU1(meradims_u1[1][i])

    timer1 = TensorKitTimers.mera_timer(; Vmera = U1Space((a=>b for (a,b) in Dmera)...))
    timer2 = ITensorsTimers.mera_timer(;  Vmera = Index((QN(a)=>b for (a,b) in Dmera)...))

    mera_u1_times[:, 1, i] = timer1(; outer = K)
    empty!(TensorOperations.cache)
    ITensors.disable_combine_contract()
    mera_u1_times[:, 2, i] = timer2(; outer = K)

    t1 = sum(mera_u1_times[:, 1, i])/K
    t2 = sum(mera_u1_times[:, 2, i])/K

    println("MERA Contraction, U1 symmetry: case $i out of $N finished at average time $t1, $t2")
end

@save "mera_times.jld" mera_triv_times, mera_z2_times, mera_u1_times
