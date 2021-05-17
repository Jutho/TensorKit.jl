include("bonds.txt")
include("timers.jl")
include("symmetries.jl")

import TensorKit
using TensorKit: ℂ, Z2Space, U1Space
import ITensors
using ITensors: Index, QN
using JLD2

K = 1 # number of repititions; scale up for more reliable testing.

# ## MPO Contractions
#
# # TRIVIAL = no symmetry
# N = length(mpodims_triv[1]) # number of different cases
# mpo_triv_times = zeros(Float64, (K, 2, N))
# for i = 1:N
#     Dmps = mpodims_triv[1][i]
#     Dmpo = mpodims_triv[2][i]
#     Dphys = mpodims_triv[3][i]
#
#     timer1 = TensorKitTimers.mpo_timer(;    Vmps = ℂ^Dmps,
#                                             Vmpo = ℂ^Dmpo,
#                                             Vphys = ℂ^Dphys)
#     timer2 = ITensorsTimers.mpo_timer(;     Vmps = Index(Dmps),
#                                             Vmpo = Index(Dmpo),
#                                             Vphys = Index(Dphys))
#
#     mpo_triv_times[:, 1, i] = timer1(; outer = K)
#     mpo_triv_times[:, 2, i] = timer2(; outer = K)
#
#     t1 = sum(mpo_triv_times[:, 1, i])/K
#     t2 = sum(mpo_triv_times[:, 2, i])/K
#
#     println("MPO Contraction, trivial symmetry: case $i out of $N finished at average time $t1 and $t2")
# end
#
# # Z2 Symmetry
# N = length(mpodims_z2[1]) # number of different cases
# K = 2 # number of times
# mpo_z2_times = zeros(Float64, (K, 3, N))
# for i = 1:N
#     Dmps = distributeZ2(mpodims_z2[1][i])
#     Dmpo = distributeZ2(mpodims_z2[2][i])
#     Dphys = distributeZ2(mpodims_z2[3][i])
#
#     timer1 = TensorKitTimers.mpo_timer(; Vmps = Z2Space((a=>b for (a,b) in Dmps)...),
#                                          Vmpo = Z2Space((a=>b for (a,b) in Dmpo)...),
#                                          Vphys = Z2Space((a=>b for (a,b) in Dphys)...))
#     timer2 = ITensorsTimers.mpo_timer(;  Vmps = Index((QN(a,2)=>b for (a,b) in Dmps)...),
#                                          Vmpo = Index((QN(a,2)=>b for (a,b) in Dmpo)...),
#                                          Vphys = Index((QN(a,2)=>b for (a,b) in Dphys)...))
#
#     mpo_z2_times[:, 1, i] = timer1(; outer = K)
#     ITensors.disable_combine_contract()
#     mpo_z2_times[:, 2, i] = timer2(; outer = K)
#     ITensors.enable_combine_contract()
#     mpo_z2_times[:, 3, i] = timer2(; outer = K)
#
#     t1 = sum(mpo_z2_times[:, 1, i])/K
#     t2 = sum(mpo_z2_times[:, 2, i])/K
#     t3 = sum(mpo_z2_times[:, 3, i])/K
#
#     println("MPO Contraction, Z2 symmetry: case $i out of $N finished at average time $t1, $t2, $t3")
# end
#
# # U1 Symmetry
# N = length(mpodims_u1[1]) # number of different cases
# mpo_u1_times = zeros(Float64, (K, 3, N))
# for i = 1:N
#     Dmps = distributeU1(mpodims_u1[1][i])
#     Dmpo = distributeU1(mpodims_u1[2][i])
#     Dphys = distributeU1(mpodims_u1[3][i])
#
#     timer1 = TensorKitTimers.mpo_timer(; Vmps = U1Space((a=>b for (a,b) in Dmps)...),
#                                          Vmpo = U1Space((a=>b for (a,b) in Dmpo)...),
#                                          Vphys = U1Space((a=>b for (a,b) in Dphys)...))
#     timer2 = ITensorsTimers.mpo_timer(;  Vmps = Index((QN(a)=>b for (a,b) in Dmps)...),
#                                          Vmpo = Index((QN(a)=>b for (a,b) in Dmpo)...),
#                                          Vphys = Index((QN(a)=>b for (a,b) in Dphys)...))
#
#     mpo_u1_times[:, 1, i] = timer1(; outer = K)
#     ITensors.disable_combine_contract()
#     mpo_u1_times[:, 2, i] = timer2(; outer = K)
#     ITensors.enable_combine_contract()
#     mpo_u1_times[:, 3, i] = timer2(; outer = K)
#
#     t1 = sum(mpo_u1_times[:, 1, i])/K
#     t2 = sum(mpo_u1_times[:, 2, i])/K
#     t3 = sum(mpo_u1_times[:, 3, i])/K
#
#     println("MPO Contraction, U1 symmetry: case $i out of $N finished at average time $t1, $t2, $t3")
# end

## PEPO Contractions
N = length(pepodims_triv[1]) # number of different cases

pepo_triv_times = zeros(Float64, (K, 2, N))
for i = 1:N
    Dpeps = pepodims[1][i]
    Dpepo = pepodims[2][i]
    Dphys = pepodims[3][i]
    Denv = pepodims[4][i]

    timer1 = TensorKitTimers.pepo_timer(;   Vpeps = ℂ^Dpeps,
                                            Vpepo = ℂ^Dpepo,
                                            Vphys = ℂ^Dphys,
                                            Venv = ℂ^Denv)
    timer2 = ITensorsTimers.pepo_timer(;    Vpeps = Index(Dpeps),
                                            Vpepo = Index(Dpepo),
                                            Vphys = Index(Dphys),
                                            Venv = Index(Denv))

    pepo_triv_times[:, 1, i] = timer1(; outer = K)
    pepo_triv_times[:, 2, i] = timer2(; outer = K)

    println("PEPO Contraction, trivial symmetry: case $i out of $N finished")
end

@save "triv_times.jld" mpo_triv_times pepo_triv_times
