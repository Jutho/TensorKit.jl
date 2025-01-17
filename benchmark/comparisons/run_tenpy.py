import mkl
mkl.set_num_threads(1)
from bonds import *
from benchtools import *
import tenpy_timers
import numpy as np

K = 100 # number of repititions; scale up for more reliable testing.

Triv = tenpy_timers.Triv
Z2 = tenpy_timers.Z2
U1 = tenpy_timers.U1

# MPO Contractions
# TRIVIAL = no symmetry
N = len(mpodims_triv[0])
mpo_triv_times = np.zeros((K, N))
for i in range(0, N) :
    Dmps = mpodims_triv[0][i]
    Dmpo = mpodims_triv[1][i]
    Dphys = mpodims_triv[2][i]
    #
    Vmps = tenpy_timers.createLegCharge(Triv, Dmps)
    Vmpo = tenpy_timers.createLegCharge(Triv, Dmpo)
    Vphys = tenpy_timers.createLegCharge(Triv, Dphys)
    #
    mpo_timer = tenpy_timers.mpo_timer(Vmps = Vmps, Vmpo = Vmpo, Vphys = Vphys)
    mpo_triv_times[:, i] = timer(mpo_timer, outer = K, inner = 1)
    #
    tavg = np.sum(mpo_triv_times[:, i])/K
    print('MPO Contraction, trivial symmetry: case ' + repr(i+1) + ' out of ' + repr(N) + ' finished at average time ' + repr(tavg))

np.savetxt('results/tenpy_mpo_triv.txt', mpo_triv_times)

# Z2 symmetry
N = len(mpodims_z2[0])
mpo_z2_times = np.zeros((K, N))
for i in range(0, N) :
    Dmps = distributeZ2(mpodims_z2[0][i])
    Dmpo = distributeZ2(mpodims_z2[1][i])
    Dphys = distributeZ2(mpodims_z2[2][i])
    #
    Vmps = tenpy_timers.createLegCharge(Z2, Dmps)
    Vmpo = tenpy_timers.createLegCharge(Z2, Dmpo)
    Vphys = tenpy_timers.createLegCharge(Z2, Dphys)
    #
    mpo_timer = tenpy_timers.mpo_timer(Vmps = Vmps, Vmpo = Vmpo, Vphys = Vphys)
    mpo_z2_times[:, i] = timer(mpo_timer, outer = K, inner = 1)
    #
    tavg = np.sum(mpo_z2_times[:, i])/K
    print('MPO Contraction, Z2 symmetry: case ' + repr(i+1) + ' out of ' + repr(N) + ' finished at average time ' + repr(tavg))

np.savetxt('results/tenpy_mpo_z2.txt', mpo_z2_times)

# U1 symmetry
N = len(mpodims_u1[0])
mpo_u1_times = np.zeros((K, N))
for i in range(0, N) :
    Dmps = distributeU1(mpodims_u1[0][i])
    Dmpo = distributeU1(mpodims_u1[1][i])
    Dphys = distributeU1(mpodims_u1[2][i])
    #
    Vmps = tenpy_timers.createLegCharge(U1, Dmps)
    Vmpo = tenpy_timers.createLegCharge(U1, Dmpo)
    Vphys = tenpy_timers.createLegCharge(U1, Dphys)
    #
    mpo_timer = tenpy_timers.mpo_timer(Vmps = Vmps, Vmpo = Vmpo, Vphys = Vphys)
    mpo_u1_times[:, i] = timer(mpo_timer, outer = K, inner = 1)
    #
    tavg = np.sum(mpo_u1_times[:, i])/K
    print('MPO Contraction, U1 symmetry: case ' + repr(i+1) + ' out of ' + repr(N) + ' finished at average time ' + repr(tavg))

np.savetxt('results/tenpy_mpo_u1.txt', mpo_u1_times)

# PEPO Contractions
N = len(pepodims_triv[0])
pepo_triv_times = np.zeros((K, N))
for i in range(0, N) :
    Dpeps = pepodims_triv[0][i]
    Dpepo = pepodims_triv[1][i]
    Dphys = pepodims_triv[2][i]
    Denv = pepodims_triv[3][i]
    #
    Vpeps = tenpy_timers.createLegCharge(Triv, Dpeps)
    Vpepo = tenpy_timers.createLegCharge(Triv, Dpepo)
    Vphys = tenpy_timers.createLegCharge(Triv, Dphys)
    Venv = tenpy_timers.createLegCharge(Triv, Denv)
    #
    pepo_timer = tenpy_timers.pepo_timer(Vpeps = Vpeps, Vpepo = Vpepo, Vphys = Vphys, Venv = Venv)
    pepo_triv_times[:, i] = timer(pepo_timer, outer = K, inner = 1)
    #
    tavg = np.sum(pepo_triv_times[:, i])/K
    print('PEPO Contraction, trivial symmetry: case ' + repr(i+1) + ' out of ' + repr(N) + ' finished at average time ' + repr(tavg))

np.savetxt('results/tenpy_pepo_triv.txt', pepo_triv_times)

# Z2 symmetry
N = len(pepodims_z2[0])
pepo_z2_times = np.zeros((K, N))
for i in range(0, N) :
    Dpeps = distributeZ2(pepodims_z2[0][i])
    Dpepo = distributeZ2(pepodims_z2[1][i])
    Dphys = distributeZ2(pepodims_z2[2][i])
    Denv = distributeZ2(pepodims_z2[3][i])
    #
    Vpeps = tenpy_timers.createLegCharge(Z2, Dpeps)
    Vpepo = tenpy_timers.createLegCharge(Z2, Dpepo)
    Vphys = tenpy_timers.createLegCharge(Z2, Dphys)
    Venv = tenpy_timers.createLegCharge(Z2, Denv)
    #
    pepo_timer = tenpy_timers.pepo_timer(Vpeps = Vpeps, Vpepo = Vpepo, Vphys = Vphys, Venv = Venv)
    pepo_z2_times[:, i] = timer(pepo_timer, outer = K, inner = 1)
    #
    tavg = np.sum(pepo_z2_times[:, i])/K
    print('PEPO Contraction, Z2 symmetry: case ' + repr(i+1) + ' out of ' + repr(N) + ' finished at average time ' + repr(tavg))

np.savetxt('results/tenpy_pepo_z2.txt', pepo_z2_times)

# U1 symmetry
N = len(pepodims_u1[0])
pepo_u1_times = np.zeros((K, N))
for i in range(0, N) :
    Dpeps = distributeU1(pepodims_u1[0][i])
    Dpepo = distributeU1(pepodims_u1[1][i])
    Dphys = distributeU1(pepodims_u1[2][i])
    Denv = distributeU1(pepodims_u1[3][i])
    #
    Vpeps = tenpy_timers.createLegCharge(U1, Dpeps)
    Vpepo = tenpy_timers.createLegCharge(U1, Dpepo)
    Vphys = tenpy_timers.createLegCharge(U1, Dphys)
    Venv = tenpy_timers.createLegCharge(U1, Denv)
    #
    pepo_timer = tenpy_timers.pepo_timer(Vpeps = Vpeps, Vpepo = Vpepo, Vphys = Vphys, Venv = Venv)
    pepo_u1_times[:, i] = timer(pepo_timer, outer = K, inner = 1)
    #
    tavg = np.sum(pepo_u1_times[:, i])/K
    print('PEPO Contraction, U1 symmetry: case ' + repr(i+1) + ' out of ' + repr(N) + ' finished at average time ' + repr(tavg))

np.savetxt('results/tenpy_pepo_u1.txt', pepo_u1_times)

# MERA Contractions
# TRIVIAL = no symmetry
N = len(meradims_triv[0])
mera_triv_times = np.zeros((K, N))
for i in range(0, N) :
    Dmera = meradims_triv[0][i]
    #
    Vmera = tenpy_timers.createLegCharge(Triv, Dmera)
    #
    mera_timer = tenpy_timers.mera_timer(Vmera = Vmera)
    mera_triv_times[:, i] = timer(mera_timer, outer = K, inner = 1)
    #
    tavg = np.sum(mera_triv_times[:, i])/K
    print('MERA Contraction, trivial symmetry: case ' + repr(i+1) + ' out of ' + repr(N) + ' finished at average time ' + repr(tavg))

np.savetxt('results/tenpy_mera_triv.txt', mera_triv_times)

# Z2 symmetry
N = len(meradims_z2[0])
mera_z2_times = np.zeros((K, N))
for i in range(0, N) :
    Dmera = distributeZ2(meradims_z2[0][i])
    #
    Vmera = tenpy_timers.createLegCharge(Z2, Dmera)
    #
    mera_timer = tenpy_timers.mera_timer(Vmera = Vmera)
    mera_z2_times[:, i] = timer(mera_timer, outer = K, inner = 1)
    #
    tavg = np.sum(mera_z2_times[:, i])/K
    print('MERA Contraction, Z2 symmetry: case ' + repr(i+1) + ' out of ' + repr(N) + ' finished at average time ' + repr(tavg))

np.savetxt('results/tenpy_mera_z2.txt', mera_z2_times)

# U1 symmetry
N = len(meradims_u1[0])
mera_u1_times = np.zeros((K, N))
for i in range(0, N) :
    Dmera = distributeU1(meradims_u1[0][i])
    #
    Vmera = tenpy_timers.createLegCharge(U1, Dmera)
    #
    mera_timer = tenpy_timers.mera_timer(Vmera = Vmera)
    mera_u1_times[:, i] = timer(mera_timer, outer = K, inner = 1)
    #
    tavg = np.sum(mera_u1_times[:, i])/K
    print('MERA Contraction, U1 symmetry: case ' + repr(i+1) + ' out of ' + repr(N) + ' finished at average time ' + repr(tavg))

np.savetxt('results/tenpy_mera_u1.txt', mera_u1_times)
