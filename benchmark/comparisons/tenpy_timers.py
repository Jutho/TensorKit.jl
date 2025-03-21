import tenpy
import tenpy.linalg.np_conserved as npc
import tenpy.algorithms.network_contractor as nc
import numpy as np

Triv = npc.ChargeInfo([])
U1 = npc.ChargeInfo([1])
Z2 = npc.ChargeInfo([2])

def createLegCharge(charge, dims):
    if isinstance(dims, int) :
        return npc.LegCharge.from_trivial(dims)
    else :
        charges = [[d[0]] for d in dims]
        slices = np.append([0], np.cumsum([d[1] for d in dims]))
        return npc.LegCharge.from_qind(charge, slices, charges)

TeNPyTensor = npc.Array.from_func

def mpo_timer(Vmps, Vphys, Vmpo, f = np.random.standard_normal, dtype = np.float64):
    Vmpsc = Vmps.conj()
    Vphysc = Vphys.conj()
    Vmpoc = Vmpo.conj()
    #
    A = TeNPyTensor(f, [Vmps, Vphys, Vmpsc], dtype = dtype)
    M = TeNPyTensor(f, [Vmpo, Vphys, Vphysc, Vmpoc], dtype = dtype)
    FL = TeNPyTensor(f, [Vmps, Vmpoc, Vmpsc], dtype = dtype)
    FR = TeNPyTensor(f, [Vmps, Vmpo, Vmpsc], dtype = dtype)
    #
    def mpo_contract():
        C = nc.ncon([FL, A, M, A.conj(), FR], [[4,2,1], [1,3,6], [2,5,3,7], [4,5,8], [6,7,8]], list(range(1,9)))
        return C
    return mpo_contract

def pepo_timer(Vpepo, Vpeps, Venv, Vphys, f = np.random.standard_normal, dtype = np.float64):
    Vpepsc = Vpeps.conj()
    Venvc = Venv.conj()
    Vpepoc = Vpepo.conj()
    Vphysc = Vphys.conj()
    #
    A = TeNPyTensor(f, [Vpeps, Vpeps, Vphys, Vpepsc, Vpepsc], dtype = dtype)
    P = TeNPyTensor(f, [Vpepo, Vpepo, Vphys, Vphysc, Vpepoc, Vpepoc], dtype = dtype)
    FL = TeNPyTensor(f, [Venv, Vpeps, Vpepoc, Vpepsc, Venvc], dtype = dtype)
    FD = TeNPyTensor(f, [Venv, Vpeps, Vpepoc, Vpepsc, Venvc], dtype = dtype)
    FR = TeNPyTensor(f, [Venv, Vpeps, Vpepo, Vpepsc, Venvc], dtype = dtype)
    FU = TeNPyTensor(f, [Venv, Vpeps, Vpepo, Vpepsc, Venvc], dtype = dtype)
    #
    def pepo_contract():
        C = nc.ncon([FL, FU, A, P, A.conj(), FR, FD], [[18,7,4,2,1], [1,3,6,9,10],
                    [2,17,5,3,11], [4,16,8,5,6,12], [7,15,8,9,13], [10,11,12,13,14], [14,15,16,17,18]], list(range(1,19)))
        return C
    return pepo_contract

def mera_timer(Vmera, f = np.random.standard_normal, dtype = np.float64):
    Vmerac = Vmera.conj()
    #
    u = TeNPyTensor(f, [Vmera, Vmera, Vmerac, Vmerac], dtype = dtype)
    w = TeNPyTensor(f, [Vmera, Vmera, Vmerac], dtype = dtype)
    rho = TeNPyTensor(f, [Vmera, Vmera, Vmera, Vmerac, Vmerac, Vmerac], dtype = dtype)
    h = TeNPyTensor(f, [Vmera, Vmera, Vmera, Vmerac, Vmerac, Vmerac], dtype = dtype)
    #
    def mera_contract():
        C = nc.ncon([h, u, u.conj(), u, w, u.conj(), w.conj(), w, w.conj(), rho, w, w.conj()], [[9,3,4,5,1,2], [1,2,7,12], [3,4,11,13], [8,5,15,6], [6,7,19], [8,9,17,10], [10,11,22], [12,14,20], [13,14,23], [18,19,20,21,22,23], [16,15,18], [16,17,21]], [1,2,3,4,6,5,7,10,8,9,11,14,12,13,19,20,22,23,15,18,16,17,21])
        return C
    return mera_contract
