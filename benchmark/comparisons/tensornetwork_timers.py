import tensornetwork as tn
from tensornetwork import BlockSparseTensor, Index
import numpy as np

def createIndex(charge, dims):
    if isinstance(charge, int) :
        return dims
    else :
        charges = []
        for d in dims:
            charges += [d[0]] * d[1]
        return Index(charge(charges), flow = False)

def tensor(legs, dtype = np.float64):
    if isinstance(legs[0], int):
        t = np.empty(tuple(legs), dtype = dtype)
        t[:] = np.random.standard_normal(t.shape)
    else:
        t = BlockSparseTensor.random(legs, dtype = dtype)
    return t

def mpo_timer(Vmps, Vphys, Vmpo, dtype = np.float64):
    if isinstance(Vmps, int) :
        Vmpsc = Vmps
        Vphysc = Vphys
        Vmpoc = Vmpo
        backend = 'numpy'
    else :
        Vmpsc = Vmps.flip_flow()
        Vphysc = Vphys.flip_flow()
        Vmpoc = Vmpo.flip_flow()
        backend = 'symmetric'
    #
    A = tensor([Vmps, Vphys, Vmpsc], dtype = dtype)
    M = tensor([Vmpo, Vphys, Vphysc, Vmpoc], dtype = dtype)
    FL = tensor([Vmps, Vmpoc, Vmpsc], dtype = dtype)
    FR = tensor([Vmps, Vmpo, Vmpsc], dtype = dtype)
    #
    def mpo_contract():
        C = tn.ncon([FL, A, M, A.conj(), FR], [[4,2,1], [1,3,6], [2,5,3,7], [4,5,8], [6,7,8]], con_order=list(range(1,9)), backend = backend)
        return C
    return mpo_contract

def pepo_timer(Vpepo, Vpeps, Venv, Vphys, dtype = np.float64):
    if isinstance(Vpeps, int) :
        Vpepsc = Vpeps
        Venvc = Venv
        Vpepoc = Vpepo
        Vphysc = Vphys
        backend = 'numpy'
    else :
        Vpepsc = Vpeps.flip_flow()
        Venvc = Venv.flip_flow()
        Vpepoc = Vpepo.flip_flow()
        Vphysc = Vphys.flip_flow()
        backend = 'symmetric'
    #
    A = tensor([Vpeps, Vpeps, Vphys, Vpepsc, Vpepsc], dtype = dtype)
    P = tensor([Vpepo, Vpepo, Vphys, Vphysc, Vpepoc, Vpepoc], dtype = dtype)
    FL = tensor([Venv, Vpeps, Vpepoc, Vpepsc, Venvc], dtype = dtype)
    FD = tensor([Venv, Vpeps, Vpepoc, Vpepsc, Venvc], dtype = dtype)
    FR = tensor([Venv, Vpeps, Vpepo, Vpepsc, Venvc], dtype = dtype)
    FU = tensor([Venv, Vpeps, Vpepo, Vpepsc, Venvc], dtype = dtype)
    #
    def pepo_contract():
        C = tn.ncon([FL, FU, A, P, A.conj(), FR, FD], [[18,7,4,2,1], [1,3,6,9,10],
                    [2,17,5,3,11], [4,16,8,5,6,12], [7,15,8,9,13], [10,11,12,13,14], [14,15,16,17,18]], con_order=list(range(1,19)), backend = backend)
        return C
    return pepo_contract

def mera_timer(Vmera, dtype = np.float64):
    if isinstance(Vmera, int) :
        Vmerac = Vmera
        backend = 'numpy'
    else :
        Vmerac = Vmera.flip_flow()
        backend = 'symmetric'
    #
    u = tensor([Vmera, Vmera, Vmerac, Vmerac], dtype = dtype)
    w = tensor([Vmera, Vmera, Vmerac], dtype = dtype)
    rho = tensor([Vmera, Vmera, Vmera, Vmerac, Vmerac, Vmerac], dtype = dtype)
    h = tensor([Vmera, Vmera, Vmera, Vmerac, Vmerac, Vmerac], dtype = dtype)
    #
    def mera_contract():
        C = tn.ncon([h, u, u.conj(), u, w, u.conj(), w.conj(), w, w.conj(), rho, w, w.conj()], [[9,3,4,5,1,2], [1,2,7,12], [3,4,11,13], [8,5,15,6], [6,7,19], [8,9,17,10], [10,11,22], [12,14,20], [13,14,23], [18,19,20,21,22,23], [16,15,18], [16,17,21]], con_order = [1,2,3,4,6,5,7,10,8,9,11,14,12,13,19,20,22,23,15,18,16,17,21], backend = backend)
        return C
    return mera_contract
