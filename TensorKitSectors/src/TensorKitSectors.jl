# Superselection sectors (quantum numbers):
# for defining graded vector spaces and invariant subspaces of tensor products
#==========================================================================================#

module TensorKitSectors

# exports
# -------
export Sector, Group, AbstractIrrep
export Irrep

export Nsymbol, Fsymbol, Rsymbol, Asymbol, Bsymbol
export dim, sqrtdim, invsqrtdim, frobeniusschur, twist, fusiontensor, dual
export otimes, deligneproduct, times
export FusionStyle, UniqueFusion, MultipleFusion, SimpleFusion, GenericFusion,
       MultiplicityFreeFusion
export BraidingStyle, NoBraiding, SymmetricBraiding, Bosonic, Fermionic, Anyonic
export SectorSet, SectorValues, findindex, vertex_ind2label, vertex_labeltype

export pentagon_equation, hexagon_equation

export Trivial, Z2Irrep, Z3Irrep, Z4Irrep, ZNIrrep, U1Irrep, SU2Irrep, CU1Irrep
export ProductSector
export FermionParity, FermionNumber, FermionSpin
export PlanarTrivial, FibonacciAnyon, IsingAnyon

# unicode exports
# ---------------
export ⊠, ⊗, ×
export ℤ, ℤ₂, ℤ₃, ℤ₄, U₁, SU, SU₂, CU₁
export fℤ₂, fU₁, fSU₂

# imports
# -------
using Base: SizeUnknown, HasLength, IsInfinite
using Base: HasEltype, EltypeUnknown
using Base.Iterators: product, filter
using Base: tuple_type_head, tuple_type_tail

using LinearAlgebra: tr
using TensorOperations
using HalfIntegers
using WignerSymbols

# includes
# --------
include("auxiliary.jl")
include("sectors.jl")
include("trivial.jl")
include("groups.jl")
include("irreps.jl")    # irreps of symmetry groups, with bosonic braiding
include("product.jl")   # direct product of different sectors
include("fermions.jl")  # irreps with defined fermionparity and fermionic braiding
include("anyons.jl")    # non-group sectors

# precompile
# ----------
include("precompile.jl")

function __precompile__()
    for I in (Trivial, Z2Irrep, Z3Irrep, Z4Irrep, ZNIrrep, U1Irrep, SU2Irrep, CU1Irrep,
              FermionParity, FermionNumber, FermionSpin, PlanarTrivial, FibonacciAnyon,
              IsingAnyon)
        precompile_sector(I)
    end
end

end # module TensorKitSectors
