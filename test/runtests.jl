using Test
using TestExtras
using Random
using TensorKit
using Combinatorics
using TensorKit: ProductSector, fusiontensor, pentagon_equation, hexagon_equation
using TensorOperations
using Base.Iterators: take, product
# using SUNRepresentations: SUNIrrep
# const SU3Irrep = SUNIrrep{3}
using LinearAlgebra: LinearAlgebra

include("newsectors.jl")
using .NewSectors

const TK = TensorKit

Random.seed!(12345)

smallset(::Type{I}) where {I<:Sector} = take(values(I), 5)
function smallset(::Type{ProductSector{Tuple{I1,I2}}}) where {I1,I2}
    iter = product(smallset(I1), smallset(I2))
    s = collect(i ⊠ j for (i, j) in iter if dim(i) * dim(j) <= 6)
    return length(s) > 6 ? rand(s, 6) : s
end
function smallset(::Type{ProductSector{Tuple{I1,I2,I3}}}) where {I1,I2,I3}
    iter = product(smallset(I1), smallset(I2), smallset(I3))
    s = collect(i ⊠ j ⊠ k for (i, j, k) in iter if dim(i) * dim(j) * dim(k) <= 6)
    return length(s) > 6 ? rand(s, 6) : s
end
function randsector(::Type{I}) where {I<:Sector}
    s = collect(smallset(I))
    a = rand(s)
    while a == one(a) # don't use trivial label
        a = rand(s)
    end
    return a
end
function hasfusiontensor(I::Type{<:Sector})
    try
        fusiontensor(one(I), one(I), one(I))
        return true
    catch e
        if e isa MethodError
            return false
        else
            rethrow(e)
        end
    end
end

sectorlist = (Z2Irrep, Z3Irrep, Z4Irrep, U1Irrep, CU1Irrep, SU2Irrep, NewSU2Irrep, # SU3Irrep,
              FibonacciAnyon, IsingAnyon, FermionParity, FermionParity ⊠ FermionParity,
              Z3Irrep ⊠ Z4Irrep, FermionParity ⊠ U1Irrep ⊠ SU2Irrep,
              FermionParity ⊠ SU2Irrep ⊠ SU2Irrep, NewSU2Irrep ⊠ NewSU2Irrep,
              NewSU2Irrep ⊠ SU2Irrep, FermionParity ⊠ SU2Irrep ⊠ NewSU2Irrep,
              Z2Irrep ⊠ FibonacciAnyon ⊠ FibonacciAnyon)

Ti = time()
include("sectors.jl")
include("fusiontrees.jl")
include("spaces.jl")
include("tensors.jl")
include("planar.jl")
include("ad.jl")
Tf = time()
printstyled("Finished all tests in ",
            string(round((Tf - Ti) / 60; sigdigits=3)),
            " minutes."; bold=true, color=Base.info_color())
println()
