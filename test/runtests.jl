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
using Zygote: Zygote

const TK = TensorKit

Random.seed!(1234)

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

sectorlist = (Z2Irrep, Z3Irrep, Z4Irrep, Z3Irrep ⊠ Z4Irrep,
              U1Irrep, CU1Irrep, SU2Irrep,
              FermionParity, FermionParity ⊠ FermionParity,
              FermionParity ⊠ U1Irrep ⊠ SU2Irrep, FermionParity ⊠ SU2Irrep ⊠ SU2Irrep, # Hubbard-like
              FibonacciAnyon, IsingAnyon,
              Z2Irrep ⊠ FibonacciAnyon ⊠ FibonacciAnyon)

Ti = time()
# include("fusiontrees.jl")
# include("spaces.jl")
include("tensors.jl")
include("diagonal.jl")
include("planar.jl")
if !(Sys.isapple()) # TODO: remove once we know why this is so slow on macOS
    include("ad.jl")
end
include("bugfixes.jl")
Tf = time()
printstyled("Finished all tests in ",
            string(round((Tf - Ti) / 60; sigdigits=3)),
            " minutes."; bold=true, color=Base.info_color())
println()

@testset "Aqua" verbose = true begin
    using Aqua
    Aqua.test_all(TensorKit)
end
