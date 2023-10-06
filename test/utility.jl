export smallset, randsector, hasfusiontensor, NewSU2Irrep
export smallspace, sectorlist, fusiontensor, force_planar
export LinearAlgebra

using Base.Iterators: take, product
using TensorKit
using TensorKit: ProductSector, Trivial, fusiontensor
using TensorKit: ℙ, PlanarTrivial
using Random
using LinearAlgebra: LinearAlgebra
using Combinatorics
using CategoryData

include("newsectors.jl")
using .NewSectors

# TODO: add some sector with multiplicity
# sectorlist = (Z2Irrep, Z3Irrep, Z4Irrep, U1Irrep, CU1Irrep, SU2Irrep, NewSU2Irrep,
#               FibonacciAnyon, IsingAnyon, FermionParity, FermionNumber, FermionSpin,
#               Z3Irrep ⊠ Z4Irrep, FermionParity ⊠ U1Irrep ⊠ SU2Irrep,
#               FermionParity ⊠ SU2Irrep ⊠ SU2Irrep, NewSU2Irrep ⊠ NewSU2Irrep,
#               Z2Irrep ⊠ FibonacciAnyon ⊠ FibonacciAnyon, Object{E6})

const spacelist = Dict(Trivial => (ℂ^3, (ℂ^4)', ℂ^5, ℂ^6, (ℂ^7)'),
                       Z2Irrep => (ℂ[Z2Irrep](0 => 1, 1 => 1), ℂ[Z2Irrep](0 => 1, 1 => 2)',
                                   ℂ[Z2Irrep](0 => 3, 1 => 2)',
                                   ℂ[Z2Irrep](0 => 2, 1 => 3), ℂ[Z2Irrep](0 => 2, 1 => 5)),
                       FermionParity => (ℂ[FermionParity](0 => 1, 1 => 1),
                                         ℂ[FermionParity](0 => 1, 1 => 2)',
                                         ℂ[FermionParity](0 => 3, 1 => 2)',
                                         ℂ[FermionParity](0 => 2, 1 => 3),
                                         ℂ[FermionParity](0 => 2, 1 => 5)),
                       Z3Irrep => (ℂ[Z3Irrep](0 => 1, 1 => 2, 2 => 2),
                                   ℂ[Z3Irrep](0 => 3, 1 => 1, 2 => 1),
                                   ℂ[Z3Irrep](0 => 2, 1 => 2, 2 => 1)',
                                   ℂ[Z3Irrep](0 => 1, 1 => 2, 2 => 3),
                                   ℂ[Z3Irrep](0 => 1, 1 => 3, 2 => 3)'),
                       U1Irrep => (ℂ[U1Irrep](0 => 1, 1 => 2, -1 => 2),
                                   ℂ[U1Irrep](0 => 3, 1 => 1, -1 => 1),
                                   ℂ[U1Irrep](0 => 2, 1 => 2, -1 => 1)',
                                   ℂ[U1Irrep](0 => 1, 1 => 2, -1 => 3),
                                   ℂ[U1Irrep](0 => 1, 1 => 3, -1 => 3)'),
                       FermionNumber => (ℂ[FermionNumber](0 => 1, 1 => 2, -1 => 2),
                                         ℂ[FermionNumber](0 => 3, 1 => 1, -1 => 1),
                                         ℂ[FermionNumber](0 => 2, 1 => 2, -1 => 1)',
                                         ℂ[FermionNumber](0 => 1, 1 => 2, -1 => 3),
                                         ℂ[FermionNumber](0 => 1, 1 => 3, -1 => 3)'),
                       CU1Irrep => (ℂ[CU1Irrep]((0, 0) => 1, (0, 1) => 2, 1 => 1),
                                    ℂ[CU1Irrep]((0, 0) => 3, (0, 1) => 0, 1 => 1),
                                    ℂ[CU1Irrep]((0, 0) => 1, (0, 1) => 0, 1 => 2)',
                                    ℂ[CU1Irrep]((0, 0) => 2, (0, 1) => 2, 1 => 1),
                                    ℂ[CU1Irrep]((0, 0) => 2, (0, 1) => 1, 1 => 2)'),
                       SU2Irrep => (ℂ[SU2Irrep](0 => 3, 1 // 2 => 1),
                                    ℂ[SU2Irrep](0 => 2, 1 => 1),
                                    ℂ[SU2Irrep](1 // 2 => 1, 1 => 1)',
                                    ℂ[SU2Irrep](0 => 2, 1 // 2 => 2),
                                    ℂ[SU2Irrep](0 => 1, 1 // 2 => 1, 3 // 2 => 1)'),
                       FermionSpin => (ℂ[FermionSpin](0 => 3, 1 // 2 => 1),
                                       ℂ[FermionSpin](0 => 2, 1 => 1),
                                       ℂ[FermionSpin](1 // 2 => 1, 1 => 1)',
                                       ℂ[FermionSpin](0 => 2, 1 // 2 => 2),
                                       ℂ[FermionSpin](0 => 1, 1 // 2 => 1, 3 // 2 => 1)'),
                       Object{E6} => (ℂ[Object{E6}](1 => 1, 3 => 1),
                                      ℂ[Object{E6}](2 => 1, 3 => 1)',
                                      ℂ[Object{E6}](1 => 1, 3 => 1),
                                      ℂ[Object{E6}](1 => 1, 3 => 1),
                                      ℂ[Object{E6}](1 => 1, 2 => 1, 3 => 1)'),
                       FibonacciAnyon => (GradedSpace{FibonacciAnyon}(:I => 2, :τ => 2),
                                          GradedSpace{FibonacciAnyon}(:I => 1, :τ => 2)',
                                          GradedSpace{FibonacciAnyon}(:I => 2, :τ => 2),
                                          GradedSpace{FibonacciAnyon}(:I => 2, :τ => 1),
                                          GradedSpace{FibonacciAnyon}(:I => 2, :τ => 2)'),
                       IsingAnyon => (GradedSpace{IsingAnyon}(:I => 2, :psi => 1,
                                                              :sigma => 1),
                                      GradedSpace{IsingAnyon}(:I => 2,
                                                              :sigma => 1)',
                                      GradedSpace{IsingAnyon}(:I => 2, :psi => 1,
                                                              :sigma => 1),
                                      GradedSpace{IsingAnyon}(:I => 2, :psi => 1),
                                      GradedSpace{IsingAnyon}(:I => 2, :psi => 1,
                                                              :sigma => 1)'))

function smallspace(::Type{I}) where {I<:Sector}
    return get(spacelist, I, nothing)
end

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

"""
    force_planar(obj)

Replace an object with a planar equivalent -- i.e. one that disallows braiding.
"""
force_planar(V::ComplexSpace) = isdual(V) ? (ℙ^dim(V))' : ℙ^dim(V)
function force_planar(V::GradedSpace)
    return GradedSpace((c ⊠ PlanarTrivial() => dim(V, c) for c in sectors(V))..., isdual(V))
end
force_planar(V::ProductSpace) = mapreduce(force_planar, ⊗, V)
function force_planar(tsrc::TensorMap{ComplexSpace})
    tdst = TensorMap(undef, scalartype(tsrc),
                     force_planar(codomain(tsrc)) ← force_planar(domain(tsrc)))
    copyto!(blocks(tdst)[PlanarTrivial()], blocks(tsrc)[Trivial()])
    return tdst
end
function force_planar(tsrc::TensorMap{<:GradedSpace})
    if BraidingStyle(sectortype(tsrc)) isa TensorKit.SymmetricBraiding
        return tsrc
    end
    tdst = TensorMap(undef, scalartype(tsrc),
                     force_planar(codomain(tsrc)) ← force_planar(domain(tsrc)))
    for (c, b) in blocks(tsrc)
        copyto!(blocks(tdst)[c ⊠ PlanarTrivial()], b)
    end
    return tdst
end
