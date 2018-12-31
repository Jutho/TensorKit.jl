using Test
using Random
using LinearAlgebra
using Combinatorics
using TensorKit
using TensorKit: ProductSector, fusiontensor
using TensorOperations
using TupleTools
using TupleTools: StaticLength

smallset(::Type{ZNIrrep{N}}) where {N} = map(ZNIrrep{N}, 1:N)
smallset(::Type{CU₁}) =
    map(t->CU₁(t[1],t[2]), [(0,0), (0,1), (1//2,2), (1,2), (3//2,2), (2,2), (5//2,2),
    (3,2), (7//2,2), (4,2), (9//2,2), (5,2)])
smallset(::Type{U₁}) = map(U₁, -10:10)
smallset(::Type{SU₂}) = map(SU₂, 1//2:1//2:2) # no zero, such that always non-trivial
smallset(::Type{ProductSector{Tuple{G1,G2}}}) where {G1,G2} =
    rand([i × j for i in smallset(G1), j in smallset(G2) if dim(i)*dim(j) <= 6], 5)

randsector(::Type{ZNIrrep{N}}) where {N} = rand(smallset(ZNIrrep{N}))
randsector(::Type{CU₁}) = rand(smallset(CU₁))
randsector(::Type{U₁}) = rand(smallset(U₁))
randsector(::Type{SU₂}) = rand(smallset(SU₂))
randsector(::Type{ProductSector{Tuple{G1,G2}}}) where {G1,G2} =
    rand([i × j for i in smallset(G1), j in smallset(G2) if dim(i)*dim(j) <= 6])
randsector(::Type{ProductSector{Tuple{G1,G2,G3}}}) where {G1,G2,G3} =
    randsector(G1) × randsector(G2) × randsector(G3)

include("sectors.jl")
include("spaces.jl")
include("tensors.jl")
