if VERSION < v"0.7.0-DEV.2005"
    const Test = Base.Test
else
    import Test
end

@static if !isdefined(Base, :ComplexF32)
    const ComplexF32 = Complex64
    const ComplexF64 = Complex128
end

using Test
using TensorKit

smallset(::Type{ZNIrrep{N}}) where {N} = map(ZNIrrep{N}, 1:N)
smallset(::Type{CU₁}) = map(t->CU₁(t[1],t[2]), [(0,0),(0,1),(1//2,2),(1,2),(3//2,2),(2,2),(5//2,2),(3,2),(7//2,2),(4,2),(9//2,2),(5,2)])
smallset(::Type{U₁}) = map(U₁, -10:10)
smallset(::Type{SU₂}) = map(SU₂, 1//2:1//2:2) # no zero, such that always non-trivial

randsector(::Type{ZNIrrep{N}}) where {N} = rand(smallset(ZNIrrep{N}))
randsector(::Type{CU₁}) = rand(smallset(CU₁))
randsector(::Type{U₁}) = rand(smallset(U₁))
randsector(::Type{SU₂}) = rand(smallset(SU₂))
randsector(P::Type{<:TensorKit.ProductSector}) = P(map(randsector, (P.parameters[1].parameters...,)))

include("sectors.jl")
include("spaces.jl")
include("tensors.jl")
