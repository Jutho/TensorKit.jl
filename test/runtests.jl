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

randsector(::Type{ZNIrrep{N}}) where {N} = ZNIrrep{N}(rand(1:N))
randsector(::Type{U₁}) = U₁(rand(-10:10))
randsector(::Type{SU₂}) = SU₂(rand(0:1//2:2))
randsector(P::Type{<:TensorKit.ProductSector}) = P(map(randsector, (P.parameters[1].parameters...,)))

include("sectors.jl")
include("spaces.jl")
include("tensors.jl")
