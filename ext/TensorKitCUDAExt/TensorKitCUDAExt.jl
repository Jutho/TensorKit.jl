module TensorKitCUDAExt

using CUDA
using cuTENSOR: cuTENSOR

using TensorKit
using TensorKit: TrivialTensor
using TensorKit: SectorDict
using TensorKit: OrthogonalFactorizationAlgorithm,
                 QL, QLpos, QR, QRpos, LQ, LQpos, RQ, RQpos, SVD, SDD, Polar
using TensorKit: TruncationCutoff, FusionTree

using LinearAlgebra: LinearAlgebra, Adjoint, Diagonal, BlasFloat
using Random: Random

include("cutensormap.jl")
include("cumatrixalgebra.jl")
include("cuda_fixes.jl")

function TensorKit._truncate!(V::SectorDict{I,<:CuVector}, trunc::TruncationCutoff,
                              p=2) where {I<:Sector}
    @assert trunc.add_back == 0 "add_back not supported for CuVector"
    truncdim = SectorDict{I,Int}(c => something(findfirst(<(trunc.ϵ), v), length(v) + 1) - 1
                                 for (c, v) in V)

    truncerr = TensorKit._norm((c => @view(v[(truncdim[c] + 1):end]) for (c, v) in V), p,
                               zero(real(eltype(valtype(V)))))
    for (c, v) in V
        resize!(v, truncdim[c])
    end
    return V, truncerr
end

for N in (1, 2)
    @eval function Base.convert(::Type{CuArray}, f::TensorKit.FusionTree{I,$N}) where {I}
        return convert(CuArray{float(TensorKit.sectorscalartype(I))}, f)
    end
end
function Base.convert(::Type{CuArray}, f::FusionTree{I,N}) where {I,N}
    return convert(CuArray{float(TensorKit.sectorscalartype(I))}, f)
end

function Base.getindex(t::CuTensorMap{T,S,N₁,N₂,I},
                       f₁::FusionTree{I,N₁},
                       f₂::FusionTree{I,N₂}) where {T,S,N₁,N₂,I<:Sector}
    @info "hi"
    c = f₁.coupled
    @boundscheck begin
        c == f₂.coupled || throw(SectorMismatch())
        haskey(t.rowr[c], f₁) || throw(SectorMismatch())
        haskey(t.colr[c], f₂) || throw(SectorMismatch())
    end
    @inbounds begin
        d = (dims(codomain(t), f₁.uncoupled)..., dims(domain(t), f₂.uncoupled)...)
        return reshape(view(t.data[c], t.rowr[c][f₁], t.colr[c][f₂]), d)
    end
end

end
