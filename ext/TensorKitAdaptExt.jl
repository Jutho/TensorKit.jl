"""
    module TensorKitAdaptExt

Extension module for supporting Adapt.jl with TensorKit types.
"""
module TensorKitAdaptExt

using TensorKit
using TensorKit: AdjointTensorMap
using Adapt

function Adapt.adapt_structure(::Type{T}, t::TrivialTensorMap) where {T}
    return TensorMap(adapt(T, t.data), codomain(t), domain(t))
end
function Adapt.adapt_structure(::Type{T}, t::TensorMap) where {T}
    data = TensorKit.SectorDict(c => adapt(T, b) for (c, b) in t.data)
    return TensorMap(data, codomain(t), domain(t))
end
function Adapt.adapt_structure(::Type{T}, t::AdjointTensorMap) where {T}
    return adjoint(adapt(T, parent(t)))
end

end
