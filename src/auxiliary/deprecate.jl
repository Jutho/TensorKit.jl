import Base: transpose

#! format: off
@deprecate permute(t::AbstractTensorMap, p1::IndexTuple, p2::IndexTuple; copy::Bool=false) permute(t, (p1, p2); copy=copy)
@deprecate transpose(t::AbstractTensorMap, p1::IndexTuple, p2::IndexTuple; copy::Bool=false) transpose(t, (p1, p2); copy=copy)
@deprecate braid(t::AbstractTensorMap, p1::IndexTuple, p2::IndexTuple, levels; copy::Bool=false) braid(t, (p1, p2), levels; copy=copy)

import LinearAlgebra: svd, svd!

Base.@deprecate(svd(t::AbstractTensorMap, leftind::IndexTuple, rightind::IndexTuple;
                    trunc::TruncationScheme=notrunc(), p::Real=2, alg::SVDAlg=SDD()),
                tsvd(t, (leftind, rightind); trunc=trunc, p=p, alg=alg))
Base.@deprecate(svd(t::AbstractTensorMap;
                    trunc::TruncationScheme=notrunc(), p::Real=2, alg::SVDAlg=SDD()),
                tsvd(t; trunc=trunc, p=p, alg=alg))
Base.@deprecate(svd!(t::AbstractTensorMap;
                     trunc::TruncationScheme=notrunc(), p::Real=2, alg::SVDAlg=SDD()),
                tsvd(t; trunc=trunc, p=p, alg=alg))

# TODO: deprecate

tsvd(t::AbstractTensorMap, p₁::IndexTuple, p₂::IndexTuple; kwargs...) = tsvd(t, (p₁, p₂); kwargs...)
leftorth(t::AbstractTensorMap, p₁::IndexTuple, p₂::IndexTuple; kwargs...) = leftorth(t, (p₁, p₂); kwargs...)
rightorth(t::AbstractTensorMap, p₁::IndexTuple, p₂::IndexTuple; kwargs...) = rightorth(t, (p₁, p₂); kwargs...)
leftnull(t::AbstractTensorMap, p₁::IndexTuple, p₂::IndexTuple; kwargs...) = leftnull(t, (p₁, p₂); kwargs...)
rightnull(t::AbstractTensorMap, p₁::IndexTuple, p₂::IndexTuple; kwargs...) = rightnull(t, (p₁, p₂); kwargs...)
LinearAlgebra.eigen(t::AbstractTensorMap, p₁::IndexTuple, p₂::IndexTuple; kwargs...) = LinearAlgebra.eigen(t, (p₁, p₂); kwargs...)
eig(t::AbstractTensorMap, p₁::IndexTuple, p₂::IndexTuple; kwargs...) = eig(t, (p₁, p₂); kwargs...)
eigh(t::AbstractTensorMap, p₁::IndexTuple, p₂::IndexTuple; kwargs...) = eigh(t, (p₁, p₂); kwargs...)

#! format: on
