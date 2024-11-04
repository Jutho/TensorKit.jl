module TensorKitKrylovKitExt

using TensorKit, KrylovKit

using TensorKit: SectorDict,
                _empty_svdtensors, _create_svdtensors,
                 TruncationScheme, NoTruncation, TruncationSpace
using KrylovKit: Selector

# AbstractTensorMap as KrylovKit operator
KrylovKit.apply(A::AbstractTensorMap, x::AbstractTensorMap) = A * x
KrylovKit.apply_normal(A::AbstractTensorMap, x::AbstractTensorMap) = A * x
KrylovKit.apply_adjoint(A::AbstractTensorMap, x::AbstractTensorMap) = A' * x

# svdsolve!
# ---------
function KrylovKit.svdsolve(t::AbstractTensorMap, howmany::Int=1, which::Selector=:LR, T::Type=eltype(t); kwargs...)
    v₀ = rand!(similar(T, codomain(t)))
    return svdsolve(t, v₀, howmany, which; kwargs...)
end
function KrylovKit.svdsolve(t, V::VectorSpace, howmany::Int=1, which::Selector=:LR, T::Type=Float64; kwargs...)
    return svdsolve(t, rand(T, V), howmany, which; kwargs...)
end

# tsvd!
# -----
function TensorKit._tsvd!(t::AbstractTensorMap, alg::GKL, trunc::TruncationScheme,
                          p::Real=2)
    # unsupported truncation schemes:
    @assert (trunc isa NoTruncation || trunc isa TruncationSpace) "`tsvd!` with `GKL` currently does not support `trunc`."

    # early return
    if isempty(blocksectors(t))
        truncerr = zero(real(scalartype(t)))
        return TensorKit._empty_svdtensors(t)..., truncerr
    end

    # compute SVD factorization for each block
    generator = Base.Iterators.map(c -> block_tsvd!(t, c, alg, trunc), blocksectors(t))
    SVDdata = SectorDict(generator)
    dims = SectorDict(c => length(Σ) for (c, (_, Σ, _)) in SVDdata)
    U, Σ, V⁺ = TensorKit._create_svdtensors(t, SVDdata, dims)

    # for now no way of computing truncation error so just return NaN
    Ttrunc = real(scalartype(t))
    truncerr = trunc === NoTruncation() ? zero(Ttrunc) : Ttrunc(NaN)

    return U, Σ, V⁺, truncerr
end

function block_tsvd!(t, c, alg, trunc)
    which, howmany = _find_svd_blocksize(t, c, trunc)
    v₀ = rand!(similar(t, codomain(t) ← Vect[sectortype(t)](c => 1)))
    svals, Uvecs, Vvecs, info = svdsolve(t, v₀, howmany, which, alg)
    info.converged >= howmany ||
        @warn "SVD for block $c did not converge" info
    resize!.((svals, Uvecs, Vvecs), howmany)

    U = stack(block_c, Uvecs; dims=2)
    Vt = stack(block_c, Vvecs; dims=2)'
    
    return c => (U, svals, Vt)
end

# Determine the `howmany` parameter for KrylovKit
function _find_svd_blocksize(t, c, ::NoTruncation)
    return :LR, min(blockdim(domain(t), c), blockdim(codomain(t), c))
end
function _find_svd_blocksize(t, c, trunc::TruncationSpace)
    return :LR,
           min(blockdim(domain(t), c),
               blockdim(codomain(t), c),
               blockdim(trunc.space, c))
end

end