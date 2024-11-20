module TensorKitKrylovKitExt

using TensorKit, KrylovKit

using TensorKit: SectorDict,
                 _empty_svdtensors, _create_svdtensors,
                 TruncationScheme, NoTruncation, TruncationSpace
import TensorKit.MatrixAlgebra: MatrixAlgebra
using KrylovKit: Selector

# AbstractTensorMap as KrylovKit operator
KrylovKit.apply(A::AbstractTensorMap, x::AbstractTensorMap) = A * x
KrylovKit.apply_normal(A::AbstractTensorMap, x::AbstractTensorMap) = A * x
KrylovKit.apply_adjoint(A::AbstractTensorMap, x::AbstractTensorMap) = A' * x

# MatrixAlgebra svd with KrylovKit
# --------------------------------
function MatrixAlgebra.svd!(A::AbstractMatrix, alg::GKL;
                            howmany::Int=min(size(A)...), which::Selector=:LR)
    x₀ = randn(size(A, 1))
    svals, Uvecs, Vvecs, info = svdsolve(A, x₀, howmany, which, alg)
    info.converged >= howmany || @warn "SVD did not converge" info
    resize!.((svals, Uvecs, Vvecs), howmany)
    U = reduce(hcat, Uvecs)
    Vt = reduce(hcat, Vvecs)'
    return U, svals, Vt
end

# svdsolve!
# ---------
function KrylovKit.svdsolve(t::AbstractTensorMap, howmany::Int=1, which::Selector=:LR,
                            T::Type=eltype(t); kwargs...)
    v₀ = rand!(similar(T, codomain(t)))
    return svdsolve(t, v₀, howmany, which; kwargs...)
end
function KrylovKit.svdsolve(t, V::VectorSpace, howmany::Int=1, which::Selector=:LR,
                            T::Type=Float64; kwargs...)
    return svdsolve(t, rand(T, V), howmany, which; kwargs...)
end

# tsvd!
# -----
function TensorKit._tsvd!(t::TensorMap, alg::GKL, trunc::TruncationScheme, p::Real=2)
    # unsupported truncation schemes:
    @assert (trunc isa NoTruncation || trunc isa TruncationSpace) "`tsvd!` with `GKL` currently does not support `trunc`."

    # early return
    if isempty(blocksectors(t))
        truncerr = zero(real(scalartype(t)))
        return TensorKit._empty_svdtensors(t)..., truncerr
    end

    # compute (truncated) SVD factorization for each block
    generator = Base.Iterators.map(blocks(t)) do (c, b)
        howmany = min(blockdim(domain(t), c), blockdim(codomain(t), c))
        if trunc isa TruncationSpace
            howmany = min(howmany, blockdim(trunc.space, c))
        end
        return MatrixAlgebra.svd!(b, alg; howmany=_find_svd_blocksize(t, c, trunc)...)
    end
    SVDdata = SectorDict(generator)
    dims = SectorDict(c => length(Σ) for (c, (_, Σ, _)) in SVDdata)
    U, Σ, V⁺ = TensorKit._create_svdtensors(t, SVDdata, dims)

    # for now no way of computing truncation error so just return NaN
    Ttrunc = real(scalartype(t))
    truncerr = trunc === NoTruncation() ? zero(Ttrunc) : Ttrunc(NaN)

    return U, Σ, V⁺, truncerr
end

# AbstractTensorMap might not have blocks as matrix, want to use only action of the operator
function TensorKit._tsvd!(t::AbstractTensorMap, alg::GKL, trunc::TruncationScheme,
                          p::Real=2)
    # unsupported truncation schemes:
    @assert (trunc isa NoTruncation || trunc isa TruncationSpace) "`tsvd!` with `GKL` currently does not support `trunc`."

    # early return
    if isempty(blocksectors(t))
        truncerr = zero(real(scalartype(t)))
        return TensorKit._empty_svdtensors(t)..., truncerr
    end

    # compute (truncated) SVD factorization for each block
    generator = Base.Iterators.map(blocksectors(t)) do c
        howmany = min(blockdim(domain(t), c), blockdim(codomain(t), c))
        if trunc isa TruncationSpace
            howmany = min(howmany, blockdim(trunc.space, c))
        end
        V = codomain(t) ← spacetype(t)(c => 1)
        x₀ = rand!(similar(t, V))
        svals, Uvecs, Vvecs, info = svdsolve(t, x₀, howmany, :LR, alg)
        info.converged >= howmany || @warn "SVD did not converge" info
        resize!.((svals, Uvecs, Vvecs), howmany)
        S = DiagonalTensorMap(svals, spacetype(t)(c => howmany))
        U = reduce(catdomain, Uvecs)
        V = reduce(catdomain, Vvecs)'
        return U, S, V
    end
    U = mapreduce(Base.Fix2(getindex, 1), catdomain, generator)
    Σ = mapreduce(Base.Fix2(getindex, 2), ⊕, generator)
    V⁺ = mapreduce(Base.Fix2(getindex, 3), catcodomain, generator)

    # for now no way of computing truncation error so just return NaN
    Ttrunc = real(scalartype(t))
    truncerr = trunc === NoTruncation() ? zero(Ttrunc) : Ttrunc(NaN)

    return U, Σ, V⁺, truncerr
end

end
