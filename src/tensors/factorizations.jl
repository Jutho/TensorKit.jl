# Tensor factorization
#----------------------
"""
    svd(t::AbstractTensorMap, leftind::Tuple, rightind::Tuple; truncation::TruncationScheme = notrunc(), p::Real = 2, alg::Union{SVD,SDD} = SDD()) -> U,S,V,truncerr

Compute the singular value decomposition such that `permute(t,leftind,rightind) = U * S *V`.

If `leftind` and `rightind` are not specified, the current partition of left and right indices
of `t` is used. In that case, less memory is allocated if one allows the data in `t` to
be destroyed/overwritten, by using `svd!(t, truncation = notrunc(), p = 2)`.

A truncation parameter can be specified for the new internal dimension, in which case
a truncated singular value decomposition will be computed. Choices are:
*   `notrunc()`: no truncation (default);
*   `truncerr(ϵ)`: truncates such that the p-norm of the truncated singular values is smaller than `ϵ` times the p-norm of all singular values;
*   `truncdim(χ)`: truncates such that the equivalent total dimension of the internal vector space is no larger than `χ`;
*   `truncspace(V)`: truncates such that the dimension of the internal vector space is smaller than that of `V` in any sector.
*   `trunbelow(ϵ)`: truncates such that every singular value is larger then `ϵ` ;

The `svd` also returns the truncation error `truncerr`, computed as the `p` norm of the
singular values that were truncated.

THe keyword `alg` can be equal to `SVD()` or `SDD()`, corresponding to the underlying LAPACK
algorithm that computes the decomposition (`_gesvd` or `_gesdd`).

Orthogonality requires `spacetype(t)<:InnerProductSpace`, and `svd(!)` is currently
only implemented for `spacetype(t)<:EuclideanSpace`.
"""
LinearAlgebra.svd(t::AbstractTensorMap, p1::IndexTuple, p2::IndexTuple; trunc::TruncationScheme = NoTruncation(), p::Real = 2, alg::Union{SVD,SDD} = SDD()) = svd!(permuteind(t, p1, p2; copy = true); trunc = trunc, p = p, alg = alg)

"""
    leftorth(t::AbstractTensorMap, leftind::Tuple, rightind::Tuple; alg::OrthogonalFactorizationAlgorithm = QRpos()) -> Q, R

Create orthonormal basis `Q` for indices in `leftind`, and remainder `R` such that
`permute(t,leftind,rightind) = Q*R`.

If `leftind` and `rightind` are not specified, the current partition of left and right indices
of `t` is used. In that case, less memory is allocated if one allows the data in `t` to
be destroyed/overwritten, by using `leftorth!(t, alg = QRpos())`.

Different algorithms are available, namely `QR()`, `QRpos()`, `SVD()` and `Polar()`.
`QR()` and `QRpos()` use a standard QR decomposition, producing an upper triangular
matrix `R`. `Polar()` produces a Hermitian and positive semidefinite `R`. `QRpos()`
corrects the standard QR decomposition such that the diagonal elements of `R`
are positive. Only `QRpos()` and `Polar()` are uniqe (no residual freedom) so that
they always return the same result for the same input tensor `t`.

Orthogonality requires `spacetype(t)<:InnerProductSpace`, and `leftorth(!)` is currently
only implemented for `spacetype(t)<:EuclideanSpace`.
"""
leftorth(t::AbstractTensorMap, p1::IndexTuple, p2::IndexTuple; alg::OrthogonalFactorizationAlgorithm = QRpos()) = leftorth!(permuteind(t, p1, p2; copy = true); alg = alg)

"""
    rightorth(t::AbstractTensorMap, leftind::Tuple, rightind::Tuple; alg::OrthogonalFactorizationAlgorithm = LQpos()) -> L, Q

Create orthonormal basis `Q` for indices in `rightind`, and remainder `L` such that
`permute(t,leftind,rightind) = L*Q`.

If `leftind` and `rightind` are not specified, the current partition of left and right indices
of `t` is used. In that case, less memory is allocated if one allows the data in `t` to
be destroyed/overwritten, by using `rightorth!(t, alg = LQpos())`.

Different algorithms are available, namely `LQ()`, `LQpos()`, `RQ()`, `RQpos()`,
`SVD()` and `Polar()`. `LQ()` and `LQpos()` produce a lower triangular matrix `L`
and are computed using a QR decomposition of the transpose. `RQ()` and `RQpos()`
produce an upper triangular remainder `L` and only works if the total left dimension
is smaller than or equal to the total right dimension. `LQpos()` and `RQpos()` add
an additional correction such that the diagonal elements of `L` are positive.
`Polar()` produces a Hermitian and positive semidefinite `L`. Only `LQpos()`, `RQpos()`
and `Polar()` are uniqe (no residual freedom) so that they always return the same
result for the same input tensor `t`.

Orthogonality requires `spacetype(t)<:InnerProductSpace`, and `rightorth(!)` is currently
only implemented for `spacetype(t)<:EuclideanSpace`.
"""
rightorth(t::AbstractTensorMap, p1::IndexTuple, p2::IndexTuple; alg::OrthogonalFactorizationAlgorithm = LQpos()) = rightorth!(permuteind(t, p1, p2; copy = true); alg = alg)

"""
    leftnull(t::AbstractTensor, leftind::Tuple, rightind::Tuple; alg::OrthogonalFactorizationAlgorithm = QRpos()) -> N

Create orthonormal basis for the orthogonal complement of the support of the indices in
`leftind`, such that `N' * permute(t, leftind, rightind) = 0`.

If `leftind` and `rightind` are not specified, the current partition of left and right indices
of `t` is used. In that case, less memory is allocated if one allows the data in `t` to
be destroyed/overwritten, by using `leftnull!(t, alg = QRpos())`.

Different algorithms are available, namely `QR()` and `SVD()`. The former assumes
that the matrix is full rank, the latter does not. For `leftnull`, there is no
distinction between `QR()` and `QRpos()`.

Orthogonality requires `spacetype(t)<:InnerProductSpace`, and `leftnull(!)` is currently
only implemented for `spacetype(t)<:EuclideanSpace`.
"""
leftnull(t::AbstractTensorMap, p1::IndexTuple, p2::IndexTuple; alg::OrthogonalFactorizationAlgorithm = QR()) = leftnull!(permuteind(t, p1, p2; copy = true); alg = alg)

"""
    rightnull(t::AbstractTensor, leftind::Tuple, rightind::Tuple; alg::OrthogonalFactorizationAlgorithm = LQ()) -> N

Create orthonormal basis for the orthogonal complement of the support of the indices in
`rightind`, such that `permute(t, leftind, rightind)*N' = 0`.

If `leftind` and `rightind` are not specified, the current partition of left and right indices
of `t` is used. In that case, less memory is allocated if one allows the data in `t` to
be destroyed/overwritten, by using `rightnull!(t, alg = LQpos())`.

Different algorithms are available, namely `LQ()` and `SVD()`. The former assumes
that the matrix is full rank, the latter does not. For `rightnull`, there is no
distinction between `LQ()` and `LQpos()`.

Orthogonality requires `spacetype(t)<:InnerProductSpace`, and `rightnull(!)` is currently
only implemented for `spacetype(t)<:EuclideanSpace`.
"""
rightnull(t::AbstractTensorMap, p1::IndexTuple, p2::IndexTuple; alg::OrthogonalFactorizationAlgorithm = LQ()) = rightnull!(permuteind(t, p1, p2; copy = true); alg = alg)

"""
    eig(t::AbstractTensor, leftind::Tuple, rightind::Tuple) -> D, V

Compute eigenvalue factorization of tensor `t` as linear map from `rightind` to `leftind`.

If `leftind` and `rightind` are not specified, the current partition of left and right indices
of `t` is used. In that case, less memory is allocated if one allows the data in `t` to
be destroyed/overwritten, by using `eig!(t)`.
"""
eig(t::AbstractTensorMap, p1::IndexTuple, p2::IndexTuple) = eig!(permuteind(t, p1, p2; copy = true))

LinearAlgebra.svd(t::AbstractTensorMap; trunc::TruncationScheme = NoTruncation(), p::Real = 2, alg::Union{SVD,SDD} = SDD()) = svd!(copy(t); trunc = trunc, p = p, alg = alg)
leftorth(t::AbstractTensorMap; alg::OrthogonalFactorizationAlgorithm = QRpos()) = leftorth!(copy(t); alg = alg)
rightorth(t::AbstractTensorMap; alg::OrthogonalFactorizationAlgorithm = LQpos()) = rightorth!(copy(t); alg = alg)
leftnull(t::AbstractTensorMap; alg::OrthogonalFactorizationAlgorithm = QRpos()) = leftnull!(copy(t); alg = alg)
rightnull(t::AbstractTensorMap; alg::OrthogonalFactorizationAlgorithm = LQpos()) = rightnull!(copy(t); alg = alg)
eig(t::AbstractTensorMap) = eig!(copy(t))

# Orthogonal factorizations (mutation for recycling memory): only correct if Euclidean inner product
#----------------------------------------------------------------------------------------------------
function leftorth!(t::AdjointTensorMap{S}; alg::OrthogonalFactorizationAlgorithm = QRpos()) where {S<:EuclideanSpace}
    return map(adjoint, reverse(rightorth!(adjoint(t); alg = alg')))
end
function rightorth!(t::AdjointTensorMap{S}; alg::OrthogonalFactorizationAlgorithm = QRpos()) where {S<:EuclideanSpace}
    return map(adjoint, reverse(leftorth!(adjoint(t); alg = alg')))
end
function leftnull!(t::AdjointTensorMap{S}; alg::OrthogonalFactorizationAlgorithm = QRpos()) where {S<:EuclideanSpace}
    return adjoint(rightnull!(adjoint(t); alg = alg'))
end
function rightnull!(t::AdjointTensorMap{S}; alg::OrthogonalFactorizationAlgorithm = QRpos()) where {S<:EuclideanSpace}
    return adjoint(leftnull!(adjoint(t); alg = alg'))
end
function LinearAlgebra.svd!(t::AdjointTensorMap{S}; trunc::TruncationScheme = NoTruncation(), p::Real = 2, alg::Union{SVD,SDD} = SDD()) where {S<:EuclideanSpace}
    return map(adjoint, reverse(svd!(adjoint(t); trunc = trunc, p = p, alg = alg)))
end

function leftorth!(t::TensorMap{S}; alg::OrthogonalFactorizationAlgorithm = QRpos()) where {S<:EuclideanSpace}
    if sectortype(t) === Trivial
        Q, R = _leftorth!(block(t, Trivial()), alg)
        V = S(size(Q,2))
        return TensorMap(Q, codomain(t)←V), TensorMap(R, V←domain(t))
    else
        Qdata = empty(t.data)
        Rdata = empty(t.data)
        dims = SectorDict{sectortype(t), Int}()
        for c in blocksectors(t)
            Q, R = _leftorth!(block(t,c), alg)
            Qdata[c] = Q
            Rdata[c] = R
            dims[c] = size(Q,2)
        end
        if length(domain(t)) == 1
            V = domain(t)[1]
            if V.dims != dims
                V = S(dims)
            end
        elseif length(codomain(t)) == 1
            V = codomain(t)[1]
            if V.dims != dims
                V = S(dims)
            end
        else
            V = S(dims)
        end
        return TensorMap(Qdata, codomain(t)←V), TensorMap(Rdata, V←domain(t))
    end
end
function leftnull!(t::TensorMap{S}; alg::OrthogonalFactorizationAlgorithm = QRpos()) where {S<:EuclideanSpace}
    if sectortype(t) === Trivial
        N = _leftnull!(block(t, Trivial()), alg)
        W = S(size(N, 2))
        return TensorMap(N, codomain(t)←W)
    else
        V = codomain(t)
        Ndata = empty(t.data)
        dims = SectorDict{sectortype(t), Int}()
        for c in blocksectors(V)
            N = _leftnull!(block(t,c), alg)
            Ndata[c] = N
            dims[c] = size(N,2)
        end
        W = S(dims)
        return TensorMap(Ndata, V←W)
    end
end
function rightorth!(t::TensorMap{S}; alg::OrthogonalFactorizationAlgorithm = LQpos()) where {S<:EuclideanSpace}
    if sectortype(t) === Trivial
        L, Q = _rightorth!(block(t, Trivial()), alg)
        V = S(size(Q, 1))
        return TensorMap(L, codomain(t)←V), TensorMap(Q, V←domain(t))
    else
        Ldata = empty(t.data)
        Qdata = empty(t.data)
        dims = SectorDict{sectortype(t), Int}()
        for c in blocksectors(t)
            L, Q = _rightorth!(block(t,c), alg)
            Ldata[c] = L
            Qdata[c] = Q
            dims[c] = size(Q,1)
        end
        if length(domain(t)) == 1
            V = domain(t)[1]
            if V.dims != dims
                V = S(dims)
            end
        elseif length(codomain(t)) == 1
            V = codomain(t)[1]
            if V.dims != dims
                V = S(dims)
            end
        else
            V = S(dims)
        end
        return TensorMap(Ldata, codomain(t)←V), TensorMap(Qdata, V←domain(t))
    end
end
function rightnull!(t::TensorMap{S}; alg::OrthogonalFactorizationAlgorithm = LQpos()) where {S<:EuclideanSpace}
    if sectortype(t) === Trivial
        N = _rightnull!(block(t, Trivial()), alg)
        W = S(size(N, 1))
        return TensorMap(N, W←domain(t))
    else
        V = domain(t)
        Ndata = empty(t.data)
        A = valtype(Ndata)
        dims = SectorDict{sectortype(t), Int}()
        for c in blocksectors(V)
            N = _rightnull!(block(t,c), alg)
            Ndata[c] = N
            dims[c] = size(N,1)
        end
        W = S(dims)
        return TensorMap(Ndata, W←V)
    end
end
function LinearAlgebra.svd!(t::TensorMap{S}; trunc::TruncationScheme = NoTruncation(), p::Real = 2, alg::Union{SVD,SDD} = SDD()) where {S<:EuclideanSpace}
    if sectortype(t) === Trivial
        U,Σ,V = _svd!(block(t, Trivial()), alg)
        dmax = length(Σ)
        Σ, truncerr = _truncate!(Σ, trunc, p)
        d = length(Σ)
        W = S(d)
        if d < dmax
            U = U[:,1:d]
            V = V[1:d,:]
        end
        Σm = copyto!(similar(Σ, (d,d)), Diagonal(Σ))
        return TensorMap(U, codomain(t)←W), TensorMap(Σm, W←W), TensorMap(V, W←domain(t)), truncerr
        #TODO: make this work with Diagonal(Σ) in such a way that it is type stable and
        # robust for all further operations on that tensor
    else
        G = sectortype(t)
        it = blocksectors(t)
        dims = SectorDict{sectortype(t), Int}()
        next = iterate(it)
        if next === nothing
            emptyrealdata = SectorDict{G,similarstoragetype(t, real(eltype(t)))}()
            W = S(dims)
            truncerr = abs(zero(eltype(t)))
            return TensorMap(empty(t.data), codomain(t)←W), TensorMap(emptyrealdata, W←W), TensorMap(empty(t.data), W←domain(t)), truncerr
        end
        c, s = next
        U,Σ,V = _svd!(block(t,c), alg)
        Udata = SectorDict(c=>U)
        Σdata = SectorDict(c=>Σ)
        Vdata = SectorDict(c=>V)
        dims[c] = length(Σ)
        next = iterate(it, s)
        while next !== nothing
            c, s = next
            U,Σ,V = _svd!(block(t,c), alg)
            Udata[c] = U
            Σdata[c] = Σ
            Vdata[c] = V
            dims[c] = length(Σ)
            next = iterate(it, s)
        end
        if !isa(trunc, NoTruncation)
            Σdata, truncerr = _truncate!(Σdata, trunc, p)
            truncdims = SectorDict{sectortype(t), Int}()
            for c in blocksectors(t)
                truncdim = length(Σdata[c])
                if truncdim != 0
                    truncdims[c] = truncdim
                    if truncdim != dims[c]
                        Udata[c] = Udata[c][:, 1:truncdim]
                        Vdata[c] = Vdata[c][1:truncdim, :]
                    end
                else
                    delete!(Udata, c)
                    delete!(Vdata, c)
                    delete!(Σdata, c)
                end
            end
            dims = truncdims
            W = S(dims)
        else
            if length(domain(t)) == 1
                W = domain(t)[1]
            elseif length(codomain(t)) == 1
                W = codomain(t)[1]
            else
                W = S(dims)
            end
            truncerr = abs(zero(eltype(t)))
        end
        Σmdata = SectorDict(c=>copyto!(similar(Σ, length(Σ), length(Σ)), Diagonal(Σ)) for (c,Σ) in Σdata)
        return TensorMap(Udata, codomain(t)←W), TensorMap(Σmdata, W←W), TensorMap(Vdata, W←domain(t)), truncerr
    end
end
