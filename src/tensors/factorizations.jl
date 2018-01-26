# Factorization
#---------------
"""
    svd(t::AbstractTensorMap, leftind::Tuple, rightind::Tuple, truncation::TruncationScheme = notrunc(), p::Real = 2) -> U,S,V,truncerr'

Performs the singular value decomposition such that `permute(t,leftind,rightind) = U * S *V`.

If `leftind` and `rightind` are not specified, the current partition of left and right indices
of `t` is used. In that case, less memory is allocated if one allows the data in `t` to
be destroyed/overwritten, by using `svd!(t, truncation = notrunc(), p = 2)`.

A truncation parameter can be specified for the new internal dimension, in which case
a truncated singular value decomposition will be computed. Choices are:
*   `notrunc()`: no truncation (default);
*   `truncerr(ϵ)`: truncates such that the p-norm of the truncated singular values is smaller than `ϵ` times the p-norm of all singular values;
*   `truncdim(χ)`: truncates such that the equivalent total dimension of the internal vector space is no larger than `χ`;
*   `truncspace(V)`: truncates such that the dimension of the internal vector space is smaller than that of `V` in any sector.

The `svd` also returns the truncation error `truncerr`, computed as the `p` norm of the
singular values that were truncated.

Orthogonality requires `spacetype(t)<:InnerProductSpace`, and `svd(!)` is currently
only implemented for `spacetype(t)<:EuclideanSpace`.
"""
Base.svd(t::AbstractTensorMap, p1::IndexTuple, p2::IndexTuple, trunc::TruncationScheme = NoTruncation(), p::Real = 2) = svd!(permuteind(t, p1, p2; copy = true), trunc, p)

"""
    leftorth(t::AbstractTensorMap, leftind::Tuple, rightind::Tuple, alg::OrthogonalFactorizationAlgorithm = QRpos()) -> Q, R

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
leftorth(t::AbstractTensorMap, p1::IndexTuple, p2::IndexTuple, alg::OrthogonalFactorizationAlgorithm = QRpos()) = leftorth!(permuteind(t, p1, p2; copy = true), alg)

"""
    rightorth(t::AbstractTensorMap, leftind::Tuple, rightind::Tuple, alg::OrthogonalFactorizationAlgorithm = LQpos()) -> L, Q

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
rightorth(t::AbstractTensorMap, p1::IndexTuple, p2::IndexTuple, alg::OrthogonalFactorizationAlgorithm = LQpos()) = rightorth!(permuteind(t, p1, p2; copy = true), alg)

"""
    leftnull(t::AbstractTensor, leftind::Tuple, rightind::Tuple, alg::OrthogonalFactorizationAlgorithm = QRpos()) -> N

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
leftnull(t::AbstractTensorMap, p1::IndexTuple, p2::IndexTuple, alg::OrthogonalFactorizationAlgorithm = QR()) = leftnull!(permuteind(t, p1, p2; copy = true), alg)

"""
    rightnull(t::AbstractTensor, leftind::Tuple, rightind::Tuple, alg::OrthogonalFactorizationAlgorithm = LQ()) -> N

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
rightnull(t::AbstractTensorMap, p1::IndexTuple, p2::IndexTuple, alg::OrthogonalFactorizationAlgorithm = LQ()) = rightnull!(permuteind(t, p1, p2; copy = true), alg)

"""
    eig(t::AbstractTensor, leftind::Tuple, rightind::Tuple) -> D, V

Compute eigenvalue factorization of tensor `t` as linear map from `rightind` to `leftind`.

If `leftind` and `rightind` are not specified, the current partition of left and right indices
of `t` is used. In that case, less memory is allocated if one allows the data in `t` to
be destroyed/overwritten, by using `eig!(t)`.
"""
Base.eig(t::AbstractTensorMap, p1::IndexTuple, p2::IndexTuple) = eig!(permuteind(t, p1, p2; copy = true))

Base.svd(t::AbstractTensorMap, trunc::TruncationScheme = NoTruncation(), p::Real = 2) = svd!(copy(t), trunc, p)
leftorth(t::AbstractTensorMap, alg::OrthogonalFactorizationAlgorithm = QRpos()) = leftorth!(copy(t), alg)
rightorth(t::AbstractTensorMap, alg::OrthogonalFactorizationAlgorithm = LQpos()) = rightorth!(copy(t), alg)
leftnull(t::AbstractTensorMap, alg::OrthogonalFactorizationAlgorithm = QRpos()) = leftnull!(copy(t), alg)
rightnull(t::AbstractTensorMap, alg::OrthogonalFactorizationAlgorithm = LQpos()) = rightnull!(copy(t), alg)
Base.eig(t::AbstractTensorMap) = eig!(copy(t))

Base.exp(t::AbstractTensorMap) = exp!(copy(t))
