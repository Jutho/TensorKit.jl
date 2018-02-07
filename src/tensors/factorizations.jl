# Tensor factorization
#----------------------
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
*   `trunbelow(ϵ)`: truncates such that every singular value is larger then `ϵ` ;

The `svd` also returns the truncation error `truncerr`, computed as the `p` norm of the
singular values that were truncated.

Orthogonality requires `spacetype(t)<:InnerProductSpace`, and `svd(!)` is currently
only implemented for `spacetype(t)<:EuclideanSpace`.
"""
svd(t::AbstractTensorMap, p1::IndexTuple, p2::IndexTuple, trunc::TruncationScheme = NoTruncation(), p::Real = 2) = svd!(permuteind(t, p1, p2; copy = true), trunc, p)

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
eig(t::AbstractTensorMap, p1::IndexTuple, p2::IndexTuple) = eig!(permuteind(t, p1, p2; copy = true))

svd(t::AbstractTensorMap, trunc::TruncationScheme = NoTruncation(), p::Real = 2) = svd!(copy(t), trunc, p)
leftorth(t::AbstractTensorMap, alg::OrthogonalFactorizationAlgorithm = QRpos()) = leftorth!(copy(t), alg)
rightorth(t::AbstractTensorMap, alg::OrthogonalFactorizationAlgorithm = LQpos()) = rightorth!(copy(t), alg)
leftnull(t::AbstractTensorMap, alg::OrthogonalFactorizationAlgorithm = QRpos()) = leftnull!(copy(t), alg)
rightnull(t::AbstractTensorMap, alg::OrthogonalFactorizationAlgorithm = LQpos()) = rightnull!(copy(t), alg)
eig(t::AbstractTensorMap) = eig!(copy(t))

# Orthogonal factorizations (mutation for recycling memory): only correct if Euclidean inner product
#----------------------------------------------------------------------------------------------------
function leftorth!(t::AdjointTensorMap{S}, alg::OrthogonalFactorizationAlgorithm = QRpos()) where {S<:EuclideanSpace}
    return map(adjoint, reverse(rightorth!(adjoint(t), alg')))
end
function rightorth!(t::AdjointTensorMap{S}, alg::OrthogonalFactorizationAlgorithm = QRpos()) where {S<:EuclideanSpace}
    return map(adjoint, reverse(leftorth!(adjoint(t), alg')))
end
function leftnull!(t::AdjointTensorMap{S}, alg::OrthogonalFactorizationAlgorithm = QRpos()) where {S<:EuclideanSpace}
    return adjoint(rightnull!(adjoint(t), alg'))
end
function rightnull!(t::AdjointTensorMap{S}, alg::OrthogonalFactorizationAlgorithm = QRpos()) where {S<:EuclideanSpace}
    return adjoint(leftnull!(adjoint(t), alg'))
end
function svd!(t::AdjointTensorMap{S}, trunc::TruncationScheme = NoTruncation(), p::Real = 2) where {S<:EuclideanSpace}
    return map(adjoint, reverse(svd!(adjoint(t), trunc, p)))
end

function leftorth!(t::TensorMap{S}, alg::OrthogonalFactorizationAlgorithm = QRpos()) where {S<:EuclideanSpace}
    if sectortype(t) == Trivial
        Q, R = leftorth!(block(t, Trivial()), alg)
        V = S(size(Q,2))
        return TensorMap(Q, codomain(t)←V), TensorMap(R, V←domain(t))
    else
        Qdata = empty(t.data)
        Rdata = empty(t.data)
        dims = SectorDict{sectortype(t), Int}()
        for c in blocksectors(t)
            Q, R = leftorth!(block(t,c), alg)
            Qdata[c] = Q
            Rdata[c] = R
            dims[c] = size(Q,2)
        end
        if length(domain(t)) == 1
            V = domain(t)[1]
        elseif length(codomain(t)) == 1
            V = codomain(t)[1]
        else
            V = S(dims)
        end
        return TensorMap(Qdata, codomain(t)←V), TensorMap(Rdata, V←domain(t))
    end
end
function leftnull!(t::TensorMap{S}, alg::OrthogonalFactorizationAlgorithm = QRpos()) where {S<:EuclideanSpace}
    if sectortype(t) == Trivial
        N = leftnull!(block(t, Trivial()), alg)
        W = S(size(N, 2))
        return TensorMap(N, codomain(t)←W)
    else
        V = codomain(t)
        Ndata = empty(t.data)
        dims = SectorDict{sectortype(t), Int}()
        for c in blocksectors(V)
            N = leftnull!(block(t,c), alg)
            Ndata[c] = N
            dims[c] = size(N,2)
        end
        W = S(dims)
        return TensorMap(Ndata, V←W)
    end
end
function rightorth!(t::TensorMap{S}, alg::OrthogonalFactorizationAlgorithm = LQpos()) where {S<:EuclideanSpace}
    if sectortype(t) == Trivial
        L, Q = rightorth!(block(t, Trivial()), alg)
        V = S(size(Q, 1))
        return TensorMap(L, codomain(t)←V), TensorMap(Q, V←domain(t))
    else
        Ldata = empty(t.data)
        Qdata = empty(t.data)
        dims = SectorDict{sectortype(t), Int}()
        for c in blocksectors(t)
            L, Q = rightorth!(block(t,c), alg)
            Ldata[c] = L
            Qdata[c] = Q
            dims[c] = size(Q,1)
        end
        if length(domain(t)) == 1
            V = domain(t)[1]
        elseif length(codomain(t)) == 1
            V = codomain(t)[1]
        else
            V = S(dims)
        end
        return TensorMap(Ldata, codomain(t)←V), TensorMap(Qdata, V←domain(t))
    end
end
function rightnull!(t::TensorMap{S}, alg::OrthogonalFactorizationAlgorithm = LQpos()) where {S<:EuclideanSpace}
    if sectortype(t) == Trivial
        N = rightnull!(block(t, Trivial()), alg)
        W = S(size(N, 1))
        return TensorMap(N, W←domain(t))
    else
        V = domain(t)
        Ndata = empty(t.data)
        A = valtype(Ndata)
        dims = SectorDict{sectortype(t), Int}()
        for c in blocksectors(V)
            N = rightnull!(block(t,c), alg)
            Ndata[c] = N
            dims[c] = size(N,1)
        end
        W = S(dims)
        return TensorMap(Ndata, W←V)
    end
end
function svd!(t::TensorMap{S}, trunc::TruncationScheme = NoTruncation(), p::Real = 2) where {S<:EuclideanSpace}
    if sectortype(t) == Trivial
        U,Σ,V = svd!(block(t, Trivial()))
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
        s = start(it)
        if done(it, s)
            fakerealdata = real(storagetype(t)(uninitialized, (0,0)))
            emptyrealdata = SectorDict{G,typeof(fakerealdata)}()
            W = S(dims)
            truncerr = abs(zero(eltype(t)))
            return TensorMap(empty(t.data), codomain(t)←W), TensorMap(emptyrealdata, W←W), TensorMap(empty(t.data), W←domain(t)), truncerr
        end
        c, s = next(it, s)
        U,Σ,V = svd!(block(t,c))
        Udata = SectorDict(c=>U)
        Σdata = SectorDict(c=>Σ)
        Vdata = SectorDict(c=>V)
        dims[c] = length(Σ)
        while !done(it, s)
            c, s = next(it, s)
            U,Σ,V = svd!(block(t,c))
            Udata[c] = U
            Σdata[c] = Σ
            Vdata[c] = V
            dims[c] = length(Σ)
        end
        if !isa(trunc, NoTruncation)
            Σdata, truncerr = _truncate!(Σdata, trunc, p)
            truncdims = SectorDict{sectortype(t), Int}()
            for c in blocksectors(t)
                truncdim = length(Σdata[c])
                truncdims[c] = truncdim
                if truncdim != dims[c]
                    Udata[c] = Udata[c][:, 1:truncdim]
                    Vdata[c] = Vdata[c][1:truncdim, :]
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

function _truncate!(v::AbstractVector, trunc::TruncationScheme, p::Real = 2)
    fullnorm = vecnorm(v, p)
    truncerr = zero(fullnorm)
    if isa(trunc, NoTruncation)
        # don't do anything
    elseif isa(trunc, TruncationError)
        dmax = length(v)
        dtrunc = dmax
        while true
            dtrunc -= 1
            truncerr = vecnorm(view(v, dtrunc+1:dmax), p)
            if truncerr / fullnorm > trunc.ϵ
                dtrunc += 1
                break
            end
        end
        truncerr = vecnorm(view(v, dtrunc+1:dmax), p)
        resize!(v, dtrunc)
    elseif isa(trunc, TruncationDimension)
        dmax = length(v)
        dtrunc = min(dmax, trunc.dim)
        truncerr = vecnorm(view(v, dtrunc+1:dmax), p)
        resize!(v, dtrunc)
    elseif isa(trunc, TruncateBelow)
        dtrunc   = findlast(x->(x>trunc.ϵ), v)
        truncerr = vecnorm(view(v, dtrunc+1:length(v)), p)
        resize!(v, dtrunc)
    else
        error("unknown truncation scheme")
    end
    return v, truncerr
end
function _truncate!(V::SectorDict{G,<:AbstractVector}, trunc::TruncationScheme, p = 2) where {G<:Sector}
    it = keys(V)
    fullnorm = _vecnorm((c=>V[c] for c in it), p)
    truncerr = zero(fullnorm)
    T = eltype(valtype(V))
    if isa(trunc, NoTruncation)
        # don't do anything
    elseif isa(trunc, TruncationError)
        truncdim = SectorDict{G,Int}(c=>length(v) for (c,v) in V)
        maxdim = copy(truncdim)
        while true
            s = start(it)
            c, s = next(it, s)
            cmin = c
            vmin = dim(c)^(1/p)*V[c][truncdim[c]]
            while !done(it, s)
                c, s = next(it, s)
                if truncdim[c] > 0
                    v = dim(c)^(1/p)*V[c][truncdim[c]]
                    if v < vmin
                        cmin, vmin = c, v
                    end
                end
            end
            truncdim[cmin] -= 1
            truncerr = _vecnorm((c=>view(V[c],truncdim[c]+1:maxdim[c]) for c in it), p)
            if truncerr / fullnorm > trunc.ϵ
                truncdim[cmin] += 1
                break
            end
        end
        truncerr = vecnorm((convert(T, dim(c))^(1/p)*vecnorm(view(Σdata[c],truncdim[c]+1:maxdim[c]), p) for c in it), p)
        for c in it
            resize!(V[c], truncdim[c])
        end
    elseif isa(trunc, TruncationDimension)
        truncdim = SectorDcit{G,Int}(c=>length(v) for (c,v) in V)
        maxdim = copy(truncdim)
        while sum(c->dim(c)*truncdim[c], it) > trunc.dim
            s = start(it)
            c, s = next(it, s)
            cmin = c
            vmin = dim(c)^(1/p)*V[c][truncdim[c]]
            while !done(it, s)
                c, s = next(it, s)
                if truncdim[c] > 0
                    v = dim(c)^(1/p)*V[c][truncdim[c]]
                    if v < vmin
                        cmin, vmin = c, v
                    end
                end
            end
            truncdim[cmin] -= 1
        end
        truncerr = vecnorm((convert(T, dim(c))^(1/p)*vecnorm(view(V[c],truncdim[c]+1:maxdim[c]), p) for c in it), p)
        for c in it
            resize!(V[c], truncdim[c])
        end
    elseif isa(trunc, TruncationSpace)
        for c in it
            if length(V[c]) > dim(trunc.space, c)
                resize!(V[c], dim(trunc.space, c))
            end
        end
    elseif isa(trunc, TruncateBelow)
        truncdim = SectorDict{G,Int}(c=>length(v) for (c,v) in V)
        maxdim = copy(truncdim)
        for c in it
            newdim = findlast(x->(x>trunc.ϵ), V[c] )
            if newdim != 0
                truncdim[c] = newdim
            else
                delete!(truncdim, c)
            end
        end
        truncerr = vecnorm((convert(T, dim(c))^(1/p)*vecnorm(view(V[c],truncdim[c]+1:maxdim[c]), p) for c in keys(truncdim)), p)
        for c in keys(truncdim)
            resize!(V[c], truncdim[c])
        end
    else
        error("unknown truncation scheme")
    end
    return V, truncerr
end
