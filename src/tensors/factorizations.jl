# Tensor factorization
#----------------------
"""
    tsvd(t::AbstractTensorMap, (leftind, rightind)::Index2Tuple;
        trunc::TruncationScheme = notrunc(), p::Real = 2, alg::Union{SVD, SDD} = SDD())
        -> U, S, V, ϵ

Compute the (possibly truncated) singular value decomposition such that
`norm(permute(t, (leftind, rightind)) - U * S * V) ≈ ϵ`, where `ϵ` thus represents the truncation error.

If `leftind` and `rightind` are not specified, the current partition of left and right
indices of `t` is used. In that case, less memory is allocated if one allows the data in
`t` to be destroyed/overwritten, by using `tsvd!(t, trunc = notrunc(), p = 2)`.

A truncation parameter `trunc` can be specified for the new internal dimension, in which
case a truncated singular value decomposition will be computed. Choices are:
*   `notrunc()`: no truncation (default);
*   `truncerr(η::Real)`: truncates such that the p-norm of the truncated singular values is
    smaller than `η` times the p-norm of all singular values;
*   `truncdim(χ::Int)`: truncates such that the equivalent total dimension of the internal
    vector space is no larger than `χ`;
*   `truncspace(V)`: truncates such that the dimension of the internal vector space is
    smaller than that of `V` in any sector.
*   `truncbelow(χ::Real)`: truncates such that every singular value is larger then `χ` ;

The method `tsvd` also returns the truncation error `ϵ`, computed as the `p` norm of the
singular values that were truncated.

THe keyword `alg` can be equal to `SVD()` or `SDD()`, corresponding to the underlying LAPACK
algorithm that computes the decomposition (`_gesvd` or `_gesdd`).

Orthogonality requires `InnerProductStyle(t) <: HasInnerProduct`, and `tsvd(!)`
is currently only implemented for `InnerProductStyle(t) === EuclideanProduct()`.
"""
function tsvd(t::AbstractTensorMap, (p₁, p₂)::Index2Tuple; kwargs...)
    return tsvd!(permute(t, (p₁, p₂); copy=true); kwargs...)
end

"""
    leftorth(t::AbstractTensorMap, (leftind, rightind)::Index2Tuple;
                alg::OrthogonalFactorizationAlgorithm = QRpos()) -> Q, R

Create orthonormal basis `Q` for indices in `leftind`, and remainder `R` such that
`permute(t, (leftind, rightind)) = Q*R`.

If `leftind` and `rightind` are not specified, the current partition of left and right
indices of `t` is used. In that case, less memory is allocated if one allows the data in `t`
to be destroyed/overwritten, by using `leftorth!(t, alg = QRpos())`.

Different algorithms are available, namely `QR()`, `QRpos()`, `SVD()` and `Polar()`. `QR()`
and `QRpos()` use a standard QR decomposition, producing an upper triangular matrix `R`.
`Polar()` produces a Hermitian and positive semidefinite `R`. `QRpos()` corrects the
standard QR decomposition such that the diagonal elements of `R` are positive. Only
`QRpos()` and `Polar()` are unique (no residual freedom) so that they always return the same
result for the same input tensor `t`.

Orthogonality requires `InnerProductStyle(t) <: HasInnerProduct`, and
`leftorth(!)` is currently only implemented for 
    `InnerProductStyle(t) === EuclideanProduct()`.
"""
function leftorth(t::AbstractTensorMap, (p₁, p₂)::Index2Tuple; kwargs...)
    return leftorth!(permute(t, (p₁, p₂); copy=true); kwargs...)
end

"""
    rightorth(t::AbstractTensorMap, (leftind, rightind)::Index2Tuple;
                alg::OrthogonalFactorizationAlgorithm = LQpos()) -> L, Q

Create orthonormal basis `Q` for indices in `rightind`, and remainder `L` such that
`permute(t, (leftind, rightind)) = L*Q`.

If `leftind` and `rightind` are not specified, the current partition of left and right
indices of `t` is used. In that case, less memory is allocated if one allows the data in `t`
to be destroyed/overwritten, by using `rightorth!(t, alg = LQpos())`.

Different algorithms are available, namely `LQ()`, `LQpos()`, `RQ()`, `RQpos()`, `SVD()` and
`Polar()`. `LQ()` and `LQpos()` produce a lower triangular matrix `L` and are computed using
a QR decomposition of the transpose. `RQ()` and `RQpos()` produce an upper triangular
remainder `L` and only works if the total left dimension is smaller than or equal to the
total right dimension. `LQpos()` and `RQpos()` add an additional correction such that the
diagonal elements of `L` are positive. `Polar()` produces a Hermitian and positive
semidefinite `L`. Only `LQpos()`, `RQpos()` and `Polar()` are unique (no residual freedom)
so that they always return the same result for the same input tensor `t`.

Orthogonality requires `InnerProductStyle(t) <: HasInnerProduct`, and
`rightorth(!)` is currently only implemented for 
`InnerProductStyle(t) === EuclideanProduct()`.
"""
function rightorth(t::AbstractTensorMap, (p₁, p₂)::Index2Tuple; kwargs...)
    return rightorth!(permute(t, (p₁, p₂); copy=true); kwargs...)
end

"""
    leftnull(t::AbstractTensor, (leftind, rightind)::Index2Tuple;
                alg::OrthogonalFactorizationAlgorithm = QRpos()) -> N

Create orthonormal basis for the orthogonal complement of the support of the indices in
`leftind`, such that `N' * permute(t, (leftind, rightind)) = 0`.

If `leftind` and `rightind` are not specified, the current partition of left and right
indices of `t` is used. In that case, less memory is allocated if one allows the data in `t`
to be destroyed/overwritten, by using `leftnull!(t, alg = QRpos())`.

Different algorithms are available, namely `QR()` (or equivalently, `QRpos()`), `SVD()` and
`SDD()`. The first assumes that the matrix is full rank and requires `iszero(atol)` and
`iszero(rtol)`. With `SVD()` and `SDD()`, `rightnull` will use the corresponding singular
value decomposition, and one can specify an absolute or relative tolerance for which
singular values are to be considered zero, where `max(atol, norm(t)*rtol)` is used as upper
bound.

Orthogonality requires `InnerProductStyle(t) <: HasInnerProduct`, and
`leftnull(!)` is currently only implemented for 
`InnerProductStyle(t) === EuclideanProduct()`.
"""
function leftnull(t::AbstractTensorMap, (p₁, p₂)::Index2Tuple; kwargs...)
    return leftnull!(permute(t, (p₁, p₂); copy=true); kwargs...)
end

"""
    rightnull(t::AbstractTensor, (leftind, rightind)::Index2Tuple;
                alg::OrthogonalFactorizationAlgorithm = LQ(),
                atol::Real = 0.0,
                rtol::Real = eps(real(float(one(scalartype(t)))))*iszero(atol)) -> N

Create orthonormal basis for the orthogonal complement of the support of the indices in
`rightind`, such that `permute(t, (leftind, rightind))*N' = 0`.

If `leftind` and `rightind` are not specified, the current partition of left and right
indices of `t` is used. In that case, less memory is allocated if one allows the data in `t`
to be destroyed/overwritten, by using `rightnull!(t, alg = LQpos())`.

Different algorithms are available, namely `LQ()` (or equivalently, `LQpos`), `SVD()` and
`SDD()`. The first assumes that the matrix is full rank and requires `iszero(atol)` and
`iszero(rtol)`. With `SVD()` and `SDD()`, `rightnull` will use the corresponding singular
value decomposition, and one can specify an absolute or relative tolerance for which
singular values are to be considered zero, where `max(atol, norm(t)*rtol)` is used as upper
bound.

Orthogonality requires `InnerProductStyle(t) <: HasInnerProduct`, and
`rightnull(!)` is currently only implemented for 
`InnerProductStyle(t) === EuclideanProduct()`.
"""
function rightnull(t::AbstractTensorMap, (p₁, p₂)::Index2Tuple; kwargs...)
    return rightnull!(permute(t, (p₁, p₂); copy=true); kwargs...)
end

"""
    eigen(t::AbstractTensor, (leftind, rightind)::Index2Tuple; kwargs...) -> D, V

Compute eigenvalue factorization of tensor `t` as linear map from `rightind` to `leftind`.

If `leftind` and `rightind` are not specified, the current partition of left and right
indices of `t` is used. In that case, less memory is allocated if one allows the data in `t`
to be destroyed/overwritten, by using `eigen!(t)`. Note that the permuted tensor on which
`eigen!` is called should have equal domain and codomain, as otherwise the eigenvalue
decomposition is meaningless and cannot satisfy
```
permute(t, (leftind, rightind)) * V = V * D
```

Accepts the same keyword arguments `scale` and `permute` as `eigen` of dense
matrices. See the corresponding documentation for more information.

See also `eig` and `eigh`
"""
function LinearAlgebra.eigen(t::AbstractTensorMap, (p₁, p₂)::Index2Tuple;
                             kwargs...)
    return eigen!(permute(t, (p₁, p₂); copy=true); kwargs...)
end

"""
    eig(t::AbstractTensor, (leftind, rightind)::Index2Tuple; kwargs...) -> D, V

Compute eigenvalue factorization of tensor `t` as linear map from `rightind` to `leftind`.
The function `eig` assumes that the linear map is not hermitian and returns type stable
complex valued `D` and `V` tensors for both real and complex valued `t`. See `eigh` for
hermitian linear maps

If `leftind` and `rightind` are not specified, the current partition of left and right
indices of `t` is used. In that case, less memory is allocated if one allows the data in
`t` to be destroyed/overwritten, by using `eig!(t)`. Note that the permuted tensor on
which `eig!` is called should have equal domain and codomain, as otherwise the eigenvalue
decomposition is meaningless and cannot satisfy
```
permute(t, (leftind, rightind)) * V = V * D
```

Accepts the same keyword arguments `scale` and `permute` as `eigen` of dense
matrices. See the corresponding documentation for more information.

See also `eigen` and `eigh`.
"""
function eig(t::AbstractTensorMap, (p₁, p₂)::Index2Tuple; kwargs...)
    return eig!(permute(t, (p₁, p₂); copy=true); kwargs...)
end

"""
    eigh(t::AbstractTensorMap, (leftind, rightind)::Index2Tuple) -> D, V

Compute eigenvalue factorization of tensor `t` as linear map from `rightind` to `leftind`.
The function `eigh` assumes that the linear map is hermitian and `D` and `V` tensors with
the same `scalartype` as `t`. See `eig` and `eigen` for non-hermitian tensors. Hermiticity
requires that the tensor acts on inner product spaces, and the current implementation
requires `InnerProductStyle(t) === EuclideanProduct()`.

If `leftind` and `rightind` are not specified, the current partition of left and right
indices of `t` is used. In that case, less memory is allocated if one allows the data in
`t` to be destroyed/overwritten, by using `eigh!(t)`. Note that the permuted tensor on
which `eigh!` is called should have equal domain and codomain, as otherwise the eigenvalue
decomposition is meaningless and cannot satisfy
```
permute(t, (leftind, rightind)) * V = V * D
```

See also `eigen` and `eig`.
"""
function eigh(t::AbstractTensorMap, (p₁, p₂)::Index2Tuple)
    return eigh!(permute(t, (p₁, p₂); copy=true))
end

"""
    isposdef(t::AbstractTensor, (leftind, rightind)::Index2Tuple) -> ::Bool

Test whether a tensor `t` is positive definite as linear map from `rightind` to `leftind`.

If `leftind` and `rightind` are not specified, the current partition of left and right
indices of `t` is used. In that case, less memory is allocated if one allows the data in
`t` to be destroyed/overwritten, by using `isposdef!(t)`. Note that the permuted tensor on
which `isposdef!` is called should have equal domain and codomain, as otherwise it is
meaningless.
"""
function LinearAlgebra.isposdef(t::AbstractTensorMap, (p₁, p₂)::Index2Tuple)
    return isposdef!(permute(t, (p₁, p₂); copy=true))
end

function tsvd(t::AbstractTensorMap; trunc::TruncationScheme=NoTruncation(),
              p::Real=2, alg::Union{SVD,SDD}=SDD())
    return tsvd!(copy(t); trunc=trunc, p=p, alg=alg)
end
function leftorth(t::AbstractTensorMap; alg::OFA=QRpos(), kwargs...)
    return leftorth!(copy(t); alg=alg, kwargs...)
end
function rightorth(t::AbstractTensorMap; alg::OFA=LQpos(), kwargs...)
    return rightorth!(copy(t); alg=alg, kwargs...)
end
function leftnull(t::AbstractTensorMap; alg::OFA=QR(), kwargs...)
    return leftnull!(copy(t); alg=alg, kwargs...)
end
function rightnull(t::AbstractTensorMap; alg::OFA=LQ(), kwargs...)
    return rightnull!(copy(t); alg=alg, kwargs...)
end
LinearAlgebra.eigen(t::AbstractTensorMap; kwargs...) = eigen!(copy(t); kwargs...)
eig(t::AbstractTensorMap; kwargs...) = eig!(copy(t); kwargs...)
eigh(t::AbstractTensorMap; kwargs...) = eigh!(copy(t); kwargs...)
LinearAlgebra.isposdef(t::AbstractTensorMap) = isposdef!(copy(t))

# Orthogonal factorizations (mutation for recycling memory):
# only correct if Euclidean inner product
#------------------------------------------------------------------------------------------
function leftorth!(t::TensorMap;
                   alg::Union{QR,QRpos,QL,QLpos,SVD,SDD,Polar}=QRpos(),
                   atol::Real=zero(float(real(scalartype(t)))),
                   rtol::Real=(alg ∉ (SVD(), SDD())) ? zero(float(real(scalartype(t)))) :
                              eps(real(float(one(scalartype(t))))) * iszero(atol))
    InnerProductStyle(t) === EuclideanProduct() || throw_invalid_innerproduct(:leftorth!)
    if !iszero(rtol)
        atol = max(atol, rtol * norm(t))
    end
    I = sectortype(t)
    S = spacetype(t)
    A = storagetype(t)
    Qdata = SectorDict{I,A}()
    Rdata = SectorDict{I,A}()
    dims = SectorDict{I,Int}()
    for c in blocksectors(domain(t))
        isempty(block(t, c)) && continue
        Q, R = MatrixAlgebra.leftorth!(block(t, c), alg, atol)
        Qdata[c] = Q
        Rdata[c] = R
        dims[c] = size(Q, 2)
    end
    V = S(dims)
    if alg isa Polar
        @assert V ≅ domain(t)
        W = domain(t)
    elseif length(domain(t)) == 1 && domain(t) ≅ V
        W = domain(t)
    elseif length(codomain(t)) == 1 && codomain(t) ≅ V
        W = codomain(t)
    else
        W = ProductSpace(V)
    end
    return TensorMap(Qdata, codomain(t) ← W), TensorMap(Rdata, W ← domain(t))
end

function leftnull!(t::TensorMap;
                   alg::Union{QR,QRpos,SVD,SDD}=QRpos(),
                   atol::Real=zero(float(real(scalartype(t)))),
                   rtol::Real=(alg ∉ (SVD(), SDD())) ? zero(float(real(scalartype(t)))) :
                              eps(real(float(one(scalartype(t))))) * iszero(atol))
    InnerProductStyle(t) === EuclideanProduct() || throw_invalid_innerproduct(:leftnull!)
    if !iszero(rtol)
        atol = max(atol, rtol * norm(t))
    end
    I = sectortype(t)
    S = spacetype(t)
    A = storagetype(t)
    V = codomain(t)
    Ndata = SectorDict{I,A}()
    dims = SectorDict{I,Int}()
    for c in blocksectors(V)
        N = MatrixAlgebra.leftnull!(block(t, c), alg, atol)
        Ndata[c] = N
        dims[c] = size(N, 2)
    end
    W = S(dims)
    return TensorMap(Ndata, V ← W)
end

function rightorth!(t::TensorMap;
                    alg::Union{LQ,LQpos,RQ,RQpos,SVD,SDD,Polar}=LQpos(),
                    atol::Real=zero(float(real(scalartype(t)))),
                    rtol::Real=(alg ∉ (SVD(), SDD())) ? zero(float(real(scalartype(t)))) :
                               eps(real(float(one(scalartype(t))))) * iszero(atol))
    InnerProductStyle(t) === EuclideanProduct() || throw_invalid_innerproduct(:rightorth!)
    if !iszero(rtol)
        atol = max(atol, rtol * norm(t))
    end
    I = sectortype(t)
    S = spacetype(t)
    A = storagetype(t)
    Ldata = SectorDict{I,A}()
    Qdata = SectorDict{I,A}()
    dims = SectorDict{I,Int}()
    for c in blocksectors(codomain(t))
        isempty(block(t, c)) && continue
        L, Q = MatrixAlgebra.rightorth!(block(t, c), alg, atol)
        Ldata[c] = L
        Qdata[c] = Q
        dims[c] = size(Q, 1)
    end
    V = S(dims)
    if alg isa Polar
        @assert V ≅ codomain(t)
        W = codomain(t)
    elseif length(codomain(t)) == 1 && codomain(t) ≅ V
        W = codomain(t)
    elseif length(domain(t)) == 1 && domain(t) ≅ V
        W = domain(t)
    else
        W = ProductSpace(V)
    end
    return TensorMap(Ldata, codomain(t) ← W), TensorMap(Qdata, W ← domain(t))
end

function rightnull!(t::TensorMap;
                    alg::Union{LQ,LQpos,SVD,SDD}=LQpos(),
                    atol::Real=zero(float(real(scalartype(t)))),
                    rtol::Real=(alg ∉ (SVD(), SDD())) ? zero(float(real(scalartype(t)))) :
                               eps(real(float(one(scalartype(t))))) * iszero(atol))
    InnerProductStyle(t) === EuclideanProduct() || throw_invalid_innerproduct(:rightnull!)
    if !iszero(rtol)
        atol = max(atol, rtol * norm(t))
    end
    I = sectortype(t)
    S = spacetype(t)
    A = storagetype(t)
    V = domain(t)
    Ndata = SectorDict{I,A}()
    dims = SectorDict{I,Int}()
    for c in blocksectors(V)
        N = MatrixAlgebra.rightnull!(block(t, c), alg, atol)
        Ndata[c] = N
        dims[c] = size(N, 1)
    end
    W = S(dims)
    return TensorMap(Ndata, W ← V)
end

function leftorth!(t::AdjointTensorMap; alg::OFA=QRpos())
    InnerProductStyle(t) === EuclideanProduct() || throw_invalid_innerproduct(:leftorth!)
    return map(adjoint, reverse(rightorth!(adjoint(t); alg=alg')))
end

function rightorth!(t::AdjointTensorMap; alg::OFA=LQpos())
    InnerProductStyle(t) === EuclideanProduct() || throw_invalid_innerproduct(:rightorth!)
    return map(adjoint, reverse(leftorth!(adjoint(t); alg=alg')))
end

function leftnull!(t::AdjointTensorMap; alg::OFA=QR(), kwargs...)
    InnerProductStyle(t) === EuclideanProduct() || throw_invalid_innerproduct(:leftnull!)
    return adjoint(rightnull!(adjoint(t); alg=alg', kwargs...))
end

function rightnull!(t::AdjointTensorMap; alg::OFA=LQ(), kwargs...)
    InnerProductStyle(t) === EuclideanProduct() || throw_invalid_innerproduct(:rightnull!)
    return adjoint(leftnull!(adjoint(t); alg=alg', kwargs...))
end

#------------------------------#
# Singular value decomposition #
#------------------------------#
function tsvd!(t::AdjointTensorMap;
               trunc::TruncationScheme=NoTruncation(),
               p::Real=2,
               alg::Union{SVD,SDD}=SDD())
    u, s, vt, err = tsvd!(adjoint(t); trunc=trunc, p=p, alg=alg)
    return adjoint(vt), adjoint(s), adjoint(u), err
end

function tsvd!(t::TensorMap;
               trunc::TruncationScheme=NoTruncation(),
               p::Real=2,
               alg::Union{SVD,SDD}=SDD())
    #early return
    if isempty(blocksectors(t))
        truncerr = zero(real(scalartype(t)))
        return _empty_svdtensors(t)..., truncerr
    end

    S = spacetype(t)
    Udata, Σdata, Vdata, dims = _compute_svddata!(t, alg)
    if !isa(trunc, NoTruncation)
        Σdata, truncerr = _truncate!(Σdata, trunc, p)
        Udata, Σdata, Vdata, dims = _implement_svdtruncation!(t, Udata, Σdata, Vdata, dims)
        W = S(dims)
    else
        truncerr = abs(zero(scalartype(t)))
        W = S(dims)
        if length(domain(t)) == 1 && domain(t)[1] ≅ W
            W = domain(t)[1]
        elseif length(codomain(t)) == 1 && codomain(t)[1] ≅ W
            W = codomain(t)[1]
        end
    end
    return _create_svdtensors(t, Udata, Σdata, Vdata, W)..., truncerr
end

# helper functions

function _compute_svddata!(t::TensorMap, alg::Union{SVD,SDD};
                           numthreads::Int=Threads.nthreads())
    InnerProductStyle(t) === EuclideanProduct() || throw_invalid_innerproduct(:tsvd!)
    I = sectortype(t)
    A = storagetype(t)
    Udata = SectorDict{I,A}()
    Vdata = SectorDict{I,A}()
    dims = SectorDict{I,Int}()
    if numthreads == 1 || length(blocksectors(t)) == 1
        local Σdata
        for (c, b) in blocks(t)
            U, Σ, V = MatrixAlgebra.svd!(b, alg)
            Udata[c] = U
            Vdata[c] = V
            if @isdefined Σdata # cannot easily infer the type of Σ, so use this construction
                Σdata[c] = Σ
            else
                Σdata = SectorDict(c => Σ)
            end
            dims[c] = length(Σ)
        end
    elseif numthreads == -1
        tasks = map(blocksectors(t)) do c
            Threads.@spawn MatrixAlgebra.svd!(blocks(t)[c], alg)
        end
        for (c, task) in zip(blocksectors(t), tasks)
            U, Σ, V = fetch(task)
            Udata[c] = U
            Vdata[c] = V
            if @isdefined Σdata
                Σdata[c] = Σ
            else
                Σdata = SectorDict(c => Σ)
            end
            dims[c] = length(Σ)
        end
    else
        Σdata = SectorDict{I,Vector{real(scalartype(t))}}()

        # sort sectors by size
        lsc = blocksectors(t)
        lsD3 = map(lsc) do c
            # O(D1^2D2) or O(D1D2^2)
            return min(size(block(t, c), 1)^2 * size(block(t, c), 2),
                       size(block(t, c), 1) * size(block(t, c), 2)^2)
        end
        lsc = lsc[sortperm(lsD3; rev=true)]

        # producer
        taskref = Ref{Task}()
        ch = Channel(; taskref=taskref, spawn=true) do ch
            for c in vcat(lsc, fill(nothing, numthreads))
                put!(ch, c)
            end
        end

        # consumers
        Lock = Threads.SpinLock()
        tasks = map(1:numthreads) do _
            task = Threads.@spawn while true
                c = take!(ch)
                isnothing(c) && break
                U, Σ, V = MatrixAlgebra.svd!(blocks(t)[c], alg)

                # note inserting keys to dict is not thread safe
                lock(Lock)
                Udata[c] = U
                Vdata[c] = V
                Σdata[c] = Σ
                dims[c] = length(Σ)
                unlock(Lock)
            end
            return errormonitor(task)
        end

        wait.(tasks)
        wait(taskref[])
    end
    return Udata, Σdata, Vdata, dims
end

function _implement_svdtruncation!(t, Udata, Σdata, Vdata, dims)
    for c in blocksectors(t)
        truncdim = length(Σdata[c])
        if truncdim == 0
            delete!(Udata, c)
            delete!(Vdata, c)
            delete!(Σdata, c)
            delete!(dims, c)
        elseif truncdim < dims[c]
            dims[c] = truncdim
            Udata[c] = Udata[c][:, 1:truncdim]
            Vdata[c] = Vdata[c][1:truncdim, :]
        end
    end
    return Udata, Σdata, Vdata, dims
end

function _empty_svdtensors(t)
    I = sectortype(t)
    S = spacetype(t)
    A = storagetype(t)
    Ar = similarstoragetype(t, real(scalartype(t)))
    Udata = SectorDict{I,A}()
    Σmdata = SectorDict{I,Ar}()
    Vdata = SectorDict{I,A}()
    dims = SectorDict{I,Int}()
    W = S(dims)
    return TensorMap(Udata, codomain(t) ← W), TensorMap(Σmdata, W ← W),
           TensorMap(Vdata, W ← domain(t))
end

function _create_svdtensors(t, Udata, Σdata, Vdata, W)
    I = sectortype(t)
    Ar = similarstoragetype(t, real(scalartype(t)))
    Σmdata = SectorDict{I,Ar}() # this will contain the singular values as matrix
    for (c, Σ) in Σdata
        Σmdata[c] = copyto!(similar(Σ, length(Σ), length(Σ)), Diagonal(Σ))
    end
    return TensorMap(Udata, codomain(t) ← W), TensorMap(Σmdata, W ← W),
           TensorMap(Vdata, W ← domain(t))
end

#--------------------------#
# Eigenvalue decomposition #
#--------------------------#
LinearAlgebra.eigen!(t::TensorMap) = ishermitian(t) ? eigh!(t) : eig!(t)

function eigh!(t::TensorMap)
    InnerProductStyle(t) === EuclideanProduct() || throw_invalid_innerproduct(:eigh!)
    domain(t) == codomain(t) ||
        throw(SpaceMismatch("`eigh!` requires domain and codomain to be the same"))
    S = spacetype(t)
    I = sectortype(t)
    A = storagetype(t)
    Ar = similarstoragetype(t, real(scalartype(t)))
    Ddata = SectorDict{I,Ar}()
    Vdata = SectorDict{I,A}()
    dims = SectorDict{I,Int}()
    for (c, b) in blocks(t)
        values, vectors = MatrixAlgebra.eigh!(b)
        d = length(values)
        Ddata[c] = copyto!(similar(values, (d, d)), Diagonal(values))
        Vdata[c] = vectors
        dims[c] = d
    end
    if length(domain(t)) == 1
        W = domain(t)[1]
    else
        W = S(dims)
    end
    return TensorMap(Ddata, W ← W), TensorMap(Vdata, domain(t) ← W)
end

function eig!(t::TensorMap; kwargs...)
    domain(t) == codomain(t) ||
        throw(SpaceMismatch("`eig!` requires domain and codomain to be the same"))
    S = spacetype(t)
    I = sectortype(t)
    T = complex(scalartype(t))
    Ac = similarstoragetype(t, T)
    Ddata = SectorDict{I,Ac}()
    Vdata = SectorDict{I,Ac}()
    dims = SectorDict{I,Int}()
    for (c, b) in blocks(t)
        values, vectors = MatrixAlgebra.eig!(b; kwargs...)
        d = length(values)
        Ddata[c] = copy!(similar(values, T, (d, d)), Diagonal(values))
        Vdata[c] = vectors
        dims[c] = d
    end
    if length(domain(t)) == 1
        W = domain(t)[1]
    else
        W = S(dims)
    end
    return TensorMap(Ddata, W ← W), TensorMap(Vdata, domain(t) ← W)
end

#--------------------------------------------------#
# Checks for hermiticity and positive definiteness #
#--------------------------------------------------#
function LinearAlgebra.ishermitian(t::TensorMap)
    domain(t) == codomain(t) || return false
    InnerProductStyle(t) === EuclideanProduct() || return false # hermiticity only defined for euclidean
    for (c, b) in blocks(t)
        ishermitian(b) || return false
    end
    return true
end

function LinearAlgebra.isposdef!(t::TensorMap)
    domain(t) == codomain(t) ||
        throw(SpaceMismatch("`isposdef` requires domain and codomain to be the same"))
    InnerProductStyle(spacetype(t)) === EuclideanProduct() || return false
    for (c, b) in blocks(t)
        isposdef!(b) || return false
    end
    return true
end
