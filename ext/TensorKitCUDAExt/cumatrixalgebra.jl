using TensorKit: MatrixAlgebra

function MatrixAlgebra.matmul!(C::AnyCuMatrix, A::AnyCuMatrix, B::AnyCuMatrix,
                               α::Number, β::Number)
    return LinearAlgebra.mul!(C, A, B, α, β)
end

function MatrixAlgebra.one!(A::CuMatrix)
    length(A) > 0 || return A
    copyto!(A, LinearAlgebra.I)
    return A
end

function MatrixAlgebra._fix_diag_sign!(Q::CuMatrix, R::CuMatrix)
    S = Diagonal(MatrixAlgebra.safesign.(LinearAlgebra.diag(R)))
    rmul!(Q, S)
    lmul!(S, R)
    return Q, R
end

# TODO: is this the most efficient way to do this?
function MatrixAlgebra._init_leftnull(A::CuMatrix)
    m, n = size(A)
    N = CUDA.zeros(eltype(A), m, max(0, m - n))
    inds = ((1:(m - n)) .+ n) .+ ((0:(m - n - 1)) .* stride(N, 2))
    N[inds] .= one(eltype(A))
    return N
end
function MatrixAlgebra._init_rightnull(A::CuMatrix)
    m, n = size(A)
    N = CUDA.zeros(eltype(A), max(n - m, 0), n)
    inds = (1:(n - m)) .+ (((0:(n - m - 1)) .+ m) .* stride(N, 2))
    N[inds] .= one(eltype(A))
    return N
end

# TODO: can we avoid computing U and V when possible?
function MatrixAlgebra.svd!(A::CuMatrix{<:BlasFloat}, alg::Union{SVD,SDD,Polar})
    USV = LinearAlgebra.svd!(A;
                             alg=alg isa SVD ? CUDA.CUSOLVER.QRAlgorithm() :
                                 CUDA.CUSOLVER.JacobiAlgorithm())
    return USV.U, USV.S, USV.Vt
end
function MatrixAlgebra.svd_US!(A::CuMatrix, alg::Union{SVD,SDD,Polar})
    U, S, _ = MatrixAlgebra.svd!(A, alg)
    return U, S
end
function MatrixAlgebra.svd_SV!(A::CuMatrix, alg::Union{SVD,SDD,Polar})
    _, S, V = MatrixAlgebra.svd!(A, alg)
    return S, V
end

# function MatrixAlgebra.eig!(A::CuMatrix; permute::Bool=true, scale::Bool=true)
#     D, V = LinearAlgebra.eigen!(A; permute=permute, scale=scale)
#     return D, V
# end

# GPU eigensolver supports only Hermitian or Symmetric matrices, and in-place fails?
function MatrixAlgebra.eigh!(A::CuMatrix)
    D, V = LinearAlgebra.eigen(LinearAlgebra.Hermitian(A))
    return D, V
end

# technically not in matrixalgebra
function TensorKit.scalar(t::CuTensorMap)
    return dim(codomain(t)) == dim(domain(t)) == 1 ?
           CUDA.@allowscalar(only(blocks(t))[2][1]) : throw(DimensionMismatch())
end
