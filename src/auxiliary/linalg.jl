import Base.LinAlg: BlasFloat, Char, BlasInt, LAPACKException,
    DimensionMismatch, SingularException, PosDefException, chkstride1, checksquare
import Base.BLAS: @blasfunc, libblas, BlasReal, BlasComplex
import Base.LAPACK: liblapack, chklapackerror

# custom wrappers for BLAS and LAPACK routines, together with some custom definitions

# MATRIX factorizations
#-----------------------
abstract type FactorizationAlgorithm end
abstract type OrthogonalFactorizationAlgorithm <: FactorizationAlgorithm end

struct QRpos <: OrthogonalFactorizationAlgorithm
end
struct QR <: OrthogonalFactorizationAlgorithm
end
struct QLPos <: OrthogonalFactorizationAlgorithm
end
struct QL <: OrthogonalFactorizationAlgorithm
end
struct LQ <: OrthogonalFactorizationAlgorithm
end
struct LQpos <: OrthogonalFactorizationAlgorithm
end
struct RQ <: OrthogonalFactorizationAlgorithm
end
struct RQpos <: OrthogonalFactorizationAlgorithm
end
struct SVD{T<:Real} <: OrthogonalFactorizationAlgorithm
    tol::T
end
SVD() = SVD(0)
struct Polar <: OrthogonalFactorizationAlgorithm
end

_safesign(s::Real) = ifelse(s<zero(s), -one(s), +one(s))
_safesign(s::Complex) = ifelse(iszero(s), one(s), s/abs(s))

function leftorth!(A::StridedMatrix{<:BlasFloat}, alg::Union{QR,QRpos,QL,QLpos} = QRpos())
    m, n = size(A)
    k = min(m, n)

    if isa(alg, QR) || isa(alg, QRpos)
        A, T = LAPACK.geqrt!(A, min(minimum(size(A)), 36))
        Q = LAPACK.gemqrt!('L', 'N', A, T, eye(eltype(A), m, k))
        R = triu!(A[1:k, :])

        if isa(alg, QRpos)
            @inbounds for j = 1:k
                s = _safesign(R[j,j])
                @simd for i = 1:m
                    Q[i,j] *= s
                end
            end
            @inbounds for j = size(R,2):-1:1
                for i = 1:min(k,j)
                    R[i,j] = R[i,j]*conj(_safesign(R[i,i]))
                end
            end
        end
        return Q, R
    else
        @assert m >= n

        nhalf = div(n,2)
        #swap colomns in A
        @inbounds for j = 1:nhalf, i = 1:m
            A[i,j], A[i,n+1-j] = A[i,n+1-j], A[i,j]
        end
        Q, R = leftorth!(A, isa(alg, QL) ? QR() : QRpos() )
        #swap comumns in Q
        @inbounds for j = 1:nhalf, i = 1:m
            Q[i,j], Q[i,n+1-j] = Q[i,n+1-j], Q[i,j]
        end
        #swap rows and columns in R
        @inbounds for j = 1:nhalf, i = 1:m
            R[i,j], R[n+1-i,n+1-j] = R[n+1-i,n+1-j], R[i,j]
        end
        return Q, R
    end
end
# TODO: reconsider the following implementation
# Unfortunately, geqrfp seems a bit slower than geqrt in the intermediate region
# around matrix size 100, which is the interesting region. => Investigate and maybe fix
# function leftorth!(A::StridedMatrix{<:BlasFloat})
#     m, n = size(A)
#     A, τ = geqrfp!(A)
#     Q = LAPACK.ormqr!('L','N',A, τ, eye(eltype(A), m, min(m,n)))
#     R = triu!(A[1:min(m,n), :])
#     return Q, R
# end

function leftorth!(A::StridedMatrix{<:BlasFloat}, alg::Union{SVD,Polar})
    U, S, V = LAPACK.gesdd!('S', A)
    if isa(alg, SVD)
        # TODO: implement truncation based on tol in SVD
        return U, Diagonal(S)*V
    else
        # TODO: check Lapack to see if we can recycle memory of A
        return U*V, V'*Diagonal(S)*V
    end
end

function leftnull!(A::StridedMatrix{<:BlasFloat}, alg::Union{QR,QRpos} = QR())
    m, n = size(A)
    m >= n || throw(ArgumentError("no null space if less rows than columns"))

    A, T = LAPACK.geqrt!(A, min(minimum(size(A)), 36))
    N = similar(A, m, max(0, m-n));
    fill!(N, 0)
    for k = 1:m-n
        N[n+k,k] = 1
    end
    N = LAPACK.gemqrt!('L', 'N', A, T, N)
end

function leftnull!(A::StridedMatrix{<:BlasFloat}, alg::SVD)
    U, S, V = LAPACK.gesdd!('A', A)
    # TODO
end

function rightorth!(A::StridedMatrix{<:BlasFloat}, alg::Union{LQ,LQpos, RQ, RQpos} = LQpos())
    # TODO: geqrfp seems a bit slower than geqrt in the intermediate region around
    # matrix size 100, which is the interesting region. => Investigate and fix
    m, n = size(A)
    k = min(m,n)
    At = transpose!(similar(A,n,m), A)

    if isa(alg, RQ) || isa(alg, RQpos)
        @assert m <= n

        mhalf = div(m,2)
        # swap columns in At
        @inbounds for j = 1:mhalf, i = 1:n
            At[i,j], At[i,m+1-j] = At[i,m+1-j], At[i,j]
        end
        Qt, Rt = leftorth!(At, isa(alg, RQ) ? QR() : QRpos())

        @inbounds for j = 1:mhalf, i = 1:n
            Qt[i,j], Qt[i,m+1-j] = Qt[i,m+1-j], Qt[i,j]
        end
        @inbounds for j = 1:mhalf, i = 1:m
            Rt[i,j], Rt[m+1-i,m+1-j] = Rt[m+1-i,m+1-j], Rt[i,j]
        end
        if isodd(m)
            j = mhalf+1
            @inbounds for i = 1:mhalf
                Rt[i,j], Rt[m+1-i,j] = Rt[m+1-i,j], Rt[i,j]
            end
        end
        Q = transpose!(A, Qt)
        R = transpose(Rt) # TODO: efficient in place
        return R, Q
    else
        Qt, Lt = leftorth!(At, isa(alg, LQ) ? QR() : QRpos())
        if m > n
            L = transpose!(A, Lt)
            Q = transpose(Qt)
        else
            Q = transpose!(A, Qt)
            L = transpose(Lt) # TODO: efficient in place
        end
        return L, Q
    end
end

function rightorth!(A::StridedMatrix{<:BlasFloat}, alg::Union{SVD,Polar})
    U, S, V = LAPACK.gesdd!('S', A)
    if isa(alg, SVD)
        # TODO: implement truncation based on tol in SVD
        return scale!(U, S), V
    else
        # TODO: check Lapack to see if we can recycle memory of A
        return scale!(U,S)*U', U*V
    end
end

function rightnull!(A::StridedMatrix{<:BlasFloat}, alg::Union{LQ,LQpos} = LQpos())
    m, n = size(A)
    k = min(m,n)
    At = adjoint!(similar(A,n,m), A)
    At, T = LAPACK.geqrt!(At, min(k, 36))
    N = similar(A, max(n-m,0), n);
    fill!(N, 0)
    for k = 1:n-m
        N[k,m+k] = 1
    end
    N = LAPACK.gemqrt!('R', eltype(At) <: Real ? 'T' : 'C', At, T, N)
end

function rightnull!(A::StridedMatrix{<:BlasFloat}, alg::SVD)
    U, S, V = LAPACK.gesdd!('A', A)
    # TODO
end


function svd!(A::StridedMatrix{<:BlasFloat}, alg::SVD = SVD(0))
    U, S, V = LAPACK.gesdd!('S', A)
    # TODO: implement truncation based on tol
    return U, S, V
end

# TODO: override Julia's eig interface

# eig!(A::StridedMatrix{<:BlasFloat}) = LinAlg.LAPACK.gees!('V', A)
#
#
#
# function eig!(A::StridedMatrix{T}; permute::Bool=true, scale::Bool=true) where T<:BlasReal
#     n = size(A, 2)
#     n == 0 && return Eigen(zeros(T, 0), zeros(T, 0, 0))
#     issymmetric(A) && return eigfact!(Symmetric(A))
#     A, WR, WI, VL, VR, _ = LAPACK.geevx!(permute ? (scale ? 'B' : 'P') : (scale ? 'S' : 'N'), 'N', 'V', 'N', A)
#     evec = zeros(Complex{T}, n, n)
#     j = 1
#     while j <= n
#         if WI[j] == 0
#             evec[:,j] = view(VR, :, j)
#         else
#             for i = 1:n
#                 evec[i,j]   = VR[i,j] + im*VR[i,j+1]
#                 evec[i,j+1] = VR[i,j] - im*VR[i,j+1]
#             end
#             j += 1
#         end
#         j += 1
#     end
#     return Eigen(complex.(WR, WI), evec)
# end
#
# function eigfact!(A::StridedMatrix{T}; permute::Bool=true, scale::Bool=true) where T<:BlasComplex
#     n = size(A, 2)
#     n == 0 && return Eigen(zeros(T, 0), zeros(T, 0, 0))
#     ishermitian(A) && return eigfact!(Hermitian(A))
#     return Eigen(LAPACK.geevx!(permute ? (scale ? 'B' : 'P') : (scale ? 'S' : 'N'), 'N', 'V', 'N', A)[[2,4]]...)
# end
#
#
# eigfact!(A::RealHermSymComplexHerm{<:BlasReal,<:StridedMatrix}) = Eigen(LAPACK.syevr!('V', 'A', A.uplo, A.data, 0.0, 0.0, 0, 0, -1.0)...)
#
#
#


# Modified / missing Lapack wrappers
#------------------------------------
# geqrfp!: computes qrpos factorization, missing in Base
geqrfp!(A::StridedMatrix{<:BlasFloat}) = ((m,n) = size(A); geqrfp!(A, similar(A, min(m, n))))

for (geqrfp, elty, relty) in
    ((:dgeqrfp_,:Float64,:Float64), (:sgeqrfp_,:Float32,:Float32), (:zgeqrfp_,:Complex128,:Float64), (:cgeqrfp_,:Complex64,:Float32))
    @eval begin
        function geqrfp!(A::StridedMatrix{$elty}, tau::StridedVector{$elty})
            chkstride1(A,tau)
            m, n  = size(A)
            if length(tau) != min(m,n)
                throw(DimensionMismatch("tau has length $(length(tau)), but needs length $(min(m,n))"))
            end
            work  = Vector{$elty}(1)
            lwork = BlasInt(-1)
            info  = Ref{BlasInt}()
            for i = 1:2                # first call returns lwork as work[1]
                ccall((@blasfunc($geqrfp), liblapack), Void,
                      (Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$elty}, Ptr{BlasInt},
                       Ptr{$elty}, Ptr{$elty}, Ptr{BlasInt}, Ptr{BlasInt}),
                      Ref(m), Ref(n), A, Ref(max(1,stride(A,2))), tau, work, Ref(lwork), info)
                chklapackerror(info[])
                if i == 1
                    lwork = BlasInt(real(work[1]))
                    resize!(work, lwork)
                end
            end
            A, tau
        end
    end
end
