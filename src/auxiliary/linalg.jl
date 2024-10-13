# Simple reference to getting and setting BLAS threads
#------------------------------------------------------
set_num_blas_threads(n::Integer) = LinearAlgebra.BLAS.set_num_threads(n)
get_num_blas_threads() = LinearAlgebra.BLAS.get_num_threads()

# Factorization algorithms
#--------------------------
abstract type FactorizationAlgorithm end
abstract type OrthogonalFactorizationAlgorithm <: FactorizationAlgorithm end

struct QRpos <: OrthogonalFactorizationAlgorithm
end
struct QR <: OrthogonalFactorizationAlgorithm
end
struct QL <: OrthogonalFactorizationAlgorithm
end
struct QLpos <: OrthogonalFactorizationAlgorithm
end
struct LQ <: OrthogonalFactorizationAlgorithm
end
struct LQpos <: OrthogonalFactorizationAlgorithm
end
struct RQ <: OrthogonalFactorizationAlgorithm
end
struct RQpos <: OrthogonalFactorizationAlgorithm
end
struct SDD <: OrthogonalFactorizationAlgorithm # lapack's default divide and conquer algorithm
end
struct SVD <: OrthogonalFactorizationAlgorithm
end
struct Polar <: OrthogonalFactorizationAlgorithm
end

Base.adjoint(::QRpos) = LQpos()
Base.adjoint(::QR) = LQ()
Base.adjoint(::LQpos) = QRpos()
Base.adjoint(::LQ) = QR()

Base.adjoint(::QLpos) = RQpos()
Base.adjoint(::QL) = RQ()
Base.adjoint(::RQpos) = QLpos()
Base.adjoint(::RQ) = QL()

Base.adjoint(alg::Union{SVD,SDD,Polar}) = alg

const OFA = OrthogonalFactorizationAlgorithm
const SVDAlg = Union{SVD,SDD}

# Matrix algebra: entrypoint for calling matrix methods from within tensor implementations
#------------------------------------------------------------------------------------------
module MatrixAlgebra
# TODO: all methods tha twe define here will need an extended version for CuMatrix in the
# CUDA package extension.

# TODO: other methods to include here:
# mul! (possibly call matmul! instead)
# adjoint!
# sylvester
# exp!
# schur!?
# 

using LinearAlgebra
using LinearAlgebra: BlasFloat, BlasReal, BlasComplex, BlasInt, Char, LAPACK,
                     LAPACKException, SingularException, PosDefException,
                     checksquare, chkstride1, require_one_based_indexing, triu!

using ..TensorKit: OrthogonalFactorizationAlgorithm,
                   QL, QLpos, QR, QRpos, LQ, LQpos, RQ, RQpos, SVD, SDD, Polar

# TODO: define for CuMatrix if we support this
function one!(A::StridedMatrix)
    length(A) > 0 || return A
    copyto!(A, LinearAlgebra.I)
    return A
end

# safesign(s::Real) = ifelse(s < zero(s), -one(s), +one(s))
# safesign(s::Complex) = ifelse(iszero(s), one(s), s / abs(s))

# function _check_leftorthargs(A, Q, R, alg::Union{QR,QRpos,QL,QLpos})
#     m, n = size(A)
#     m == size(Q, 1) || throw(DimensionMismatch("size mismatch between A and Q"))
#     n == size(R, 2) || throw(DimensionMismatch("size mismatch between A and R"))
#     size(Q, 2) == size(R, 1) || throw(DimensionMismatch("size mismatch between Q and R"))
#     size(Q, 2) <= m || throw(DimensionMismatch("Q has more columns than rows"))
#     k = min(m, n)
#     size(Q, 2) >= k || @warn "Q has too few columns, truncating QR factorization"
#     if alg isa Union{QL,QLpos}
#         size(R, 1) == n || throw(DimensionMismatch("QL factorisation requires square R"))
#     end
#     return m, n, k
# end

# function leftorth!(A::StridedMatrix{S}, Q::StridedMatrix{S}, R::StridedMatrix{S}, alg::Union{QR,QRpos}, atol::Real) where {S<:BlasFloat}
#     iszero(atol) || throw(ArgumentError("nonzero atol not supported by $alg"))
#     m, n, k = _check_leftorthargs(A, Q, R, alg)

#     A, T = LAPACK.geqrt!(A, min(k, 36))
#     Q = LAPACK.gemqrt!('L', 'N', A, T, one!(Q))
#     R = copy!(R, triu!(view(A, axes(R)...)))

#     if isa(alg, QRpos)
#         @inbounds for j in 1:k
#             s = safesign(R[j, j])
#             @simd for i in 1:m
#                 Q[i, j] *= s
#             end
#         end
#         @inbounds for j in size(R, 2):-1:1
#             for i in 1:min(k, j)
#                 R[i, j] = R[i, j] * conj(safesign(R[i, i]))
#             end
#         end
#     end
#     return Q, R
# end

# function leftorth!(A::StridedMatrix{S}, Q::StridedMatrix{S}, R::StridedMatrix{S}, alg::Union{QL,QLpos}, atol::Real) where {S<:BlasFloat}
#     iszero(atol) || throw(ArgumentError("nonzero atol not supported by $alg"))
#     m, n, k = _check_leftorthargs(A, Q, R, alg)

#     nhalf = div(n, 2)
#     #swap columns in A
#     @inbounds for j in 1:nhalf, i in 1:m
#         A[i, j], A[i, n + 1 - j] = A[i, n + 1 - j], A[i, j]
#     end

#     # perform QR factorization
#     leftorth!(A, Q, R, isa(alg, QL) ? QR() : QRpos(), atol)

#     #swap columns in Q
#     @inbounds for j in 1:nhalf, i in 1:m
#         Q[i, j], Q[i, n + 1 - j] = Q[i, n + 1 - j], Q[i, j]
#     end
#     #swap rows and columns in R
#     @inbounds for j in 1:nhalf, i in 1:n
#         R[i, j], R[n + 1 - i, n + 1 - j] = R[n + 1 - i, n + 1 - j], R[i, j]
#     end
#     if isodd(n)
#         j = nhalf + 1
#         @inbounds for i in 1:nhalf
#             R[i, j], R[n + 1 - i, j] = R[n + 1 - i, j], R[i, j]
#         end
#     end
#     return Q, R
# end

# function leftorth!(A::StridedMatrix{<:BlasFloat}, alg::Union{SVD,SDD,Polar}, atol::Real)
#     U, S, V = alg isa SVD ? LAPACK.gesvd!('S', 'S', A) : LAPACK.gesdd!('S', A)
#     if isa(alg, Union{SVD,SDD})
#         n = count(s -> s .> atol, S)
#         if n != length(S)
#             return U[:, 1:n], lmul!(Diagonal(S[1:n]), V[1:n, :])
#         else
#             return U, lmul!(Diagonal(S), V)
#         end
#     else
#         iszero(atol) || throw(ArgumentError("nonzero atol not supported by $alg"))
#         # TODO: check Lapack to see if we can recycle memory of A
#         Q = mul!(A, U, V)
#         Sq = map!(sqrt, S, S)
#         SqV = lmul!(Diagonal(Sq), V)
#         R = SqV' * SqV
#         return Q, R
#     end
# end

# function leftnull!(A::StridedMatrix{<:BlasFloat}, alg::Union{QR,QRpos}, atol::Real)
#     iszero(atol) || throw(ArgumentError("nonzero atol not supported by $alg"))
#     m, n = size(A)
#     m >= n || throw(ArgumentError("no null space if less rows than columns"))

#     A, T = LAPACK.geqrt!(A, min(minimum(size(A)), 36))
#     N = similar(A, m, max(0, m - n))
#     fill!(N, 0)
#     for k in 1:(m - n)
#         N[n + k, k] = 1
#     end
#     return N = LAPACK.gemqrt!('L', 'N', A, T, N)
# end

# function leftnull!(A::StridedMatrix{<:BlasFloat}, alg::Union{SVD,SDD}, atol::Real)
#     size(A, 2) == 0 && return one!(similar(A, (size(A, 1), size(A, 1))))
#     U, S, V = alg isa SVD ? LAPACK.gesvd!('A', 'N', A) : LAPACK.gesdd!('A', A)
#     indstart = count(>(atol), S) + 1
#     return U[:, indstart:end]
# end

# function rightorth!(A::StridedMatrix{<:BlasFloat}, alg::Union{LQ,LQpos,RQ,RQpos},
#                     atol::Real)
#     iszero(atol) || throw(ArgumentError("nonzero atol not supported by $alg"))
#     # TODO: geqrfp seems a bit slower than geqrt in the intermediate region around
#     # matrix size 100, which is the interesting region. => Investigate and fix
#     m, n = size(A)
#     k = min(m, n)
#     At = transpose!(similar(A, n, m), A)

#     if isa(alg, RQ) || isa(alg, RQpos)
#         @assert m <= n

#         mhalf = div(m, 2)
#         # swap columns in At
#         @inbounds for j in 1:mhalf, i in 1:n
#             At[i, j], At[i, m + 1 - j] = At[i, m + 1 - j], At[i, j]
#         end
#         Qt, Rt = leftorth!(At, isa(alg, RQ) ? QR() : QRpos(), atol)

#         @inbounds for j in 1:mhalf, i in 1:n
#             Qt[i, j], Qt[i, m + 1 - j] = Qt[i, m + 1 - j], Qt[i, j]
#         end
#         @inbounds for j in 1:mhalf, i in 1:m
#             Rt[i, j], Rt[m + 1 - i, m + 1 - j] = Rt[m + 1 - i, m + 1 - j], Rt[i, j]
#         end
#         if isodd(m)
#             j = mhalf + 1
#             @inbounds for i in 1:mhalf
#                 Rt[i, j], Rt[m + 1 - i, j] = Rt[m + 1 - i, j], Rt[i, j]
#             end
#         end
#         Q = transpose!(A, Qt)
#         R = transpose!(similar(A, (m, m)), Rt) # TODO: efficient in place
#         return R, Q
#     else
#         Qt, Lt = leftorth!(At, alg', atol)
#         if m > n
#             L = transpose!(A, Lt)
#             Q = transpose!(similar(A, (n, n)), Qt) # TODO: efficient in place
#         else
#             Q = transpose!(A, Qt)
#             L = transpose!(similar(A, (m, m)), Lt) # TODO: efficient in place
#         end
#         return L, Q
#     end
# end

# function rightorth!(A::StridedMatrix{<:BlasFloat}, alg::Union{SVD,SDD,Polar}, atol::Real)
#     U, S, V = alg isa SVD ? LAPACK.gesvd!('S', 'S', A) : LAPACK.gesdd!('S', A)
#     if isa(alg, Union{SVD,SDD})
#         n = count(s -> s .> atol, S)
#         if n != length(S)
#             return rmul!(U[:, 1:n], Diagonal(S[1:n])), V[1:n, :]
#         else
#             return rmul!(U, Diagonal(S)), V
#         end
#     else
#         iszero(atol) || throw(ArgumentError("nonzero atol not supported by $alg"))
#         Q = mul!(A, U, V)
#         Sq = map!(sqrt, S, S)
#         USq = rmul!(U, Diagonal(Sq))
#         L = USq * USq'
#         return L, Q
#     end
# end

# function rightnull!(A::StridedMatrix{<:BlasFloat}, alg::Union{LQ,LQpos}, atol::Real)
#     iszero(atol) || throw(ArgumentError("nonzero atol not supported by $alg"))
#     m, n = size(A)
#     k = min(m, n)
#     At = adjoint!(similar(A, n, m), A)
#     At, T = LAPACK.geqrt!(At, min(k, 36))
#     N = similar(A, max(n - m, 0), n)
#     fill!(N, 0)
#     for k in 1:(n - m)
#         N[k, m + k] = 1
#     end
#     return N = LAPACK.gemqrt!('R', eltype(At) <: Real ? 'T' : 'C', At, T, N)
# end

# function rightnull!(A::StridedMatrix{<:BlasFloat}, alg::Union{SVD,SDD}, atol::Real)
#     size(A, 1) == 0 && return one!(similar(A, (size(A, 2), size(A, 2))))
#     U, S, V = alg isa SVD ? LAPACK.gesvd!('N', 'A', A) : LAPACK.gesdd!('A', A)
#     indstart = count(>(atol), S) + 1
#     return V[indstart:end, :]
# end

# function svd!(A::StridedMatrix{T}, U::StridedMatrix{T}, S::StridedVector{T},
#               V::StridedMatrix{T}; alg = SVD()) where {T<:BlasFloat}    
#     U, S, V = alg isa SVD ? custom_gesvd!(A, U, S, V) : custom_gesdd!(A, U, S, V)
#     return U, S, V
# end

# function eig!(A::StridedMatrix{T}; permute::Bool=true, scale::Bool=true) where {T<:BlasReal}
#     n = checksquare(A)
#     n == 0 && return zeros(Complex{T}, 0), zeros(Complex{T}, 0, 0)

#     A, DR, DI, VL, VR, _ = LAPACK.geevx!(permute ? (scale ? 'B' : 'P') :
#                                          (scale ? 'S' : 'N'), 'N', 'V', 'N', A)
#     D = complex.(DR, DI)
#     V = zeros(Complex{T}, n, n)
#     j = 1
#     while j <= n
#         if DI[j] == 0
#             vr = view(VR, :, j)
#             s = conj(sign(argmax(abs, vr)))
#             V[:, j] .= s .* vr
#         else
#             vr = view(VR, :, j)
#             vi = view(VR, :, j + 1)
#             s = conj(sign(argmax(abs, vr))) # vectors coming from lapack have already real absmax component
#             V[:, j] .= s .* (vr .+ im .* vi)
#             V[:, j + 1] .= s .* (vr .- im .* vi)
#             j += 1
#         end
#         j += 1
#     end
#     return D, V
# end

# function eig!(A::StridedMatrix{T}; permute::Bool=true,
#               scale::Bool=true) where {T<:BlasComplex}
#     n = checksquare(A)
#     n == 0 && return zeros(T, 0), zeros(T, 0, 0)
#     D, V = LAPACK.geevx!(permute ? (scale ? 'B' : 'P') : (scale ? 'S' : 'N'), 'N', 'V', 'N',
#                          A)[[2, 4]]
#     for j in 1:n
#         v = view(V, :, j)
#         s = conj(sign(argmax(abs, v)))
#         v .*= s
#     end
#     return D, V
# end

# function eigh!(A::StridedMatrix{T}) where {T<:BlasFloat}
#     n = checksquare(A)
#     n == 0 && return zeros(real(T), 0), zeros(T, 0, 0)
#     D, V = LAPACK.syevr!('V', 'A', 'U', A, 0.0, 0.0, 0, 0, -1.0)
#     for j in 1:n
#         v = view(V, :, j)
#         s = conj(sign(argmax(abs, v)))
#         v .*= s
#     end
#     return D, V
# end

# # Type stable definitions of matrix functions:

# # functions that map ℝ to (a subset of) ℝ
# for f in (:cos, :sin, :tan, :cot, :cosh, :sinh, :tanh, :coth, :atan, :acot, :asinh)
#     @eval $f(A::StridedMatrix{<:Real}) = real(LinearAlgebra.$f(A))
#     @eval $f(A::StridedMatrix{<:Complex}) = LinearAlgebra.$f(A)
# end
# # functions that don't map ℝ to (a subset of) ℝ
# for f in (:sqrt, :log, :asin, :acos, :acosh, :atanh, :acoth)
#     @eval $f(A::StridedMatrix) = complex(LinearAlgebra.$f(A))
# end

# # redefined Lapack wrappers
# using LinearAlgebra: chklapackerror, @blasfunc, libblastrampoline

# for (gesvd, gesdd, elty, relty) in
#     ((:dgesvd_, :dgesdd_, :Float64, :Float64),
#      (:sgesvd_, :sgesdd_, :Float32, :Float32),
#      (:zgesvd_, :zgesdd_, :ComplexF64, :Float64),
#      (:cgesvd_, :cgesdd_, :ComplexF32, :Float32))
#     @eval begin
#         function custom_gesdd!(A::AbstractMatrix{$elty}, U::AbstractMatrix{$elty}, VT::AbstractMatrix{$elty}, S::AbstractVector{$relty})
#             require_one_based_indexing(A)
#             chkstride1(A)
#             m, n = size(A)
#             minmn = min(m, n)

#             if length(U) == 0 && length(VT) == 0
#                 job = 'N'
#             else
#                 size(U, 1) == m || throw(DimensionMismatch("row size mismatch between A and U"))
#                 size(VT, 2) == n || throw(DimensionMismatch("column size mismatch between A and VT"))
#                 length(S) == minmn || throw(DimensionMismatch("length mismatch between A and S"))
#                 if size(U, 2) == m && size(VT, 1) == n
#                     job = 'A'
#                 elseif size(U, 2) == minmn && size(VT, 1) == minmn
#                     job = 'S'
#                 else
#                     throw(DimensionMismatch("invalid column size of U or row size of VT"))
#                 end
#             end
#             cmplx = eltype(A) <: Complex
#             if cmplx
#                 rwork = Vector{$relty}(undef,
#                                        job == 'N' ? 7 * minmn :
#                                        minmn *
#                                        max(5 * minmn + 7, 2 * max(m, n) + 2 * minmn + 1))
#             end
#             iwork = Vector{BlasInt}(undef, 8 * minmn)
#             info = Ref{BlasInt}()
#             for i in 1:2  # first call returns lwork as work[1]
#                 if cmplx
#                     ccall((@blasfunc($gesdd), libblastrampoline), Cvoid,
#                           (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
#                            Ref{BlasInt}, Ptr{$relty}, Ptr{$elty}, Ref{BlasInt},
#                            Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
#                            Ptr{$relty}, Ptr{BlasInt}, Ptr{BlasInt}, Clong),
#                           job, m, n, A, max(1, stride(A, 2)), S, U, max(1, stride(U, 2)),
#                           VT, max(1, stride(VT, 2)),
#                           work, lwork, rwork, iwork, info, 1)
#                 else
#                     ccall((@blasfunc($gesdd), libblastrampoline), Cvoid,
#                           (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
#                            Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt},
#                            Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
#                            Ptr{BlasInt}, Ptr{BlasInt}, Clong),
#                           job, m, n, A, max(1, stride(A, 2)), S, U, max(1, stride(U, 2)),
#                           VT, max(1, stride(VT, 2)),
#                           work, lwork, iwork, info, 1)
#                 end
#                 chklapackerror(info[])
#                 if i == 1
#                     # Work around issue with truncated Float32 representation of lwork in
#                     # sgesdd by using nextfloat. See
#                     # http://icl.cs.utk.edu/lapack-forum/viewtopic.php?f=13&t=4587&p=11036&hilit=sgesdd#p11036
#                     # and
#                     # https://github.com/scipy/scipy/issues/5401
#                     lwork = round(BlasInt, nextfloat(real(work[1])))
#                     resize!(work, lwork)
#                 end
#             end
#             return (U, S, VT)
#         end
#         function custom_gesvd!(A::AbstractMatrix{$elty}, U::AbstractMatrix{$elty}, VT::AbstractMatrix{$elty}, S::AbstractVector{$relty})
#             require_one_based_indexing(A)
#             chkstride1(A)
#             m, n = size(A)
#             minmn = min(m, n)
#             if length(U) == 0
#                 jobu = 'N'
#             else
#                 size(U, 1) == m ||
#                     throw(DimensionMismatch("row size mismatch between A and U"))
#                 if size(U, 2) == m
#                     jobu = 'A'
#                 elseif size(U, 2) == minmn
#                     jobu = 'S'
#                 else
#                     throw(DimensionMismatch("invalid column size of U"))
#                 end
#             end
#             if length(VT) == 0
#                 jobv = 'N'
#             else
#                 size(VT, 2) == n ||
#                     throw(DimensionMismatch("column size mismatch between A and VT"))
#                 if size(VT, 1) == n
#                     jobv = 'A'
#                 elseif size(VT, 1) == minmn
#                     jobv = 'S'
#                 else
#                     throw(DimensionMismatch("invalid row size of VT"))
#                 end
#             end
#             length(S) == minmn ||
#                 throw(DimensionMismatch("length mismatch between A and S"))

#             work = Vector{$elty}(undef, 1)
#             cmplx = eltype(A) <: Complex
#             if cmplx
#                 rwork = Vector{$relty}(undef, 5minmn)
#             end
#             lwork = BlasInt(-1)
#             info = Ref{BlasInt}()
#             for i in 1:2  # first call returns lwork as work[1]
#                 if cmplx
#                     ccall((@blasfunc($gesvd), libblastrampoline), Cvoid,
#                           (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
#                            Ptr{$elty}, Ref{BlasInt}, Ptr{$relty}, Ptr{$elty},
#                            Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
#                            Ref{BlasInt}, Ptr{$relty}, Ptr{BlasInt}, Clong, Clong),
#                           jobu, jobvt, m, n, A, max(1, stride(A, 2)), S, U,
#                           max(1, stride(U, 2)), VT, max(1, stride(VT, 2)),
#                           work, lwork, rwork, info, 1, 1)
#                 else
#                     ccall((@blasfunc($gesvd), libblastrampoline), Cvoid,
#                           (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
#                            Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ptr{$elty},
#                            Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
#                            Ref{BlasInt}, Ptr{BlasInt}, Clong, Clong),
#                           jobu, jobvt, m, n, A, max(1, stride(A, 2)), S, U,
#                           max(1, stride(U, 2)), VT, max(1, stride(VT, 2)),
#                           work, lwork, info, 1, 1)
#                 end
#                 chklapackerror(info[])
#                 if i == 1
#                     lwork = BlasInt(real(work[1]))
#                     resize!(work, lwork)
#                 end
#             end
#             return (U, S, VT)
#         end
#     end
# end

# for (geevx, ggev, ggev3, elty) in
#     ((:dgeevx_, :dggev_, :dggev3_, :Float64),
#      (:sgeevx_, :sggev_, :sggev3_, :Float32))
#     @eval begin
#         #     SUBROUTINE DGEEVX( BALANC, JOBVL, JOBVR, SENSE, N, A, LDA, WR, WI,
#         #                          VL, LDVL, VR, LDVR, ILO, IHI, SCALE, ABNRM,
#         #                          RCONDE, RCONDV, WORK, LWORK, IWORK, INFO )
#         #
#         #       .. Scalar Arguments ..
#         #       CHARACTER          BALANC, JOBVL, JOBVR, SENSE
#         #       INTEGER            IHI, ILO, INFO, LDA, LDVL, LDVR, LWORK, N
#         #       DOUBLE PRECISION   ABNRM
#         #       ..
#         #       .. Array Arguments ..
#         #       INTEGER            IWORK( * )
#         #       DOUBLE PRECISION   A( LDA, * ), RCONDE( * ), RCONDV( * ),
#         #      $                   SCALE( * ), VL( LDVL, * ), VR( LDVR, * ),
#         #      $                   WI( * ), WORK( * ), WR( * )
#         function geevx!(balanc::AbstractChar, jobvl::AbstractChar, jobvr::AbstractChar,
#                         sense::AbstractChar, A::AbstractMatrix{$elty})
#             require_one_based_indexing(A)
#             @chkvalidparam 1 balanc ('N', 'P', 'S', 'B')
#             @chkvalidparam 4 sense ('N', 'E', 'V', 'B')
#             if sense ∈ ('E', 'B') && !(jobvl == jobvr == 'V')
#                 throw(ArgumentError(lazy"sense = '$sense' requires jobvl = 'V' and jobvr = 'V'"))
#             end
#             n = checksquare(A)
#             ldvl = 0
#             if jobvl == 'V'
#                 ldvl = n
#             elseif jobvl == 'N'
#                 ldvl = 0
#             else
#                 throw(ArgumentError(lazy"jobvl must be 'V' or 'N', but $jobvl was passed"))
#             end
#             ldvr = 0
#             if jobvr == 'V'
#                 ldvr = n
#             elseif jobvr == 'N'
#                 ldvr = 0
#             else
#                 throw(ArgumentError(lazy"jobvr must be 'V' or 'N', but $jobvr was passed"))
#             end
#             chkfinite(A) # balancing routines don't support NaNs and Infs
#             lda = max(1, stride(A, 2))
#             wr = similar(A, $elty, n)
#             wi = similar(A, $elty, n)
#             VL = similar(A, $elty, ldvl, n)
#             VR = similar(A, $elty, ldvr, n)
#             ilo = Ref{BlasInt}()
#             ihi = Ref{BlasInt}()
#             scale = similar(A, $elty, n)
#             abnrm = Ref{$elty}()
#             rconde = similar(A, $elty, n)
#             rcondv = similar(A, $elty, n)
#             work = Vector{$elty}(undef, 1)
#             lwork = BlasInt(-1)
#             iworksize = 0
#             if sense == 'N' || sense == 'E'
#                 iworksize = 0
#             elseif sense == 'V' || sense == 'B'
#                 iworksize = 2 * n - 2
#             else
#                 throw(ArgumentError(lazy"sense must be 'N', 'E', 'V' or 'B', but $sense was passed"))
#             end
#             iwork = Vector{BlasInt}(undef, iworksize)
#             info = Ref{BlasInt}()
#             for i in 1:2  # first call returns lwork as work[1]
#                 ccall((@blasfunc($geevx), libblastrampoline), Cvoid,
#                       (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{UInt8},
#                        Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
#                        Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
#                        Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$elty},
#                        Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, Ptr{$elty},
#                        Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt},
#                        Clong, Clong, Clong, Clong),
#                       balanc, jobvl, jobvr, sense,
#                       n, A, lda, wr,
#                       wi, VL, max(1, ldvl), VR,
#                       max(1, ldvr), ilo, ihi, scale,
#                       abnrm, rconde, rcondv, work,
#                       lwork, iwork, info,
#                       1, 1, 1, 1)
#                 chklapackerror(info[])
#                 if i == 1
#                     lwork = BlasInt(work[1])
#                     resize!(work, lwork)
#                 end
#             end
#             return A, wr, wi, VL, VR, ilo[], ihi[], scale, abnrm[], rconde, rcondv
#         end

#         #       SUBROUTINE DGGEV( JOBVL, JOBVR, N, A, LDA, B, LDB, ALPHAR, ALPHAI,
#         #      $                  BETA, VL, LDVL, VR, LDVR, WORK, LWORK, INFO )
#         # *     .. Scalar Arguments ..
#         #       CHARACTER          JOBVL, JOBVR
#         #       INTEGER            INFO, LDA, LDB, LDVL, LDVR, LWORK, N
#         # *     ..
#         # *     .. Array Arguments ..
#         #       DOUBLE PRECISION   A( LDA, * ), ALPHAI( * ), ALPHAR( * ),
#         #      $                   B( LDB, * ), BETA( * ), VL( LDVL, * ),
#         #      $                   VR( LDVR, * ), WORK( * )
#         function ggev!(jobvl::AbstractChar, jobvr::AbstractChar, A::AbstractMatrix{$elty},
#                        B::AbstractMatrix{$elty})
#             require_one_based_indexing(A, B)
#             chkstride1(A, B)
#             n, m = checksquare(A, B)
#             if n != m
#                 throw(DimensionMismatch(lazy"A has dimensions $(size(A)), and B has dimensions $(size(B)), but A and B must have the same size"))
#             end
#             ldvl = 0
#             if jobvl == 'V'
#                 ldvl = n
#             elseif jobvl == 'N'
#                 ldvl = 1
#             else
#                 throw(ArgumentError(lazy"jobvl must be 'V' or 'N', but $jobvl was passed"))
#             end
#             ldvr = 0
#             if jobvr == 'V'
#                 ldvr = n
#             elseif jobvr == 'N'
#                 ldvr = 1
#             else
#                 throw(ArgumentError(lazy"jobvr must be 'V' or 'N', but $jobvr was passed"))
#             end
#             lda = max(1, stride(A, 2))
#             ldb = max(1, stride(B, 2))
#             alphar = similar(A, $elty, n)
#             alphai = similar(A, $elty, n)
#             beta = similar(A, $elty, n)
#             vl = similar(A, $elty, ldvl, n)
#             vr = similar(A, $elty, ldvr, n)
#             work = Vector{$elty}(undef, 1)
#             lwork = BlasInt(-1)
#             info = Ref{BlasInt}()
#             for i in 1:2  # first call returns lwork as work[1]
#                 ccall((@blasfunc($ggev), libblastrampoline), Cvoid,
#                       (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty},
#                        Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
#                        Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt},
#                        Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
#                        Ref{BlasInt}, Clong, Clong),
#                       jobvl, jobvr, n, A,
#                       lda, B, ldb, alphar,
#                       alphai, beta, vl, ldvl,
#                       vr, ldvr, work, lwork,
#                       info, 1, 1)
#                 chklapackerror(info[])
#                 if i == 1
#                     lwork = BlasInt(work[1])
#                     resize!(work, lwork)
#                 end
#             end
#             return alphar, alphai, beta, vl, vr
#         end

#         #       SUBROUTINE DGGEV3( JOBVL, JOBVR, N, A, LDA, B, LDB, ALPHAR, ALPHAI,
#         #      $                   BETA, VL, LDVL, VR, LDVR, WORK, LWORK, INFO )
#         # *     .. Scalar Arguments ..
#         #       CHARACTER          JOBVL, JOBVR
#         #       INTEGER            INFO, LDA, LDB, LDVL, LDVR, LWORK, N
#         # *     ..
#         # *     .. Array Arguments ..
#         #       DOUBLE PRECISION   A( LDA, * ), ALPHAI( * ), ALPHAR( * ),
#         #      $                   B( LDB, * ), BETA( * ), VL( LDVL, * ),
#         #      $                   VR( LDVR, * ), WORK( * )
#         function ggev3!(jobvl::AbstractChar, jobvr::AbstractChar, A::AbstractMatrix{$elty},
#                         B::AbstractMatrix{$elty})
#             require_one_based_indexing(A, B)
#             chkstride1(A, B)
#             n, m = checksquare(A, B)
#             if n != m
#                 throw(DimensionMismatch(lazy"A has dimensions $(size(A)), and B has dimensions $(size(B)), but A and B must have the same size"))
#             end
#             ldvl = 0
#             if jobvl == 'V'
#                 ldvl = n
#             elseif jobvl == 'N'
#                 ldvl = 1
#             else
#                 throw(ArgumentError(lazy"jobvl must be 'V' or 'N', but $jobvl was passed"))
#             end
#             ldvr = 0
#             if jobvr == 'V'
#                 ldvr = n
#             elseif jobvr == 'N'
#                 ldvr = 1
#             else
#                 throw(ArgumentError(lazy"jobvr must be 'V' or 'N', but $jobvr was passed"))
#             end
#             lda = max(1, stride(A, 2))
#             ldb = max(1, stride(B, 2))
#             alphar = similar(A, $elty, n)
#             alphai = similar(A, $elty, n)
#             beta = similar(A, $elty, n)
#             vl = similar(A, $elty, ldvl, n)
#             vr = similar(A, $elty, ldvr, n)
#             work = Vector{$elty}(undef, 1)
#             lwork = BlasInt(-1)
#             info = Ref{BlasInt}()
#             for i in 1:2  # first call returns lwork as work[1]
#                 ccall((@blasfunc($ggev3), libblastrampoline), Cvoid,
#                       (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty},
#                        Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
#                        Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt},
#                        Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
#                        Ref{BlasInt}, Clong, Clong),
#                       jobvl, jobvr, n, A,
#                       lda, B, ldb, alphar,
#                       alphai, beta, vl, ldvl,
#                       vr, ldvr, work, lwork,
#                       info, 1, 1)
#                 chklapackerror(info[])
#                 if i == 1
#                     lwork = BlasInt(work[1])
#                     resize!(work, lwork)
#                 end
#             end
#             return alphar, alphai, beta, vl, vr
#         end
#     end
# end

## Old stuff and experiments

# using LinearAlgebra: BlasFloat, Char, BlasInt, LAPACK, LAPACKException,
#                      DimensionMismatch, SingularException, PosDefException, chkstride1,
#                      checksquare,
#                      triu!

# TODO: reconsider the following implementation
# Unfortunately, geqrfp seems a bit slower than geqrt in the intermediate region
# around matrix size 100, which is the interesting region. => Investigate and maybe fix
# function _leftorth!(A::StridedMatrix{<:BlasFloat})
#     m, n = size(A)
#     A, τ = geqrfp!(A)
#     Q = LAPACK.ormqr!('L', 'N', A, τ, eye(eltype(A), m, min(m, n)))
#     R = triu!(A[1:min(m, n), :])
#     return Q, R
# end

# geqrfp!: computes qrpos factorization, missing in Base
# geqrfp!(A::StridedMatrix{<:BlasFloat}) =
#     ((m, n) = size(A); geqrfp!(A, similar(A, min(m, n))))
#
# for (geqrfp, elty, relty) in
#     ((:dgeqrfp_, :Float64, :Float64), (:sgeqrfp_, :Float32, :Float32),
#         (:zgeqrfp_, :ComplexF64, :Float64), (:cgeqrfp_, :ComplexF32, :Float32))
#     @eval begin
#         function geqrfp!(A::StridedMatrix{$elty}, tau::StridedVector{$elty})
#             chkstride1(A, tau)
#             m, n  = size(A)
#             if length(tau) != min(m, n)
#                 throw(DimensionMismatch("tau has length $(length(tau)), but needs length $(min(m, n))"))
#             end
#             work  = Vector{$elty}(1)
#             lwork = BlasInt(-1)
#             info  = Ref{BlasInt}()
#             for i = 1:2                # first call returns lwork as work[1]
#                 ccall((@blasfunc($geqrfp), liblapack), Nothing,
#                       (Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$elty}, Ptr{BlasInt},
#                        Ptr{$elty}, Ptr{$elty}, Ptr{BlasInt}, Ptr{BlasInt}),
#                       Ref(m), Ref(n), A, Ref(max(1, stride(A, 2))),
#                       tau, work, Ref(lwork), info)
#                 chklapackerror(info[])
#                 if i == 1
#                     lwork = BlasInt(real(work[1]))
#                     resize!(work, lwork)
#                 end
#             end
#             A, tau
#         end
#     end
# end

end
