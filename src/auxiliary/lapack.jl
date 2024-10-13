module YALAPACK # Yet another lapack wrapper

using LinearAlgebra: BlasFloat, BlasReal, BlasComplex, BlasInt, Char, LAPACK,
                     LAPACKException, SingularException, PosDefException,
                     checksquare, chkstride1, require_one_based_indexing, triu!,
                     issymmetric, ishermitian, isposdef

using LinearAlgebra.BLAS: @blasfunc, libblastrampoline
using LinearAlgebra.LAPACK: chkfinite, chkuplofinite, chklapackerror

# LU factorisation
for (getrf, getrs, elty) in ((:dgetrf_, :dgetrs_, :Float64),
                             (:sgetrf_, :sgetrs_, :Float32),
                             (:zgetrf_, :zgetrs_, :ComplexF64),
                             (:cgetrf_, :cgetrs_, :ComplexF32))
    @eval begin
        function getrf!(A::AbstractMatrix{$elty}, ipiv::AbstractVector{BlasInt};
                        check::Bool=true)
            require_one_based_indexing(A, ipiv)
            chkstride1(A, ipiv)
            chkfinite(A)
            m, n = size(A)

            lda = max(1, stride(A, 2))
            info = Ref{BlasInt}()
            ccall((@blasfunc($getrf), libblastrampoline), Cvoid,
                  (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
                   Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
                  m, n, A, lda, ipiv, info)
            chkargsok(info[])
            return A, ipiv, info[] #Error code is stored in LU factorization type
        end
        function getrs!(trans::AbstractChar, A::AbstractMatrix{$elty},
                        ipiv::AbstractVector{BlasInt}, B::AbstractVecOrMat{$elty})
            require_one_based_indexing(A, ipiv, B)
            chktrans(trans)
            chkstride1(A, B, ipiv)
            n = checksquare(A)
            if n != size(B, 1)
                throw(DimensionMismatch(lazy"B has leading dimension $(size(B,1)), but needs $n"))
            end
            if n != length(ipiv)
                throw(DimensionMismatch(lazy"ipiv has length $(length(ipiv)), but needs to be $n"))
            end
            nrhs = size(B, 2)
            info = Ref{BlasInt}()
            ccall((@blasfunc($getrs), libblastrampoline), Cvoid,
                  (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                   Ptr{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Clong),
                  trans, n, size(B, 2), A, max(1, stride(A, 2)), ipiv, B,
                  max(1, stride(B, 2)), info, 1)
            chklapackerror(info[])
            return B
        end
    end
end

# QR factorisation
#! format: off

const DEFAULT_QR_BLOCKSIZE = 36
default_qr_blocksize(A::AbstractMatrix) = min(size(A)..., DEFAULT_QR_BLOCKSIZE)

for (gelqf, geqlf, geqrf, geqp3, geqrt, geqrt3, gerqf, getrf, elty, relty) in
    ((:dgelqf_, :dgeqlf_, :dgeqrf_, :dgeqp3_, :dgeqrt_, :dgeqrt3_, :dgerqf_, :dgetrf_, :Float64, :Float64),
     (:sgelqf_, :sgeqlf_, :sgeqrf_, :sgeqp3_, :sgeqrt_, :sgeqrt3_, :sgerqf_, :sgetrf_, :Float32, :Float32),
     (zgelqf_, :zgeqlf_, :zgeqrf_, :zgeqp3_, :zgeqrt_, :zgeqrt3_, :zgerqf_, :zgetrf_, :ComplexF64, :Float64),
     (:cgelqf_, :cgeqlf_, :cgeqrf_, :cgeqp3_, :cgeqrt_, :cgeqrt3_, :cgerqf_, :cgetrf_, :ComplexF32, :Float32))
#! format: on
    @eval begin
        # LQ
        function gelqf!(A::AbstractMatrix{$elty},
                        tau::AbstractVector{$elty}=similar(A, $elty, min(size(A)...)))
            require_one_based_indexing(A, tau)
            chkstride1(A, tau)
            m, n = size(A)
            length(tau) == min(m, n) ||
                throw(DimensionMismatch(lazy"tau has length $(length(tau)), but needs length $(min(m,n))"))
            n == 0 && return A, tau

            lda = max(1, stride(A, 2))
            lwork = BlasInt(-1)
            work = Vector{$elty}(undef, 1)
            info = Ref{BlasInt}()
            for i in 1:2  # first call returns lwork as work[1]
                ccall((@blasfunc($gelqf), libblastrampoline), Cvoid,
                      (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
                      m, n, A, lda,
                      tau, work, lwork, info)
                chklapackerror(info[])
                if i == 1
                    lwork = BlasInt(real(work[1]))
                    resize!(work, lwork)
                end
            end
            return A, tau
        end
        # QL
        function geqlf!(A::AbstractMatrix{$elty},
                        tau::AbstractVector{$elty}=similar(A, $elty, min(size(A)...)))
            require_one_based_indexing(A, tau)
            chkstride1(A, tau)
            m, n = size(A)
            length(tau) == min(m, n) ||
                throw(DimensionMismatch(lazy"tau has length $(length(tau)), but needs length $(min(m,n))"))
            n == 0 && return A, tau

            lda = max(1, stride(A, 2))
            lwork = BlasInt(-1)
            work = Vector{$elty}(undef, 1)
            info = Ref{BlasInt}()
            for i in 1:2  # first call returns lwork as work[1]
                ccall((@blasfunc($geqlf), libblastrampoline), Cvoid,
                      (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
                      m, n, A, lda,
                      tau, work, lwork, info)
                chklapackerror(info[])
                if i == 1
                    lwork = BlasInt(real(work[1]))
                    resize!(work, lwork)
                end
            end
            return A, tau
        end
        # QR
        function geqrf!(A::AbstractMatrix{$elty},
                        tau::AbstractVector{$elty}=similar(A, $elty, min(size(A)...)))
            require_one_based_indexing(A, tau)
            chkstride1(A, tau)
            m, n = size(A)
            length(tau) == min(m, n) ||
                throw(DimensionMismatch(lazy"tau has length $(length(tau)), but needs length $(min(m,n))"))
            n == 0 && return A, tau

            lda = max(1, stride(A, 2))
            work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            info = Ref{BlasInt}()
            for i in 1:2                # first call returns lwork as work[1]
                ccall((@blasfunc($geqrf), libblastrampoline), Cvoid,
                      (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
                      m, n, A, lda,
                      tau, work, lwork, info)
                chklapackerror(info[])
                if i == 1
                    lwork = max(BlasInt(1), BlasInt(real(work[1])))
                    resize!(work, lwork)
                end
            end
            return A, tau
        end
        # RQ
        function gerqf!(A::AbstractMatrix{$elty},
                        tau::AbstractVector{$elty}=similar(A, $elty, min(size(A)...)))
            require_one_based_indexing(A, tau)
            chkstride1(A, tau)
            m, n = size(A)
            length(tau) == min(m, n) ||
                throw(DimensionMismatch(lazy"tau has length $(length(tau)), but needs length $(min(m,n))"))
            n == 0 && return A, tau

            lda = max(1, stride(A, 2))
            lwork = BlasInt(-1)
            work = Vector{$elty}(undef, 1)
            info = Ref{BlasInt}()
            for i in 1:2                # first call returns lwork as work[1]
                ccall((@blasfunc($gerqf), libblastrampoline), Cvoid,
                      (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
                      m, n, A, lda,
                      tau, work, lwork, info)
                chklapackerror(info[])
                if i == 1
                    lwork = max(BlasInt(m), BlasInt(real(work[1])))
                    resize!(work, lwork)
                end
            end
            return A, tau
        end
        # QR with column pivoting
        function geqp3!(A::AbstractMatrix{$elty},
                        tau::AbstractVector{$elty}=similar(A, $elty, min(size(A)...)),
                        jpvt::AbstractVector{BlasInt}=similar(A, BlasInt, size(A, 2)))
            require_one_based_indexing(A, jpvt, tau)
            chkstride1(A, jpvt, tau)
            m, n = size(A)
            length(tau) == min(m, n) ||
                throw(DimensionMismatch(lazy"tau has length $(length(tau)), but needs length $(min(m,n))"))
            length(jpvt) == n ||
                throw(DimensionMismatch(lazy"jpvt has length $(length(jpvt)), but needs length $n"))
            n == 0 && return A, tau, jpvt

            lda = max(1, stride(A, 2))
            work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            cmplx = eltype(A) <: Complex
            if cmplx
                rwork = Vector{$relty}(undef, 2n)
            end
            info = Ref{BlasInt}()
            for i in 1:2  # first call returns lwork as work[1]
                if cmplx
                    ccall((@blasfunc($geqp3), libblastrampoline), Cvoid,
                          (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{BlasInt}, Ptr{$elty},
                           Ptr{$elty}, Ref{BlasInt}, Ptr{$relty},
                           Ptr{BlasInt}),
                          m, n, A, lda,
                          jpvt, tau,
                          work, lwork, rwork,
                          info)
                else
                    ccall((@blasfunc($geqp3), libblastrampoline), Cvoid,
                          (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{BlasInt}, Ptr{$elty},
                           Ptr{$elty}, Ref{BlasInt},
                           Ptr{BlasInt}),
                          m, n, A, lda,
                          jpvt, tau,
                          work, lwork,
                          info)
                end
                chklapackerror(info[])
                if i == 1
                    lwork = BlasInt(real(work[1]))
                    resize!(work, lwork)
                end
            end
            return A, tau, jpvt
        end
        # QR with block reflectors
        #! format: off
        function geqrt!(A::AbstractMatrix{$elty},
                        T::AbstractMatrix{$elty}=similar(A, $elty, default_qr_blocksize(A), min(size(A)...)))
        #! format: on
            require_one_based_indexing(A, T)
            chkstride1(A)
            m, n = size(A)
            minmn = min(m, n)
            nb = size(T, 1)
            nb <= minmn ||
                throw(ArgumentError(lazy"block size $nb > $minmn too large"))
            size(T, 2) == minmn ||
                throw(DimensionMismatch(lazy"block reflector T should have size ($nb,$minmn)"))
            n == 0 && return A, T

            lda = max(1, stride(A, 2))
            ldt = max(1, stride(T, 2))
            work = Vector{$elty}(undef, nb * n)
            if minmn > 0
                info = Ref{BlasInt}()
                ccall((@blasfunc($geqrt), libblastrampoline), Cvoid,
                      (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                       Ptr{BlasInt}),
                      m, n, nb, A, lda,
                      T, ldt, work,
                      info)
                chklapackerror(info[])
            end
            return A, T
        end
        # Recursive QR with block reflectors
        #! format: off
        function geqrt3!(A::AbstractMatrix{$elty},
                        T::AbstractMatrix{$elty}=similar(A, $elty, size(A, 2), size(A, 2)))
        #! format: on
            require_one_based_indexing(A, T)
            chkstride1(A, T)
            m, n = size(A)
            m >= n ||
                throw(DimensionMismatch(lazy"input matrix A has dimensions ($m,$n), but should have more rows than columns"))
            n == chksquare(T) ||
                throw(DimensionMismatch(lazy"block reflector T should have size ($n,$n)"))
            n == 0 && return A, T

            info = Ref{BlasInt}()
            ccall((@blasfunc($geqrt3), libblastrampoline), Cvoid,
                  (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                   Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
                  m, n, A, max(1, stride(A, 2)),
                  T, max(1, stride(T, 2)), info)
            chklapackerror(info[])
            return A, T
        end

        # SUBROUTINE DGETRF( M, N, A, LDA, IPIV, INFO )
        # *     .. Scalar Arguments ..
        #       INTEGER            INFO, LDA, M, N
        # *     .. Array Arguments ..
        #       INTEGER            IPIV( * )
        #       DOUBLE PRECISION   A( LDA, * )
        function getrf!(A::AbstractMatrix{$elty}, ipiv::AbstractVector{BlasInt};
                        check::Bool=true)
            require_one_based_indexing(A)
            check && chkfinite(A)
            chkstride1(A)
            m, n = size(A)
            lda = max(1, stride(A, 2))
            info = Ref{BlasInt}()
            ccall((@blasfunc($getrf), libblastrampoline), Cvoid,
                  (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
                   Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
                  m, n, A, lda, ipiv, info)
            chkargsok(info[])
            return A, ipiv, info[] #Error code is stored in LU factorization type
        end
    end
end

# Symmetric / Hermitian eigenvalue decomposition
for (heev, heevr, heevd, hegvd, elty, relty) in
    ((:dsyev_, :dsyevr_, :dsyevd_, :dsygvd_, :Float64, :Float64),
     (:ssyev_, :ssyevr_, :ssyevd_, :ssygvd_, :Float32, :Float32),
     (:zheev_, :zheevr_, :zheevd_, :zhegvd_, :ComplexF64, :Float64),
     (:cheev_, :cheevr_, :cheevd_, :chegvd_, :ComplexF32, :Float32))
    @eval begin
        function heev!(A::AbstractMatrix{$elty},
                       W::AbstractVector{$relty}=similar(A, $relty, size(A, 1)),
                       V::AbstractMatrix{$elty}=A)
            require_one_based_indexing(A, V, W)
            chkstride1(A, V, W)
            n = checksquare(A)
            if $elty <: Real
                issymmetric(A) || throw(ArgumentError("A must be symmetric"))
            else
                ishermitian(A) || throw(ArgumentError("A must be Hermitian"))
            end
            uplo = 'U' # shouldn't matter but 'U' seems slightly faster than 'L'
            chkuplofinite(A, uplo)
            n == length(W) || throw(DimensionMismatch("length mismatch between A and W"))
            if length(V) == 0
                jobz = 'N'
            else
                n == checksquare(V) ||
                    throw(DimensionMismatch("square size mismatch between A and V"))
                jobz = 'V'
            end

            lda = max(1, stride(A, 2))
            work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            info = Ref{BlasInt}()
            if $elty <: Real
                for i in 1:2  # first call returns lwork as work[1]
                    ccall((@blasfunc($heev), libblastrampoline), Cvoid,
                          (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$relty}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{BlasInt}, Clong, Clong),
                          jobz, uplo, n, A, lda,
                          W, work, lwork,
                          info, 1, 1)
                    chklapackerror(info[])
                    if i == 1
                        lwork = BlasInt(real(work[1]))
                        resize!(work, lwork)
                    end
                end
            else
                rwork = Vector{$relty}(undef, max(1, 3n - 2))
                for i in 1:2  # first call returns lwork as work[1]
                    ccall((@blasfunc($heev), libblastrampoline), Cvoid,
                          (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$relty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$relty},
                           Ptr{BlasInt}, Clong, Clong),
                          jobz, uplo, n, A, lda,
                          W, work, lwork, rwork,
                          info, 1, 1)
                    chklapackerror(info[])
                    if i == 1
                        lwork = BlasInt(real(work[1]))
                        resize!(work, lwork)
                    end
                end
            end
            if jobz == 'V' && V !== A
                copy!(V, A)
            end
            return W, V
        end
        function heevr!(A::AbstractMatrix{$elty},
                        W::AbstractVector{$relty}=similar(A, $relty, size(A, 1)),
                        V::AbstractMatrix{$elty}=similar(A);
                        kwargs...)
            require_one_based_indexing(A, V, W)
            chkstride1(A, V, W)
            n = checksquare(A)
            if $elty <: Real
                issymmetric(A) || throw(ArgumentError("A must be symmetric"))
            else
                ishermitian(A) || throw(ArgumentError("A must be Hermitian"))
            end
            uplo = 'U' # shouldn't matter but 'U' seems slightly faster than 'L'
            chkuplofinite(A, uplo)
            if haskey(kwargs, :irange)
                il = first(irange)
                iu = last(irange)
                vl = vu = zero($relty)
                range = 'I'
            elseif haskey(kwargs, :vl) || haskey(kwargs, :vu)
                vl = get(kwargs, :ul, $relty(-Inf))
                vu = get(kwargs, :vu, $relty(+Inf))
                il = iu = 0
                range = 'V'
            else
                il = iu = 0
                vl = vu = zero($relty)
                range = 'A'
            end
            length(W) == n || throw(DimensionMismatch("length mismatch between A and W"))
            if length(V) == 0
                jobz = 'N'
            else
                jobz = 'V'
                size(V, 1) == n || throw(DimensionMismatch("size mismatch between A and V"))
                if range == 'I'
                    size(V, 2) >= iu - il + 1 ||
                        throw(DimensionMismatch("number of columns of V must correspond to number of requested eigenvalues"))
                else
                    size(V, 2) == n ||
                        throw(DimensionMismatch("size mismatch between A and V"))
                end
            end

            lda = max(1, stride(A, 2))
            ldv = max(1, stride(V, 2))
            abstol = -one($relty)
            m = Ref{BlasInt}()
            isuppz = similar(A, BlasInt, 2 * n)
            work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            iwork = Vector{BlasInt}(undef, 1)
            liwork = BlasInt(-1)
            info = Ref{BlasInt}()
            if $elty <: Real
                for i in 1:2  # first call returns lwork as work[1] and liwork as iwork[1]
                    ccall((@blasfunc($heevr), libblastrampoline), Cvoid,
                          (Ref{UInt8}, Ref{UInt8}, Ref{UInt8},
                           Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ref{$elty}, Ref{$elty}, Ref{BlasInt}, Ref{BlasInt}, Ref{$elty},
                           Ptr{BlasInt},
                           Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                           Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt},
                           Ptr{BlasInt}, Clong, Clong, Clong),
                          jobz, range, uplo,
                          n, A, lda,
                          vl, vu, il, iu, abstol, m,
                          W, V, ldv, isuppz,
                          work, lwork, iwork, liwork,
                          info, 1, 1, 1)
                    chklapackerror(info[])
                    if i == 1
                        lwork = BlasInt(real(work[1]))
                        resize!(work, lwork)
                        liwork = iwork[1]
                        resize!(iwork, liwork)
                    end
                end
            else
                rwork = Vector{$relty}(undef, 1)
                lrwork = BlasInt(-1)
                for i in 1:2  # first call returns lwork as work[1], lrwork as rwork[1] and liwork as iwork[1]
                    ccall((@blasfunc($heevr), libblastrampoline), Cvoid,
                          (Ref{UInt8}, Ref{UInt8}, Ref{UInt8},
                           Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ref{$elty}, Ref{$elty}, Ref{BlasInt}, Ref{BlasInt}, Ref{$elty},
                           Ptr{BlasInt},
                           Ptr{$relty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                           Ptr{$elty}, Ref{BlasInt}, Ptr{$relty}, Ref{BlasInt},
                           Ptr{BlasInt}, Ref{BlasInt},
                           Ptr{BlasInt}, Clong, Clong, Clong),
                          jobz, range, uplo,
                          n, A, lda,
                          vl, vu, il, iu, abstol, m,
                          W, V, ldv, isuppz,
                          work, lwork, rwork, lrwork, iwork, liwork,
                          info, 1, 1, 1)
                    chklapackerror(info[])
                    if i == 1
                        lwork = BlasInt(real(work[1]))
                        resize!(work, lwork)
                        lrwork = BlasInt(rwork[1])
                        resize!(rwork, lrwork)
                        liwork = iwork[1]
                        resize!(iwork, liwork)
                    end
                end
            end
            return W, V, m[]
        end

        function heevd!(A::AbstractMatrix{$elty},
                        W::AbstractVector{$relty}=similar(A, $relty, size(A, 1)),
                        V::AbstractMatrix{$elty}=A)
            require_one_based_indexing(A, V, W)
            chkstride1(A, V, W)
            n = checksquare(A)
            if $elty <: Real
                issymmetric(A) || throw(ArgumentError("A must be symmetric"))
            else
                ishermitian(A) || throw(ArgumentError("A must be Hermitian"))
            end
            uplo = 'U' # shouldn't matter but 'U' seems slightly faster than 'L'
            chkuplofinite(A, uplo)
            n == length(W) || throw(DimensionMismatch("length mismatch between A and W"))
            if length(V) == 0
                jobz = 'N'
            else
                n == checksquare(V) ||
                    throw(DimensionMismatch("square size mismatch between A and V"))
                jobz = 'V'
            end

            lda = max(1, stride(A, 2))
            work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            iwork = Vector{BlasInt}(undef, 1)
            liwork = BlasInt(-1)
            info = Ref{BlasInt}()
            if $elty <: Real
                for i in 1:2  # first call returns lwork as work[1] and liwork as iwork[1]
                    ccall((@blasfunc($heevd), libblastrampoline), Cvoid,
                          (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$relty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                           Ref{BlasInt},
                           Ptr{BlasInt}, Clong, Clong),
                          jobz, uplo, n, A, lda,
                          W, work, lwork, iwork, liwork,
                          info, 1, 1)
                    chklapackerror(info[])
                    if i == 1
                        lwork = BlasInt(real(work[1]))
                        resize!(work, lwork)
                        liwork = iwork[1]
                        resize!(iwork, liwork)
                    end
                end
            else
                rwork = Vector{$relty}(undef, 1)
                lrwork = BlasInt(-1)
                for i in 1:2  # first call returns lwork as work[1], lrwork as rwork[1] and liwork as iwork[1]
                    ccall((@blasfunc($heevd), libblastrampoline), Cvoid,
                          (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$relty},
                           Ptr{$elty}, Ref{BlasInt}, Ptr{$relty}, Ref{BlasInt},
                           Ptr{BlasInt}, Ref{BlasInt},
                           Ptr{BlasInt}, Clong, Clong),
                          jobz, uplo, n, A, lda,
                          W,
                          work, lwork, rwork, lrwork, iwork, liwork,
                          info, 1, 1)
                    chklapackerror(info[])
                    if i == 1
                        lwork = BlasInt(real(work[1]))
                        resize!(work, lwork)
                        lrwork = BlasInt(rwork[1])
                        resize!(rwork, lrwork)
                        liwork = iwork[1]
                        resize!(iwork, liwork)
                    end
                end
            end
            if jobz == 'V' && V !== A
                copy!(V, A)
            end
            return W, V
        end

        #         # Generalized eigenproblem
        #         #           SUBROUTINE DSYGVD( ITYPE, JOBZ, UPLO, N, A, LDA, B, LDB, W, WORK,
        #         #      $                   LWORK, IWORK, LIWORK, INFO )
        #         # *     .. Scalar Arguments ..
        #         #       CHARACTER          JOBZ, UPLO
        #         #       INTEGER            INFO, ITYPE, LDA, LDB, LIWORK, LWORK, N
        #         # *     ..
        #         # *     .. Array Arguments ..
        #         #       INTEGER            IWORK( * )
        #         #       DOUBLE PRECISION   A( LDA, * ), B( LDB, * ), W( * ), WORK( * )
        #         function sygvd!(itype::Integer, jobz::AbstractChar, uplo::AbstractChar,
        #                         A::AbstractMatrix{$elty}, B::AbstractMatrix{$elty})
        #             require_one_based_indexing(A, B)
        #             @chkvalidparam 1 itype 1:3
        #             @chkvalidparam 2 jobz ('N', 'V')
        #             chkuplo(uplo)
        #             chkstride1(A, B)
        #             n, m = checksquare(A, B)
        #             if n != m
        #                 throw(DimensionMismatch(lazy"dimensions of A, ($n,$n), and B, ($m,$m), must match"))
        #             end
        #             lda = max(1, stride(A, 2))
        #             ldb = max(1, stride(B, 2))
        #             w = similar(A, $elty, n)
        #             work = Vector{$elty}(undef, 1)
        #             lwork = BlasInt(-1)
        #             iwork = Vector{BlasInt}(undef, 1)
        #             liwork = BlasInt(-1)
        #             info = Ref{BlasInt}()
        #             for i in 1:2  # first call returns lwork as work[1] and liwork as iwork[1]
        #                 ccall((@blasfunc($sygvd), libblastrampoline), Cvoid,
        #                       (Ref{BlasInt}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt},
        #                        Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
        #                        Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
        #                        Ref{BlasInt}, Ptr{BlasInt}, Clong, Clong),
        #                       itype, jobz, uplo, n,
        #                       A, lda, B, ldb,
        #                       w, work, lwork, iwork,
        #                       liwork, info, 1, 1)
        #                 chkargsok(info[])
        #                 if i == 1
        #                     lwork = BlasInt(work[1])
        #                     resize!(work, lwork)
        #                     liwork = iwork[1]
        #                     resize!(iwork, liwork)
        #                 end
        #             end
        #             chkposdef(info[])
        #             return w, A, B
        #         end
    end
end

# General eigenvalue decomposition
for (geev, geevx, ggev, elty, celty, relty) in
    ((:sgeev_, :sgeevx_, :sggev_, :Float32, :ComplexF32, :Float32),
     (:dgeev_, :dgeevx_, :dggev_, :Float64, :ComplexF64, :Float64),
     (:cgeev_, :cgeevx_, :cggev_, :ComplexF32, :ComplexF32, :Float32),
     (:zgeev_, :zgeevx_, :zggev_, :ComplexF64, :ComplexF64, :Float64))
    @eval begin
        function geev!(A::AbstractMatrix{$elty},
                       W::AbstractVector{$celty}=similar(A, $celty, size(A, 1)),
                       V::AbstractMatrix{$celty}=similar(A, $celty))
            require_one_based_indexing(A, V, W)
            chkstride1(A, V, W)
            n = checksquare(A)
            chkfinite(A) # balancing routines don't support NaNs and Infs
            n == length(W) || throw(DimensionMismatch("length mismatch between A and W"))
            if length(V) == 0
                jobvr = 'N'
            else
                n == checksquare(V) ||
                    throw(DimensionMismatch("square size mismatch between A and VR"))
                jobvr = 'V'
            end
            jobvl = 'N'

            lda = max(1, stride(A, 2))
            work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            info = Ref{BlasInt}()
            VL = similar(A, n, 0)
            ldvl = max(1, stride(VL, 2))

            if eltype(A) <: Real
                W2 = reinterpret($elty, W)
                # reuse memory, we will have to reorder afterwards to bring real and imaginary
                # components in the order as required for the Complex type
                WR = view(W2, 1:n)
                WI = view(W2, (n + 1):(2n))
                VR = reinterpret($elty, V)
                ldvr = max(1, stride(VR, 2))
                for i in 1:2  # first call returns lwork as work[1]
                    #! format: off
                    ccall((@blasfunc($geev), libblastrampoline), Cvoid,
                          (Ref{UInt8}, Ref{UInt8},
                           Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$elty}, Ref{BlasInt},
                           Ptr{BlasInt}, Clong, Clong),
                          jobvl, jobvr,
                          n, A, lda,
                          WR, WI, VL, ldvl, VR, ldvr,
                          work, lwork,
                          info, 1, 1)
                    #! format: on
                    chklapackerror(info[])
                    if i == 1
                        lwork = BlasInt(real(work[1]))
                        resize!(work, lwork)
                    end
                end
            else
                VR = V
                ldvr = max(1, stride(VR, 2))
                rwork = Vector{$relty}(undef, 2n)
                for i in 1:2
                    #! format: off
                    ccall((@blasfunc($geev), libblastrampoline), Cvoid,
                          (Ref{UInt8}, Ref{UInt8},
                           Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$elty}, Ref{BlasInt}, Ptr{$relty},
                           Ptr{BlasInt}, Clong, Clong),
                          jobvl, jobvr,
                          n, A, lda, 
                          W, VL, ldvl, VR, ldvr,
                          work, lwork, rwork,
                          info, 1, 1)
                    #! format: on
                    chklapackerror(info[])
                    if i == 1
                        lwork = BlasInt(real(work[1]))
                        resize!(work, lwork)
                    end
                end
            end

            # Cleanup the output in the real case
            if eltype(A) <: Real
                _reorder_realeigendecomposition!(W, WR, WI, work, VR, jobvr)
            end
            return W, V
        end
        function geevx!(A::AbstractMatrix{$elty},
                        W::AbstractVector{$celty}=similar(A, $celty, size(A, 1)),
                        V::AbstractMatrix{$celty}=similar(A, $celty);
                        scale::Bool=true, permute::Bool=true)
            require_one_based_indexing(A, V, W)
            chkstride1(A, V, W)
            n = checksquare(A)
            chkfinite(A) # balancing routines don't support NaNs and Infs
            n == length(W) || throw(DimensionMismatch("length mismatch between A and W"))
            if length(V) == 0
                jobvr = 'N'
            else
                n == checksquare(V) ||
                    throw(DimensionMismatch("square size mismatch between A and VR"))
                jobvr = 'V'
            end
            jobvl = 'N'

            if scale && permute
                balanc = 'B'
            elseif scale
                balanc = 'S'
            elseif permute
                balanc = 'P'
            else
                balanc = 'N'
            end
            sense = 'N'

            lda = max(1, stride(A, 2))
            ilo = Ref{BlasInt}()
            ihi = Ref{BlasInt}()
            scale = similar(A, $relty, n)
            abnrm = Ref{$relty}()
            rconde = similar(A, $relty, n)
            rcondv = similar(A, $relty, n)
            work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            iworksize = 0
            iwork = Vector{BlasInt}(undef, iworksize)
            info = Ref{BlasInt}()
            VL = similar(A, n, 0)
            ldvl = max(1, stride(VL, 2))

            if eltype(A) <: Real
                W2 = reinterpret($elty, W)
                # reuse memory, we will have to reorder afterwards to bring real and imaginary
                # components in the order as required for the Complex type
                WR = view(W2, 1:n)
                WI = view(W2, (n + 1):(2n))
                VR = reinterpret($elty, V)
                ldvr = max(1, stride(VR, 2))
                for i in 1:2  # first call returns lwork as work[1]
                    #! format: off
                    ccall((@blasfunc($geevx), libblastrampoline), Cvoid,
                          (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{UInt8},
                           Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$elty}, Ptr{$elty},
                           Ptr{$elty}, Ptr{$elty},
                           Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt},
                           Clong, Clong, Clong, Clong),
                          balanc, jobvl, jobvr, sense,
                          n, A, lda,
                          WR, WI, VL, ldvl, VR, ldvr,
                          ilo, ihi, scale, abnrm,
                          rconde, rcondv,
                          work, lwork, iwork, info,
                          1, 1, 1, 1)
                    #! format: on
                    chklapackerror(info[])
                    if i == 1
                        lwork = BlasInt(real(work[1]))
                        resize!(work, lwork)
                    end
                end
            else
                VR = V
                ldvr = max(1, stride(VR, 2))
                rwork = Vector{$relty}(undef, 2n)
                for i in 1:2
                    #! format: off
                    ccall((@blasfunc($geevx), libblastrampoline), Cvoid,
                          (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{UInt8},
                           Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$relty}, Ptr{$relty},
                           Ptr{$relty}, Ptr{$relty},
                           Ptr{$elty}, Ref{BlasInt}, Ptr{$relty}, Ref{BlasInt},
                           Clong, Clong, Clong, Clong),
                          balanc, jobvl, jobvr, sense,
                          n, A, lda,
                          W, VL, ldvl, VR, ldvr,
                          ilo, ihi, scale, abnrm,
                          rconde, rcondv,
                          work, lwork, rwork, info,
                          1, 1, 1, 1)
                    #! format: on
                    chklapackerror(info[])
                    if i == 1
                        lwork = BlasInt(real(work[1]))
                        resize!(work, lwork)
                    end
                end
            end

            # Cleanup the output in the real case
            if eltype(A) <: Real
                _reorder_realeigendecomposition!(W, WR, WI, work, VR, jobvr)
            end
            return W, V
        end
        function ggev!(A::AbstractMatrix{$elty}, B::AbstractMatrix{$elty},
                       W::AbstractVector{$celty}=similar(A, $celty, size(A, 1)),
                       V::AbstractMatrix{$celty}=similar(A, $celty))
            require_one_based_indexing(A, B, V, W)
            chkstride1(A, B, V, W)
            n = checksquare(A)
            n == checksquare(B) || throw(DimensionMismatch("size mismatch between A and B"))
            n == length(W) || throw(DimensionMismatch("length mismatch between A and W"))
            if length(V) == 0
                jobvr = 'N'
            else
                n == checksquare(V) ||
                    throw(DimensionMismatch("square size mismatch between A and VR"))
                jobvr = 'V'
            end
            jobvl = 'N'

            lda = max(1, stride(A, 2))
            ldb = max(1, stride(B, 2))
            work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            info = Ref{BlasInt}()
            VL = similar(A, n, 0)
            ldvl = stride(VL, 2)

            if eltype(A) <: Real
                W2 = reinterpret($elty, W)
                # reuse memory, we will have to reorder afterwards to bring real and imaginary
                # components in the order as required for the Complex type
                WR = view(W2, 1:n)
                WI = view(W2, (n + 1):(2n))
                VR = reinterpret($elty, V)
                ldvr = stride(VR, 2)
                for i in 1:2  # first call returns lwork as work[1]
                    #! format: off
                    ccall((@blasfunc($geev), libblastrampoline), Cvoid,
                          (Ref{UInt8}, Ref{UInt8},
                           Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$elty}, Ref{BlasInt},
                           Ptr{BlasInt}, Clong, Clong),
                          jobvl, jobvr,
                          n, A, lda,
                          WR, WI, VL, ldvl, VR, ldvr,
                          work, lwork,
                          info, 1, 1)
                    #! format: on
                    chklapackerror(info[])
                    if i == 1
                        lwork = BlasInt(real(work[1]))
                        resize!(work, lwork)
                    end
                end
            else
                VR = V
                ldvr = stride(VR, 2)
                rwork = Vector{$relty}(undef, 2n)
                for i in 1:2
                    #! format: off
                    ccall((@blasfunc($geev), libblastrampoline), Cvoid,
                          (Ref{UInt8}, Ref{UInt8},
                           Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$elty}, Ref{BlasInt}, Ptr{$relty},
                           Ptr{BlasInt}, Clong, Clong),
                          jobvl, jobvr,
                          n, A, lda, 
                          W, VL, ldvl, VR, ldvr,
                          work, lwork, rwork,
                          info, 1, 1)
                    #! format: on
                    chklapackerror(info[])
                    if i == 1
                        lwork = BlasInt(real(work[1]))
                        resize!(work, lwork)
                    end
                end
            end

            # Cleanup the output in the real case
            if eltype(A) <: Real
                _reorder_realeigendecomposition!(W, WR, WI, work, VR, jobvr)
            end
            return W, V
        end
    end
end

function _reorder_realeigendecomposition!(W, WR, WI, work, VR, jobvr)
    # first reorder eigenvalues and recycle work as temporary buffer to efficiently implement the permutation
    n = size(W, 1)
    resize!(work, n)
    copy!(work, WI)
    for i in n:-1:1
        W[i] = WR[i] + im * work[i]
    end
    if jobvr == 'V' # also reorganise vectors
        i = 1
        while i <= n
            if iszero(imag(W[i])) # real eigenvalue => real eigenvector
                for j in n:-1:1
                    VR[2 * j, i] = 0
                    VR[2 * j - 1, i] = VR[j, i]
                end
                i += 1
            else # complex eigenvalue => complex eigenvector and conjugate
                @assert i != n
                for j in n:-1:1
                    VR[2 * j, i] = VR[j, i + 1]
                    VR[2 * j - 1, i] = VR[j, i]
                    VR[2 * j, i + 1] = -VR[j, i + 1]
                    VR[2 * j - 1, i + 1] = VR[j, i]
                end
                i += 2
            end
        end
    end
end

# SVD
for (gesvd, gesdd, elty, relty) in
    ((:dgesvd_, :dgesdd_, :Float64, :Float64),
     (:sgesvd_, :sgesdd_, :Float32, :Float32),
     (:zgesvd_, :zgesdd_, :ComplexF64, :Float64),
     (:cgesvd_, :cgesdd_, :ComplexF32, :Float32))
    @eval begin
        #! format: off
        function gesdd!(A::AbstractMatrix{$elty},
                        S::AbstractVector{$relty}=similar(A, $relty, min(size(A)...)),
                        U::AbstractMatrix{$elty}=similar(A, $elty, size(A, 1), min(size(A)...)),
                        Vᴴ::AbstractMatrix{$elty}=similar(A, $elty, min(size(A)...), size(A, 2)))
        #! format: on
            require_one_based_indexing(A, U, Vᴴ, S)
            chkstride1(A, U, Vᴴ, S)
            m, n = size(A)
            minmn = min(m, n)

            if length(U) == 0 && length(Vᴴ) == 0
                job = 'N'
            else
                size(U, 1) == m ||
                    throw(DimensionMismatch("row size mismatch between A and U"))
                size(Vᴴ, 2) == n ||
                    throw(DimensionMismatch("column size mismatch between A and Vᴴ"))
                length(S) == minmn ||
                    throw(DimensionMismatch("length mismatch between A and S"))
                if size(U, 2) == m && size(Vᴴ, 1) == n
                    job = 'A'
                elseif size(U, 2) == minmn && size(Vᴴ, 1) == minmn
                    if m >= n && U === A
                        job = 'O'
                    elseif m < n && Vᴴ === A
                        job = 'O'
                    else
                        job = 'S'
                    end
                else
                    throw(DimensionMismatch("invalid column size of U or row size of Vᴴ"))
                end
            end

            lda = max(1, stride(A, 2))
            ldu = max(1, stride(U, 2))
            ldv = max(1, stride(Vᴴ, 2))
            work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            cmplx = eltype(A) <: Complex
            if cmplx
                lrwork = job == 'N' ? 7 * minmn :
                         minmn * max(5 * minmn + 7, 2 * max(m, n) + 2 * minmn + 1)
                rwork = Vector{$relty}(undef, lrwork)
            end
            iwork = Vector{BlasInt}(undef, 8 * minmn)
            info = Ref{BlasInt}()
            for i in 1:2  # first call returns lwork as work[1]
                #! format: off
                if cmplx
                    ccall((@blasfunc($gesdd), libblastrampoline), Cvoid,
                          (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$relty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$elty}, Ref{BlasInt}, Ptr{$relty}, Ptr{BlasInt},
                           Ptr{BlasInt}, Clong),
                          job, m, n, A, lda,
                          S, U, ldu, Vᴴ, ldv,
                          work, lwork, rwork, iwork,
                          info, 1)
                else
                    ccall((@blasfunc($gesdd), libblastrampoline), Cvoid,
                          (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                           Ptr{BlasInt}, Clong),
                          job, m, n, A, lda,
                          S, U, ldu, Vᴴ, ldv,
                          work, lwork, iwork,
                          info, 1)
                end
                #! format: on
                chklapackerror(info[])
                if i == 1
                    # Work around issue with truncated Float32 representation of lwork in
                    # sgesdd by using nextfloat. See
                    # http://icl.cs.utk.edu/lapack-forum/viewtopic.php?f=13&t=4587&p=11036&hilit=sgesdd#p11036
                    # and
                    # https://github.com/scipy/scipy/issues/5401
                    lwork = round(BlasInt, nextfloat(real(work[1])))
                    resize!(work, lwork)
                end
            end
            return (S, U, Vᴴ)
        end
        #! format: off
        function gesvd!(A::AbstractMatrix{$elty},
                        S::AbstractVector{$relty}=similar(A, $relty, min(size(A)...)),
                        U::AbstractMatrix{$elty}=similar(A, $elty, size(A, 1), min(size(A)...)),
                        Vᴴ::AbstractMatrix{$elty}=similar(A, $elty, min(size(A)...), size(A, 2)))
        #! format: on
            require_one_based_indexing(A, U, Vᴴ, S)
            chkstride1(A, U, Vᴴ, S)
            m, n = size(A)
            minmn = min(m, n)
            if length(U) == 0
                jobu = 'N'
            else
                size(U, 1) == m ||
                    throw(DimensionMismatch("row size mismatch between A and U"))
                if size(U, 2) == minmn
                    if U === A
                        jobu = 'O'
                    else
                        jobu = 'S'
                    end
                elseif size(U, 2) == m
                    jobu = 'A'
                else
                    throw(DimensionMismatch("invalid column size of U"))
                end
            end
            if length(Vᴴ) == 0
                jobvt = 'N'
            else
                size(Vᴴ, 2) == n ||
                    throw(DimensionMismatch("column size mismatch between A and Vᴴ"))
                if size(Vᴴ, 1) == minmn
                    if Vᴴ === A
                        jobvt = 'O'
                    else
                        jobvt = 'S'
                    end
                elseif size(Vᴴ, 1) == n
                    jobvt = 'A'
                else
                    throw(DimensionMismatch("invalid row size of Vᴴ"))
                end
            end
            length(S) == minmn ||
                throw(DimensionMismatch("length mismatch between A and S"))

            lda = max(1, stride(A, 2))
            ldu = max(1, stride(U, 2))
            ldv = max(1, stride(Vᴴ, 2))
            work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            cmplx = eltype(A) <: Complex
            if cmplx
                rwork = Vector{$relty}(undef, 5minmn)
            end
            info = Ref{BlasInt}()
            for i in 1:2  # first call returns lwork as work[1]
                #! format: off
                if cmplx
                    ccall((@blasfunc($gesvd), libblastrampoline), Cvoid,
                          (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$relty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$elty}, Ref{BlasInt}, Ptr{$relty},
                           Ptr{BlasInt}, Clong, Clong),
                          jobu, jobvt, m, n, A, lda,
                          S, U, ldu, Vᴴ, ldv,
                          work, lwork, rwork,
                          info, 1, 1)
                else
                    ccall((@blasfunc($gesvd), libblastrampoline), Cvoid,
                          (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$elty}, Ref{BlasInt},
                           Ptr{BlasInt}, Clong, Clong),
                          jobu, jobvt, m, n, A, lda,
                          S, U, ldu, Vᴴ, ldv,
                          work, lwork,
                          info, 1, 1)
                end
                #! format: on
                chklapackerror(info[])
                if i == 1
                    lwork = BlasInt(real(work[1]))
                    resize!(work, lwork)
                end
            end
            return (S, U, Vᴴ)
        end
    end
end

end
