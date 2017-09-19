# Scalar product and norm: only valid for EuclideanSpace
Base.dot{S<:EuclideanSpace}(t1::Tensor{S},t2::Tensor{S})= (space(t1)==space(t2) ? dot(vec(t1),vec(t2)) : throw(SpaceError()))
Base.vecnorm{S<:EuclideanSpace}(t::Tensor{S})=vecnorm(t.data) # frobenius norm


# # Factorizations:
# #-----------------
# for (S,TT) in ((CartesianSpace,CartesianTensor),(ComplexSpace,ComplexTensor))
#     @eval function svd!(t::$TT,n::Int,::NoTruncation)
#         # Perform singular value decomposition corresponding to bipartion of the
#         # tensor indices into the left indices 1:n and remaining right indices,
#         # thereby destroying the original tensor.
#         N=numind(t)
#         spacet=space(t)
#         leftspace=spacet[1:n]
#         rightspace=spacet[n+1:N]
#         leftdim=dim(leftspace)
#         rightdim=dim(rightspace)
#         data=reshape(t.data,(leftdim,rightdim))
#         F=svdfact!(data)
#         newdim=length(F[:S])
#         newspace=$S(newdim)
#         U=tensor(F[:U],leftspace*newspace')
#         S=tensor(diagm(F[:S]),newspace*newspace')
#         V=tensor(F[:Vt],newspace*rightspace)
#         return U,S,V,abs(zero(eltype(t)))
#     end
#
#     @eval function svd!(t::$TT,n::Int,trunc::Union(TruncationDimension,TruncationSpace))
#         # Truncate rank corresponding to bipartition into left indices 1:n
#         # and remain right indices, based on singular value decomposition,
#         # thereby destroying the original tensor.
#         # Truncation parameters are given as keyword arguments: trunctol should
#         # always be one of the possible arguments for specifying truncation, but
#         # truncdim can be replaced with different parameters for other types of tensors.
#
#         N=numind(t)
#         spacet=space(t)
#         leftspace=spacet[1:n]
#         rightspace=spacet[n+1:N]
#         leftdim=dim(leftspace)
#         rightdim=dim(rightspace)
#         dim(trunc) >= min(leftdim,rightdim) && return svd!(t,n)
#         data=reshape(t.data,(leftdim,rightdim))
#         F=svdfact!(data)
#         sing=F[:S]
#
#         # compute truncation dimension
#         if eps(trunc)!=0
#             sing=F[:S]
#             normsing=vecnorm(sing)
#             truncdim=0
#             while truncdim<length(sing) && vecnorm(sing[(truncdim+1):end])>eps(trunc)*normsing
#                 truncdim+=1
#             end
#             truncdim=min(truncdim,dim(trunc))
#         else
#             truncdim=dim(trunc)
#         end
#         newspace=$S(truncdim)
#
#         # truncate
#         truncerr=vecnorm(sing[(truncdim+1):end])/vecnorm(sing)
#         U=tensor(F[:U][:,1:truncdim],leftspace*newspace')
#         S=tensor(diagm(sing[1:truncdim]),newspace*newspace')
#         V=tensor(F[:Vt][1:truncdim,:],newspace*rightspace)
#         return U,S,V,truncerr
#     end
#
#     @eval function svd!(t::$TT,n::Int,trunc::TruncationError)
#         # Truncate rank corresponding to bipartition into left indices 1:n
#         # and remain right indices, based on singular value decomposition,
#         # thereby destroying the original tensor.
#         # Truncation parameters are given as keyword arguments: trunctol should
#         # always be one of the possible arguments for specifying truncation, but
#         # truncdim can be replaced with different parameters for other types of tensors.
#
#         N=numind(t)
#         spacet=space(t)
#         leftspace=spacet[1:n]
#         rightspace=spacet[n+1:N]
#         leftdim=dim(leftspace)
#         rightdim=dim(rightspace)
#         data=reshape(t.data,(leftdim,rightdim))
#         F=svdfact!(data)
#
#         # find truncdim based on trunctolinfo
#         sing=F[:S]
#         normsing=vecnorm(sing)
#         truncdim=0
#         while truncdim<length(sing) && vecnorm(sing[(truncdim+1):end])>eps(trunc)*normsing
#             truncdim+=1
#         end
#         newspace=$S(truncdim)
#
#         # truncate
#         truncerr=vecnorm(sing[(truncdim+1):end])/normsing
#         U=tensor(F[:U][:,1:truncdim],leftspace*newspace')
#         S=tensor(diagm(sing[1:truncdim]),newspace*newspace')
#         V=tensor(F[:Vt][1:truncdim,:],newspace*rightspace)
#         return U,S,V,truncerr
#     end
#
#     @eval function leftorth!(t::$TT,n::Int,::NoTruncation)
#         # Create orthogonal basis U for indices 1:n, and remainder C for right
#         # indices, thereby destroying the original tensor.
#         # Decomposition should be unique, such that it always returns the same
#         # result for the same input tensor t. UC = QR is fastest but only unique
#         # after correcting for phases.
#         N=numind(t)
#         spacet=space(t)
#         leftspace=spacet[1:n]
#         rightspace=spacet[n+1:N]
#         leftdim=dim(leftspace)
#         rightdim=dim(rightspace)
#         data=reshape(t.data,(leftdim,rightdim))
#         if leftdim>rightdim
#             newdim=rightdim
#             tau=zeros(eltype(data),(newdim,))
#             Base.LinAlg.LAPACK.geqrf!(data,tau)
#
#             C=zeros(eltype(t),(newdim,newdim))
#             for j in 1:newdim
#                 for i in 1:j
#                     @inbounds C[i,j]=data[i,j]
#                 end
#             end
#             Base.LinAlg.LAPACK.orgqr!(data,tau)
#             U=data
#             for i=1:newdim
#                 tau[i]=sign(C[i,i])
#             end
#             scale!(U,tau)
#             scale!(conj!(tau),C)
#         else
#             newdim=leftdim
#             C=data
#             U=eye(eltype(data),newdim)
#         end
#
#         newspace=$S(newdim)
#         return tensor(U,leftspace*newspace'), tensor(C,newspace*rightspace), abs(zero(eltype(t)))
#     end
#
#     @eval function leftorth!(t::$TT,n::Int,trunc::Union(TruncationDimension,TruncationSpace))
#         # Truncate rank corresponding to bipartition into left indices 1:n
#         # and remain right indices, based on singular value decomposition,
#         # thereby destroying the original tensor.
#         # Truncation parameters are given as keyword arguments: trunctol should
#         # always be one of the possible arguments for specifying truncation, but
#         # truncdim can be replaced with different parameters for other types of tensors.
#
#         N=numind(t)
#         spacet=space(t)
#         leftspace=spacet[1:n]
#         rightspace=spacet[n+1:N]
#         leftdim=dim(leftspace)
#         rightdim=dim(rightspace)
#         dim(trunc) >= min(leftdim,rightdim) && return leftorth!(t,n)
#         data=reshape(t.data,(leftdim,rightdim))
#         F=svdfact!(data)
#         sing=F[:S]
#
#         # compute truncation dimension
#         if eps(trunc)!=0
#             sing=F[:S]
#             normsing=vecnorm(sing)
#             truncdim=0
#             while truncdim<length(sing) && vecnorm(sing[(truncdim+1):end])>eps(trunc)*normsing
#                 truncdim+=1
#             end
#             truncdim=min(truncdim,dim(trunc))
#         else
#             truncdim=dim(trunc)
#         end
#         newspace=$S(truncdim)
#
#         # truncate
#         truncerr=vecnorm(sing[(truncdim+1):end])/vecnorm(sing)
#         U=tensor(F[:U][:,1:truncdim],leftspace*newspace')
#         C=tensor(scale!(sing[1:truncdim],F[:Vt][1:truncdim,:]),newspace*rightspace)
#         return U,C,truncerr
#     end
#
#     @eval function leftorth!(t::$TT,n::Int,trunc::TruncationError)
#         # Truncate rank corresponding to bipartition into left indices 1:n
#         # and remain right indices, based on singular value decomposition,
#         # thereby destroying the original tensor.
#         # Truncation parameters are given as keyword arguments: trunctol should
#         # always be one of the possible arguments for specifying truncation, but
#         # truncdim can be replaced with different parameters for other types of tensors.
#
#         N=numind(t)
#         spacet=space(t)
#         leftspace=spacet[1:n]
#         rightspace=spacet[n+1:N]
#         leftdim=dim(leftspace)
#         rightdim=dim(rightspace)
#         data=reshape(t.data,(leftdim,rightdim))
#         F=svdfact!(data)
#
#         # compute truncation dimension
#         sing=F[:S]
#         normsing=vecnorm(sing)
#         truncdim=0
#         while truncdim<length(sing) && vecnorm(sing[(truncdim+1):end])>eps(trunc)*normsing
#             truncdim+=1
#         end
#         newspace=$S(truncdim)
#
#         # truncate
#         truncerr=vecnorm(sing[(truncdim+1):end])/vecnorm(sing)
#         U=tensor(F[:U][:,1:truncdim],leftspace*newspace')
#         C=tensor(scale!(sing[1:truncdim],F[:Vt][1:truncdim,:]),newspace*rightspace)
#
#         return U,C,truncerr
#     end
#
#     @eval function rightorth!(t::$TT,n::Int,::NoTruncation)
#         # Create orthogonal basis U for right indices, and remainder C for left
#         # indices. Decomposition should be unique, such that it always returns the
#         # same result for the same input tensor t. CU = LQ is typically fastest but only
#         # unique after correcting for phases.
#         N=numind(t)
#         spacet=space(t)
#         leftspace=spacet[1:n]
#         rightspace=spacet[n+1:N]
#         leftdim=dim(leftspace)
#         rightdim=dim(rightspace)
#         data=reshape(t.data,(leftdim,rightdim))
#         if leftdim<rightdim
#             newdim=leftdim
#             tau=zeros(eltype(data),(newdim,))
#             datat=transpose(data)
#             Base.LinAlg.LAPACK.geqrf!(datat,tau)
#
#             C=zeros(eltype(t),(newdim,newdim))
#             for j in 1:newdim
#                 for i in 1:j
#                     @inbounds C[j,i]=datat[i,j]
#                 end
#             end
#             Base.LinAlg.LAPACK.orgqr!(datat,tau)
#             Base.transpose!(data,datat)
#             U=data
#
#             for i=1:newdim
#                 tau[i]=sign(C[i,i])
#             end
#             scale!(tau,U)
#             scale!(C,conj!(tau))
#         else
#             newdim=rightdim
#             C=data
#             U=eye(eltype(data),newdim)
#         end
#
#         newspace=$S(newdim)
#         return tensor(C,leftspace ⊗ dual(newspace)), tensor(U,newspace ⊗ rightspace), abs(zero(eltype(t)))
#     end
#
#     @eval function rightorth!(t::$TT,n::Int,trunc::Union(TruncationDimension,TruncationSpace))
#         # Truncate rank corresponding to bipartition into left indices 1:n
#         # and remain right indices, based on singular value decomposition,
#         # thereby destroying the original tensor.
#         # Truncation parameters are given as keyword arguments: trunctol should
#         # always be one of the possible arguments for specifying truncation, but
#         # truncdim can be replaced with different parameters for other types of tensors.
#
#         N=numind(t)
#         spacet=space(t)
#         leftspace=spacet[1:n]
#         rightspace=spacet[n+1:N]
#         leftdim=dim(leftspace)
#         rightdim=dim(rightspace)
#         dim(trunc) >= min(leftdim,rightdim) && return rightorth!(t,n)
#         data=reshape(t.data,(leftdim,rightdim))
#         F=svdfact!(data)
#         sing=F[:S]
#
#         # compute truncation dimension
#         if eps(trunc)!=0
#             sing=F[:S]
#             normsing=vecnorm(sing)
#             truncdim=0
#             while truncdim<length(sing) && vecnorm(sing[(truncdim+1):end])>eps(trunc)*normsing
#                 truncdim+=1
#             end
#             truncdim=min(truncdim,dim(trunc))
#         else
#             truncdim=dim(trunc)
#         end
#         newspace=$S(truncdim)
#
#         # truncate
#         truncerr=vecnorm(sing[(truncdim+1):end])/vecnorm(sing)
#         C=tensor(scale!(F[:U][:,1:truncdim],sing[1:truncdim]),leftspace*newspace')
#         U=tensor(F[:Vt][1:truncdim,:],newspace*rightspace)
#         return C,U,truncerr
#     end
#
#     @eval function rightorth!(t::$TT,n::Int,trunc::TruncationError)
#         # Truncate rank corresponding to bipartition into left indices 1:n
#         # and remain right indices, based on singular value decomposition,
#         # thereby destroying the original tensor.
#         # Truncation parameters are given as keyword arguments: trunctol should
#         # always be one of the possible arguments for specifying truncation, but
#         # truncdim can be replaced with different parameters for other types of tensors.
#
#         N=numind(t)
#         spacet=space(t)
#         leftspace=spacet[1:n]
#         rightspace=spacet[n+1:N]
#         leftdim=dim(leftspace)
#         rightdim=dim(rightspace)
#         data=reshape(t.data,(leftdim,rightdim))
#         F=svdfact!(data)
#
#         # compute truncation dimension
#         sing=F[:S]
#         normsing=vecnorm(sing)
#         truncdim=0
#         while truncdim<length(sing) && vecnorm(sing[(truncdim+1):end])>eps(trunc)*normsing
#             truncdim+=1
#         end
#         newspace=$S(truncdim)
#
#         # truncate
#         truncerr=vecnorm(sing[(truncdim+1):end])/vecnorm(sing)
#         C=tensor(scale!(F[:U][:,1:truncdim],sing[1:truncdim]),leftspace*newspace')
#         U=tensor(F[:Vt][1:truncdim,:],newspace*rightspace)
#         return C,U,truncerr
#     end
# end
#
# # Methods below are only implemented for CartesianMatrix or ComplexMatrix:
# #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# typealias ComplexMatrix{T} ComplexTensor{T,2}
# typealias CartesianMatrix{T} CartesianTensor{T,2}
#
# function Base.pinv(t::Union(ComplexMatrix,CartesianMatrix))
#     # Compute pseudo-inverse
#     spacet=space(t)
#     data=copy(t.data)
#     leftdim=dim(spacet[1])
#     rightdim=dim(spacet[2])
#
#     F=svdfact!(data)
#     Sinv=F[:S]
#     for k=1:length(Sinv)
#         if Sinv[k]>eps(Sinv[1])*max(leftdim,rightdim)
#             Sinv[k]=one(Sinv[k])/Sinv[k]
#         end
#     end
#     data=F[:V]*scale(F[:S],F[:U]')
#     return tensor(data,spacet')
# end
#
# function Base.eig(t::Union(ComplexMatrix,CartesianMatrix))
#     # Compute eigenvalue decomposition.
#     spacet=space(t)
#     spacet[1] == spacet[2]' || throw(SpaceError("eigenvalue factorization only exists if left and right index space are dual"))
#     data=copy(t.data)
#
#     F=eigfact!(data)
#
#     Lambda=tensor(diagm(F[:values]),spacet)
#     V=tensor(F[:vectors],spacet)
#     return Lambda, V
# end
#
# function Base.inv(t::Union(ComplexMatrix,CartesianMatrix))
#     # Compute inverse.
#     spacet=space(t)
#     spacet[1] == spacet[2]' || throw(SpaceError("inverse only exists if left and right index space are dual"))
#
#     return tensor(inv(t.data),spacet)
# end
