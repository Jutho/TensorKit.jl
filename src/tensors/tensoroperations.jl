# # Tensor Operations
# #-------------------
# scalar(t::Tensor)=iscnumber(space(t)) ? t.data[1] : throw(SpaceError("Not a scalar"))
#
# function tensorcopy!(t1::Tensor,labels1,t2::Tensor,labels2)
#     # Replaces tensor t2 with t1
#     N1=numind(t1)
#     perm=indexin(labels2,labels1)
#
#     length(perm) == N1 || throw(TensorOperations.LabelError("invalid label specification"))
#     isperm(perm) || throw(TensorOperations.LabelError("invalid label specification"))
#     for i = 1:N1
#         space(t1,i) == space(t2,perm[i]) || throw(SpaceError())
#     end
#     N1==0 && (t2.data[1]=t1.data[1]; return t2)
#     perm==[1:N1] && return copy!(t2,t1)
#     TensorOperations.tensorcopy_native!(t1.data,t2.data,perm)
#     return t2
# end
# function tensoradd!(alpha::Number,t1::Tensor,labels1,beta::Number,t2::Tensor,labels2)
#     # Replaces tensor t2 with beta*t2+alpha*t1
#     N1=numind(t1)
#     perm=indexin(labels2,labels1)
#
#     length(perm) == N1 || throw(TensorOperations.LabelError("invalid label specification"))
#     isperm(perm) || throw(TensorOperations.LabelError("invalid label specification"))
#     for i = 1:N1
#         space(t1,perm[i]) == space(t2,i) || throw(SpaceError("incompatible index spaces of tensors"))
#     end
#     N1==0 && (t2.data[1]=beta*t2.data[1]+alpha*t1.data[1]; return t2)
#     perm==[1:N1] && return (beta==0 ? scale!(copy!(t2,t1),alpha) : Base.LinAlg.axpy!(alpha,t1,scale!(t2,beta)))
#     beta==0 && (TensorOperations.tensorcopy_native!(t1.data,t2.data,perm);return scale!(t2,alpha))
#     TensorOperations.tensoradd_native!(alpha,t1.data,beta,t2.data,perm)
#     return t2
# end
# function tensortrace!(alpha::Number,A::Tensor,labelsA,beta::Number,C::Tensor,labelsC)
#     NA=numind(A)
#     NC=numind(C)
#     (length(labelsA)==NA && length(labelsC)==NC) || throw(LabelError("invalid label specification"))
#     NA==NC && return tensoradd!(alpha,A,labelsA,beta,C,labelsC) # nothing to trace
#
#     oindA=indexin(labelsC,labelsA)
#     clabels=unique(setdiff(labelsA,labelsC))
#     NA==NC+2*length(clabels) || throw(LabelError("invalid label specification"))
#
#     cindA1=Array(Int,length(clabels))
#     cindA2=Array(Int,length(clabels))
#     for i=1:length(clabels)
#         cindA1[i]=findfirst(labelsA,clabels[i])
#         cindA2[i]=findnext(labelsA,clabels[i],cindA1[i]+1)
#     end
#     isperm(vcat(oindA,cindA1,cindA2)) || throw(LabelError("invalid label specification"))
#
#     for i = 1:NC
#         space(A,oindA[i]) == space(C,i) || throw(SpaceError("space mismatch"))
#     end
#     for i = 1:div(NA-NC,2)
#         space(A,cindA1[i]) == dual(space(A,cindA2[i])) || throw(SpaceError("space mismatch"))
#     end
#
#     TensorOperations.tensortrace_native!(alpha,A.data,beta,C.data,oindA,cindA1,cindA2)
#     return C
# end
# function tensorcontract!(alpha::Number,A::Tensor,labelsA,conjA::Char,B::Tensor,labelsB,conjB::Char,beta::Number,C::Tensor,labelsC;method=:BLAS,buffer::TCBuffer=defaultcontractbuffer)
#     # Get properties of input arrays
#     NA=numind(A)
#     NB=numind(B)
#     NC=numind(C)
#
#     # Process labels, do some error checking and analyse problem structure
#     if NA!=length(labelsA) || NB!=length(labelsB) || NC!=length(labelsC)
#         throw(TensorOperations.LabelError("invalid label specification"))
#     end
#     ulabelsA=unique(labelsA)
#     ulabelsB=unique(labelsB)
#     ulabelsC=unique(labelsC)
#     if NA!=length(ulabelsA) || NB!=length(ulabelsB) || NC!=length(ulabelsC)
#         throw(TensorOperations.LabelError("tensorcontract requires unique label for every index of the tensor, handle inner contraction first with tensortrace"))
#     end
#
#     clabels=intersect(ulabelsA,ulabelsB)
#     numcontract=length(clabels)
#     olabelsA=intersect(ulabelsA,ulabelsC)
#     numopenA=length(olabelsA)
#     olabelsB=intersect(ulabelsB,ulabelsC)
#     numopenB=length(olabelsB)
#
#     if numcontract+numopenA!=NA || numcontract+numopenB!=NB || numopenA+numopenB!=NC
#         throw(LabelError("invalid contraction pattern"))
#     end
#
#     # Compute and contraction indices and check size compatibility
#     cindA=indexin(clabels,ulabelsA)
#     oindA=indexin(olabelsA,ulabelsA)
#     oindCA=indexin(olabelsA,ulabelsC)
#     cindB=indexin(clabels,ulabelsB)
#     oindB=indexin(olabelsB,ulabelsB)
#     oindCB=indexin(olabelsB,ulabelsC)
#
#     # check size compatibility
#     spaceA=space(A)
#     spaceB=space(B)
#     spaceC=space(C)
#
#     ospaceA=spaceA[oindA]
#     ospaceB=spaceB[oindB]
#
#     conjA=='C' || conjA=='N' || throw(ArgumentError("conjA should be 'C' or 'N'."))
#     conjB=='C' || conjB=='N' || throw(ArgumentError("conjB should be 'C' or 'N'."))
#
#     if conjA == conjB
#         for (i,j) in zip(cindA,cindB)
#             spaceA[i] == dual(spaceB[j]) || throw(SpaceError("incompatible index space for label $(ulabelsA[i])"))
#         end
#     else
#         for (i,j) in zip(cindA,cindB)
#             spaceA[i] == dual(conj(spaceB[j])) || throw(SpaceError("incompatible index space for label $(ulabelsA[i])"))
#         end
#     end
#     for (i,j) in zip(oindA,oindCA)
#         spaceC[j] == (conjA=='C' ? conj(spaceA[i]) : spaceA[i]) || throw(SpaceError("incompatible index space for label $(ulabelsA[i])"))
#     end
#     for (i,j) in zip(oindB,oindCB)
#         spaceC[j] == (conjB=='C' ? conj(spaceB[i]) : spaceB[i]) || throw(SpaceError("incompatible index space for label $(ulabelsB[i])"))
#     end
#
#     if method==:BLAS
#         TensorOperations.tensorcontract_blas!(alpha,A.data,conjA,B.data,conjB,beta,C.data,buffer,oindA,cindA,oindB,cindB,oindCA,oindCB)
#     elseif method==:native
#         if NA>=NB
#             TensorOperations.tensorcontract_native!(alpha,A.data,conjA,B.data,conjB,beta,C.data,oindA,cindA,oindB,cindB,oindCA,oindCB)
#         else
#             TensorOperations.tensorcontract_native!(alpha,B.data,conjB,A.data,conjA,beta,C.data,oindB,cindB,oindA,cindA,oindCB,oindCA)
#         end
#     else
#         throw(ArgumentError("method should be :BLAS or :native"))
#     end
#     return C
# end
#
# function tensorproduct!(alpha::Number,A::Tensor,labelsA,B::Tensor,labelsB,beta::Number,C::Tensor,labelsC)
#     # Get properties of input arrays
#     NA=numind(A)
#     NB=numind(B)
#     NC=numind(C)
#
#     # Process labels, do some error checking and analyse problem structure
#     if NA!=length(labelsA) || NB!=length(labelsB) || NC!=length(labelsC)
#         throw(TensorOperations.LabelError("invalid label specification"))
#     end
#     NC==NA+NB || throw(TensorOperations.LabelError("invalid label specification for tensor product"))
#
#     tensorcontract!(alpha,A,labelsA,'N',B,labelsB,'N',beta,C,labelsC;method=:native)
# end
#
