# abstracttensor.jl
#
# Defines AbstractTensor, an abstract tensor to start a type hierarchy for
# representing tensors and extending/uniformizing Julia's built-in
# functionality for working with multilinear objects. A tensor is interpreted
# as a multilinear map whose indices are associated to vector spaces corresponding
# IndexSpace objects.

# Abstract Tensor type
#----------------------
abstract AbstractTensor{S<:IndexSpace,P<:TensorSpace,T,N}
# Any implementation of AbstractTensor should have method definitions for the
# same set of methods which are defined for the dense implementation Tensor
# defined in tensor.jl.

# tensor characteristics
spacetype{S}(::AbstractTensor{S})=S
spacetype{S<:IndexSpace}(::Type{AbstractTensor{S}})=S
spacetype{S<:IndexSpace,P<:TensorSpace}(::Type{AbstractTensor{S,P}})=S
spacetype{S<:IndexSpace,P<:TensorSpace,T}(::Type{AbstractTensor{S,P,T}})=S
spacetype{S<:IndexSpace,P<:TensorSpace,T,N}(::Type{AbstractTensor{S,P,T,N}})=S
spacetype{TT<:AbstractTensor}(::Type{TT})=spacetype(super(TT))

tensortype{S,P}(::AbstractTensor{S,P})=P
tensortype{S<:IndexSpace,P<:TensorSpace}(::Type{AbstractTensor{S,P}})=P
tensortype{S<:IndexSpace,P<:TensorSpace,T}(::Type{AbstractTensor{S,P,T}})=P
tensortype{S<:IndexSpace,P<:TensorSpace,T,N}(::Type{AbstractTensor{S,P,T,N}})=P
tensortype{TT<:AbstractTensor}(::Type{TT})=spacetype(super(TT))

Base.eltype{S,P,T}(::AbstractTensor{S,P,T}) = T
Base.eltype{S<:IndexSpace,P<:TensorSpace,T}(::Type{AbstractTensor{S,P,T}})=T
Base.eltype{S<:IndexSpace,P<:TensorSpace,T,N}(::Type{AbstractTensor{S,P,T,N}})=T
Base.eltype{TT<:AbstractTensor}(::Type{TT}) = eltype(super(TT))

numind{S,P,T,N}(::AbstractTensor{S,P,T,N})=N
numind{S,P<:TensorSpace,T,N}(::Type{AbstractTensor{S,P,T,N}})=N
numind{TT<:AbstractTensor}(::Type{TT})=numind(super(TT))

dim(t::AbstractTensor)=dim(space(t))
sectors(t::AbstractTensor)=sectors(space(t))

order=numind

# check whether a tensor describes a valid state in a Hilbert space encoded as a QuantumSystem object
# by comparing space(t) to the tensor product structure of the Hilbert space
Base.in(t::AbstractTensor,V::VectorSpace)=issubspace(space(t),V)

tensor(t::AbstractTensor)=t
tensor(t::AbstractTensor,P::TensorSpace)= (space(t) == P ? t : throw(SpaceError("tensor not in $P")))

Base.promote_rule{S,P,T1,T2,N}(::Type{AbstractTensor{S,P,T1,N}},::Type{AbstractTensor{S,P,T2,N}})=AbstractTensor{S,P,promote_type(T1,T2),N}
Base.promote_rule{S,P,T1,T2,N1,N2}(::Type{AbstractTensor{S,P,T1,N1}},::Type{AbstractTensor{S,P,T2,N2}})=AbstractTensor{S,P,promote_type(T1,T2)}
Base.promote_rule{S,P,T1,T2}(::Type{AbstractTensor{S,P,T1}},::Type{AbstractTensor{S,P,T2}})=AbstractTensor{S,P,promote_type(T1,T2)}

Base.convert{S,P,T,N}(::Type{AbstractTensor{S,P,T,N}},t::AbstractTensor{S,P,T,N})=t
Base.convert{S,P,T1,T2,N}(::Type{AbstractTensor{S,P,T1,N}},t::AbstractTensor{S,P,T2,N})=copy!(similar(t,T1),t)
Base.convert{S,P,T}(::Type{AbstractTensor{S,P,T}},t::AbstractTensor{S,P,T})=t
Base.convert{S,P,T1,T2}(::Type{AbstractTensor{S,P,T1}},t::AbstractTensor{S,P,T2})=copy!(similar(t,T1),t)

# Basic algebra
#---------------
Base.copy(t::AbstractTensor)=Base.copy!(similar(t),t)

*(t::AbstractTensor,a::Number)=scale(t,a)
*(a::Number,t::AbstractTensor)=scale(t,a)
/(t::AbstractTensor,a::Number)=scale(t,one(a)/a)
\(a::Number,t::AbstractTensor)=scale(t,one(a)/a)
Base.scale(a::Number,t::AbstractTensor)=scale(t,a)
function Base.scale(t::AbstractTensor,a::Number)
    tnew=similar(t,promote_type(eltype(t),typeof(a)))
    scale!(tnew,t,a)
end
Base.scale!(t::AbstractTensor,a::Number)=scale!(t,t,a)
Base.scale!(a::Number,t::AbstractTensor)=scale!(t,a,t)

Base.conj(t::AbstractTensor)=Base.conj!(similar(t,conj(space(t))),t)
Base.transpose(t::AbstractTensor)=Base.transpose!(similar(t,space(t).'),t)
Base.ctranspose(t::AbstractTensor)=Base.ctranspose!(similar(t,space(t)'),t)

# Tensor operations
#-------------------
# convenience definition which works for vectors and matrices but also sometimes useful in general case
*{S,P,T1,T2,N1,N2}(t1::AbstractTensor{S,P,T1,N1},t2::AbstractTensor{S,P,T2,N2})=(t3=similar(t1,promote_type(T1,T2),space(t1)[1:N1-1] ⊗ space(t2)[2:N2]);tensorcontract!(1,t1,vcat(1:N1-1,0),'N',t2,vcat(0,numind(t1)-1+(1:N2-1)),'N',0,t3,1:(N1+N2-2)))
Base.At_mul_B{S,P,T1,T2,N1,N2}(t1::AbstractTensor{S,P,T1,N1},t2::AbstractTensor{S,P,T2,N2})=(t3=similar(t1,promote_type(T1,T2),space(t1)[2:N1].' ⊗ space(t2)[2:N2]);tensorcontract!(1,t1,vcat(0,reverse(1:N1-1)),'N',t2,vcat(0,N1-1+(1:N2-1)),'N',0,t3,1:(numind(t1)+numind(t2)-2)))
Base.Ac_mul_B{S,P,T1,T2,N1,N2}(t1::AbstractTensor{S,P,T1,N1},t2::AbstractTensor{S,P,T2,N2})=(t3=similar(t1,promote_type(T1,T2),space(t1)[2:N1]' ⊗ space(t2)[2:N2]);tensorcontract!(1,t1,vcat(0,reverse(1:N1-1)),'C',t2,vcat(0,N1-1+(1:N2-1)),'N',0,t3,1:(numind(t1)+numind(t2)-2)))

⊗{S,P}(t1::AbstractTensor{S,P},t2::AbstractTensor{S,P})=tensorproduct(t1,1:numind(t1),t2,numind(t1)+(1:numind(t2)))
Base.trace{S,P,T}(t::AbstractTensor{S,P,T,2})=scalar(tensortrace(t,[1,1],[]))

# general tensor operations: no error checking, pass to mutating methods
function tensorcopy(A::AbstractTensor,labelsA,outputlabels=labelsA)
    spaceA=space(A)
    spaceC=spaceA[indexin(outputlabels,labelsA)]
    C=similar(A,spaceC)
    tensorcopy!(A,labelsA,C,outputlabels)
    return C
end
function tensoradd{S,P,TA,TB,N}(A::AbstractTensor{S,P,TA,N},labelsA,B::AbstractTensor{S,P,TB,N},labelsB,outputlabels=labelsA)
    spaceA=space(A)
    spaceC=spaceA[indexin(outputlabels,labelsA)]
    T=promote_type(TA,TB)
    C=similar(A,T,spaceC)
    tensorcopy!(A,labelsA,C,outputlabels)
    tensoradd!(1,B,labelsB,1,C,outputlabels)
    return C
end
function tensortrace(A::AbstractTensor,labelsA,outputlabels)
    T=eltype(A)
    spaceA=space(A)
    spaceC=spaceA[indexin(outputlabels,labelsA)]
    C=similar(A,spaceC)
    tensortrace!(1,A,labelsA,0,C,outputlabels)
    return C
end
function tensortrace(A::AbstractTensor,labelsA) # there is no one-line method to compute the default outputlabels
    ulabelsA=unique(labelsA)
    labelsC=similar(labelsA,0)
    sizehint(labelsC,length(ulabelsA))
    for j=1:length(ulabelsA)
        ind=findfirst(labelsA,ulabelsA[j])
        if findnext(labelsA,ulabelsA[j],ind+1)==0
            push!(labelsC,ulabelsA[j])
        end
    end
    return tensortrace(A,labelsA,labelsC)
end
function tensorcontract{S}(A::AbstractTensor{S},labelsA,conjA,B::AbstractTensor{S},labelsB,conjB,outputlabels=symdiff(labelsA,labelsB);method::Symbol=:BLAS,buffer::TCBuffer=defaultcontractbuffer)
    spaceA=space(A)
    spaceB=space(B)
    spaceC=(spaceA ⊗ spaceB)[indexin(outputlabels,vcat(labelsA,labelsB))]
    T=promote_type(eltype(A),eltype(B))
    C=similar(A,T,spaceC)
    tensorcontract!(1,A,labelsA,conjA,B,labelsB,conjB,0,C,outputlabels;method=method,buffer=buffer)
    return C
end
tensorcontract{S}(A::AbstractTensor{S},labelsA,B::AbstractTensor{S},labelsB,outputlabels=symdiff(labelsA,labelsB);
    method::Symbol=:BLAS,buffer::TCBuffer=defaultcontractbuffer)=tensorcontract(A,labelsA,'N',B,labelsB,'N',outputlabels;method=method,buffer=buffer)

function tensorproduct{S}(A::AbstractTensor{S},labelsA,B::AbstractTensor{S},labelsB,outputlabels=vcat(labelsA,labelsB))
    spaceA=space(A)
    spaceB=space(B)

    spaceC=(spaceA ⊗ spaceB)[indexin(outputlabels,vcat(labelsA,labelsB))]
    T=promote_type(eltype(A),eltype(B))
    C=similar(A,T,spaceC)
    tensorproduct!(1,A,labelsA,B,labelsB,0,C,outputlabels)
    return C
end

Base.eye{S<:ElementarySpace,P<:TensorSpace,T}(t::AbstractTensor{S,P,T},V::S)=eye(T,P,V)

# Factorization
#---------------
Base.svd{S,P,T}(t::AbstractTensor{S,P,T,2})=svd(t,1,2)
leftorth{S,P,T}(t::AbstractTensor{S,P,T,2})=leftorth(t,1,2)
rightorth{S,P,T}(t::AbstractTensor{S,P,T,2})=rightorth(t,1,2)

# general tensor factorizations: permute to correct order and pass to in place methods
Base.svd(t::AbstractTensor,leftind,truncation::TruncationScheme=notrunc())=svd(t,leftind,setdiff(1:numind(t),leftind),truncation)
function Base.svd(t::AbstractTensor,leftind,rightind,truncation::TruncationScheme=notrunc())
    # Perform singular value decomposition corresponding to bipartion of the
    # tensor indices into leftind and rightind.
    N=numind(t)
    p=vcat(leftind,rightind)
    (isperm(p) && length(p)==N) || throw(IndexError("Not a valid bipartation of the tensor indices"))
    newt=tensorcopy(t,1:N,p)
    return svd!(newt,length(leftind),truncation)
end

leftorth(t::AbstractTensor,leftind,truncation::TruncationScheme=notrunc())=leftorth(t,leftind,setdiff(1:numind(t),leftind),truncation)
function leftorth(t::AbstractTensor,leftind,rightind=setdiff(1:numind(t),leftind),truncation::TruncationScheme=notrunc())
    # Create orthogonal basis U for left indices, and remainder R for right
    # indices. Decomposition should be unique, such that it always returns the
    # same result for the same input tensor t. QR is fastest but only unique
    # after correcting for phases.
    N=numind(t)
    p=vcat(leftind,rightind)
    (isperm(p) && length(p)==N) || throw(IndexError("Not a valid bipartation of the tensor indices"))
    newt=tensorcopy(t,1:N,p)
    return leftorth!(newt,length(leftind),truncation)
end

rightorth(t::AbstractTensor,leftind,truncation::TruncationScheme=notrunc())=rightorth(t,leftind,setdiff(1:numind(t),leftind),truncation)
function rightorth(t::AbstractTensor,leftind,rightind=setdiff(1:numind(t),leftind),truncation::TruncationScheme=notrunc())
    # Create orthogonal basis U for left indices, and remainder R for right
    # indices. Decomposition should be unique, such that it always returns the
    # same result for the same input tensor t. QR is fastest but only unique
    # after correcting for phases.
    N=numind(t)
    p=vcat(leftind,rightind)
    (isperm(p) && length(p)==N) || throw(IndexError("Not a valid bipartation of the tensor indices"))
    newt=tensorcopy(t,1:N,p)
    return rightorth!(newt,length(leftind),truncation)
end

svd!(t::AbstractTensor,n::Int) = svd!(t,n,notrunc())
leftorth!(t::AbstractTensor,n::Int) = leftorth!(t,n,notrunc())
rightorth!(t::AbstractTensor,n::Int) = rightorth!(t,n,notrunc())
