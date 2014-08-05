# base/abstracttensor.jl
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
numind{T<:TensorSpace}(::Type{T})=numind(super(T))

order=numind

# check whether a tensor describes a valid state in a Hilbert space encoded as a QuantumSystem object
# by comparing space(t) to the tensor product structure of the Hilbert space
Base.in(t::AbstractTensor,V::VectorSpace)= issubspace(space(t),V)

tensor(t::AbstractTensor)=t
tensor(t::AbstractTensor,P::TensorSpace)= (space(t) == P ? t : throw(SpaceError("tensor not in $P")))

Base.promote_rule{S,P,T1,T2,N}(::Type{AbstractTensor{S,P,T1,N}},::Type{AbstractTensor{S,P,T2,N}})=AbstractTensor{S,P,promote_type(T1,T2),N}
Base.promote_rule{S,P,T1,T2,N1,N2}(::Type{AbstractTensor{S,P,T1,N1}},::Type{AbstractTensor{S,P,T2,N2}})=AbstractTensor{S,P,promote_type(T1,T2)}
Base.promote_rule{S,P,T1,T2}(::Type{AbstractTensor{S,P,T1}},::Type{AbstractTensor{S,P,T2}})=AbstractTensor{S,P,promote_type(T1,T2)}

Base.convert{S,P,T,N}(::Type{AbstractTensor{S,P,T,N}},t::AbstractTensor{S,P,T,N})=t
Base.convert{S,P,T1,T2,N}(::Type{AbstractTensor{S,P,T1,N}},t::AbstractTensor{S,P,T2,N})=copy!(similar(t,T2),t)
Base.convert{S,P,T}(::Type{AbstractTensor{S,P,T}},t::AbstractTensor{S,P,T})=t
Base.convert{S,P,T1,T2}(::Type{AbstractTensor{S,P,T1}},t::AbstractTensor{S,P,T2})=copy!(similar(t,T2),t)

# Common functionality for AbstractTensor{S,P,T,N} and AbstractTensor{S,P,T,2}:
#-------------------------------------------------------------------------------
*(t::AbstractTensor,a::Number)=scale(t,a)
*(a::Number,t::AbstractTensor)=scale(t,a)
/(t::AbstractTensor,a::Number)=scale(t,one(a)/a)
\(a::Number,t::AbstractTensor)=scale(t,one(a)/a)
Base.scale(a::Number,t::AbstractTensor)=scale(t,a)
function Base.scale(t::AbstractTensor,a::Number)
    tnew=similar(t,promote_type(eltype(t),typeof(a)))
    copy!(tnew,t)
    scale!(tnew,a)
end

# convenience definition which works for vectors and matrices but also sometimes useful in general case
*{S}(t1::AbstractTensor{S},t2::AbstractTensor{S})=tensorcontract(t1,vcat(1:numind(t1)-1,0),t2,vcat(0,-(1:numind(t2)-1)))
⊗{S}(t1::AbstractTensor{S},t2::AbstractTensor{S})=tensorproduct(t1,1:numind(t1),t2,numind(t1)+(1:numind(t2)))

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
    tensoradd!(one(T),B,labelsB,one(T),C,outputlabels)
    return C
end
function tensortrace(A::AbstractTensor,labelsA,outputlabels)
    T=eltype(A)
    spaceA=space(A)
    spaceC=spaceA[indexin(outputlabels,labelsA)]
    C=similar(A,spaceC)
    fill!(C,zero(T))
    tensortrace!(one(T),A,labelsA,zero(T),C,outputlabels)
    return C
end
function tensortrace(A::AbstractTensor,labelsA) # there is no one-line method to compute the default outputlabels
    ulabelsA=unique(labelsA)
    labelsC=similar(labelsA,0)
    sizehint(labelsC,length(labelsA))
    for j=1:length(ulabelsA)
        ind=findfirst(labelsA,ulabelsA[j])
        if findnext(labelsA,ulabelsA[j],ind+1)==0
            push!(labelsC,ulabelsA[j])
        end
    end
    return tensortrace(A,labelsA,labelsC)
end
function tensorcontract{S}(A::AbstractTensor{S},labelsA,B::AbstractTensor{S},labelsB,outputlabels=symdiff(labelsA,labelsB);method::Symbol=:BLAS)
    spaceA=space(A)
    spaceB=space(B)

    spaceC=(spaceA ⊗ spaceB)[indexin(outputlabels,vcat(labelsA,labelsB))]
    T=promote_type(eltype(A),eltype(B))
    C=similar(A,T,spaceC)
    fill!(C,zero(T))
    tensorcontract!(one(T),A,labelsA,'N',B,labelsB,'N',zero(T),C,outputlabels;method=method)
    return C
end
function tensorproduct{S}(A::AbstractTensor{S},labelsA,B::AbstractTensor{S},labelsB,outputlabels=symdiff(labelsA,labelsB))
    spaceA=space(A)
    spaceB=space(B)

    spaceC=(spaceA ⊗ spaceB)[indexin(outputlabels,vcat(labelsA,labelsB))]
    T=promote_type(eltype(A),eltype(B))
    C=similar(A,T,spaceC)
    fill!(C,zero(T))
    tensorproduct!(one(T),A,labelsA,B,labelsB,zero(T),C,outputlabels)
    return C
end

# general tensor factorizations: permute to correct order and pass to in place methods
function Base.svd(t::AbstractTensor,leftind,rightind=setdiff(1:numind(t),leftind))
    # Perform singular value decomposition corresponding to bipartion of the
    # tensor indices into leftind and rightind.
    N=numind(t)
    p=vcat(leftind,rightind)
    (isperm(p) && length(P)==N) || throw(IndexError("Not a valid bipartation of the tensor indices"))
    newt=tensorcopy(t,1:N,p)
    return svd!(t,length(leftind))
end

function svdtrunc(t::AbstractTensor,leftind=codomainind(t),rightind=setdiff(1:numind(t),leftind);kwargs...)
    # Truncate tensor rank corresponding to bipartition into leftind and
    # rightind, based on singular value decomposition. Truncation parameters
    # are given as  keyword arguments: trunctol should always be one of the
    # possible arguments for specifying truncation, but truncdim can be
    # replaced with different parameters for other types of tensors.
    N=numind(t)
    p=vcat(leftind,rightind)
    (isperm(p) && length(P)==N) || throw(IndexError("Not a valid bipartation of the tensor indices"))
    newt=tensorcopy(t,1:N,p)
    return svdtrunc!(t,length(leftind);kwargs...)
end

function leftorth(t::AbstractTensor,leftind,rightind=setdiff(1:numind(t),leftind))
    # Create orthogonal basis U for left indices, and remainder R for right
    # indices. Decomposition should be unique, such that it always returns the
    # same result for the same input tensor t. QR is fastest but only unique
    # after correcting for phases.
    N=numind(t)
    p=vcat(leftind,rightind)
    (isperm(p) && length(P)==N) || throw(IndexError("Not a valid bipartation of the tensor indices"))
    newt=tensorcopy(t,1:N,p)
    return leftorth!(t,length(leftind))
end

function rightorth(t::AbstractTensor,leftind,rightind=setdiff(1:numind(t),leftind))
    # Create orthogonal basis U for left indices, and remainder R for right
    # indices. Decomposition should be unique, such that it always returns the
    # same result for the same input tensor t. QR is fastest but only unique
    # after correcting for phases.
    N=numind(t)
    p=vcat(leftind,rightind)
    (isperm(p) && length(P)==N) || throw(IndexError("Not a valid bipartation of the tensor indices"))
    newt=tensorcopy(t,1:N,p)
    return rightorth!(t,length(leftind))
end
