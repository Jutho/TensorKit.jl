# invarianttensor.jl
#
# ITensor provides a dense implementation of an AbstractTensor type living
# in an InvariantSpace, i.e. the invariant subspace of the tensor product
# of its index spaces.
# Currenlty only defined for abelian sectors.

#+++++++++++++++++++++++
# InvariantTensor type:
#+++++++++++++++++++++++
# Type definition and constructors:
#-----------------------------------
immutable InvariantTensor{G<:Abelian,S<:AbelianSpace,T,N} <: AbstractTensor{S,InvariantSpace,T,N}
    data::Vector{T}
    space::InvariantSpace{G,S,N}
    _datasectors::Dict{NTuple{N,G},Array{T,N}}
    function Tensor(data::Array{T},space::InvariantSpace{G,S,N})
        if length(data)!=dim(space)
            throw(DimensionMismatch("data not of right size"))
        end
        if promote_type(T,eltype(S)) != eltype(S)
            error("For a tensor in $(space), the entries cannot be of type $(T)")
        end
        _datasectors=Dict{NTuple{N,G},Array{T,N}}()
        ind=1
        for s in sectors(space)
            dims=ntuple(N,n->dim(space[n],s[n]))
            _datasectors[s]=pointer_to_array(pointer(data,ind),dims)
            ind+=prod(dims)
        end
        return new(reshape(data,map(dim,space)),space)
    end
end

# Show method:
#-------------
function Base.show{S,T,N}(io::IO,t::InvariantTensor{G,S,T,N})
    print(io," InvariantTensor ∈ $T")
    print(io,"[")
    for n=1:N
        n==1 || print(io, " ⊗ ")
        show(io,space(t,n))
    end
    println(io,"]:")
    Base.showarray(io,t.data;header=false)
end

# Basic methods for characterising a tensor:
#--------------------------------------------
space(t::Tensor,ind::Int)=t.space[ind]
space(t::Tensor)=t.space

sectors(t::Tensor)=sectors(space(t))

# General constructors
#---------------------
# with data
tensor{G,S,T,N}(data::Array{T},P::InvariantSpace{G,S,N})=InvariantTensor{G,S,T,N}(data,P)

# without data
tensor{T}(::Type{T},P::InvariantSpace)=tensor(Array(T,dim(P)),P)
tensor(P::InvariantSpace)=tensor(Float64,V)

Base.similar{G,S,T,N}(t::InvariantTensor{G,S},::Type{T},P::InvariantSpace{G,S,N}=space(t))=tensor(similar(t.data,T,dim(P)),P)
Base.similar{G,S,T,N}(t::InvariantTensor{G,S},::Type{T},P::ProductSpace{S,N}=space(t))=tensor(t,T,invariant(P))

Base.similar{S,T}(t::Tensor{S},::Type{T},V::S)=similar(t,T,⊗(V))
Base.similar{S,N}(t::Tensor{S},P::ProductSpace{S,N}=space(t))=similar(t,eltype(t),P)
Base.similar{S}(t::Tensor{S},V::S)=similar(t,eltype(t),V)

Base.zero(t::Tensor)=tensor(zero(t.data),space(t))

Base.zeros{T}(::Type{T},P::ProductSpace)=tensor(zeros(T,dim(P)),P)
Base.zeros{T}(::Type{T},V::IndexSpace)=zeros(T,⊗(V))
Base.zeros(V::Union(ProductSpace,IndexSpace))=zeros(Float64,V)

Base.rand{T}(::Type{T},P::ProductSpace)=tensor(rand(T,dim(P)),P)
Base.rand{T}(::Type{T},V::IndexSpace)=rand(T,⊗(V))
Base.rand(V::Union(ProductSpace,IndexSpace))=rand(Float64,V)

Base.eye{T}(::Type{T},P::ProductSpace)=tensor(eye(T,dim(P)),P⊗dual(P))
Base.eye{T}(::Type{T},V::IndexSpace)=eye(T,⊗(V))
Base.eye(V::Union(ProductSpace,IndexSpace))=eye(Float64,V)

# tensors from concatenation
function tensorcat{S}(catind, X::Tensor{S}...)
    catind = collect(catind)
    isempty(catind) && error("catind should not be empty")
    # length(unique(catdims)) != length(catdims) && error("every dimension should appear only once")

    nargs = length(X)
    numindX = map(numind, X)

    all(n->(n == numindX[1]), numindX) || throw(SpaceError("all tensors should have the same number of indices for concatenation"))

    numindC = numindX[1]
    ncatind = setdiff(1:numindC,catind)
    spaceCvec = Array(S, numindC)
    for n = 1:numindC
        spaceCvec[n] = space(X[1],n)
    end
    for i = 2:nargs
        for n in catind
            spaceCvec[n] = directsum(spaceCvec[n], space(X[i],n))
        end
        for n in ncatind
            spaceCvec[n] == space(X[i],n) || throw(SpaceError("space mismatch for index $n"))
        end
    end
    spaceC = ⊗(spaceCvec...)
    typeC = mapreduce(eltype, promote_type, X)
    dataC = zeros(typeC, map(dim,spaceC))

    offset = zeros(Int,numindC)
    for i=1:nargs
        currentdims=ntuple(numindC,n->dim(space(X[i],n)))
        currentrange=[offset[n]+(1:currentdims[n]) for n=1:numindC]
        dataC[currentrange...] = X[i].data
        for n in catind
            offset[n]+=currentdims[n]
        end
    end
    return tensor(dataC,spaceC)
end

# Copy and fill tensors:
#------------------------
function Base.copy!(tdest::Tensor,tsource::Tensor)
    # Copies data of tensor tsource to tensor tdest if compatible
    space(tdest)==space(tsource) || throw(SpaceError("tensor spaces don't match"))
    copy!(tdest.data,tsource.data)
    return tdest
end
Base.fill!{S,T}(tdest::Tensor{S,T},value::Number)=fill!(tdest.data,convert(T,value))

# Vectorization:
#----------------
Base.vec(t::Tensor)=vec(t.data)
# Convert the non-trivial degrees of freedom in a tensor to a vector to be passed to eigensolvers etc.

# Conversion and promotion:
#---------------------------
Base.full(t::Tensor)=t.data

Base.promote_rule{S,T1,T2,N}(::Type{Tensor{S,T1,N}},::Type{Tensor{S,T2,N}})=Tensor{S,promote_type(T1,T2),N}
Base.promote_rule{S,T1,T2,N1,N2}(::Type{Tensor{S,T1,N1}},::Type{Tensor{S,T2,N2}})=Tensor{S,promote_type(T1,T2)}
Base.promote_rule{S,T1,T2}(::Type{Tensor{S,T1}},::Type{Tensor{S,T2}})=Tensor{S,promote_type(T1,T2)}

Base.promote_rule{S,T1,T2,N}(::Type{AbstractTensor{S,ProductSpace,T1,N}},::Type{Tensor{S,T2,N}})=AbstractTensor{S,ProductSpace,promote_type(T1,T2),N}
Base.promote_rule{S,T1,T2,N1,N2}(::Type{AbstractTensor{S,ProductSpace,T1,N1}},::Type{Tensor{S,T2,N2}})=AbstractTensor{S,ProductSpace,promote_type(T1,T2)}
Base.promote_rule{S,T1,T2}(::Type{AbstractTensor{S,ProductSpace,T1}},::Type{Tensor{S,T2}})=AbstractTensor{S,ProductSpace,promote_type(T1,T2)}

Base.convert{S,T,N}(::Type{Tensor{S,T,N}},t::Tensor{S,T,N})=t
Base.convert{S,T1,T2,N}(::Type{Tensor{S,T1,N}},t::Tensor{S,T2,N})=copy!(similar(t,T1),t)
Base.convert{S,T}(::Type{Tensor{S,T}},t::Tensor{S,T})=t
Base.convert{S,T1,T2}(::Type{Tensor{S,T1}},t::Tensor{S,T2})=copy!(similar(t,T1),t)

Base.float{S,T<:FloatingPoint}(t::Tensor{S,T})=t
Base.float(t::Tensor)=tensor(float(t.data),space(t))

Base.real{S,T<:Real}(t::Tensor{S,T})=t
Base.real(t::Tensor)=tensor(real(t.data),space(t))
Base.complex{S,T<:Complex}(t::Tensor{S,T})=t
Base.complex(t::Tensor)=tensor(complex(t.data),space(t))

for (f,T) in ((:float32,    Float32),
              (:float64,    Float64),
              (:complex64,  Complex64),
              (:complex128, Complex128))
    @eval (Base.$f){S}(t::Tensor{S,$T}) = t
    @eval (Base.$f)(t::Tensor) = tensor(($f)(t.data),space(t))
end

# Basic algebra:
#----------------
function Base.conj!(t1::Tensor,t2::Tensor)
    space(t1)==conj(space(t2)) || throw(SpaceError())
    copy!(t1.data,t2.data)
    conj!(t1.data)
    return t1
end

# transpose inverts order of indices, compatible with graphical notation
function Base.transpose!(tdest::Tensor,tsource::Tensor)
    space(tdest)==space(tsource).' || throw(SpaceError())
    N=numind(tsource)
    TensorOperations.tensorcopy!(tsource.data,1:N,tdest.data,reverse(1:N))
    return tdest
end
function Base.ctranspose!(tdest::Tensor,tsource::Tensor)
    space(tdest)==space(tsource)' || throw(SpaceError())
    N=numind(tsource)
    TensorOperations.tensorcopy!(tsource.data,1:N,tdest.data,reverse(1:N))
    conj!(tdest.data)
    return tdest
end

Base.scale!(t::Tensor,a::Number)=(scale!(t.data,a);return t)
Base.scale!(a::Number,t::Tensor)=(scale!(a,t.data);return t)
Base.scale!{S,T,N}(t1::Tensor{S,T,N},t2::Tensor{S,T,N},a::Number)=(space(t1)==space(t2) ? scale!(t1.data,t2.data,a) : throw(SpaceError());return t1)
Base.scale!{S,T,N}(t1::Tensor{S,T,N},a::Number,t2::Tensor{S,T,N})=(space(t1)==space(t2) ? scale!(t1.data,a,t2.data) : throw(SpaceError());return t1)

Base.LinAlg.axpy!(a::Number,x::Tensor,y::Tensor)=(space(x)==space(y) ? Base.LinAlg.axpy!(a,x.data,y.data) : throw(SpaceError()); return y)

-(t::Tensor)=tensor(-t.data,space(t))
+(t1::Tensor,t2::Tensor)= space(t1)==space(t2) ? tensor(t1.data+t2.data,space(t1)) : throw(SpaceError())
-(t1::Tensor,t2::Tensor)= space(t1)==space(t2) ? tensor(t1.data-t2.data,space(t1)) : throw(SpaceError())

# Scalar product and norm: only valid for EuclideanSpace
Base.dot{S<:EuclideanSpace}(t1::Tensor{S},t2::Tensor{S})= (space(t1)==space(t2) ? dot(vec(t1),vec(t2)) : throw(SpaceError()))
Base.vecnorm{S<:EuclideanSpace}(t::Tensor{S})=vecnorm(t.data) # frobenius norm

# Indexing
#----------
# linear indexing using ProductBasisVector
Base.getindex{S,T,N}(t::Tensor{S,T,N},b::ProductBasisVector{N,S})=getindex(t.data,Base.to_index(b))
Base.setindex!{S,T,N}(t::Tensor{S,T,N},value,b::ProductBasisVector{N,S})=setindex!(t.data,value,Base.to_index(b))

# Tensor Operations
#-------------------
scalar{S,T}(t::Tensor{S,T,0})=TensorOperations.scalar(t.data)

function tensorcopy!(t1::Tensor,labels1,t2::Tensor,labels2)
    # Replaces tensor t2 with t1
    N1=numind(t1)
    perm=indexin(labels2,labels1)

    length(perm) == N1 || throw(TensorOperations.LabelError("invalid label specification"))
    isperm(perm) || throw(TensorOperations.LabelError("invalid label specification"))
    for i = 1:N1
        space(t1,i) == space(t2,perm[i]) || throw(SpaceError())
    end
    N1==0 && (t2.data[1]=t1.data[1]; return t2)
    perm==[1:N1] && return copy!(t2,t1)
    TensorOperations.tensorcopy_native!(t1.data,t2.data,perm)
    return t2
end
function tensoradd!(alpha::Number,t1::Tensor,labels1,beta::Number,t2::Tensor,labels2)
    # Replaces tensor t2 with beta*t2+alpha*t1
    N1=numind(t1)
    perm=indexin(labels2,labels1)

    length(perm) == N1 || throw(TensorOperations.LabelError("invalid label specification"))
    isperm(perm) || throw(TensorOperations.LabelError("invalid label specification"))
    for i = 1:N1
        space(t1,perm[i]) == space(t2,i) || throw(SpaceError("incompatible index spaces of tensors"))
    end
    N1==0 && (t2.data[1]=beta*t2.data[1]+alpha*t1.data[1]; return t2)
    perm==[1:N1] && return (beta==0 ? scale!(copy!(t2,t1),alpha) : Base.LinAlg.axpy!(alpha,t1,scale!(t2,beta)))
    beta==0 && (TensorOperations.tensorcopy_native!(t1.data,t2.data,perm);return scale!(t2,alpha))
    TensorOperations.tensoradd_native!(alpha,t1.data,beta,t2.data,perm)
    return t2
end
function tensortrace!(alpha::Number,A::Tensor,labelsA,beta::Number,C::Tensor,labelsC)
    NA=numind(A)
    NC=numind(C)
    (length(labelsA)==NA && length(labelsC)==NC) || throw(LabelError("invalid label specification"))
    NA==NC && return tensoradd!(alpha,A,labelsA,beta,C,labelsC) # nothing to trace

    oindA=indexin(labelsC,labelsA)
    clabels=unique(setdiff(labelsA,labelsC))
    NA==NC+2*length(clabels) || throw(LabelError("invalid label specification"))

    cindA1=Array(Int,length(clabels))
    cindA2=Array(Int,length(clabels))
    for i=1:length(clabels)
        cindA1[i]=findfirst(labelsA,clabels[i])
        cindA2[i]=findnext(labelsA,clabels[i],cindA1[i]+1)
    end
    isperm(vcat(oindA,cindA1,cindA2)) || throw(LabelError("invalid label specification"))

    for i = 1:NC
        space(A,oindA[i]) == space(C,i) || throw(SpaceError("space mismatch"))
    end
    for i = 1:div(NA-NC,2)
        space(A,cindA1[i]) == dual(space(A,cindA2[i])) || throw(SpaceError("space mismatch"))
    end

    TensorOperations.tensortrace_native!(alpha,A.data,beta,C.data,oindA,cindA1,cindA2)
    return C
end
function tensorcontract!(alpha::Number,A::Tensor,labelsA,conjA::Char,B::Tensor,labelsB,conjB::Char,beta::Number,C::Tensor,labelsC;method=:BLAS,buffer::TCBuffer=defaultcontractbuffer)
    # Get properties of input arrays
    NA=numind(A)
    NB=numind(B)
    NC=numind(C)

    # Process labels, do some error checking and analyse problem structure
    if NA!=length(labelsA) || NB!=length(labelsB) || NC!=length(labelsC)
        throw(TensorOperations.LabelError("invalid label specification"))
    end
    ulabelsA=unique(labelsA)
    ulabelsB=unique(labelsB)
    ulabelsC=unique(labelsC)
    if NA!=length(ulabelsA) || NB!=length(ulabelsB) || NC!=length(ulabelsC)
        throw(TensorOperations.LabelError("tensorcontract requires unique label for every index of the tensor, handle inner contraction first with tensortrace"))
    end

    clabels=intersect(ulabelsA,ulabelsB)
    numcontract=length(clabels)
    olabelsA=intersect(ulabelsA,ulabelsC)
    numopenA=length(olabelsA)
    olabelsB=intersect(ulabelsB,ulabelsC)
    numopenB=length(olabelsB)

    if numcontract+numopenA!=NA || numcontract+numopenB!=NB || numopenA+numopenB!=NC
        throw(LabelError("invalid contraction pattern"))
    end

    # Compute and contraction indices and check size compatibility
    cindA=indexin(clabels,ulabelsA)
    oindA=indexin(olabelsA,ulabelsA)
    oindCA=indexin(olabelsA,ulabelsC)
    cindB=indexin(clabels,ulabelsB)
    oindB=indexin(olabelsB,ulabelsB)
    oindCB=indexin(olabelsB,ulabelsC)

    # check size compatibility
    spaceA=space(A)
    spaceB=space(B)
    spaceC=space(C)

    cspaceA=spaceA[cindA]
    cspaceB=spaceB[cindB]
    ospaceA=spaceA[oindA]
    ospaceB=spaceB[oindB]

    conjA=='C' || conjA=='N' || throw(ArgumentError("conjA should be 'C' or 'N'."))
    conjB=='C' || conjB=='N' || throw(ArgumentError("conjB should be 'C' or 'N'."))

    if conjA == conjB
        for i=1:numcontract
            cspaceA[i] == dual(cspaceB[i]) || throw(SpaceError("incompatible index space for label $(clabels[i])"))
        end
    else
        for i=1:numcontract
            cspaceA[i] == dual(conj(cspaceB[i])) || throw(SpaceError("incompatible index space for label $(clabels[i])"))
        end
    end
    for i=1:numopenA
        spaceC[oindCA[i]] == (conjA=='C' ? conj(ospaceA[i]) : ospaceA[i]) || throw(SpaceError("incompatible index space for label $(olabelsA[i])"))
    end
    for i=1:numopenB
        spaceC[oindCB[i]] == (conjB=='C' ? conj(ospaceB[i]) : ospaceB[i]) || throw(SpaceError("incompatible index space for label $(olabelsB[i])"))
    end

    method==:BLAS && TensorOperations.tensorcontract_blas!(alpha,A.data,conjA,B.data,conjB,beta,C.data,buffer,oindA,cindA,oindB,cindB,oindCA,oindCB)
    method==:native && return NA>=NB ? 
        TensorOperations.tensorcontract_native!(alpha,A.data,conjA,B.data,conjB,beta,C.data,oindA,cindA,oindB,cindB,oindCA,oindCB) :
        TensorOperations.tensorcontract_native!(alpha,B.data,conjB,A.data,conjA,beta,C.data,oindB,cindB,oindA,cindA,oindCB,oindCA)
    return C
end

function tensorproduct!(alpha::Number,A::Tensor,labelsA,B::Tensor,labelsB,beta::Number,C::Tensor,labelsC)
    # Get properties of input arrays
    NA=numind(A)
    NB=numind(B)
    NC=numind(C)

    # Process labels, do some error checking and analyse problem structure
    if NA!=length(labelsA) || NB!=length(labelsB) || NC!=length(labelsC)
        throw(TensorOperations.LabelError("invalid label specification"))
    end
    NC==NA+NB || throw(TensorOperations.LabelError("invalid label specification for tensor product"))

    tensorcontract!(alpha,A,labelsA,'N',B,labelsB,'N',beta,C,labelsC;method=:native)
end

# Methods below are only implemented for CartesianTensor or ComplexTensor:
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
typealias ComplexTensor{T,N} Tensor{ComplexSpace,T,N}
typealias CartesianTensor{T,N} Tensor{CartesianSpace,T,N}

# Index methods
#---------------
@eval function insertind{S}(t::Tensor{S},ind::Int,V::S)
    N=numind(t)
    0<=ind<=N || throw(IndexError("index out of range"))
    iscnumber(V) || throw(SpaceError("can only insert index with c-number index space"))
    spacet=space(t)
    newspace=spacet[1:ind] ⊗ V ⊗ spacet[ind+1:N]
    return tensor(t.data,newspace)
end
@eval function deleteind(t::Tensor,ind::Int)
    1<=ind<=numind(t) || throw(IndexError("index out of range"))
    iscnumber(space(t,ind)) || throw(SpaceError("can only delete index with c-number index space"))
    spacet=space(t)
    newspace=spacet[1:ind-1] ⊗ spacet[ind+1:N]
    return tensor(t.data,newspace)
end

for (S,TT) in ((CartesianSpace,CartesianTensor),(ComplexSpace,ComplexTensor))
    @eval function fuseind(t::$TT,ind1::Int,ind2::Int,V::$S)
        N=numind(t)
        ind2==ind1+1 || throw(IndexError("only neighbouring indices can be fused"))
        1<=ind1<=N-1 || throw(IndexError("index out of range"))
        fuse(space(t,ind1),space(t,ind2),V) || throw(SpaceError("index spaces $(space(t,ind1)) and $(space(t,ind2)) cannot be fused to $V"))
        spacet=space(t)
        newspace=spacet[1:ind1-1]*V*spacet[ind2+1:N]
        return tensor(t.data,newspace)
    end
    @eval function splitind(t::$TT,ind::Int,V1::$S,V2::$S)
        1<=ind<=numind(t) || throw(IndexError("index out of range"))
        fuse(V1,V2,space(t,ind)) || throw(SpaceError("index space $(space(t,ind)) cannot be split into $V1 and $V2"))
        spacet=space(t)
        newspace=spacet[1:ind-1]*V1*V2*spacet[ind+1:N]
        return tensor(t.data,newspace)
    end
end

# Factorizations:
#-----------------
for (S,TT) in ((CartesianSpace,CartesianTensor),(ComplexSpace,ComplexTensor))
    @eval function svd!(t::$TT,n::Int,::NoTruncation)
        # Perform singular value decomposition corresponding to bipartion of the
        # tensor indices into the left indices 1:n and remaining right indices,
        # thereby destroying the original tensor.
        N=numind(t)
        spacet=space(t)
        leftspace=spacet[1:n]
        rightspace=spacet[n+1:N]
        leftdim=dim(leftspace)
        rightdim=dim(rightspace)
        data=reshape(t.data,(leftdim,rightdim))
        F=svdfact!(data)
        newdim=length(F[:S])
        newspace=$S(newdim)
        U=tensor(F[:U],leftspace*newspace')
        Sigma=tensor(diagm(F[:S]),newspace*newspace')
        V=tensor(F[:Vt],newspace*rightspace)
        return U,Sigma,V,abs(zero(eltype(t)))
    end

    @eval function svd!(t::$TT,n::Int,trunc::Union(TruncationDimension,TruncationSpace))
        # Truncate rank corresponding to bipartition into left indices 1:n
        # and remain right indices, based on singular value decomposition,
        # thereby destroying the original tensor.
        # Truncation parameters are given as keyword arguments: trunctol should
        # always be one of the possible arguments for specifying truncation, but
        # truncdim can be replaced with different parameters for other types of tensors.

        N=numind(t)
        spacet=space(t)
        leftspace=spacet[1:n]
        rightspace=spacet[n+1:N]
        leftdim=dim(leftspace)
        rightdim=dim(rightspace)
        dim(trunc) >= min(leftdim,rightdim) && return svd!(t,n)
        data=reshape(t.data,(leftdim,rightdim))
        F=svdfact!(data)
        sing=F[:S]
        truncdim=dim(trunc)
        newspace=$S(truncdim)

        # truncate
        truncerr=vecnorm(sing[(truncdim+1):end])/vecnorm(sing)
        U=tensor(F[:U][:,1:truncdim],leftspace*newspace')
        Sigma=tensor(diagm(sing[1:truncdim]),newspace*newspace')
        V=tensor(F[:Vt][1:truncdim,:],newspace*rightspace)
        return U,Sigma,V,truncerr
    end

    @eval function svd!(t::$TT,n::Int,trunc::TruncationError)
        # Truncate rank corresponding to bipartition into left indices 1:n
        # and remain right indices, based on singular value decomposition,
        # thereby destroying the original tensor.
        # Truncation parameters are given as keyword arguments: trunctol should
        # always be one of the possible arguments for specifying truncation, but
        # truncdim can be replaced with different parameters for other types of tensors.

        N=numind(t)
        spacet=space(t)
        leftspace=spacet[1:n]
        rightspace=spacet[n+1:N]
        leftdim=dim(leftspace)
        rightdim=dim(rightspace)
        data=reshape(t.data,(leftdim,rightdim))
        F=svdfact!(data)

        # find truncdim based on trunctolinfo
        sing=F[:S]
        normsing=vecnorm(sing)
        truncdim=0
        while truncdim<length(sing) && vecnorm(sing[(truncdim+1):end])>eps(trunc)*normsing
            truncdim+=1
        end
        newspace=$S(truncdim)

        # truncate
        truncerr=vecnorm(sing[(truncdim+1):end])/normsing
        U=tensor(F[:U][:,1:truncdim],leftspace*newspace')
        Sigma=tensor(diagm(sing[1:truncdim]),newspace*newspace')
        V=tensor(F[:Vt][1:truncdim,:],newspace*rightspace)
        return U,Sigma,V,truncerr
    end

    @eval function leftorth!(t::$TT,n::Int,::NoTruncation)
        # Create orthogonal basis U for indices 1:n, and remainder C for right
        # indices, thereby destroying the original tensor.
        # Decomposition should be unique, such that it always returns the same
        # result for the same input tensor t. UC = QR is fastest but only unique
        # after correcting for phases.
        N=numind(t)
        spacet=space(t)
        leftspace=spacet[1:n]
        rightspace=spacet[n+1:N]
        leftdim=dim(leftspace)
        rightdim=dim(rightspace)
        data=reshape(t.data,(leftdim,rightdim))
        if leftdim>rightdim
            newdim=rightdim
            tau=zeros(eltype(data),(newdim,))
            Base.LinAlg.LAPACK.geqrf!(data,tau)

            C=zeros(eltype(t),(newdim,newdim))
            for j in 1:newdim
                for i in 1:j
                    @inbounds C[i,j]=data[i,j]
                end
            end
            Base.LinAlg.LAPACK.orgqr!(data,tau)
            U=data
            for i=1:newdim
                tau[i]=sign(C[i,i])
            end
            scale!(U,tau)
            scale!(conj!(tau),C)
        else
            newdim=leftdim
            C=data
            U=eye(eltype(data),newdim)
        end

        newspace=$S(newdim)
        return tensor(U,leftspace*newspace'), tensor(C,newspace*rightspace), abs(zero(eltype(t)))
    end

    @eval function leftorth!(t::$TT,n::Int,trunc::Union(TruncationDimension,TruncationSpace))
        # Truncate rank corresponding to bipartition into left indices 1:n
        # and remain right indices, based on singular value decomposition,
        # thereby destroying the original tensor.
        # Truncation parameters are given as keyword arguments: trunctol should
        # always be one of the possible arguments for specifying truncation, but
        # truncdim can be replaced with different parameters for other types of tensors.

        N=numind(t)
        spacet=space(t)
        leftspace=spacet[1:n]
        rightspace=spacet[n+1:N]
        leftdim=dim(leftspace)
        rightdim=dim(rightspace)
        dim(trunc) >= min(leftdim,rightdim) && return leftorth!(t,n)
        data=reshape(t.data,(leftdim,rightdim))
        F=svdfact!(data)
        sing=F[:S]
        
        # compute truncation dimension
        if eps(trunc)!=0
            sing=F[:S]
            normsing=vecnorm(sing)
            truncdim=0
            while truncdim<length(sing) && vecnorm(sing[(truncdim+1):end])>eps(trunc)*normsing
                truncdim+=1
            end
            truncdim=min(truncdim,dim(trunc))
        else
            truncdim=dim(trunc)
        end
        newspace=$S(truncdim)

        # truncate
        truncerr=vecnorm(sing[(truncdim+1):end])/vecnorm(sing)
        U=tensor(F[:U][:,1:truncdim],leftspace*newspace')
        C=tensor(scale!(sing[1:truncdim],F[:Vt][1:truncdim,:]),newspace*rightspace)
        return U,C,truncerr
    end

    @eval function leftorth!(t::$TT,n::Int,trunc::TruncationError)
        # Truncate rank corresponding to bipartition into left indices 1:n
        # and remain right indices, based on singular value decomposition,
        # thereby destroying the original tensor.
        # Truncation parameters are given as keyword arguments: trunctol should
        # always be one of the possible arguments for specifying truncation, but
        # truncdim can be replaced with different parameters for other types of tensors.

        N=numind(t)
        spacet=space(t)
        leftspace=spacet[1:n]
        rightspace=spacet[n+1:N]
        leftdim=dim(leftspace)
        rightdim=dim(rightspace)
        data=reshape(t.data,(leftdim,rightdim))
        F=svdfact!(data)

        # compute truncation dimension
        sing=F[:S]
        normsing=vecnorm(sing)
        truncdim=0
        while truncdim<length(sing) && vecnorm(sing[(truncdim+1):end])>eps(trunc)*normsing
            truncdim+=1
        end
        newspace=$S(truncdim)

        # truncate
        truncerr=vecnorm(sing[(truncdim+1):end])/vecnorm(sing)
        U=tensor(F[:U][:,1:truncdim],leftspace*newspace')
        C=tensor(scale!(sing[1:truncdim],F[:Vt][1:truncdim,:]),newspace*rightspace)

        return U,C,truncerr
    end

    @eval function rightorth!(t::$TT,n::Int,::NoTruncation)
        # Create orthogonal basis U for right indices, and remainder C for left
        # indices. Decomposition should be unique, such that it always returns the
        # same result for the same input tensor t. CU = LQ is typically fastest but only
        # unique after correcting for phases.
        N=numind(t)
        spacet=space(t)
        leftspace=spacet[1:n]
        rightspace=spacet[n+1:N]
        leftdim=dim(leftspace)
        rightdim=dim(rightspace)
        data=reshape(t.data,(leftdim,rightdim))
        if leftdim<rightdim
            newdim=leftdim
            tau=zeros(eltype(data),(newdim,))
            Base.LinAlg.LAPACK.gelqf!(data,tau)

            C=zeros(eltype(data),(newdim,newdim))
            for j=1:newdim
                for i=j:newdim
                    @inbounds C[i,j]=data[i,j]
                end
            end
            Base.LinAlg.LAPACK.orglq!(data,tau)
            U=data
            for i=1:newdim
                tau[i]=sign(C[i,i])
            end
            scale!(tau,U)
            scale!(C,conj!(tau))
        else
            newdim=rightdim
            C=data
            U=eye(eltype(data),newdim)
        end

        newspace=$S(newdim)
        return tensor(C,leftspace ⊗ dual(newspace)), tensor(U,newspace ⊗ rightspace), abs(zero(eltype(t)))
    end

    @eval function rightorth!(t::$TT,n::Int,trunc::Union(TruncationDimension,TruncationSpace))
        # Truncate rank corresponding to bipartition into left indices 1:n
        # and remain right indices, based on singular value decomposition,
        # thereby destroying the original tensor.
        # Truncation parameters are given as keyword arguments: trunctol should
        # always be one of the possible arguments for specifying truncation, but
        # truncdim can be replaced with different parameters for other types of tensors.

        N=numind(t)
        spacet=space(t)
        leftspace=spacet[1:n]
        rightspace=spacet[n+1:N]
        leftdim=dim(leftspace)
        rightdim=dim(rightspace)
        dim(trunc) >= min(leftdim,rightdim) && return rightorth!(t,n)
        data=reshape(t.data,(leftdim,rightdim))
        F=svdfact!(data)
        sing=F[:S]

        # compute truncation dimension
        if eps(trunc)!=0
            sing=F[:S]
            normsing=vecnorm(sing)
            truncdim=0
            while truncdim<length(sing) && vecnorm(sing[(truncdim+1):end])>eps(trunc)*normsing
                truncdim+=1
            end
            truncdim=min(truncdim,dim(trunc))
        else
            truncdim=dim(trunc)
        end
        newspace=$S(truncdim)

        # truncate
        truncerr=vecnorm(sing[(truncdim+1):end])/vecnorm(sing)
        C=tensor(scale!(F[:U][:,1:truncdim],sing[1:truncdim]),leftspace*newspace')
        U=tensor(F[:Vt][1:truncdim,:],newspace*rightspace)
        return C,U,truncerr
    end

    @eval function rightorth!(t::$TT,n::Int,trunc::TruncationError)
        # Truncate rank corresponding to bipartition into left indices 1:n
        # and remain right indices, based on singular value decomposition,
        # thereby destroying the original tensor.
        # Truncation parameters are given as keyword arguments: trunctol should
        # always be one of the possible arguments for specifying truncation, but
        # truncdim can be replaced with different parameters for other types of tensors.

        N=numind(t)
        spacet=space(t)
        leftspace=spacet[1:n]
        rightspace=spacet[n+1:N]
        leftdim=dim(leftspace)
        rightdim=dim(rightspace)
        data=reshape(t.data,(leftdim,rightdim))
        F=svdfact!(data)

        # compute truncation dimension
        sing=F[:S]
        normsing=vecnorm(sing)
        truncdim=0
        while truncdim<length(sing) && vecnorm(sing[(truncdim+1):end])>eps(trunc)*normsing
            truncdim+=1
        end
        newspace=$S(truncdim)

        # truncate
        truncerr=vecnorm(sing[(truncdim+1):end])/vecnorm(sing)
        C=tensor(scale!(F[:U][:,1:truncdim],sing[1:truncdim]),leftspace*newspace')
        U=tensor(F[:Vt][1:truncdim,:],newspace*rightspace)
        return C,U,truncerr
    end
end

# Methods below are only implemented for CartesianMatrix or ComplexMatrix:
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
typealias ComplexMatrix{T} ComplexTensor{T,2}
typealias CartesianMatrix{T} CartesianTensor{T,2}

function Base.pinv(t::Union(ComplexMatrix,CartesianMatrix))
    # Compute pseudo-inverse
    spacet=space(t)
    data=copy(t.data)
    leftdim=dim(spacet[1])
    rightdim=dim(spacet[2])

    F=svdfact!(data)
    Sinv=F[:S]
    for k=1:length(Sinv)
        if Sinv[k]>eps(Sinv[1])*max(leftdim,rightdim)
            Sinv[k]=one(Sinv[k])/Sinv[k]
        end
    end
    data=F[:V]*scale(F[:S],F[:U]')
    return tensor(data,spacet')
end

function Base.eig(t::Union(ComplexMatrix,CartesianMatrix))
    # Compute eigenvalue decomposition.
    spacet=space(t)
    spacet[1] == spacet[2]' || throw(SpaceError("eigenvalue factorization only exists if left and right index space are dual"))
    data=copy(t.data)

    F=eigfact!(data)

    Lambda=tensor(diagm(F[:values]),spacet)
    V=tensor(F[:vectors],spacet)
    return Lambda, V
end

function Base.inv(t::Union(ComplexMatrix,CartesianMatrix))
    # Compute inverse.
    spacet=space(t)
    spacet[1] == spacet[2]' || throw(SpaceError("inverse only exists if left and right index space are dual"))

    return tensor(inv(t.data),spacet)
end
