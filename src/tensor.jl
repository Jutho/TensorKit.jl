# tensor.jl
#
# Tensor provides a dense implementation of an AbstractTensor type without any
# symmetry assumptions, i.e. it describes tensors living in the full tensor
# product space of its index spaces.
#
# Written by Jutho Haegeman

#++++++++++++++
# Tensor type:
#++++++++++++++
# Type definition and constructors:
#-----------------------------------
type Tensor{S,T,N} <: AbstractTensor{S,ProductSpace{S},T,N}
    data::Array{T,N}
    space::ProductSpace{S,N}
    function Tensor(data::Array{T},space::ProductSpace{S,N})
        if length(data)!=dim(space)
            throw(DimensionMismatch("data not of right size"))
        end
        if promote_type(T,eltype(S)) != eltype(S)
            error("For a tensor in $(space), the entries cannot be of type $(T)")
        end
        return new(reshape(data,map(dim,space)),space)
    end
end

# Show method:
#-------------
function Base.show{S,T,N}(io::IO,t::Tensor{S,T,N})
    print(io," Tensor ∈ $T")
    for n=1:N
        print(io, n==1 ? "[" : " ⊗ ")
        show(io,space(t,n))
    end
    println(io,"]:")
    Base.showarray(io,t.data;header=false)
end

# Basic methods for characterising a tensor:
#--------------------------------------------
space(t::Tensor,ind::Int)=t.space[ind]
space(t::Tensor)=t.space

# General constructors
#---------------------
# with data
tensor{T<:Real,N}(data::Array{T,N})=Tensor{CartesianSpace,T,N}(data,prod(CartesianSpace,size(data)))
function tensor{T<:Complex}(data::Array{T,1})
    warning("for complex array, consider specifying Euclidean index spaces")
    Tensor{ComplexEuclideanSpace,T,1}(data,prod(ComplexSpace(size(data,1))))
end
function tensor{T<:Complex}(data::Array{T,2})
    warning("for complex array, consider specifying Euclidean index spaces")
    Tensor{ComplexEuclideanSpace,T,2}(data,ComplexSpace(size(data,1))*ComplexSpace(size(data,2))')
end

tensor{S,T,N}(data::Array{T},P::ProductSpace{S,N})=Tensor{S,T,N}(data,P)

# without data
tensor{T}(::Type{T},P::ProductSpace)=tensor(Array(T,dim(P)),P)
tensor{T}(::Type{T},V::IndexSpace)=tensor(T,prod(V))
tensor(V::Union(ProductSpace,IndexSpace))=tensor(Float64,V)

Base.similar{S,T,N}(t::Tensor{S},::Type{T},P::ProductSpace{S,N}=space(t))=tensor(similar(t.data,T,dim(P)),P)
Base.similar{S,T}(t::Tensor{S},::Type{T},V::S)=similar(t,T,prod(V))

Base.similar{S,N}(t::Tensor{S},P::ProductSpace{S,N}=space(t))=similar(t,eltype(t),P)
Base.similar{S}(t::Tensor{S},V::S)=similar(t,eltype(t),V)

Base.zero(t::Tensor)=tensor(zero(t.data),space(t))

Base.zeros{T}(::Type{T},P::ProductSpace)=tensor(zeros(T,dim(P)),P)
Base.zeros{T}(::Type{T},V::IndexSpace)=zeros(T,prod(V))
Base.zeros(V::Union(ProductSpace,IndexSpace))=zeros(Float64,V)

Base.rand{T}(::Type{T},P::ProductSpace)=tensor(rand(T,dim(P)),P)
Base.rand{T}(::Type{T},V::IndexSpace)=rand(T,prod(V))
Base.rand(V::Union(ProductSpace,IndexSpace))=rand(Float64,V)

Base.eye{T}(::Type{T},V::IndexSpace)=tensor(eye(T,dim(V)),V*dual(V))
Base.eye(V::IndexSpace)=tensor(eye(dim(V)),V*dual(V))

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
    spaceC = prod(spaceCvec)
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
    if space(tdest)!=space(tsource)
        throw(SpaceError("tensor spaces don't match"))
    end
    copy!(tdest.data,tsource.data)
end
Base.fill!{S,T}(tdest::Tensor{S,T},value::Number)=fill!(tdest.data,convert(T,value))

# Vectorization:
#----------------
Base.vec(t::Tensor)=vec(t.data)
# Convert the non-trivial degrees of freedom in a tensor to a vector to be passed to eigensolvers etc.

# Conversion and promotion:
#---------------------------
Base.full(t::Tensor)=t.data

Base.promote_rule{S,T,N1,N2}(::Type{Tensor{S,T,N1}},::Type{Tensor{S,T,N2}})=Tensor{S,T}
Base.promote_rule{S,T1,T2,N}(::Type{Tensor{S,T1,N}},::Type{Tensor{S,T2,N}})=Tensor{S,promote_type(T1,T2),N}
Base.promote_rule{S,T1,T2,N1,N2}(::Type{Tensor{S,T1,N1}},::Type{Tensor{S,T2,N2}})=Tensor{S,promote_type(T1,T2)}
Base.promote_rule{S,T1,T2}(::Type{Tensor{S,T1}},::Type{Tensor{S,T2}})=Tensor{S,promote_type(T1,T2)}
Base.convert{S,T1,T2,N}(::Type{Tensor{S,T1,N}},t::Tensor{S,T2,N})=tensor(convert(Array{T1,N},t.data),space(t))
Base.convert{S,T1,T2,N}(::Type{Tensor{S,T1}},t::Tensor{S,T2,N})=tensor(convert(Array{T1},t.data),space(t))
Base.convert{S,T,N}(::Type{Tensor{S}},t::Tensor{S,T,N})=t
Base.convert{S,T,N}(::Type{Tensor},t::Tensor{S,T,N})=t

Base.convert{S,T1,T2,N}(::Type{AbstractTensor{S,ProductSpace,T1,N}},t::Tensor{S,T2,N})=tensor(convert(Array{T1,N},t.data),space(t))
Base.convert{S,T1,T2}(::Type{AbstractTensor{S,ProductSpace,T1}},t::Tensor{S,T2})=tensor(convert(Array{T1},t.data),space(t))
Base.convert{S}(::Type{AbstractTensor{S,ProductSpace}},t::Tensor{S})=t
Base.convert(::Type{AbstractTensor},t::Tensor)=t

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
    @eval (Base.$f)(t::Tensor) = Tensor(($f)(t.data),space(t))
end

# Basic algebra:
#----------------
# transpose inverts order of indices, compatible with graphical notation

Base.conj(t::Tensor) = tensor(conj(t.data),conj(space(t)))
function Base.conj!(t::Tensor)
    t.space=conj(t.space)
    conj!(t.data)
    return t
end

# SHOULD THIS BE DEFINED?
# function Base.transpose(t::Tensor)
#     tdest=similar(t,reverse(space(t)))
#     return Base.transpose!(tdest,t)
# end
# function Base.transpose!(tdest::Tensor,tsource::Tensor)
#     if space(tdest)!=reverse(space(tsource))
#         throw(SpaceError("tensor spaces don't match"))
#     end
#     N=numind(tsource)
#     TensorOperations.tensorcopy!(tsource.data,1:N,tdest.data,reverse(1:N))
#     return tdest
# end

# function Base.ctranspose(t::Tensor)
#     tdest=similar(t,reverse(conj(space(t))))
#     return Base.ctranspose!(tdest,t)
# end
# function Base.ctranspose!(tdest::Tensor,tsource::Tensor)
#     if space(tdest)!=reverse(conj(space(tsource))))
#         throw(SpaceError("tensor spaces don't match"))
#     end
#     N=numind(tsource)
#     TensorOperations.tensorcopy!(tsource.data,1:N,tdest.data,reverse(1:N))
#     conj!(tdest.data)
#     return tdest
# end

Base.scale(t::Tensor,a::Number)=tensor(scale(t.data,a),space(t))
Base.scale!(t::Tensor,a::Number)=(scale!(t.data,convert(eltype(t),a));return t)

-(t::Tensor)=tensor(-t.data,space(t))

function +(t1::Tensor,t2::Tensor)
    if space(t1)!=space(t2)
        throw(SpaceError("tensor spaces do not agree"))
    end
    return tensor(t1.data+t2.data,space(t1))
end

function -(t1::Tensor,t2::Tensor)
    if space(t1)!=space(t2)
        throw(SpaceError("tensor spaces do not agree"))
    end
    return tensor(t1.data-t2.data,space(t1))
end

Base.vecnorm(t::Tensor)=vecnorm(t.data)
# Frobenius norm of tensor

# Indexing
#----------
# linear indexing using ProductBasisVector
Base.getindex{S,T,N}(t::Tensor{S,T,N},b::ProductBasisVector{N,S})=getindex(t.data,Base.to_index(b))
Base.setindex!{S,T,N}(t::Tensor{S,T,N},value,b::ProductBasisVector{N,S})=setindex!(t.data,value,Base.to_index(b))

# Tensor Operations
#-------------------
scalar{S,T}(t::Tensor{S,T,0})=scalar(t.data)

function tensorcopy!{S,T1,T2,N}(t1::Tensor{S,T1,N},labels1,t2::Tensor{S,T2,N},labels2)
    # Replaces tensor t2 with t1
    perm=indexin(labels1,labels2)

    length(perm) == N || throw(TensorOperations.LabelError("invalid label specification"))
    isperm(perm) || throw(TensorOperations.LabelError("invalid label specification"))
    for i = 1:N
        space(t1,i) == space(t2,perm[i]) || throw(SpaceError("incompatible index spaces of tensors"))
    end

    TensorOperations.tensorcopy!(t1.data,labels1,t2.data,labels2)
    return t2
end
function tensoradd!{S,T1,T2,N}(alpha::Number,t1::Tensor{S,T1,N},labels1,beta::Number,t2::Tensor{S,T2,N},labels2)
    # Replaces tensor t2 with beta*t2+alpha*t1
    perm=indexin(labels1,labels2)

    length(perm) == N || throw(TensorOperations.LabelError("invalid label specification"))
    isperm(perm) || throw(TensorOperations.LabelError("invalid label specification"))
    for i = 1:N
        space(t1,i) == space(t2,perm[i]) || throw(SpaceError("incompatible index spaces of tensors"))
    end

    TensorOperations.tensoradd!(alpha,t1.data,labels1,beta,t2.data,labels2)
    return t2
end
function tensortrace!{S,TA,NA,TC,NC}(alpha::Number,A::Tensor{S,TA,NA},labelsA,beta::Number,C::Tensor{S,TC,NC},labelsC)
    (length(labelsA)==NA && length(labelsC)==NC) || throw(LabelError("invalid label specification"))
    NA==NC && return tensoradd!(alpha,A,labelsA,beta,C,labelsC) # nothing to trace
    
    po=indexin(labelsC,labelsA)
    clabels=unique(setdiff(labelsA,labelsC))
    NA==NC+2*length(clabels) || throw(LabelError("invalid label specification"))
    
    pc1=Array(Int,length(clabels))
    pc2=Array(Int,length(clabels))
    for i=1:length(clabels)
        pc1[i]=findfirst(labelsA,clabels[i])
        pc2[i]=findnext(labelsA,clabels[i],pc1[i]+1)
    end
    isperm(vcat(po,pc1,pc2)) || throw(LabelError("invalid label specification"))
    
    for i = 1:NC
        space(A,po[i]) == space(C,i) || throw(SpaceError("space mismatch"))
    end
    for i = 1:div(NA-NC,2)
        space(A,pc1[i]) == dual(space(A,pc2[i])) || throw(SpaceError("space mismatch"))
    end
    
    TensorOperations.tensortrace!(alpha,A.data,labelsA,beta,C.data,labelsC)
end
function tensorcontract!{S}(alpha::Number,A::Tensor{S},labelsA,conjA::Char,B::Tensor{S},labelsB,conjB::Char,beta::Number,C::Tensor{S},labelsC;method=:BLAS)
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
    conjB=='C' || conjA=='N' || throw(ArgumentError("conjB should be 'C' or 'N'."))

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

    TensorOperations.tensorcontract!(alpha,A.data,labelsA,conjA,B.data,labelsB,conjB,beta,C.data,labelsC;method=method)
    return C
end

# Methods below are only implemented for Cartesian or Euclidean tensors:
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
typealias ComplexTensor{T,N} Tensor{ComplexSpace,T,N}
typealias CartesianTensor{T,N} Tensor{CartesianSpace,T,N}

# Index methods
#---------------
for (S,TT) in ((CartesianSpace,CartesianTensor),(ComplexSpace,ComplexTensor))
    @eval function insertind(t::$TT,ind::Int,V::$S)
        N=numind(t)
        0<=ind<=N || throw(IndexError("index out of range"))
        iscnumber(V) || throw(SpaceError("can only insert index with c-number index space"))
        spacet=space(t)
        newspace=spacet[1:ind]*V*spacet[ind+1:N]
        return tensor(t.data,newspace)
    end
    @eval function deleteind(t::$TT,ind::$S)
        1<=ind<=numind(t) || throw(IndexError("index out of range"))
        iscnumber(space(t,ind)) || throw(SpaceError("can only squeeze index with c-number index space"))
        spacet=space(t)
        newspace=spacet[1:ind-1]*spacet[ind+1:N]
        return tensor(t.data,newspace)
    end
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
    @eval function svd!(t::$TT,n::Int)
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
        return U,Sigma,V
    end

    @eval function svdtrunc!(t::$TT,n::Int;trunctol::Real=0,truncdim::Int=typemax(Int),truncspace::$S=$S(truncdim))
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
        if trunctol==0
            trunctoldim=length(sing)
        else
            normsing=norm(sing)
            trunctoldim=0
            while norm(sing[(trunctoldim+1):end])>trunctol*normsing
                trunctoldim+=1
                if trunctoldim==length(sing)
                    break
                end
            end
        end

        # choose minimal truncdim
        truncdim > 0 || throw(ArgumentError("Truncation dimension should be bigger than zero"))
        truncdim=min(truncdim,trunctoldim,dim(truncspace))
        truncerr=zero(eltype(sing))
        newspace=$S(truncdim)
        if truncdim<length(sing)
            truncerr=vecnorm(sing[(truncdim+1):end])
            Sigma=Sigma[1:truncdim]
            U=tensor(F[:U][:,1:truncdim],leftspace*newspace')
            Sigma=tensor(diagm(sing[1:truncdim]),newspace*newspace')
            V=tensor(F[:Vt][1:truncdim,:],newspace*rightspace)
        else
            U=tensor(F[:U],leftspace*newspace')
            Sigma=tensor(diagm(sing),newspace*newspace')
            V=tensor(F[:Vt],newspace*rightspace)
        end

        return U,Sigma,V,truncerr
    end

    @eval function leftorth!(t::$TT,n::Int;buildC::Bool=true)
        # Create orthogonal basis U for indices 1:n, and remainder C for right
        # indices, thereby destroying the original tensor.
        # Decomposition should be unique, such that it always returns the same
        # result for the same input tensor t. UC = QR is fastest but only unique
        # after correcting for phases.
        local C::Array{eltype(t),2}
        
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
            
            phase=zeros(eltype(data),(newdim,))
            C=zeros(eltype(t),(newdim,newdim))
            for j in 1:newdim
                for i in 1:j
                    @inbounds C[i,j]=data[i,j]
                end
            end
            Base.LinAlg.LAPACK.orgqr!(data,tau)
            U=data
            for i=1:newdim
                phase[i]=C[i,i]/abs(C[i,i])
                tau[i]=1/phase[i]
            end
            scale!(tau,C)
            scale!(U,phase)
        else
            newdim=leftdim
            C=data
            U=eye(eltype(data),newdim)
        end
        
        newspace=$S(newdim)
        return tensor(U,leftspace*newspace'), tensor(C,newspace*rightspace)
    end

    @eval function rightorth!(t::$TT,n::Int)
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
            
            phase=zeros(eltype(data),(newdim,))
            C=zeros(eltype(data),(newdim,newdim))
            for j=1:newdim
                for i=j:newdim
                    @inbounds C[i,j]=data[i,j]
                end
            end
            Base.LinAlg.LAPACK.orglq!(data,tau)
            U=data
            for i=1:newdim
                phase[i]=C[i,i]/abs(C[i,i])
                tau[i]=1/phase[i]
            end
            scale!(C,tau)
            scale!(phase,U)
        else
            newdim=rightdim
            C=data
            U=eye(eltype(data),newdim)
        end
        
        newspace=$S(newdim)
        return tensor(C,leftspace*newspace'), tensor(U,newspace*rightspace)
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