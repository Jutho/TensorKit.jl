# tensornetwork.jl
#
# Definition and implementation of a TensorNetwork type that
# contains a list of tensors together with a specification of how they
# should be contracted, along with methods to perform the actual contraction
# and to optimize the contraction order.
#
type TensorNetwork{S,P,T,L} <: AbstractTensorNetwork{S,P,T}
    tensors::Vector{AbstractTensor{S,P,T}}
    labels::Vector{Vector{L}}
    conjs::Vector{Char}
    contracttree::Tuple
    outputorder::Vector{L}
    outputspaces::Dict{L,S}
end
function TensorNetwork{S,P,T,L}(tensors::Vector{AbstractTensor{S,P,T}},labels::Vector{Vector{L}},conjs::Vector{Char},contracttree::Tuple,outputorder::Vector{L}=Array(L,0))
    # basic concistency check on arguments
    numtensors=length(tensors)
    numtensors>1 || throw(Argument("number of tensors should be larger than one for a proper tensor network"))
    length(labels)==numtensors || throw(ArgumentError("number of label lists does not equal number of tensors"))
    length(conjs)==numtensors || throw(ArgumentError("number of conjugation identifiers does not equal number of tensors"))
    for k=1:numtensors
        numind(tensors[k])==length(labels[k]) || throw(ArgumentError("number of labels should equal number of indices for tensor $k"))
        conjs[k]=='N' || conjs[k]=='C' || throw(ArgumentError("conjugation identifiers should equal 'N' or 'C'"))
    end
    
    # analyse contraction network
    alllabels=vcat(labels...)
    numindices=length(alllabels)
    alllabels=unique(alllabels)
    numlabels=length(alllabels)
    
    table=zeros(Int,(numlabels,2))
    for n=1:numtensors
        length(labels[n])==length(unique(labels[n])) || throw(ArgumentError("duplicate labels for tensor $n: handle inner contraction first with tensortrace"))
        indn=findin(alllabels,labels[n])
        for i in indn
            if table[i,1]==0
                table[i,1]=n
            elseif table[i,2]==0
                table[i,2]=n
            else
                throw(ArgumentError("no label should appear more than two times"))
            end
        end
    end
    
    # build lookup table for output spaces
    outputspaces=Dict{L,S}()
    sizehint(outputspaces,2*numlabels-numindices)
    for i=1:numindices
        if table[i,2]==0
            n=table[i,1]
            l=alllabels[i]
            s=space(tensors[n],findfirst(labels[n],l))
            outputspaces[l]=conjs[n]=='C' ? conj(s) : s
        end
    end
    
    # check / create output order
    outputlabels=collect(keys(outputspaces))
    if isempty(outputorder)
        outputorder=sort!(outputlabels)
    else
        isperm(indexin(outputorder,outputlabels)) || throw(ArgumentError("output order does not match with uncontracted indices"))
    end
    
    # check contraction tree to visit every tensor once
    visited=zeros(Int,numtensors)
    visit(node::Tuple)=(visit(node[1]);visit(node[2]))
    visit(leaf::Int)=(visited[leaf]+=1)
    visit(contracttree)
    all(visited.==1) || throw(ArgumentError("invalid contractrion tree"))
        
    TensorNetwork{S,P,T,L}(tensors,labels,conjs,contracttree,outputorder,outputspaces)
end
function TensorNetwork{S,P,T,L}(tensors::Vector{AbstractTensor{S,P,T}},labels::Vector{Vector{L}},conjs::Vector{Char},contractorder::Vector{L}=Array(L,0),outputorder::Vector{L}=Array(L,0))
    # basic concistency check on arguments
    numtensors=length(tensors)
    numtensors>1 || throw(Argument("number of tensors should be larger than one for a proper tensor network"))
    length(labels)==numtensors || throw(ArgumentError("number of label lists does not equal number of tensors"))
    length(conjs)==numtensors || throw(ArgumentError("number of conjugation identifiers does not equal number of tensors"))
    for k=1:numtensors
        numind(tensors[k])==length(labels[k]) || throw(ArgumentError("number of labels should equal number of indices for tensor $k"))
        conjs[k]=='N' || conjs[k]=='C' || throw(ArgumentError("conjugation identifiers should equal 'N' or 'C'"))
    end
    
    # analyse contraction network
    alllabels=vcat(labels...)
    numindices=length(alllabels)
    alllabels=unique(alllabels)
    numlabels=length(alllabels)
    
    table=zeros(Int,(numlabels,2))
    for n=1:numtensors
        length(labels[n])==length(unique(labels[n])) || throw(ArgumentError("duplicate labels for tensor $n: handle inner contraction first with tensortrace"))
        indn=findin(alllabels,labels[n])
        for i in indn
            if table[i,1]==0
                table[i,1]=n
            elseif table[i,2]==0
                table[i,2]=n
            else
                throw(ArgumentError("no label should appear more than two times"))
            end
        end
    end
    
    # build lookup table for output spaces
    outputspaces=Dict{L,S}()
    sizehint(outputspaces,2*numlabels-numindices)
    for i=1:numindices
        if table[i,2]==0
            n=table[i,1]
            l=alllabels[i]
            s=space(tensors[n],findfirst(labels[n],l))
            outputspaces[l]=conjs[n]=='C' ? conj(s) : s
        end
    end
    
    # check / create output order
    outputlabels=collect(keys(outputspaces))
    if isempty(outputorder)
        outputorder=sort!(outputlabels)
    else
        isperm(indexin(outputorder,outputlabels)) || throw(ArgumentError("output order does not match with uncontracted indices"))
    end
    
    # check / create contraction order
    contractlabels=setdiff(alllabels,outputlabels)
    if isempty(contractorder)
        contractorder=sort!(contractlabels,rev=true)
    else
        isperm(indexin(contractorder,contractlabels)) || throw(ArgumentError("contraction order does not match with uncontracted indices"))
    end
    
    # create corresponding contraction tree
    indexorder=indexin(contractorder,alllabels)
    lookup=(Int=>Any)[n=>n for n=1:numtensors]
    for i in indexorder
        n1,n2=table[i,1],table[i,2]
        node1=lookup[n1]
        node2=lookup[n2]
        if node1!=node2
            node=tuple(node1,node2)
            lookup[n1]=node
            lookup[n2]=node
        end
    end
    nodes=unique(values(lookup))
    contracttree=nodes[1]
    for k=2:length(nodes)
        contracttree=tuple(tree,nodes[k])
    end
    
    
    TensorNetwork{S,P,T,L}(tensors,labels,conjs,contracttree,outputorder,outputspaces)
end

# Convenience constructor
#-------------------------
function network(tensors::Vector,labels::Vector,conjs::Vector{Char}=fill('N',length(tensors));contractorder::Vector=[],outputorder::Vector=[],contracttree::Tuple=())
    length(tensors)>1 || throw(ArgumentError("number of tensors should be larger than one for a proper tensor network"))
    length(tensors)==length(labels) || throw(ArgumentError("number of label lists does not equal number of tensors"))
    length(tensors)==length(conjs) || throw(ArgumentError("number of conjugation identifiers does not equal number of tensors"))
    for n=1:length(tensors)
        isa(tensors[n],AbstractTensor) || throw(ArgumentError("not a list of tensors"))
        isa(labels[n],Vector) || throw(ArgumentError("not a list of labels"))
    end
    Slist=map(spacetype,tensors)
    S=Slist[1]
    all(Slist .== S) || throw(ArgumentError("not a homogeneous list of tensors"))
    Plist=map(tensortype,tensors)
    P=Plist[1]
    all(Plist .== P) || throw(ArgumentError("not a homogeneous list of tensors"))
    T = eltype(tensors[1])
    L = eltype(labels[1])
    for n=2:length(tensors)
        T=promote_type(T,eltype(tensors[n]))
        L=promote_type(L,eltype(labels[n]))
    end
    
    if isempty(contracttree)
        return TensorNetwork(convert(Vector{AbstractTensor{S,P,T}},tensors),convert(Vector{Vector{L}},labels),convert(Vector{L},contractorder),convert(Vector{L},outputorder))
    elseif isempty(contractorder)
        return TensorNetwork(convert(Vector{AbstractTensor{S,P,T}},tensors),convert(Vector{Vector{L}},labels),contracttree,convert(Vector{L},outputorder))
    else
        throw(ArgumentError("provide contractorder or contracttree, not both"))
    end
end

# Basic properties
#------------------
Base.length(network::TensorNetwork)=length(network.tensors)

# Basic methods for characterising the output tensor:
#-----------------------------------------------------
numind(network::TensorNetwork)=length(network.outputorder)
space(network::TensorNetwork,ind::Integer)=network.outputspaces[network.outputorder[ind]]
space(network::TensorNetwork)=ntuple(numind(network),ind->space(network,ind))

# Compute the output tensor:
#----------------------------
function tensor(network::TensorNetwork)
    tree=network.contracttree
    t1,c1,labels1=_contract(network,tree[1])
    t2,c2,labels2=_contract(network,tree[2])
    return tensorcontract(t1,labels1,c1,t2,labels2,c2,network.outputorder)
end
tensorcontract(network::TensorNetwork)=tensor(network)

# Internal methods to perform the contraction
#---------------------------------------------
_contract(network::TensorNetwork,n::Int)=(network.tensors[n],network.conjs[n],network.labels[n])
function _contract(network::TensorNetwork,tree::Tuple)
    t1,conj1,labels1=_contract(network,tree[1])
    t2,conj2,labels2=_contract(network,tree[2])
    
    newlabels=symdiff(labels1,labels2)
    if conj1==conj2
        newconj=conj1
        conj1='N'
        conj2='N'
    else
        newconj='N'
    end
    
    return tensorcontract(t1,labels1,conj1,t2,labels2,conj2,newlabels),newconj,newlabels
end