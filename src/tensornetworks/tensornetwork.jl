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
    _outputspace::P
    _contracttable::Dict{L,(Int,Int)}
end
function TensorNetwork{S,P,T,L}(tensors::Vector{AbstractTensor{S,P,T}},labels::Vector{Vector{L}},conjs::Vector{Char}=fill('N',length(tensors)),contracttree::Tuple=(),outputorder::Vector{L}=Array(L,0))
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
    
    # build lookup tables
    contracttable=Dict{L,(Int,Int)}()
    outputtable=Dict{L,(Int,Int)}()
    sizehint(contracttable,numindices-numlabels)
    sizehint(outputtable,2*numlabels-numindices)
    for i=1:numindices
        if table[i,2]==0
            outputtable[alllabels[i]]=table[i,1]
        else
            contracttable[alllabels[i]]=(table[i,1],table[i,2])
        end
    end
    
    # check / create output order
    outputlabels=collect(keys(outputtable))
    if isempty(outputorder)
        outputorder=sort!(outputlabels)
    else
        isperm(indexin(outputorder,outputlabels)) || throw(ArgumentError("output order does not match with uncontracted indices"))
    end
    outputspace=P(ntuple(length(outputorder),m->begin
        l=outputorder[m]
        n=outputtable[l]
        s=space(tensors[n],findfirst(labels[n],l))
        conjs[n]=='C' ? conj(s) : s
    end))
    
    # check contraction tree to visit every tensor once
    visited=zeros(Int,numtensors)
    visit(node::Tuple)=(visit(node[1]);visit(node[2]))
    visit(leaf::Int)=(visited[leaf]+=1)
    visit(contracttree)
    all(visited.==1) || throw(ArgumentError("invalid contractrion tree"))
        
    TensorNetwork{S,P,T,L}(tensors,labels,conjs,contracttree,outputorder,outputspace,contracttable)
end
function TensorNetwork{S,P,T,L}(tensors::Vector{AbstractTensor{S,P,T}},labels::Vector{Vector{L}},conjs::Vector{Char}=fill('N',length(tensors)),contractorder::Vector{L}=Array(L,0),outputorder::Vector{L}=Array(L,0))
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
    
    # build lookup tables
    contracttable=Dict{L,(Int,Int)}()
    outputtable=Dict{L,(Int,Int)}()
    sizehint(contracttable,numindices-numlabels)
    sizehint(outputtable,2*numlabels-numindices)
    for i=1:numindices
        if table[i,2]==0
            outputtable[alllabels[i]]=table[i,1]
        else
            contracttable[alllabels[i]]=(table[i,1],table[i,2])
        end
    end
    
    # check / create output order
    outputlabels=collect(keys(outputtable))
    if isempty(outputorder)
        outputorder=sort!(outputlabels)
    else
        isperm(indexin(outputorder,outputlabels)) || throw(ArgumentError("output order does not match with uncontracted indices"))
    end
    outputspace=P(ntuple(length(outputorder),m->begin
        l=outputorder[m]
        n=outputtable[l]
        s=space(tensors[n],findfirst(labels[n],l))
        conjs[n]=='C' ? conj(s) : s
    end))
    
    # check / create contraction order
    contractlabels=collect(keys(contracttable))
    if isempty(contractorder)
        contractorder=sort!(contractlabels,rev=true)
    else
        isperm(indexin(contractorder,contractlabels)) || throw(ArgumentError("contraction order does not match with uncontracted indices"))
    end
    
    # create corresponding contraction tree
    lookup=(Int=>Any)[n=>n for n=1:numtensors]
    for l in contractorder
        i1,i2=contracttable[l]
        node1=lookup[i1]
        node2=lookup[i2]
        if node1!=node2
            node=tuple(node1,node2)
            lookup[i1]=node
            lookup[i2]=node
        end
    end
    nodes=unique(values(lookup))
    contracttree=nodes[1]
    for k=2:length(nodes)
        contracttree=tuple(tree,nodes[k])
    end
    
    TensorNetwork{S,P,T,L}(tensors,labels,conjs,contracttree,outputorder,outputspace,contracttable)
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

# Basic methods for characterising the output tensor:
#-----------------------------------------------------
numind(network::TensorNetwork)=length(network._outputspace)
space(network::TensorNetwork,ind::Integer)=network._outputspace[ind]
space(network::TensorNetwork)=network._outputspace

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