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
function TensorNetwork{S,P,T,L}(tensors::Vector{AbstractTensor{S,P,T}},labels::Vector{Vector{L}},conjs::Vector{Char}=fill('N',length(tensors)),contracttree::Tuple,outputorder::Vector{L}=Array(L,0))
    numtensors=length(tensors)
    numtensors>1 || throw(Argument("number of tensors should be larger than one for a proper tensor network"))
    length(labels)==numtensors || throw(ArgumentError("number of label lists does not equal number of tensors"))
    length(conjs)==numtensors || throw(ArgumentError("number of conjugation identifiers does not equal number of tensors"))
    for k=1:numtensors
        numind(tensors[k])==length(labels[k]) || throw(ArgumentError("number of labels should equal number of indices for tensor $k"))
    end
    
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
    
    outputlabels=collect(keys(outputtable))
    if isempty(outputorder)
        outputorder=sort!(outputlabels)
    else
        isperm(indexin(outputorder,outputlabels)) || throw(ArgumentError("output order does not match with uncontracted indices"))
    end
    outputspace=P(ntuple(length(outputorder),m->(l=outputorder[m],n=outputtable[l];space(tensors[n],findfirst(labels[n],l)))))
    
    visited=falses(numtensors)
    visit(node::Tuple)=(visit(node[1]);visit(node[2]))
    visit(leaf::Int)=(visited[leaf]=true)
    visit(contracttree)
    all(visited) || throw(ArgumentError("invalid contractrion tree"))
        
    TensorNetwork{S,P,T,L}(tensors,labels,conjs,contracttree,outputorder,outputspace,contracttable)
end
function TensorNetwork{S,P,T,L}(tensors::Vector{AbstractTensor{S,P,T}},labels::Vector{Vector{L}},conjs::Vector{Char}=fill('N',length(tensors)),contractorder::Vector{L}=Array(L,0),outputorder::Vector{L}=Array(L,0))
    numtensors=length(tensors)
    numtensors>1 || throw(Argument("number of tensors should be larger than one for a proper tensor network"))
    length(labels)==numtensors || throw(ArgumentError("number of label lists does not equal number of tensors"))
    length(conjs)==numtensors || throw(ArgumentError("number of conjugation identifiers does not equal number of tensors"))
    for k=1:numtensors
        numind(tensors[k])==length(labels[k]) || throw(ArgumentError("number of labels should equal number of indices for tensor $k"))
    end
    
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
    
    outputlabels=collect(keys(outputtable))
    if isempty(outputorder)
        outputorder=sort!(outputlabels)
    else
        isperm(indexin(outputorder,outputlabels)) || throw(ArgumentError("output order does not match with uncontracted indices"))
    end
    outputspace=P(ntuple(length(outputorder),m->(l=outputorder[m],n=outputtable[l];space(tensors[n],findfirst(labels[n],l)))))
    
    contractlabels=collect(keys(contracttable))
    if isempty(contractorder)
        contractorder=sort!(contractlabels,rev=true)
    else
        isperm(indexin(contractorder,contractlabels)) || throw(ArgumentError("contraction order does not match with uncontracted indices"))
    end
    
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

partialcontract(network::TensorNetwork,n::Int)



function tensor(network::TensorNetwork)
    tensors=copy(network.tensors)
    labels=copy(network.labels)
    conjs=copy(network.conjs)
    slots=trues(length(tensors))
    
    numcontract=length(network.contractorder)
    
    contracttable=zeros(Int,(numcontract,2))
    for k=1:numcontract
        l=network.contractorder[k]
        i,j=network._contracttable[l]
        contracttable[k,1]=min(i,j)
        contracttable[k,2]=max(i,j)
    end
    
    for k=1:numcontract
        i=contracttable[k,1]
        j=contracttable[k,2]
        
        if i!=0 && j!=0
            newlabels=symdiff(labels[i],labels[j])
            if conjs[i]==conjs[j]
                newtensor=tensorcontract(tensors[i],labels[i],'N',tensors[j],labels[j],'N',newlabels)
            else
                newtensor=tensorcontract(tensors[i],labels[i],conjs[i],tensors[j],labels[j],conjs[j],newlabels)
            end
            labels[i]=newlabels
            tensors[i]=newtensor
            if conjs[i]!=conjs[j]
                conjs[i]='N'
            end
            slots[j]=false
        
            for k2=k+1:numcontract
                if contracttable[k2,1]==i && contracttable[k2,2]==j
                    contracttable[k2,1]=0
                    contracttable[k2,2]=0
                end
                
                contracttable[k2,1]==j && (contracttable[k2,1]=i)
                contracttable[k2,2]==j && (contracttable[k2,2]=i)
            end
        end
    end
        
    components=find(slots)
    t=tensors[components[1]]
    l=labels[components[1]]
    
    for k=2:length(components)
        t2=tensors[components[k]]
        l2=labels[components[k]]
        t=tensorproduct(t,l,t2,l2,vcat(l,l2))
        l=vcat(l,l2)
    end
    
    return tensorcopy(p,l,network.outputorder)
end



# Contraction routines
function mcontract(tensors::Vector{AbstractTensor},labels::Vector{Vector{Int}};outputorder::Vector{Int}=Array(Int,0),contractorder::Vector{Int}=Array(Int,0))
    # Construct contraction table, analyze contraction graph and do some error checking
    
    table=zeros(Int,(numtensors,numlabels))
    for i=1:numtensors
        ci=findin(alllabels,labels[i])
        table[i,ci]=1
    end
    otable=table[:,1:firstclabelpos-1] # table of open labels
    if any(sum(otable,1).!=1)
        throw(LabelError("Positive labels indicate output dimensions and should appear only once"))
    end
    ctable=table[:,firstclabelpos:end] # table of contraction labels
    if any(sum(ctable,1).!=2)
        throw(LabelError("Negative labels indicate contractions and should appear exactly twice"))
    end
    adjacencymatrix=ctable*transpose(ctable)
    # not really adjacencymatrix:
    # diagonal elements tell number of contracted indices of that tensor
    # off-diagonal elements tell number of contracted bonds between those two tensors
    # onlything important is to have non-zero numbers between tensors that share a contraction
    componentlist=connectedcomponents(bool(adjacencymatrix))
    numindlist=map(component->sum(otable[component,:]),componentlist) # number of open indices of every component
    p=sortperm(numindlist)
    componentlist=componentlist[p]
    firsttensorpos=findfirst(numindlist)
    if firsttensorpos==0
        firsttensorpos=length(componentlist)+1
    end

    # Set contraction order
    if !isempty(contractorder)
        corder=indexin(clabels,contractorder)
        p=sortperm(corder)
        if corder[p[1]]==0
            throw(ArgumentError("Contraction order does not specify order for all contraction labels"))
        end
        ctable=ctable[:,p]
        clabels=clabels[p]
    end

    # Start contracting:
    # first run over scalar components
    scalar=one(mapreduce(eltype,promote_type,tensors))
    for pos=1:firsttensorpos-1
        component=componentlist[pos]
        while (c=findfirst(sum(ctable[component,:],1)))!=0
            ind1,ind2=find(ctable[:,c])
            c=intersect(labels[ind1],labels[ind2]) # should be equal to c + maybe other contraction indices shared by these two tensors
            contractind1=indexin(c,labels[ind1])
            contractind2=indexin(c,labels[ind2])
            # put in temporary object, because this could be a scalar (at end of component)
            ttemp=contract(tensors[ind1],contractind1,tensors[ind2],contractind2)
            labeltemp=vcat(setdiff(labels[ind1],c),setdiff(labels[ind2],c))
            ctable[ind1,:]=0
            ctable[ind2,:]=0
            if isempty(labeltemp) # ttemp is scalar: can only happen at the end of the scalar component
                scalar=scalar*ttemp
            else
                tensors[ind1]=ttemp
                labels[ind1]=labeltemp
                c=findin(clabels,labels[ind1])
                ctable[ind1,c]=1
            end
        end
    end
    if firsttensorpos>length(componentlist)
        return scalar
    end

    # first tensor component should create tensor object
    component=componentlist[firsttensorpos]
    ind1=component[1] # position of resulting tensor for this component (even when no contraction at all)
    while (c=findfirst(sum(ctable[component,:],1)))!=0 # findfirst guarantees that we perform contractions in specified order
        ind1,ind2=find(ctable[:,c])
        c=intersect(labels[ind1],labels[ind2])
        contractind1=indexin(c,labels[ind1])
        contractind2=indexin(c,labels[ind2])

        # this should always be tensors, so immediately put result in tensors[ind1]
        tensors[ind1]=contract(tensors[ind1],contractind1,tensors[ind2],contractind2)
        labels[ind1]=vcat(setdiff(labels[ind1],c),setdiff(labels[ind2],c))
        ctable[ind1,:]=0
        ctable[ind2,:]=0
        c=findin(clabels,labels[ind1])
        ctable[ind1,c]=1
    end
    t=tensors[ind1]
    labelt=labels[ind1]

    # construct direct product with other tensor components
    for pos=firsttensorpos+1:length(componentlist)
        component=componentlist[pos]
        ind1=component[1] # position of resulting tensor for this component (even when no contraction at all)
        while (c=findfirst(sum(ctable[component,:],1)))!=0 # findfirst guarantees that we perform contractions in specified order
            ind1,ind2=find(ctable[:,c])
            c=intersect(labels[ind1],labels[ind2])
            contractind1=indexin(c,labels[ind1])
            contractind2=indexin(c,labels[ind2])
            # this should always be tensors, so immediately put result in tensors[ind1]
            tensors[ind1]=contract(tensors[ind1],contractind1,tensors[ind2],contractind2)
            labels[ind1]=vcat(setdiff(labels[ind1],c),setdiff(labels[ind2],c))
            ctable[ind1,:]=0
            ctable[ind2,:]=0
            c=findin(clabels,labels[ind1])
            ctable[ind1,c]=1
        end
        t=tensorprod(t,tensors[ind1])
        labelt=vcat(labelt,labels[ind1])
    end
    t=scale!(t,scalar)

    # Final step: order indices of output tensor
    # Set contraction order
    if isempty(outputorder)
        outputorder=sort(labelt)
    end
    oorder=indexin(labelt,outputorder)
    p=sortperm(oorder)
    if oorder[p[1]]==0
        throw(ArgumentError("Output order does not specify order for all output labels"))
    end
    if p==1:numind(t)
        return t
    else
        return permuteind(t,p)
    end
end
# function mcontract(tensors::Vector{Any},labels::Vector{Any};outputorder::Vector{Int}=Array(Int,0),contractorder::Vector{Int}=Array(Int,0))
#     numtensors=length(tensors)
#     tensorsnew=Array(AbstractTensor,numtensors)
#     for i=1:numtensors
#         tensorsnew[i]=tensors[i]
#     end
#     return mcontract(tensorsnew,Vector{Int}[labels...];outputorder=outputorder,contractorder=contractorder)
# end
#
# function mcontractopt{TT<:AbstractTensor}(tensors::Vector{TT},labels::Vector{Vector{Int}})
#     # Finds the optimal contraction order for mcontract. Inputs tensors and labels
#     # should have the same format as the corresponding inputs of mcontract. Other
#     # inputs of mcontract are not required because contractorder will be determined
#     # and outputorder has no influence on the optimal contraction order (to first
#     # approximation). This routine does not do any actual contraction but only determines
#     # the optimal contraction order to be fed to mcontract.
#
#     # Process labels and do some error checking
#     numtensors=length(tensors)
#     if length(labels)!=numtensors
#         throw(ArgumentError("Number of label vectors does not match number of tensors"))
#     end
#     alllabels=vcat(labels...)
#     for i=1:numtensors
#         if numind(tensors[i])!=length(labels[i])
#             throw(LabelError("Length of labels[$i] does not match numind(tensors[$i])"))
#         end
#         if numind(tensors[i])!=length(unique(labels[i]))
#             throw(LabelError("Identical labels in labels[$i]: handle inner contraction first with trace"))
#         end
#     end
#     alllabels=unique(alllabels)
#     numlabels=length(alllabels)
#     sort!(alllabels,rev=true)
#     firstclabelpos=searchsortedfirst(alllabels,0,rev=true) # c is the first index in alllabels where alllabels[c] is no longer greater than zero
#     if firstclabelpos>numlabels || alllabels[firstclabelpos]==0 # make sure that first contraction label is not zero
#         throw(LabelError("Contraction labels should be nonzero"))
#     end
#     clabels=alllabels[firstclabelpos:end] # contraction labels
#
#     # Construct contraction table, analyze contraction graph and do some error checking
#     table=zeros(Int,(numtensors,numlabels))
#     for i=1:numtensors
#         ci=findin(alllabels,labels[i])
#         table[i,ci]=1
#     end
#     otable=table[:,1:firstclabelpos-1] # table of open labels
#     if any(sum(otable,1).!=1)
#         throw(LabelError("Positive labels indicate output dimensions and should appear only once"))
#     end
#     ctable=table[:,firstclabelpos:end] # table of contraction labels
#     if any(sum(ctable,1).!=2)
#         throw(LabelError("Negative labels indicate contractions and should appear exactly twice"))
#     end
#     adjacencymatrix=ctable*transpose(ctable)
#     # not really adjacencymatrix:
#     # diagonal elements tell number of contracted indices of that tensor
#     # off-diagonal elements tell number of contracted bonds between those two tensors
#     # onlything important is to have non-zero numbers between tensors that share a contraction
#     componentlist=connectedcomponents(bool(adjacencymatrix))
#     numindlist=map(component->sum(otable[component,:]),componentlist) # number of open indices of every component
#     p=sortperm(numindlist)
#     componentlist=componentlist[p]
#     numcomponent=length(componentlist)
#
#     # generate output structures
#     costlist=Array(Float64,numcomponent)
#     treelist=Array(Any,numcomponent)
#     orderlist=Array(Vector{Int},numcomponent)
#
#     # run over components
#     total=0
#     accept1=0
#     accept2=0
#     accept3=0
#     for c=1:numcomponent
#         # find optimal contraction for every component
#         component=sort(componentlist[c])
#         componentsize=length(component)
#         costdict=Array(Dict{Any,Float64},componentsize)
#         orderdict=Array(Dict{Any,Vector{Int}},componentsize)
#         treedict=Array(Dict{Any,Any},numtensors)
#         labeldict=Array(Dict{Any,Vector{Int}},componentsize)
#         spacedict=Array(Dict{Any,Any},componentsize)
#         for k=1:componentsize
#             costdict[k]=Dict{Any,Float64}()
#             orderdict[k]=Dict{Any,Vector{Int}}()
#             treedict[k]=Dict{Any,Any}()
#             labeldict[k]=Dict{Any,Vector{Int}}()
#             spacedict[k]=Dict{Any,Any}()
#         end
#
#         for i in component
#             costdict[1][[i]]=1
#             orderdict[1][[i]]=Int[]
#             treedict[1][[i]]=i
#             labeldict[1][[i]]=labels[i]
#             spacedict[1][[i]]=indspace(tensors[i])
#         end
#         for n=2:componentsize
#             println("Component $c of $numcomponent: partition size $n out of $componentsize tensors")
#             for k=1:div(n,2)
#                 ksets=sort(collect(keys(costdict[k])),order=Base.Order.Lexicographic,rev=true) # subset of k tensors
#                 if n-k==k
#                     nmksets=ksets
#                 else
#                     nmksets=sort(collect(keys(costdict[n-k])),order=Base.Order.Lexicographic,rev=true) # subset of n-k tensors
#                 end
#                 for i=1:length(ksets)
#                     for j=(n==2*k ? i+1 : 1):length(nmksets)
#                         total+=1
#                         if isempty(intersect(ksets[i],nmksets[j])) # only select subsets with no common tensors
#                             accept1+=1
#                             s1=ksets[i]
#                             s2=nmksets[j]
#                             label1=labeldict[k][s1]
#                             label2=labeldict[n-k][s2]
#                             clabels=intersect(label1,label2)
#                             if !isempty(clabels) # only select subsets with shared contraction lines
#                                 accept2+=1
#                                 selection=sort(vcat(s1,s2)) # new subset
#                                 space1=spacedict[k][s1]
#                                 space2=spacedict[n-k][s2]
#
#                                 indcontract1=indexin(clabels,label1)
#                                 indopen1=setdiff(1:length(label1),indcontract1)
#                                 indcontract2=indexin(clabels,label2)
#                                 indopen2=setdiff(1:length(label2),indcontract2)
#
#                                 cost=costdict[k][s1]+costdict[n-k][s2]+contractcost(space1,indcontract1,space2,indcontract2)
#                                 if !haskey(costdict[n],selection) || (haskey(costdict[n],selection) && costdict[n][selection]>cost)
#                                     accept3+=1
#                                     costdict[n][selection]=cost
#                                     orderdict[n][selection]=vcat(orderdict[k][s1],orderdict[n-k][s2],clabels)
#                                     treedict[n][selection]=(treedict[k][s1],treedict[n-k][s2])
#                                     labeldict[n][selection]=vcat(label1[indopen1],label2[indopen2])
#                                     spacedict[n][selection]=tuple(space1[indopen1]...,space2[indopen2]...)
#                                 end
#                             end
#                         end
#                     end
#                 end
#             end
#         end
#         costlist[c]=costdict[componentsize][component]
#         treelist[c]=treedict[componentsize][component]
#         orderlist[c]=orderdict[componentsize][component]
#     end
#
#     return orderlist,treelist,costlist,componentlist,total,accept1,accept2,accept3
# end
#
# function mcontractopt(tensors::Vector{Vector{Bool}},labels::Vector{Vector{Int}})
#     # Finds the optimal contraction order for mcontract. Inputs tensors can now be
#     # just vectors of bools, where true indicates that the corresponding index of the
#     # tensor has to be counted (e.g. a big virtual index D), whereas false indicates
#     # that the corresponding index can be ignored in the computation of the contraction
#     # cost (e.g. a small physical index d). All big indices (corresponding to true values)
#     # are assumed to have the same size and he reported contraction cost corresponds to
#     # the power of the polynomial scaling in the big index.
#
#     # Process labels and do some error checking
#     numtensors=length(tensors)
#     if length(labels)!=numtensors
#         throw(ArgumentError("Number of label vectors does not match number of tensors"))
#     end
#     alllabels=vcat(labels...)
#     for i=1:numtensors
#         if length(tensors[i])!=length(labels[i])
#             throw(LabelError("Length of labels[$i] does not match numind(tensors[$i])"))
#         end
#         if length(tensors[i])!=length(unique(labels[i]))
#             throw(LabelError("Identical labels in labels[$i]: handle inner contraction first with trace"))
#         end
#     end
#     alllabels=unique(alllabels)
#     numlabels=length(alllabels)
#     sort!(alllabels,rev=true)
#     firstclabelpos=searchsortedfirst(alllabels,0,rev=true) # c is the first index in alllabels where alllabels[c] is no longer greater than zero
#     if firstclabelpos>numlabels || alllabels[firstclabelpos]==0 # make sure that first contraction label is not zero
#         throw(LabelError("Contraction labels should be nonzero"))
#     end
#     clabels=alllabels[firstclabelpos:end] # contraction labels
#
#     # Construct contraction table, analyze contraction graph and do some error checking
#     table=zeros(Int,(numtensors,numlabels))
#     for i=1:numtensors
#         ci=findin(alllabels,labels[i])
#         table[i,ci]=1
#     end
#     otable=table[:,1:firstclabelpos-1] # table of open labels
#     if any(sum(otable,1).!=1)
#         throw(LabelError("Positive labels indicate output dimensions and should appear only once"))
#     end
#     ctable=table[:,firstclabelpos:end] # table of contraction labels
#     if any(sum(ctable,1).!=2)
#         throw(LabelError("Negative labels indicate contractions and should appear exactly twice"))
#     end
#     adjacencymatrix=ctable*transpose(ctable)
#     # not really adjacencymatrix:
#     # diagonal elements tell number of contracted indices of that tensor
#     # off-diagonal elements tell number of contracted bonds between those two tensors
#     # onlything important is to have non-zero numbers between tensors that share a contraction
#     componentlist=connectedcomponents(bool(adjacencymatrix))
#     numindlist=map(component->sum(otable[component,:]),componentlist) # number of open indices of every component
#     p=sortperm(numindlist)
#     componentlist=componentlist[p]
#     numcomponent=length(componentlist)
#
#     # generate output structures
#     costlist=Array(Int,numcomponent)
#     treelist=Array(Any,numcomponent)
#     orderlist=Array(Vector{Int},numcomponent)
#
#     # run over components
#     for c=1:numcomponent
#         # find optimal contraction for every component
#         component=sort(componentlist[c])
#         componentsize=length(component)
#         costdict=Array(Dict{Any,Int},componentsize)
#         orderdict=Array(Dict{Any,Vector{Int}},componentsize)
#         treedict=Array(Dict{Any,Any},numtensors)
#         labeldict=Array(Dict{Any,Vector{Int}},componentsize)
#         spacedict=Array(Dict{Any,Any},componentsize)
#         for k=1:componentsize
#             costdict[k]=Dict{Any,Int}()
#             orderdict[k]=Dict{Any,Vector{Int}}()
#             treedict[k]=Dict{Any,Any}()
#             labeldict[k]=Dict{Any,Vector{Int}}()
#             spacedict[k]=Dict{Any,Any}()
#         end
#
#         for i in component
#             costdict[1][[i]]=1
#             orderdict[1][[i]]=Int[]
#             treedict[1][[i]]=i
#             labeldict[1][[i]]=labels[i]
#             spacedict[1][[i]]=tensors[i]
#         end
#         for n=2:componentsize
#             println("Component $c of $numcomponent: partition size $n out of $componentsize tensors")
#             for k=1:div(n,2)
#                 ksets=sort(collect(keys(costdict[k])),order=Base.Order.Lexicographic,rev=true) # subset of k tensors
#                 if n-k==k
#                     nmksets=ksets
#                 else
#                     nmksets=sort(collect(keys(costdict[n-k])),order=Base.Order.Lexicographic,rev=true) # subset of n-k tensors
#                 end
#                 for i=1:length(ksets)
#                     for j=(n==2*k ? i+1 : 1):length(nmksets)
#                         if isempty(intersect(ksets[i],nmksets[j])) # only select subsets with no common tensors
#                             s1=ksets[i]
#                             s2=nmksets[j]
#                             label1=labeldict[k][s1]
#                             label2=labeldict[n-k][s2]
#                             clabels=intersect(label1,label2)
#                             if !isempty(clabels) # only select subsets with shared contraction lines
#                                 selection=sort(vcat(s1,s2)) # new subset
#                                 space1=spacedict[k][s1]
#                                 space2=spacedict[n-k][s2]
#
#                                 indcontract1=indexin(clabels,label1)
#                                 indopen1=setdiff(1:length(label1),indcontract1)
#                                 indcontract2=indexin(clabels,label2)
#                                 indopen2=setdiff(1:length(label2),indcontract2)
#
#                                 cost=sum(space1)+sum(space2)-sum(space1[indcontract1])
#                                 cost=max(cost,costdict[k][s1],costdict[n-k][s2])
#                                 if !haskey(costdict[n],selection) || (haskey(costdict[n],selection) && costdict[n][selection]>cost)
#                                     costdict[n][selection]=cost
#                                     orderdict[n][selection]=vcat(orderdict[k][s1],orderdict[n-k][s2],clabels)
#                                     treedict[n][selection]=(treedict[k][s1],treedict[n-k][s2])
#                                     labeldict[n][selection]=vcat(label1[indopen1],label2[indopen2])
#                                     spacedict[n][selection]=tuple(space1[indopen1]...,space2[indopen2]...)
#                                 end
#                             end
#                         end
#                     end
#                 end
#             end
#         end
#         costlist[c]=costdict[componentsize][component]
#         treelist[c]=treedict[componentsize][component]
#         orderlist[c]=orderdict[componentsize][component]
#     end
#
#     return orderlist,treelist,costlist,componentlist
# end
#
# mcontractopt(tensors::Vector{Any},labels::Vector{Any})=mcontractopt(mapreduce(typeof,typejoin,tensors)[tensors...],Vector{Int}[labels...])
#
#
#
# function mcontractopt2(tensors::Vector{Vector{Bool}},labels::Vector{Vector{Int}};maxcost::Int=200)
#     # Finds the optimal contraction order for mcontract. Inputs tensors can now be
#     # just vectors of bools, where true indicates that the corresponding index of the
#     # tensor has to be counted (e.g. a big virtual index D), whereas false indicates
#     # that the corresponding index can be ignored in the computation of the contraction
#     # cost (e.g. a small physical index d). All big indices (corresponding to true values)
#     # are assumed to have the same size and he reported contraction cost corresponds to
#     # the power of the polynomial scaling in the big index.
#
#     # Process labels and do some error checking
#     numtensors=length(tensors)
#     if length(labels)!=numtensors
#         throw(ArgumentError("Number of label vectors does not match number of tensors"))
#     end
#     alllabels=vcat(labels...)
#     for i=1:numtensors
#         if length(tensors[i])!=length(labels[i])
#             throw(LabelError("Length of labels[$i] does not match numind(tensors[$i])"))
#         end
#         if length(tensors[i])!=length(unique(labels[i]))
#             throw(LabelError("Identical labels in labels[$i]: handle inner contraction first with trace"))
#         end
#     end
#     alllabels=unique(alllabels)
#     numlabels=length(alllabels)
#     sort!(alllabels,rev=true)
#     firstclabelpos=searchsortedfirst(alllabels,0,rev=true) # c is the first index in alllabels where alllabels[c] is no longer greater than zero
#     if firstclabelpos>numlabels || alllabels[firstclabelpos]==0 # make sure that first contraction label is not zero
#         throw(LabelError("Contraction labels should be nonzero"))
#     end
#     clabels=alllabels[firstclabelpos:end] # contraction labels
#
#     # Construct contraction table, analyze contraction graph and do some error checking
#     table=zeros(Int,(numtensors,numlabels))
#     for i=1:numtensors
#         ci=findin(alllabels,labels[i])
#         table[i,ci]=1
#     end
#     otable=table[:,1:firstclabelpos-1] # table of open labels
#     if any(sum(otable,1).!=1)
#         throw(LabelError("Positive labels indicate output dimensions and should appear only once"))
#     end
#     ctable=table[:,firstclabelpos:end] # table of contraction labels
#     if any(sum(ctable,1).!=2)
#         throw(LabelError("Negative labels indicate contractions and should appear exactly twice"))
#     end
#     adjacencymatrix=ctable*transpose(ctable)
#     # not really adjacencymatrix:
#     # diagonal elements tell number of contracted indices of that tensor
#     # off-diagonal elements tell number of contracted bonds between those two tensors
#     # onlything important is to have non-zero numbers between tensors that share a contraction
#     componentlist=connectedcomponents(bool(adjacencymatrix))
#     numindlist=map(component->sum(otable[component,:]),componentlist) # number of open indices of every component
#     p=sortperm(numindlist)
#     componentlist=componentlist[p]
#     numcomponent=length(componentlist)
#
#     # generate output structures
#     costlist=Array(Int,numcomponent)
#     treelist=Array(Any,numcomponent)
#     orderlist=Array(Vector{Int},numcomponent)
#
#     # run over components
#     for c=1:numcomponent
#         # find optimal contraction for every component
#         component=sort(componentlist[c])
#         componentsize=length(component)
#         costdict=Array(Dict{Any,Int},componentsize)
#         orderdict=Array(Dict{Any,Vector{Int}},componentsize)
#         treedict=Array(Dict{Any,Any},numtensors)
#         labeldict=Array(Dict{Any,Vector{Int}},componentsize)
#         spacedict=Array(Dict{Any,Any},componentsize)
#         for k=1:componentsize
#             costdict[k]=Dict{Any,Int}()
#             orderdict[k]=Dict{Any,Vector{Int}}()
#             treedict[k]=Dict{Any,Any}()
#             labeldict[k]=Dict{Any,Vector{Int}}()
#             spacedict[k]=Dict{Any,Any}()
#         end
#
#         for i in component
#             costdict[1][[i]]=0
#             orderdict[1][[i]]=Int[]
#             treedict[1][[i]]=i
#             labeldict[1][[i]]=labels[i]
#             spacedict[1][[i]]=tensors[i]
#         end
#
#         currentcost=1
#         nextcost=maxcost
#         while nextcost<=maxcost
#             nextcost=maxcost
#             for n=2:componentsize
#                 println("Component $c of $numcomponent: current cost = $currentcost, partition size $n out of $componentsize tensors")
#                 for k=1:div(n,2)
#                     ksets=sort(collect(keys(costdict[k])),order=Base.Order.Lexicographic,rev=true) # subset of k tensors
#                     if n-k==k
#                         nmksets=ksets
#                     else
#                         nmksets=sort(collect(keys(costdict[n-k])),order=Base.Order.Lexicographic,rev=true) # subset of n-k tensors
#                     end
#                     for i=1:length(ksets)
#                         for j=(n==2*k ? i+1 : 1):length(nmksets)
#                             if isempty(intersect(ksets[i],nmksets[j])) # only select subsets with no common tensors
#                                 s1=ksets[i]
#                                 s2=nmksets[j]
#                                 label1=labeldict[k][s1]
#                                 label2=labeldict[n-k][s2]
#                                 clabels=intersect(label1,label2)
#                                 if !isempty(clabels) # only select subsets with shared contraction lines
#                                     selection=sort(vcat(s1,s2)) # new subset
#                                     space1=spacedict[k][s1]
#                                     space2=spacedict[n-k][s2]
#
#                                     indcontract1=indexin(clabels,label1)
#                                     indopen1=setdiff(1:length(label1),indcontract1)
#                                     indcontract2=indexin(clabels,label2)
#                                     indopen2=setdiff(1:length(label2),indcontract2)
#
#                                     cost=sum(space1)+sum(space2)-sum(space1[indcontract1])
#                                     cost=max(cost,costdict[k][s1],costdict[n-k][s2])
#                                     if cost<=currentcost
#                                         if !haskey(costdict[n],selection) || (haskey(costdict[n],selection) && costdict[n][selection]>cost)
#                                             costdict[n][selection]=cost
#                                             orderdict[n][selection]=vcat(orderdict[k][s1],orderdict[n-k][s2],clabels)
#                                             treedict[n][selection]=(treedict[k][s1],treedict[n-k][s2])
#                                             labeldict[n][selection]=vcat(label1[indopen1],label2[indopen2])
#                                             spacedict[n][selection]=tuple(space1[indopen1]...,space2[indopen2]...)
#                                         end
#                                     elseif cost<nextcost
#                                         nextcost=cost
#                                     end
#                                 end
#                             end
#                         end
#                     end
#                 end
#             end
#             currentcost=nextcost
#             if !isempty(costdict[componentsize])
#                 break
#             end
#         end
#         costlist[c]=costdict[componentsize][component]
#         treelist[c]=treedict[componentsize][component]
#         orderlist[c]=orderdict[componentsize][component]
#     end
#
#     return orderlist,treelist,costlist,componentlist
# end
#
# function connectedcomponents(A::AbstractMatrix{Bool})
#     # connectedcomponents(A::AbstractMatrix{Bool})
#     #
#     # For a given adjacency matrix of size n x n, connectedcomponents returns
#     # a list componentlist that contains integer vectors, where every integer
#     # vector groups the indices of the vertices of a connected component of the
#     # graph encoded by A. The number of connected components is given by
#     # length(componentlist).
#     #
#     # Used as auxiliary function to analyze contraction graph in contract.
#
#     n=size(A,1)
#     assert(size(A,2)==n)
#
#     componentlist=Array(Vector{Int},0)
#     assignedlist=falses((n,))
#
#     for i=1:n
#         if !assignedlist[i]
#             assignedlist[i]=true
#             checklist=[i]
#             currentcomponent=[i]
#             while !isempty(checklist)
#                 j=pop!(checklist)
#                 for k=find(A[j,:])
#                     if !assignedlist[k]
#                         push!(currentcomponent,k)
#                         push!(checklist,k)
#                         assignedlist[k]=true;
#                     end
#                 end
#             end
#             push!(componentlist,currentcomponent)
#         end
#     end
#     return componentlist
# end
#
