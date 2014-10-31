include("bonddimension.jl")

function optimizecontract(tensors::AbstractVector,labels::AbstractVector;maximalcost=-1,initialcost=0)
    length(tensors)==length(labels) || throw("number of tensors and number of labels don't match")
    numtensors=length(tensors)
    numtensors > 1 || throw(ArgumentError("a tensor network requires at least two tensors"))
    
    # determine type of spaces and of cost
    if all(x->isa(x,Tuple),tensors)
        T=mapreduce(x->promote_type(typeof(x)...),promote_type,tensors)
        if T<:Number || T<:Power
            P=(T,T...)
            T=typeof(T(0)+T(0))
            spaces=Array(P,numtensors)
            for n=1:numtensors
                spaces[n]=tensors[n]
            end
        else
            throw(ArgumentError("Unrecognized type of tensors"))
        end
    elseif all(x->isa(x,AbstractTensor),tensors)
        TT=mapreduce(typeof,promote_type,tensors)
        P=tensortype(TT)
        spaces=Array(P,numtensors)
        for n=1:numtensors
            spaces[n]=space(tensors[n])
        end
        T=Float64
    elseif all(x->isa(x,TensorSpace),tensors)
        P=mapreduce(typeof,promote_type,tensors)
        spaces=Array(P,numtensors)
        for n=1:numtensors
            spaces[n]=tensors[n]
        end
        T=Float64
    else
        throw(ArgumentError("Unrecognized type of tensors"))
    end
    
    # determine minimal and maximal cost
    if maximalcost==-1
        if T<:Poly
            maxcost=T(1,typemax(Int))
        else
            maxcost=typemax(T)
        end
    else
        maxcost=T(maximalcost)
    end
    if initialcost==0
        initcost=T(0)
    else
        initcost=T(initialcost)
    end
    
    return _optimizecontract(spaces,labels,initcost,maxcost)
end

# Auxiliary:
type SubGraph{T}
    tensorset::BitVector
    labelset::BitVector
    dualset::BitVector
    tree::Union(Int,Tuple)
    cost::T
end

function _optimizecontract{P,T}(spaces::Vector{P},labels::AbstractVector,initcost::T,maxcost::T)
    numtensors=length(labels)
    
    # Analyze contraction graph and detect components via adjacency matrix
    alllabels=unique(vcat(labels...))
    numlabels=length(alllabels)
    labelsets=Array(BitVector,numtensors) # map labels of tensor n to integers i and store in IntSet for faster set operations
    dualsets=Array(BitVector,numtensors) # store whether labelspaces[i] holds the space or its dual for tensor n
    
    tabletensor=zeros(Int,(numlabels,2))
    tableindex=zeros(Int,(numlabels,2))
    
    adjacencymatrix=falses(numtensors,numtensors)
    
    for n=1:numtensors
        length(labels[n])==length(unique(labels[n])) || throw(ArgumentError("duplicate labels for tensor $n: handle inner contraction first with tensortrace"))
        indn=findin(alllabels,labels[n])
        labelsets[n]=falses(numlabels)
        labelsets[n][indn]=true
        dualsets[n]=falses(numlabels)
        for i in indn
            if tabletensor[i,1]==0
                tabletensor[i,1]=n
                tableindex[i,1]=findfirst(labels[n],alllabels[i])
            elseif tabletensor[i,2]==0
                tabletensor[i,2]=n
                tableindex[i,2]=findfirst(labels[n],alllabels[i])
                n1=tabletensor[i,1]
                ind1=tableindex[i,1]
                ind2=tableindex[i,2]
                if isa(spaces[n1][ind1],Number) || isa(spaces[n1][ind1],Power)
                    spaces[n1][ind1]==spaces[n][ind2] || throw(ArgumentError("spaces don't match for label $(alllabels[i])"))
                else
                    spaces[n1][ind1]==dual(spaces[n][ind2]) || throw(ArgumentError("spaces don't match for label $(alllabels[i])"))
                end
                adjacencymatrix[n1,n]=true
                adjacencymatrix[n,n1]=true
                dualsets[n][i]=true
            else
                throw(ArgumentError("no label should appear more than two times"))
            end
        end
    end
    componentlist=connectedcomponents(adjacencymatrix)
    numcomponent=length(componentlist)
    
    # construct a list of the spaces of all labels
    labelspaces=P(ntuple(numlabels,j->spaces[tabletensor[j,1]][tableindex[j]]))

    # generate output structures
    costlist=Array(T,numcomponent)
    treelist=Array(Tuple,numcomponent)
    labellist=Array(BitVector,numcomponent)

    # compute cost and optimal contraction order for every component
    for c=1:numcomponent
        # find optimal contraction for every component
        componentsize=length(componentlist[c])
        setcost=Array(Dict{BitVector,SubGraph{T}},componentsize)
        for k=1:componentsize
            setcost[k]=Dict{BitVector,SubGraph{T}}()
        end

        for n in componentlist[c]
            tensorset=falses(numtensors)
            tensorset[n]=true
            setcost[1][tensorset]=SubGraph{T}(tensorset,labelsets[n],dualsets[n],n,T(0))
        end
        currentcost=initcost
        while true
            nextcost=maxcost+1 # corresponds to uninitialized value
            for n=2:componentsize
                println("Component $c of $numcomponent: current cost = $currentcost, partition size $n out of $componentsize tensors")
                for k=1:div(n,2)
                    ksets=collect(values(setcost[k])) # subset of k tensors
                    if n-k==k
                        nmksets=ksets
                    else
                        nmksets=collect(values(setcost[n-k])) # subset of n-k tensors
                    end
                    for i=1:length(ksets)
                        for j=(n==2*k ? i+1 : 1):length(nmksets)
                            sc1=ksets[i]
                            sc2=nmksets[j]
                            if !anyand(sc1.tensorset,sc2.tensorset) # only select subsets with no common tensors
                                if anyand(sc1.labelset,sc2.labelset) # only select subsets with shared contraction lines
                                    clabels=sc1.labelset & sc2.labelset
                                    olabels1=sc1.labelset $ clabels
                                    olabels2=sc2.labelset $ clabels
                                    
                                    cost=sc1.cost+sc2.cost+contractcost(labelspaces,olabels1,olabels2,clabels,sc1.dualset,sc2.dualset)
                                    if cost<=currentcost
                                        tensorset=sc1.tensorset | sc2.tensorset
                                        if !haskey(setcost[n],tensorset)
                                            labelset=olabels1 | olabels2
                                            dualset=(sc1.dualset | sc2.dualset) & labelset
                                            setcost[n][tensorset]=SubGraph{T}(tensorset,labelset,dualset,(sc1.tree,sc2.tree),cost)
                                        else
                                            sc=setcost[n][tensorset]
                                            if sc.cost>cost
                                                sc.tree=(sc1.tree,sc2.tree)
                                                sc.cost=cost
                                            end
                                        end
                                    elseif cost<nextcost
                                        nextcost=cost
                                    end
                                end
                            end
                        end
                    end
                end
            end
            if nextcost>maxcost
                error("no solution with cost < maximalcost = $maxcost was found")
            end
            if !isempty(setcost[componentsize])
                break
            end
            currentcost=min(2*nextcost,maxcost)
        end
        component=falses(numtensors)
        component[componentlist[c]]=true
        sc=setcost[componentsize][component]
        costlist[c]=sc.cost
        treelist[c]=sc.tree
        labellist[c]=sc.labelset
    end
    
    # collect different components
    totaltree=treelist[1]
    totalcost=costlist[1]
    outputlabels=labellist[1]
    
    for c=2:numcomponent
        totaltree=(totaltree,treelist[c])
        totalcost=totalcost+costlist[c]+contractcost(labelspaces,outputlabels,labellist[c],IntSet(),IntSet(),IntSet())
        outputlabels=union(outputlabels,labellist[c])
    end
    
    return totaltree, totalcost
end

@inline function anyand(a::BitArray,b::BitArray)
    for n in 1:length(a)
        a[n] && b[n] && return true
    end
    return false
end
@inline function contractcost{D}(spaces::(Power{D,Int},Power{D,Int}...),olabels1::BitVector,olabels2::BitVector,clabels::BitVector,dualset1::BitVector,dualset2::BitVector)
    cost=Power{D,Int}(1,0)
    for n in find(olabels1 | olabels2 | clabels)
        cost*=spaces[n]
    end
    return cost
end
@inline function contractcost{T<:Number}(spaces::(T,T...),olabels1::BitVector,olabels2::BitVector,clabels::BitVector,dualset1::BitVector,dualset2::BitVector)
    cost=T(0)
    for n in find(olabels1 | olabels2 | clabels)
        cost*=spaces[n]
    end
    return cost
end


function connectedcomponents(A::AbstractMatrix{Bool})
    # connectedcomponents(A::AbstractMatrix{Bool})
    #
    # For a given adjacency matrix of size n x n, connectedcomponents returns
    # a list componentlist that contains integer vectors, where every integer
    # vector groups the indices of the vertices of a connected component of the
    # graph encoded by A. The number of connected components is given by
    # length(componentlist).
    #
    # Used as auxiliary function to analyze contraction graph.

    n=size(A,1)
    assert(size(A,2)==n)

    componentlist=Array(Vector{Int},0)
    assignedlist=falses((n,))

    for i=1:n
        if !assignedlist[i]
            assignedlist[i]=true
            checklist=[i]
            currentcomponent=[i]
            while !isempty(checklist)
                j=pop!(checklist)
                for k=1:n
                    if A[j,k] && !assignedlist[k]
                        push!(currentcomponent,k)
                        push!(checklist,k)
                        assignedlist[k]=true;
                    end
                end
            end
            push!(componentlist,currentcomponent)
        end
    end
    return componentlist
end

