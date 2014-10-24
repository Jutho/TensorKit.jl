using Polynomials

# immutable TensorNetworkSubset
#     set::IntSet
#     cost::

function optimizecontract{P<:Union((Poly{Int}...),TensorSpace),X<:AbstractVector}(spaces::Vector{P},labels::Vector{X};maxcost::Number=0)
    # Finds the optimal contraction order
    
    length(spaces)==length(labels) || throw(ArgumentError("number of label vectors doesn't match number of tensor spaces"))

    # # Convert labels to integers and label vectors to IntSet for faster set operations
    # intlabels=Array(IntSet,length(labels))
    # alllabels=unique(vcat(labels...))
    #
    # for n=1:length(labels)
    #     intlabels[n]=IntSet(indexin(labels[n],alllabels))
    # end
    
    spaceT=(P<:(Poly{Int}...)) ? (Poly{Int}...) : P
    costT=(P<:(Poly{Int}...) ? Poly{Int} : Float64)
    local maximalcost::costT
    if costT==Float64
        maximalcost=(maxcost==0 ? Inf : convert(Float64,maxcost))
        currentcost=(maxcost==0 ? 1 : maximalcost)
    else
        maxpoly=zeros(Int,maxcost==0 ? 101 : convert(Int,maxcost+1))
        maxpoly[end]=1
        maximalcost=Poly(maxpoly)
        currentcost=(maxcost==0 ? one(Poly{Int}) : maximalcost)
    end
        
    # Build contraction graph via adjacency matrix and detect components
    numtensors=length(spaces)
    adjacencymatrix=BitArray(numtensors,numtensors)
    for n1 = 1:numtensors
        for n2 = n1+1:numtensors
            adjacencymatrix[n1,n2] = !isempty(intersect(labels[n1],labels[n2]))
            adjacencymatrix[n2,n1] = adjacencymatrix[n1,n2]
        end
    end
    componentlist=connectedcomponents(adjacencymatrix)
    numcomponent=length(componentlist)

    # generate output structures
    costlist=Array(Float64,numcomponent)
    treelist=Array(Tuple,numcomponent)

    # compute cost and optimal contraction order for every component
    for c=1:numcomponent
        # find optimal contraction for every component
        component=IntSet(componentlist[c])
        componentsize=length(component)
        costdict=Array(Dict{IntSet,costT},componentsize)
        treedict=Array(Dict{IntSet,Any},numtensors)
        labeldict=Array(Dict{IntSet,Vector{L}},componentsize)
        spacedict=Array(Dict{IntSet,spaceT},componentsize)
        adjacentdict=Array(Dict{IntSet,IntSet},componentsize)
        for k=1:componentsize
            costdict[k]=Dict{IntSet,costT}()
            treedict[k]=Dict{IntSet,Any}()
            labeldict[k]=Dict{IntSet,Vector{L}}()
            spacedict[k]=Dict{IntSet,spaceT}()
            adjacentdict[k]=Dict{IntSet,IntSet}()
        end

        for i in component
            set=IntSet(i)
            costdict[1][set]=zero(costT)
            treedict[1][set]=i
            labeldict[1][set]=labels[i]
            spacedict[1][set]=spaces[i]
            adjacentdict[1][set]=IntSet(find(adjacencymatrix[:,i]))
        end
        nextcost=maximalcost
        while nextcost<=maximalcost
            nextcost=maximalcost
            for n=2:componentsize
                println("Component $c of $numcomponent: current cost = $currentcost, partition size $n out of $componentsize tensors")
                for k=1:div(n,2)
                    ksets=collect(keys(costdict[k])) # subset of k tensors
                    if n-k==k
                        nmksets=ksets
                    else
                        nmksets=collect(keys(costdict[n-k])) # subset of n-k tensors
                    end
                    for i=1:length(ksets)
                        for j=(n==2*k ? i+1 : 1):length(nmksets)
                            if isempty(intersect(ksets[i],nmksets[j])) # only select subsets with no common tensors
                                s1=ksets[i]
                                s2=nmksets[j]
                                if !isempty(intersect(s2,adjacentdict[k][s1])) # only select subsets with shared contraction lines
                                    labels1=labeldict[k][s1]
                                    labels2=labeldict[n-k][s2]
                                    space1=spacedict[k][s1]
                                    space2=spacedict[n-k][s2]
                                    
                                    cost=costdict[k][s1]+costdict[n-k][s2]+contractcost(space1,labels1,space2,labels2)
                                    if cost<=currentcost
                                        selection=union(s1,s2)
                                        if !haskey(costdict[n],selection) || costdict[n][selection]>cost
                                            costdict[n][selection]=cost
                                            treedict[n][selection]=(treedict[k][s1],treedict[n-k][s2])
                                            
                                            openlabels=symdiff(labels1,labels2)
                                            labeldict[n][selection]=openlabels
                                            spacedict[n][selection]=joinspaces(space1,space2)[indexin(openlabels,vcat(labels1,labels2))]
                                            adjacentdict[n][selection]=union(adjacentdict[k][s1],adjacentdict[n-k][s2])
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
            if !isempty(costdict[componentsize])
                break
            end
            currentcost=min(2*nextcost,maximalcost)
        end
        costlist[c]=costdict[componentsize][component]
        treelist[c]=treedict[componentsize][component]
        labellist[c]=labeldict[componentsize][component]
        spacelist[c]=spacedict[componentsize][component]
    end
    
    # collect different components
    contracttree=treelist[1]
    cost=costlist[1]
    outputlabels=labellist[1]
    outputspace=spacelist[1]
    for c=2:numcomponent
        contracttree=(contracttree,treelist[c])
        cost=cost+costlist[c]+contractcost(outputspace,outputlabels,spacelist[c],labellist[c])
        outputlabels=vcat(outputlabels,labellist[c])
        outputspace=outputspace ⊗ spacelist[c]
    end
    
    return contracttree, cost, outputspace, outputlabels
end

# work with polynomial sizes
Base.isless(p1::Poly,p2::Poly)=(length(p1)<length(p2) || length(p1)==length(p2) && p1[end]<p2[end])
function contractcost{L}(space1::(Poly{Int}...),labels1::Vector{L},space2::(Poly{Int}...),labels2::Vector{L})
    clabels=intersect(labels1,labels2)
    cind1=indexin(clabels,labels1)
    cind2=indexin(clabels,labels2)
    oind1=setdiff(1:length(space1),cind1)
    oind2=setdiff(1:length(space2),cind2)
    cost=one(Poly{Int})
    for i in oind1
        cost*=space1[i]
    end
    for i in cind1
        cost*=space1[i]
    end
    for i in oind2
        cost*=space2[i]
    end
    return cost
end

joinspaces{P<:TensorSpace}(space1::P,space2::P) = space1 ⊗ space2
joinspaces(space1::(Poly{Int}...),space2::(Poly{Int}...)) = tuple(space1...,space2...)

# Auxiliary:
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

