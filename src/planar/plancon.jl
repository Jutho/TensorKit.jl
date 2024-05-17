"""
    plancon(tensorlist, indexlist, [conjlist]; order = ..., output = ...)

Contract the tensors in `tensorlist` (of type `Vector` or `Tuple`) according to the network
as specified by `indexlist`. Here, `indexlist` is a list (i.e. a `Vector` or `Tuple`) with
the same length as `tensorlist` whose entries are themselves lists (preferably
`Vector{Int}`) where every integer entry provides a label for corresponding index/dimension
of the corresponding tensor in `tensorlist`. Positive integers are used to label indices
that need to be contracted, and such thus appear in two different entries within
`indexlist`, whereas negative integers are used to label indices of the output tensor, and
should appear only once.

Optional arguments in another list with the same length, `conjlist`, whose entries are of
type `Bool` and indicate whether the corresponding tensor object should be conjugated
(`true`) or not (`false`). The default is `false` for all entries.

By default, contractions are performed in the order such that the indices being contracted
over are labelled by increasing integers, i.e. first the contraction corresponding to label
`1` is performed. The output tensor had an index order corresponding to decreasing
(negative, so increasing in absolute value) index labels. The keyword arguments `order` and
`output` allow to change these defaults.

See also the macro version [`@planar`](@ref).
"""
function plancon(tensors, network,
                 conjlist=fill(false, length(tensors));
                 order=nothing, output=nothing)
    length(tensors) == length(network) == length(conjlist) ||
        throw(ArgumentError("number of tensors and of index lists should be the same"))
    # TO.isnconstyle(network) || throw(ArgumentError("invalid NCON network: $network"))
    output′ = planconoutput(network, output)

    if length(tensors) == 1
        error("not implemented")
        if length(output′) == length(network[1])
            return tensorcopy(output′, tensors[1], network[1], conjlist[1] ? :C : :N)
        else
            return tensortrace(output′, tensors[1], network[1], conjlist[1] ? :C : :N)
        end
    end

    (tensors, network) = TO.resolve_traces(tensors, network)
    tree = order === nothing ? plancontree(network) : TO.indexordertree(network, order)
    return planarcontracttree(tensors, network, conjlist, tree, output′)
end

# single tensor case
function planarcontracttree(tensors, network, conjlist, tree::Int, output)
    # extract data
    A = tensors[tree]
    IA = canonicalize_labels(A, network[tree])
    conjA = conjlist[tree] ? :C : :N

    pA, qA = planartrace_indices(IA, conjA, output)

    if isempty(qA[1]) # no traced indices
        return planarcopy(A, pA, conjA)
        C = tensoralloc_add(scalartype(A), pA, A, conjA)
        return planaradd!(C, A, pA, conjA, One(), Zero())
    else
        return planartrace(A, pA, qA, conjA)
    end
end

# recursive case
function planarcontracttree(tensors, network, conjlist, tree::Int)
    # extract data
    A = tensors[tree]
    IA = canonicalize_labels(A, network[tree])
    conjA = conjlist[tree] ? :C : :N

    pA, qA = planartrace_indices(IA, conjA)

    if isempty(qA[1]) # no traced indices
        C = A
        IC = IA
        conjC = conjA
    else
        C = planartrace(A, pA, qA, conjA)
        IC = (TupleTools.getindices(linearize(IA), pA[1]),
              TupleTools.getindices(linearize(IA), pA[2]))
        conjC = :N
    end

    return C, IC, conjC
end

function planarcontracttree(tensors, network, conjlist, tree)
    @assert !(tree isa Int) "single-node tree should already have been handled"
    A, IA, CA = planarcontracttree(tensors, network, conjlist, tree[1])
    B, IB, CB = planarcontracttree(tensors, network, conjlist, tree[2])
    pA, pB, pAB = planarcontract_indices(IA, CA, IB, CB)

    C = planarcontract(A, pA, CA, B, pB, CB, pAB)

    # deduce labels of C
    IAB = (TupleTools.getindices(linearize(IA), pA[1])...,
           TupleTools.getindices(linearize(IB), pB[2])...)
    IC = (TupleTools.getindices(IAB, pAB[1]), TupleTools.getindices(IAB, pAB[2]))

    return C, IC, :N
end
# special case for last step -- dispatch on output argument
function planarcontracttree(tensors, network, conjlist, tree, output)
    @assert !(tree isa Int) "single-node tree should already have been handled"
    A, IA, CA = planarcontracttree(tensors, network, conjlist, tree[1])
    B, IB, CB = planarcontracttree(tensors, network, conjlist, tree[2])
    pA, pB, pAB = planarcontract_indices(IA, CA, IB, CB, output)

    return planarcontract(A, pA, CA, B, pB, CB, pAB)
end

function planconoutput(network, output::Union{Nothing,Tuple{Tuple,Tuple}})
    outputindices = Vector{Int}()
    for n in network
        for k in n
            if k < 0
                push!(outputindices, k)
            end
        end
    end
    isnothing(output) && return (tuple(sort(outputindices; rev=true)...), ())

    issetequal(TO.linearize(output), outputindices) ||
        throw(ArgumentError("invalid NCON network: $network -> $output"))
    return output
end

function plancontree(network)
    contractionindices = Vector{Vector{Int}}(undef, length(network))
    for k in 1:length(network)
        indices = network[k]
        # trace indices have already been removed, remove open indices by filtering on positive values
        contractionindices[k] = Base.filter(>(0), indices)
    end
    partialtrees = collect(Any, 1:length(network))
    return TO._ncontree!(partialtrees, contractionindices)
end
