"""
    TreeTransformer

Supertype for structures containing the data for a tree transformation.
"""
abstract type TreeTransformer end

struct TrivialTreeTransformer <: TreeTransformer end

const _AbelianTransformerData{T,N} = Tuple{T,StridedStructure{N},StridedStructure{N}}

struct AbelianTreeTransformer{T,N} <: TreeTransformer
    data::Vector{_AbelianTransformerData{T,N}}
end

function AbelianTreeTransformer(transform, p, Vdst, Vsrc)
    permute(Vsrc, p) == Vdst || throw(SpaceMismatch("Incompatible spaces for permuting."))
    structure_dst = fusionblockstructure(Vdst)
    structure_src = fusionblockstructure(Vsrc)

    L = length(structure_src.fusiontreelist)
    T = sectorscalartype(sectortype(Vdst))
    N = numind(Vsrc)
    data = Vector{Tuple{T,StridedStructure{N},StridedStructure{N}}}(undef, L)

    for i in 1:L
        f₁, f₂ = structure_src.fusiontreelist[i]
        (f₃, f₄), coeff = only(transform(f₁, f₂))
        j = structure_dst.fusiontreeindices[(f₃, f₄)]
        stridestructure_dst = structure_dst.fusiontreestructure[j]
        stridestructure_src = structure_src.fusiontreestructure[i]
        data[i] = (coeff, stridestructure_dst, stridestructure_src)
    end

    return AbelianTreeTransformer(data)
end

const _GenericTransformerData{T,N} = Tuple{Matrix{T},Vector{StridedStructure{N}},
                                           Vector{StridedStructure{N}}}

struct GenericTreeTransformer{T,N} <: TreeTransformer
    data::Vector{_GenericTransformerData{T,N}}
end

function GenericTreeTransformer(transform, p, Vdst, Vsrc)
    permute(Vsrc, p) == Vdst || throw(SpaceMismatch("Incompatible spaces for permuting."))
    structure_dst = fusionblockstructure(Vdst)
    structure_src = fusionblockstructure(Vsrc)
    I = sectortype(Vsrc)

    uncoupleds_src = map(structure_src.fusiontreelist) do (f₁, f₂)
        return TupleTools.vcat(f₁.uncoupled, f₂.uncoupled)
    end
    uncoupleds_src_unique = unique(uncoupleds_src)

    uncoupleds_dst = map(structure_dst.fusiontreelist) do (f₁, f₂)
        return TupleTools.vcat(f₁.uncoupled, f₂.uncoupled)
    end

    T = sectorscalartype(I)
    N = numind(Vdst)
    L = length(uncoupleds_src_unique)
    data = Vector{_GenericTransformerData{T,N}}(undef, L)

    # TODO: this can be multithreaded
    for (i, uncoupled) in enumerate(uncoupleds_src_unique)
        ids_src = findall(==(uncoupled), uncoupleds_src)
        fusiontrees_outer_src = structure_src.fusiontreelist[ids_src]

        uncoupled_dst = TupleTools.getindices(uncoupled, (p[1]..., p[2]...))
        ids_dst = findall(==(uncoupled_dst), uncoupleds_dst)

        fusiontrees_outer_dst = structure_dst.fusiontreelist[ids_dst]

        matrix = zeros(sectorscalartype(I), length(ids_dst), length(ids_src))
        for (row, (f₁, f₂)) in enumerate(fusiontrees_outer_src)
            for ((f₃, f₄), coeff) in transform(f₁, f₂)
                col = findfirst(==((f₃, f₄)), fusiontrees_outer_dst)::Int
                matrix[row, col] = coeff
            end
        end
        @debug("Created recoupling block for uncoupled: $uncoupled",
               sz = size(matrix), sparsity = count(!iszero, matrix) / length(matrix))

        data[i] = (matrix,
                   structure_dst.fusiontreestructure[ids_dst],
                   structure_src.fusiontreestructure[ids_src])
    end

    # sort by (approximate) weight to make the buffers happy
    # and use round-robin strategy for multi-threading
    sort!(data; by=_transformer_weight, rev=true)

    @debug("TreeTransformer for $Vsrc to $Vdst via $p",
           nblocks = length(data),
           sz_median = size(data[end ÷ 2][1], 1),
           sz_max = size(data[1][1], 1))

    return GenericTreeTransformer{T,N}(data)
end

# Cost model for transforming a set of subblocks with fixed uncoupled sectors:
# L x L x length(subblock) where L is the number of subblocks
# this is L input blocks each going to L output blocks of given length
# Note that it might be the case that the permutations are dominant, in which case the
# actual cost model would scale like L x length(subblock)
function _transformer_weight((matrix, structures_dst, structures_src))
    return length(matrix) * prod(structures_dst[1][1])
end

function buffersize(transformer::GenericTreeTransformer)
    return maximum(transformer.data; init=0) do (basistransform, structures_dst, _)
        return prod(structures_dst[1][1]) * size(basistransform, 1)
    end
end

function allocate_buffers(tdst::TensorMap, tsrc::TensorMap,
                          transformer::GenericTreeTransformer)
    sz = buffersize(transformer)
    return similar(tdst.data, sz), similar(tsrc.data, sz)
end

function treetransformertype(Vdst, Vsrc)
    I = sectortype(Vdst)
    I === Trivial && return TrivialTreeTransformer

    T = sectorscalartype(I)
    N = numind(Vdst)
    return FusionStyle(I) == UniqueFusion() ? AbelianTreeTransformer{T,N} :
           GenericTreeTransformer{T,N}
end

function TreeTransformer(transform::Function, p, Vdst::HomSpace{S},
                         Vsrc::HomSpace{S}) where {S}
    permute(Vsrc, p) == Vdst ||
        throw(SpaceMismatch("Incompatible spaces for permuting"))

    I = sectortype(Vdst)
    I === Trivial && return TrivialTreeTransformer()

    return FusionStyle(I) == UniqueFusion() ?
           AbelianTreeTransformer(transform, p, Vdst, Vsrc) :
           GenericTreeTransformer(transform, p, Vdst, Vsrc)
end

# braid is special because it has levels
function treebraider(::AbstractTensorMap, ::AbstractTensorMap, p::Index2Tuple, levels)
    return fusiontreetransform(f1, f2) = braid(f1, f2, levels..., p...)
end
function treebraider(tdst::TensorMap, tsrc::TensorMap, p::Index2Tuple, levels)
    return treebraider(space(tdst), space(tsrc), p, levels)
end
@cached function treebraider(Vdst::TensorMapSpace, Vsrc::TensorMapSpace, p::Index2Tuple,
                             levels)::treetransformertype(Vdst, Vsrc)
    fusiontreebraider(f1, f2) = braid(f1, f2, levels..., p...)
    return TreeTransformer(fusiontreebraider, p, Vdst, Vsrc)
end

for (transform, treetransformer) in
    ((:permute, :treepermuter), (:transpose, :treetransposer))
    @eval begin
        function $treetransformer(::AbstractTensorMap, ::AbstractTensorMap, p::Index2Tuple)
            return fusiontreetransform(f1, f2) = $transform(f1, f2, p...)
        end
        function $treetransformer(tdst::TensorMap, tsrc::TensorMap, p::Index2Tuple)
            return $treetransformer(space(tdst), space(tsrc), p)
        end
        @cached function $treetransformer(Vdst::TensorMapSpace, Vsrc::TensorMapSpace,
                                          p::Index2Tuple)::treetransformertype(Vdst, Vsrc)
            fusiontreetransform(f1, f2) = $transform(f1, f2, p...)
            return TreeTransformer(fusiontreetransform, p, Vdst, Vsrc)
        end
    end
end

# default cachestyle is GlobalLRUCache
