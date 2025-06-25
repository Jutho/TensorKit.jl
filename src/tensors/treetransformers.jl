"""
    TreeTransformer

Supertype for structures containing the data for a tree transformation.
"""
abstract type TreeTransformer end

struct TrivialTreeTransformer <: TreeTransformer end

struct AbelianTreeTransformer{T,N} <: TreeTransformer
    data::Vector{Tuple{StridedStructure{N},StridedStructure{N},T}}
end

function AbelianTreeTransformer(transform, p, Vsrc, Vdst)
    permute(Vsrc, p) == Vdst || throw(SpaceMismatch("Incompatible spaces for permuting."))
    structure_dst = fusionblockstructure(Vdst)
    structure_src = fusionblockstructure(Vsrc)

    L = length(structure_src.fusiontreelist)
    T = sectorscalartype(sectortype(Vdst))
    N = numind(Vsrc)
    data = Vector{Tuple{StridedStructure{N},StridedStructure{N},T}}(undef, L)

    for i in 1:L
        f₁, f₂ = structure_src.fusiontreelist[i]
        (f₃, f₄), coeff = only(transform(f₁, f₂))
        j = structure_dst.fusiontreeindices[(f₃, f₄)]
        stridestructure_dst = structure_dst.fusiontreestructure[j]
        stridestructure_src = structure_src.fusiontreestructure[i]
        data[i] = (stridestructure_dst, stridestructure_src, coeff)
    end

    return AbelianTreeTransformer(data)
end

struct GenericTreeTransformer{T,N} <: TreeTransformer
    matrix::SparseMatrixCSC{T,Int}
    structure_dst::Vector{StridedStructure{N}}
    structure_src::Vector{StridedStructure{N}}
end

function GenericTreeTransformer(transform, p, Vsrc, Vdst)
    permute(Vsrc, p) == Vdst || throw(SpaceMismatch("Incompatible spaces for permuting."))
    structure_dst = fusionblockstructure(Vdst)
    structure_src = fusionblockstructure(Vsrc)

    rows = Int[]
    cols = Int[]
    vals = sectorscalartype(sectortype(Vdst))[]

    for (row, (f1, f2)) in enumerate(structure_src.fusiontreelist)
        for ((f3, f4), coeff) in transform(f1, f2)
            col = structure_dst.fusiontreeindices[(f3, f4)]
            push!(rows, row)
            push!(cols, col)
            push!(vals, coeff)
        end
    end
    ldst = length(structure_dst.fusiontreelist)
    lsrc = length(structure_src.fusiontreelist)
    matrix = sparse(rows, cols, vals, ldst, lsrc)

    return GenericTreeTransformer(matrix,
                                  structure_dst.fusiontreestructure,
                                  structure_src.fusiontreestructure)
end

struct OuterTreeTransformer{T,N} <: TreeTransformer
    data::Vector{Tuple{Matrix{T},Vector{StridedStructure{N}},Vector{StridedStructure{N}}}}
end

function OuterTreeTransformer(transform, p, Vsrc, Vdst)
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

    outer_data = map(uncoupleds_src_unique) do uncoupled
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

        return (matrix,
                structure_dst.fusiontreestructure[ids_dst],
                structure_src.fusiontreestructure[ids_src])
    end
    return OuterTreeTransformer(outer_data)
end

useouter() = true

function treetransformertype(Vdst, Vsrc)
    I = sectortype(Vdst)
    I === Trivial && return TrivialTreeTransformer

    T = sectorscalartype(I)
    N = numind(Vdst)
    if useouter()
        return FusionStyle(I) == UniqueFusion() ? AbelianTreeTransformer{T,N} :
               OuterTreeTransformer{T,N}
    else
        return FusionStyle(I) == UniqueFusion() ? AbelianTreeTransformer{T,N} :
               GenericTreeTransformer{T,N}
    end
end

function TreeTransformer(transform::Function, p, Vsrc::HomSpace{S},
                         Vdst::HomSpace{S}) where {S}
    permute(Vsrc, p) == Vdst || throw(SpaceMismatch("Incompatible spaces for permuting."))

    I = sectortype(Vdst)
    I === Trivial && return TrivialTreeTransformer()

    FusionStyle(I) == UniqueFusion() &&
        return AbelianTreeTransformer(transform, p, Vsrc, Vdst)

    if useouter()
        return OuterTreeTransformer(transform, p, Vsrc, Vdst)
    else
        return GenericTreeTransformer(transform, p, Vsrc, Vdst)
    end
end

# braid is special because it has levels
const treebraidercache = LRU{Any,Any}(; maxsize=10^5)
const usetreebraidercache = Ref{Bool}(true)
@noinline function _get_treebraider(A, key)
    d::A = get!(treebraidercache, key) do
        return _treebraider(key)
    end
    return d
end
function _treebraider((Vdst, Vsrc, p, levels))
    fusiontreebraider(f1, f2) = braid(f1, f2, levels..., p...)
    return TreeTransformer(fusiontreebraider, p, Vsrc, Vdst)
end
function treebraider(::AbstractTensorMap, ::AbstractTensorMap, p, levels)
    return fusiontreetransform(f1, f2) = braid(f1, f2, levels..., p...)
end
function treebraider(tdst::TensorMap, tsrc::TensorMap, p, levels)
    if usetreebraidercache[]
        key = (space(tdst), space(tsrc), p, levels)
        A = treetransformertype(space(tdst), space(tsrc))
        return _get_treebraider(A, key)
    else
        return _treebraider((space(tdst), space(tsrc), p, levels))
    end
end

for (transform, transformer) in
    ((:permute, :permuter), (:transpose, :transposer))
    treetransformcache = Symbol("tree", transformer, "cache")
    usetreetransformcache = Symbol("usetree", transformer, "cache")
    treetransformer = Symbol("tree", transformer)
    _get_treetransformer = Symbol("_get_", treetransformer)
    _treetransformer = Symbol("_", treetransformer)

    @eval begin
        const $treetransformcache = LRU{Any,Any}(; maxsize=10^5)
        const $usetreetransformcache = Ref{Bool}(true)

        function $treetransformer(::AbstractTensorMap, ::AbstractTensorMap, p::Index2Tuple)
            return fusiontreetransform(f1, f2) = $transform(f1, f2, p...)
        end
        function $treetransformer(tdst::TensorMap, tsrc::TensorMap, p::Index2Tuple)
            if $usetreetransformcache[]
                key = (space(tdst), space(tsrc), p)
                A = treetransformertype(space(tdst), space(tsrc))
                return $_get_treetransformer(A, key)
            else
                return $_treetransformer((space(tdst), space(tsrc), p))
            end
        end
        @noinline function $_get_treetransformer(A, key)
            d::A = get!($treetransformcache, key) do
                return $_treetransformer(key)
            end
            return d
        end
        function $_treetransformer((Vdst, Vsrc, p))
            fusiontreetransform(f1, f2) = $transform(f1, f2, p...)
            return TreeTransformer(fusiontreetransform, p, Vsrc, Vdst)
        end
    end
end
