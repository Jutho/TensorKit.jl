"""
    TreeTransformer

Supertype for structures containing the data for a tree transformation.
"""
abstract type TreeTransformer end

struct TrivialTreeTransformer <: TreeTransformer end

struct AbelianTreeTransformer{T,I,N,F1,F2,F3,F4} <: TreeTransformer
    rows::Vector{Int}
    cols::Vector{Int}
    vals::Vector{T}
    structure_dst::FusionBlockStructure{I,N,F1,F2}
    structure_src::FusionBlockStructure{I,N,F3,F4}
end

struct GenericTreeTransformer{T,I,N,F1,F2,F3,F4} <: TreeTransformer
    matrix::SparseMatrixCSC{T,Int}
    structure_dst::FusionBlockStructure{I,N,F1,F2}
    structure_src::FusionBlockStructure{I,N,F3,F4}
end

function treetransformertype(Vdst, Vsrc)
    I = sectortype(Vdst)
    I === Trivial && return TrivialTreeTransformer

    N = numind(Vdst)
    F1 = fusiontreetype(I, numout(Vdst))
    F2 = fusiontreetype(I, numin(Vdst))
    F3 = fusiontreetype(I, numout(Vsrc))
    F4 = fusiontreetype(I, numin(Vsrc))

    if FusionStyle(I) isa UniqueFusion
        return AbelianTreeTransformer{sectorscalartype(I),I,N,F1,F2,F3,F4}
    else
        return GenericTreeTransformer{sectorscalartype(I),I,N,F1,F2,F3,F4}
    end
end

function TreeTransformer(transform::Function, Vsrc::HomSpace{S},
                         Vdst::HomSpace{S}) where {S}
    I = sectortype(Vdst)
    I === Trivial && return TrivialTreeTransformer()

    structure_dst = fusionblockstructure(Vdst)
    structure_src = fusionblockstructure(Vsrc)

    rows = Int[]
    cols = Int[]
    vals = sectorscalartype(sectortype(Vdst))[]

    for (f1, f2) in structure_src.fusiontreelist
        row = structure_src.fusiontreeindices[(f1, f2)]
        for ((f3, f4), coeff) in transform(f1, f2)
            col = structure_dst.fusiontreeindices[(f3, f4)]
            push!(rows, row)
            push!(cols, col)
            push!(vals, coeff)
        end
    end

    if FusionStyle(I) isa UniqueFusion
        return AbelianTreeTransformer(rows, cols, vals, structure_dst, structure_src)
    else
        ldst = length(structure_dst.fusiontreelist)
        lsrc = length(structure_src.fusiontreelist)
        matrix = sparse(rows, cols, vals, ldst, lsrc)
        return GenericTreeTransformer(matrix, structure_dst, structure_src)
    end
end

# Transpose
# ---------
const treetransposercache = LRU{Any,Any}(; maxsize=10^5)
const usetreetransposercache = Ref{Bool}(true)

function treetransposer(::AbstractTensorMap, ::AbstractTensorMap, p::Index2Tuple)
    return fusiontreetransform(f1, f2) = transpose(f1, f2, p...)
end
function treetransposer(tdst::TensorMap, tsrc::TensorMap, p::Index2Tuple)
    if usetreetransposercache[]
        key = (space(tdst), space(tsrc), p)
        A = treetransformertype(space(tdst), space(tsrc))
        return _get_treetransposer(A, key)
    else
        return _treetransposer((space(tdst), space(tsrc), p))
    end
end
@noinline function _get_treetransposer(A, key)
    d::A = get!(treetransposercache, key) do
        return _treetransposer(key)
    end
    return d
end
function _treetransposer((Vdst, Vsrc, p))
    fusiontreetransform(f1, f2) = transpose(f1, f2, p...)
    return TreeTransformer(fusiontreetransform, Vsrc, Vdst)
end

# Braid
# -----
const treebraidercache = LRU{Any,Any}(; maxsize=10^5)
const usetreebraidercache = Ref{Bool}(true)

function treebraider(::AbstractTensorMap, ::AbstractTensorMap, p::Index2Tuple,
                     l::Index2Tuple)
    return fusiontreetransform(f1, f2) = braid(f1, f2, p..., l...)
end
function treebraider(tdst::TensorMap, tsrc::TensorMap, p::Index2Tuple,
                     l::Index2Tuple)
    if usetreebraidercache[]
        key = (space(tdst), space(tsrc), p, l)
        A = treetransformertype(space(tdst), space(tsrc))
        return _get_treebraider(A, key)
    else
        return _treebraider((space(tdst), space(tsrc), p, l))
    end
end
@noinline function _get_treebraider(A, key)
    d::A = get!(treebraidercache, key) do
        return _treebraider(key)
    end
    return d
end
function _treebraider((Vdst, Vsrc, p, l))
    fusiontreetransform(f1, f2) = braid(f1, f2, p..., l...)
    return TreeTransformer(fusiontreetransform, Vsrc, Vdst)
end

# Permute
# -------
const treepermutercache = LRU{Any,Any}(; maxsize=10^5)
const usetreepermutercache = Ref{Bool}(true)

function treepermuter(::AbstractTensorMap, ::AbstractTensorMap, p::Index2Tuple)
    return fusiontreetransform(f1, f2) = permute(f1, f2, p...)
end
function treepermuter(tdst::TensorMap, tsrc::TensorMap, p::Index2Tuple)
    if usetreepermutercache[]
        key = (space(tdst), space(tsrc), p)
        A = treetransformertype(space(tdst), space(tsrc))
        return _get_treepermuter(A, key)
    else
        return _treepermuter((space(tdst), space(tsrc), p))
    end
end
@noinline function _get_treepermuter(A, key)
    d::A = get!(treepermutercache, key) do
        return _treepermuter(key)
    end
    return d
end
function _treepermuter((Vdst, Vsrc, p))
    fusiontreetransform(f1, f2) = permute(f1, f2, p...)
    return TreeTransformer(fusiontreetransform, Vsrc, Vdst)
end
