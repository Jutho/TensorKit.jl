# DiagonalTensorMap
#==========================================================#
struct DiagonalTensorMap{T,S<:IndexSpace,A<:DenseVector{T}} <: AbstractTensorMap{T,S,1,1}
    data::A
    domain::S # equals codomain

    # uninitialized constructors
    function DiagonalTensorMap{T,S,A}(::UndefInitializer,
                                      dom::S) where {T,S<:IndexSpace,A<:DenseVector{T}}
        data = A(undef, reduceddim(dom))
        return DiagonalTensorMap{T,S,A}(data, dom)
    end
    # constructors from data
    function DiagonalTensorMap{T,S,A}(data::A,
                                      dom::S) where {T,S<:IndexSpace,A<:DenseVector{T}}
        T ⊆ field(S) || @warn("scalartype(data) = $T ⊈ $(field(S)))", maxlog = 1)
        return DiagonalTensorMap{T,S,A}(data, dom)
    end
end
reduceddim(V::IndexSpace) = sum(c -> dim(V, c), sectors(V); init=0)

# Basic methods for characterising a tensor:
#--------------------------------------------
space(t::DiagonalTensorMap) = t.domain ← t.domain

"""
    storagetype(::Union{T,Type{T}}) where {T<:TensorMap} -> Type{A<:DenseVector}

Return the type of the storage `A` of the tensor map.
"""
storagetype(::Type{<:DiagonalTensorMap{T,S,A}}) where {T,S,A<:DenseVector{T}} = A

# DiagonalTensorMap constructors
#--------------------------------
# undef constructors
"""
    DiagonalTensorMap{T}(undef, domain::S) where {T,S<:IndexSpace}
    # expert mode: select storage type `A`
    DiagonalTensorMap{T,S,A}(undef, domain::S) where {T,S<:IndexSpace,A<:DenseVector{T}}

Construct a `DiagonalTensorMap` with uninitialized data.
"""
function DiagonalTensorMap{T}(::UndefInitializer, V::S) where {T,S<:IndexSpace}
    return DiagonalTensorMap{T,S,Vector{T}}(undef, V)
end

function DiagonalTensorMap{T}(data::A, V::S) where {T,S<:IndexSpace,A<:DenseVector{T}}
    length(data) == reduceddim(V) ||
        throw(DimensionMismatch("length(data) = $(length(data)) is not compatible with the space $V"))
    return DiagonalTensorMap{T,S,A}(data, V)
end

function DiagonalTensorMap(data::DenseVector{T}, V::IndexSpace) where {T}
    return DiagonalTensorMap{T}(data, V)
end

# TODO: more constructors needed?

# Special case adjoint:
#-----------------------
Base.adjoint(t::DiagonalTensorMap{<:Real}) = t
Base.adjoint(t::DiagonalTensorMap{<:Complex}) = DiagonalTensorMap(conj(t.data), t.domain)

# Efficient copy constructors
#-----------------------------
Base.copy(t::DiagonalTensorMap) = typeof(t)(copy(t.data), t.domain)

function Base.complex(t::DiagonalTensorMap)
    if scalartype(t) <: Complex
        return t
    else
        return DiagonalTensorMap(complex(t.data), t.domain)
    end
end

# Getting and setting the data at the block level
#-------------------------------------------------
blocksectors(t::DiagonalTensorMap) = blocksectors(t.domain)

function block(t::DiagonalTensorMap, s::Sector)
    sectortype(t) == typeof(s) || throw(SectorMismatch())
    offset = 0
    for c in sectors(t)
        if c < s
            offset += dim(t, c)
        elseif c == s
            r = offset .+ (1:dim(t, c))
            return Diagonal(view(t.data, r))
        else # s not in sectors(t)
            return Diagonal(view(t.data, 1:0))
        end
    end
end

# TODO: is relying on generic AbstractTensorMap blocks sufficient?

# Indexing and getting and setting the data at the subblock level
#-----------------------------------------------------------------
@inline function Base.getindex(t::DiagonalTensorMap,
                               f₁::FusionTree{I,1},
                               f₂::FusionTree{I,1}) where {I<:Sector}
    s = f₁.uncoupled[1]
    s == f₁.uncoulped == f₂.uncoupled[1] == f₂.uncoupled || throw(SectorMismatch())
    return block(t, s)
    # TODO: do we want a StridedView here? Then we need to allocate a new matrix.
end

function Base.setindex!(t::TensorMap,
                        v,
                        f₁::FusionTree{I,1},
                        f₂::FusionTree{I,1}) where {I<:Sector}
    return copy!(getindex(t, f₁, f₂), v)
end
