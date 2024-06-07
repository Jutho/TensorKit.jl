# AdjointTensorMap: lazy adjoint
#==========================================================#
"""
    struct AdjointTensorMap{E, S, N₁, N₂, ...} <: AbstractTensorMap{E, S, N₁, N₂}

Specific subtype of [`AbstractTensorMap`](@ref) that is a lazy wrapper for representing the
adjoint of an instance of [`TensorMap`](@ref).
"""
struct AdjointTensorMap{E,S,N₁,N₂,I,A,F₁,F₂} <:
       AbstractTensorMap{E,S,N₁,N₂}
    parent::TensorMap{E,S,N₂,N₁,I,A,F₂,F₁}
end

#! format: off
const AdjointTrivialTensorMap{E,S,N₁,N₂,A<:DenseMatrix} =
    AdjointTensorMap{E,S,N₁,N₂,Trivial,A,Nothing,Nothing}
#! format: on

# Constructor: construct from taking adjoint of a tensor
Base.adjoint(t::TensorMap) = AdjointTensorMap(t)
Base.adjoint(t::AdjointTensorMap) = t.parent

function Base.similar(t::AdjointTensorMap, ::Type{TorA},
                      P::TensorMapSpace) where {TorA<:MatOrNumber}
    return similar(t', TorA, P)
end

# Properties
codomain(t::AdjointTensorMap) = domain(t.parent)
domain(t::AdjointTensorMap) = codomain(t.parent)

blocksectors(t::AdjointTensorMap) = blocksectors(t.parent)

#! format: off
storagetype(::Type{<:AdjointTrivialTensorMap{E,S,N₁,N₂,A}}) where {E,S,N₁,N₂,A<:DenseMatrix} = A
storagetype(::Type{<:AdjointTensorMap{E,S,N₁,N₂,I,<:SectorDict{I,A}}}) where {E,S,N₁,N₂,I<:Sector,A<:DenseMatrix} = A
#! format: on

dim(t::AdjointTensorMap) = dim(t.parent)

# Indexing
#----------
hasblock(t::AdjointTensorMap, s::Sector) = hasblock(t.parent, s)
block(t::AdjointTensorMap, s::Sector) = block(t.parent, s)'
blocks(t::AdjointTensorMap) = (c => b' for (c, b) in blocks(t.parent))

fusiontrees(::AdjointTrivialTensorMap) = ((nothing, nothing),)
fusiontrees(t::AdjointTensorMap) = TensorKeyIterator(t.parent.colr, t.parent.rowr)

function Base.getindex(t::AdjointTensorMap{E,S,N₁,N₂,I},
                       f₁::FusionTree{I,N₁}, f₂::FusionTree{I,N₂}) where {E,S,N₁,N₂,I}
    c = f₁.coupled
    @boundscheck begin
        c == f₂.coupled || throw(SectorMismatch())
        hassector(codomain(t), f₁.uncoupled) && hassector(domain(t), f₂.uncoupled)
    end
    return sreshape((StridedView(t.parent.data[c])[t.parent.rowr[c][f₂],
                                                   t.parent.colr[c][f₁]])',
                    (dims(codomain(t), f₁.uncoupled)..., dims(domain(t), f₂.uncoupled)...))
end
@propagate_inbounds function Base.setindex!(t::AdjointTensorMap{E,S,N₁,N₂,I}, v,
                                            f₁::FusionTree{I,N₁},
                                            f₂::FusionTree{I,N₂}) where {E,S,N₁,N₂,I}
    return copy!(getindex(t, f₁, f₂), v)
end

@inline function Base.getindex(t::AdjointTrivialTensorMap)
    return sreshape(StridedView(t.parent.data)', (dims(codomain(t))..., dims(domain(t))...))
end
@inline Base.setindex!(t::AdjointTrivialTensorMap, v) = copy!(getindex(t), v)

@inline Base.getindex(t::AdjointTrivialTensorMap, ::Tuple{Nothing,Nothing}) = getindex(t)
@inline function Base.setindex!(t::AdjointTrivialTensorMap, v, ::Tuple{Nothing,Nothing})
    return setindex!(t, v)
end

# For a tensor with trivial symmetry, allow direct indexing
@inline function Base.getindex(t::AdjointTrivialTensorMap, indices::Vararg{Int})
    data = t[]
    @boundscheck checkbounds(data, indices)
    @inbounds v = data[indices...]
    return v
end
@inline function Base.setindex!(t::AdjointTrivialTensorMap, v, indices::Vararg{Int})
    data = t[]
    @boundscheck checkbounds(data, indices)
    @inbounds data[indices...] = v
    return v
end

# Show
#------
function Base.summary(t::AdjointTensorMap)
    return print("AdjointTensorMap(", codomain(t), " ← ", domain(t), ")")
end
function Base.show(io::IO, t::AdjointTensorMap)
    if get(io, :compact, false)
        print(io, "AdjointTensorMap(", codomain(t), " ← ", domain(t), ")")
        return
    end
    println(io, "AdjointTensorMap(", codomain(t), " ← ", domain(t), "):")
    if sectortype(t) === Trivial
        Base.print_array(io, t[])
        println(io)
    elseif FusionStyle(sectortype(t)) isa UniqueFusion
        for (f₁, f₂) in fusiontrees(t)
            println(io, "* Data for sector ", f₁.uncoupled, " ← ", f₂.uncoupled, ":")
            Base.print_array(io, t[f₁, f₂])
            println(io)
        end
    else
        for (f₁, f₂) in fusiontrees(t)
            println(io, "* Data for fusiontree ", f₁, " ← ", f₂, ":")
            Base.print_array(io, t[f₁, f₂])
            println(io)
        end
    end
end
