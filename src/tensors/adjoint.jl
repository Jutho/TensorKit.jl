# AdjointTensorMap: lazy adjoint
#==========================================================#
"""
    struct AdjointTensorMap{T, S, N₁, N₂, TT<:AbstractTensorMap} <: AbstractTensorMap{T, S, N₁, N₂}

Specific subtype of [`AbstractTensorMap`](@ref) that is a lazy wrapper for representing the
adjoint of an instance of [`AbstractTensorMap`](@ref).
"""
struct AdjointTensorMap{T,S,N₁,N₂,TT<:AbstractTensorMap{T,S,N₂,N₁}} <:
       AbstractTensorMap{T,S,N₁,N₂}
    parent::TT
end

#! format: off
const AdjointTrivialTensorMap{T,S,N₁,N₂,TT<:TrivialTensorMap{T,S,N₂,N₁}} =
    AdjointTensorMap{T,S,N₁,N₂,TT}
#! format: on

# Constructor: construct from taking adjoint of a tensor
Base.adjoint(t::AbstractTensorMap) = AdjointTensorMap(t)
Base.adjoint(t::AdjointTensorMap) = parent(t)

Base.parent(t::AdjointTensorMap) = t.parent
parenttype(::Type{<:AdjointTensorMap{T,S,N₁,N₂,TT}}) where {T,S,N₁,N₂,TT} = TT

function Base.similar(t::AdjointTensorMap, ::Type{TorA},
                      P::TensorMapSpace) where {TorA<:MatOrNumber}
    return similar(t', TorA, P)
end

# Properties
codomain(t::AdjointTensorMap) = domain(parent(t))
domain(t::AdjointTensorMap) = codomain(parent(t))

blocksectors(t::AdjointTensorMap) = blocksectors(parent(t))

storagetype(::Type{TT}) where {TT<:AdjointTensorMap} = storagetype(parenttype(TT))

dim(t::AdjointTensorMap) = dim(parent(t))

# Indexing
#----------
hasblock(t::AdjointTensorMap, s::Sector) = hasblock(parent(t), s)
block(t::AdjointTensorMap, s::Sector) = block(parent(t), s)'
blocks(t::AdjointTensorMap) = (c => b' for (c, b) in blocks(parent(t)))

fusiontrees(::AdjointTrivialTensorMap) = ((nothing, nothing),)
function fusiontrees(t::AdjointTensorMap{T,S,N₁,N₂,TT}) where {T,S,N₁,N₂,TT<:TensorMap}
    return TensorKeyIterator(parent(t).colr, parent(t).rowr)
end

function Base.getindex(t::AdjointTensorMap{T,S,N₁,N₂,<:TensorMap{T,S,N₁,N₂,I}},
                       f₁::FusionTree{I,N₁}, f₂::FusionTree{I,N₂}) where {T,S,N₁,N₂,I}
    c = f₁.coupled
    @boundscheck begin
        c == f₂.coupled || throw(SectorMismatch())
        hassector(codomain(t), f₁.uncoupled) && hassector(domain(t), f₂.uncoupled)
    end
    return sreshape((StridedView(parent(t).data[c])[parent(t).rowr[c][f₂],
                                                    parent(t).colr[c][f₁]])',
                    (dims(codomain(t), f₁.uncoupled)..., dims(domain(t), f₂.uncoupled)...))
end
@propagate_inbounds function Base.getindex(t::AdjointTensorMap{T,S,N₁,N₂},
                                           f₁::FusionTree{I,N₁},
                                           f₂::FusionTree{I,N₂}) where {T,S,N₁,N₂,I}
    d_cod = dims(codomain(t), f₁.uncoupled)
    d_dom = dims(domain(t), f₂.uncoupled)
    return sreshape(sreshape(StridedView(parent(t)[f₂, f₁]), (prod(d_dom), prod(d_cod)))',
                    (d_cod..., d_dom...))
end

@propagate_inbounds function Base.setindex!(t::AdjointTensorMap{T,S,N₁,N₂,I}, v,
                                            f₁::FusionTree{I,N₁},
                                            f₂::FusionTree{I,N₂}) where {T,S,N₁,N₂,I}
    return copy!(getindex(t, f₁, f₂), v)
end

@inline function Base.getindex(t::AdjointTrivialTensorMap)
    return sreshape(StridedView(parent(t).data)',
                    (dims(codomain(t))..., dims(domain(t))...))
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
function Base.summary(io::IO, t::AdjointTensorMap)
    return print(io, "AdjointTensorMap(", codomain(t), " ← ", domain(t), ")")
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
