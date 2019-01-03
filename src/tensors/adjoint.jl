# AdjointTensorMap: lazy adjoint
#==========================================================#
struct AdjointTensorMap{S<:IndexSpace, N₁, N₂, G<:Sector, A, F₁, F₂} <: AbstractTensorMap{S, N₁, N₂}
    parent::TensorMap{S,N₂,N₁,G,A,F₂,F₁}
end

const AdjointTrivialTensorMap{S<:IndexSpace, N₁, N₂, A<:DenseMatrix} = AdjointTensorMap{S, N₁, N₂, Trivial, A, Nothing, Nothing}

# Constructor: construct from taking adjoint of a tensor
Base.adjoint(t::TensorMap) = AdjointTensorMap(t)
Base.adjoint(t::AdjointTensorMap) = t.parent

# Properties
codomain(t::AdjointTensorMap) = domain(t.parent)
domain(t::AdjointTensorMap) = codomain(t.parent)

blocksectors(t::AdjointTensorMap) = blocksectors(t.parent)

Base.@pure storagetype(::Type{<:AdjointTensorMap{<:IndexSpace,N₁,N₂,Trivial,A}}) where {N₁,N₂,A<:DenseMatrix} = A
Base.@pure storagetype(::Type{<:AdjointTensorMap{<:IndexSpace,N₁,N₂,G,<:SectorDict{G,A}}}) where {N₁,N₂,G<:Sector,A<:DenseMatrix} = A

Base.length(t::AdjointTensorMap) = length(t.parent)

# Indexing
#----------
hasblock(t::AdjointTensorMap, s::Sector) = hasblock(t.parent, s)
block(t::AdjointTensorMap, s::Sector) = block(t.parent, s)'
blocks(t::AdjointTensorMap) = (c=>b' for (c,b) in blocks(t.parent))

fusiontrees(::AdjointTrivialTensorMap) = ((nothing, nothing),)
fusiontrees(t::AdjointTensorMap) = TensorKeyIterator(t.parent.colr, t.parent.rowr)

function Base.getindex(t::AdjointTensorMap{S,N₁,N₂,G}, f1::FusionTree{G,N₁}, f2::FusionTree{G,N₂}) where {S,N₁,N₂,G}
    c = f1.coupled
    @boundscheck begin
        c == f2.coupled || throw(SectorMismatch())
        hassector(codomain(t), f1.uncoupled) && hassector(domain(t), f2.uncoupled)
    end
    return sreshape((StridedView(t.parent.data[c])[t.parent.rowr[c][f2], t.parent.colr[c][f1]])', (dims(codomain(t), f1.uncoupled)..., dims(domain(t), f2.uncoupled)...))
end
@propagate_inbounds Base.setindex!(t::AdjointTensorMap{S,N₁,N₂}, v, f1::FusionTree{G,N₁}, f2::FusionTree{G,N₂}) where {S,N₁,N₂,G} = copyto!(getindex(t, f1, f2), v)

@inline Base.getindex(t::AdjointTrivialTensorMap) = sreshape(StridedView(t.parent.data)', (dims(codomain(t))..., dims(domain(t))...))
@inline Base.setindex!(t::AdjointTrivialTensorMap, v) = copyto!(getindex(t), v)

@inline Base.getindex(t::AdjointTrivialTensorMap, ::Tuple{Nothing,Nothing}) = getindex(t)
@inline Base.setindex!(t::AdjointTrivialTensorMap, v, ::Tuple{Nothing,Nothing}) = setindex!(t, v)

# For a tensor with trivial symmetry, allow direct indexing
@inline function Base.getindex(t::AdjointTrivialTensorMap, I::Vararg{Int})
    data = t[]
    @boundscheck checkbounds(data, I)
    @inbounds v = data[I...]
    return v
end
@inline function Base.setindex!(t::AdjointTrivialTensorMap, v, I::Vararg{Int})
    data = t[]
    @boundscheck checkbounds(data, I)
    @inbounds data[I...] = v
    return v
end

# Show
#------
function Base.summary(t::AdjointTensorMap)
    print("AdjointTensorMap(", codomain(t), " ← ", domain(t), ")")
end
function Base.show(io::IO, t::AdjointTensorMap{S}) where {S<:IndexSpace}
    if get(io, :compact, false)
        print(io, "AdjointTensorMap(", codomain(t), " ← ", domain(t), ")")
        return
    end
    println(io, "AdjointTensorMap(", codomain(t), " ← ", domain(t), "):")
    if sectortype(S) == Trivial
        Base.print_array(io, t[])
        println(io)
    elseif FusionStyle(sectortype(S)) isa Abelian
        for (f1,f2) in fusiontrees(t)
            println(io, "* Data for sector ", f1.uncoupled, " ← ", f2.uncoupled, ":")
            Base.print_array(io, t[f1,f2])
            println(io)
        end
    else
        for (f1,f2) in fusiontrees(t)
            println(io, "* Data for fusiontree ", f1, " ← ", f2, ":")
            Base.print_array(io, t[f1,f2])
            println(io)
        end
    end
end
