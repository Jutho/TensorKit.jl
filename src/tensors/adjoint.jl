# AdjointTensorMap: lazy adjoint
#==========================================================#
struct AdjointTensorMap{S<:IndexSpace, N₁, N₂, A, F₁, F₂} <: AbstractTensorMap{S, N₁, N₂}
    parent::TensorMap{S,N₂,N₁,A,F₂,F₁}
end
# Constructor: construct from taking adjoint of a tensor
adjoint(t::TensorMap) = AdjointTensorMap(t)
adjoint(t::AdjointTensorMap) = t.parent

# Properties
codomain(t::AdjointTensorMap) = domain(t.parent)
domain(t::AdjointTensorMap) = codomain(t.parent)

Base.eltype(::Type{<:AdjointTensorMap{<:IndexSpace,N₁,N₂,<:AbstractArray{T}}}) where {T,N₁,N₂} = T
Base.eltype(::Type{<:AdjointTensorMap{<:IndexSpace,N₁,N₂,<:AbstractDict{<:Any,<:AbstractArray{T}}}}) where {T,N₁,N₂} = T

Base.length(t::AdjointTensorMap) = length(t.parent)

Base.similar(t::AdjointTensorMap{S}, ::Type{T}, P::TensorMapSpace{S} = (domain(t)=>codomain(t))) where {T,S} = similar(t.parent, T, P)
Base.similar(t::AdjointTensorMap{S}, ::Type{T}, P::TensorSpace{S}) where {T,S} = similar(t.parent, T, P)
Base.similar(t::AdjointTensorMap{S}, P::TensorMapSpace{S} = (domain(t)=>codomain(t))) where {S} = similar(t.parent, P)
Base.similar(t::AdjointTensorMap{S}, P::TensorSpace{S}) where {S} = similar(t.parent, P)

unsafe_similar(t::AdjointTensorMap{S}, ::Type{T}, P::TensorMapSpace{S} = (domain(t)=>codomain(t))) where {T,S} = unsafe_similar(t.parent, T, P)
unsafe_similar(t::AdjointTensorMap{S}, ::Type{T}, P::TensorSpace{S}) where {T,S} = unsafe_similar(t.parent, T, P)
unsafe_similar(t::AdjointTensorMap{S}, P::TensorMapSpace{S} = (domain(t)=>codomain(t))) where {S} = unsafe_similar(t.parent, P)
unsafe_similar(t::AdjointTensorMap{S}, P::TensorSpace{S}) where {S} = unsafe_similar(t.parent, P)

# Copy
Base.copy!(tdst::TensorMap, tsrc::AdjointTensorMap) = adjoint!(tdst, tsrc.parent)
Base.copy!(tdst::AdjointTensorMap, tsrc::TensorMap) = adjoint!(tdst.parent, tsrc)
Base.copy!(tdst::AdjointTensorMap, tsrc::AdjointTensorMap) = copy!(tdst.parent, tsrc.parent)

Base.vecnorm(t::AdjointTensorMap, p::Real) = vecnorm(t.parent, p)

# Indexing
#----------
fusiontrees(t::AdjointTensorMap{S,N₁,N₂,<:AbstractDict}) where {S<:IndexSpace,N₁,N₂} = TensorTreeIterator(t.parent.colr, t.parent.rowr)

function Base.getindex(t::AdjointTensorMap{S,N₁,N₂}, f1::FusionTree{G,N₁}, f2::FusionTree{G,N₂}) where {S,N₁,N₂,G}
    c = f1.incoming
    @boundscheck begin
        c == f2.incoming || throw(SectorMismatch())
        checksectors(codomain(t), f1.outgoing) && checksectors(domain(t), f2.outgoing)
    end
    return splitdims(sview(t.parent.data[c], t.parent.rowr[c][f2], t.parent.colr[c][f1])', dims(codomain(t), f1.outgoing), dims(domain(t), f2.outgoing))
end
@propagate_inbounds Base.setindex!(t::AdjointTensorMap{S,N₁,N₂}, v, f1::FusionTree{G,N₁}, f2::FusionTree{G,N₂}) where {S,N₁,N₂,G} = copy!(getindex(t, f1, f2), v)

Base.getindex(t::AdjointTensorMap{<:Any,N₁,N₂,<:AbstractArray}) where {N₁,N₂} = splitdims(sview(t.parent.data,:,:)', dims(codomain(t)), dims(domain(t)))
Base.setindex!(t::AdjointTensorMap{<:Any,N₁,N₂,<:AbstractArray}, v) where {N₁,N₂} = copy!(splitdims(sview(t.parent.data,:,:)', dims(codomain(t)), dims(domain(t))), v)

# TensorMap multiplication:
#--------------------------
function mul!(tC::TensorMap, tA::AdjointTensorMap,  tB::TensorMap)
    (codomain(tC) == codomain(tA) && domain(tC) == domain(tB) && domain(tA) == codomain(tB)) || throw(SpaceMismatch())
    for c in blocksectors(tC)
        if hasblock(tA.parent, c) # then also tB should have such a block
            Ac_mul_B!(block(tC, c), block(tA.parent, c), block(tB, c))
        else
            fill!(block(tC, c), 0)
        end
    end
    return tC
end
function mul!(tC::TensorMap, tA::TensorMap,  tB::AdjointTensorMap)
    (codomain(tC) == codomain(tA) && domain(tC) == domain(tB) && domain(tA) == codomain(tB)) || throw(SpaceMismatch())
    for c in blocksectors(tC)
        if hasblock(tA, c) # then also tB should have such a block
            A_mul_Bc!(block(tC, c), block(tA, c), block(tB.parent, c))
        else
            fill!(block(tC, c), 0)
        end
    end
    return tC
end
function mul!(tC::TensorMap, tA::AdjointTensorMap,  tB::AdjointTensorMap)
    (codomain(tC) == codomain(tA) && domain(tC) == domain(tB) && domain(tA) == codomain(tB)) || throw(SpaceMismatch())
    for c in blocksectors(tC)
        if hasblock(tA.parent, c) # then also tB should have such a block
            Ac_mul_Bc!(block(tC, c), block(tA.parent, c), block(tB.parent, c))
        else
            fill!(block(tC, c), 0)
        end
    end
    return tC
end
