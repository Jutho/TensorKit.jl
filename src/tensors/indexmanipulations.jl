# Index manipulations
#---------------------
"""
    permuteind(tsrc::AbstractTensorMap{S}, p1::NTuple{N₁,Int}, p2::NTuple{N₂,Int} = ()) -> tdst::TensorMap{S,N₁,N₂}

Permutes the indices of `tsrc::AbstractTensorMap{S}` such that a new tensor
`tdst::TensorMap{S,N₁,N₂}` is obtained, with indices in `p1` playing the role of
the codomain or range of the map, and indices in `p2` indicating the domain.

To permute into an existing `tdst`, use `permuteind!(tdst, tsrc, p1, p2)`.
"""
function permuteind(t::AbstractTensorMap{S}, p1::IndexTuple{N₁},  p2::IndexTuple{N₂}=(); copy::Bool = false) where {S,N₁,N₂}
    if !copy
        # share data if possible
        if (p1..., p2...) == ntuple(identity, StaticLength(N₁)+StaticLength(N₂))
            if isa(t, AbstractTensorMap{S,N₁,N₂})
                return t
            elseif isa(t, TensorMap) && sectortype(S) == Trivial
                spacet = codomain(t) ⊗ dual(domain(t))
                cod = spacet[map(n->tensor2spaceindex(t,n), p1)]
                dom = dual(spacet[map(n->tensor2spaceindex(t,n), reverse(p2))])
                return TensorMap(reshape(t.data, dim(cod), dim(dom)), cod, dom)
            end
        end
    end
    # general case
    @inbounds begin
        return permuteind!(similar_from_indices(eltype(t), p1, p2, t), t, p1, p2)
    end
end

@propagate_inbounds permuteind!(tdst::AbstractTensorMap{S,N₁,N₂}, tsrc::AbstractTensorMap{S}, p1::IndexTuple{N₁},  p2::IndexTuple{N₂}=()) where {S,N₁,N₂} = add!(1,tsrc,0,tdst,p1,p2)

# Index manipulations that increase or decrease the number of indices

function splitind end#

function fuseind end

# # TODO: reconsider whether we need repartitionind!, or if we just want permuteind!
# function repartitionind!(tdst::TensorMap{S,N₁,N₂}, tsrc::TensorMap{S,N₁′,N₂′}) where {S,N₁,N₂,N₁′,N₂′}
#     space1 = codomain(tdst) ⊗ dual(domain(tdst))
#     space2 = codomain(tsrc) ⊗ dual(domain(tsrc))
#     space1 == space2 || throw(SpaceMismatch())
#     p = (ntuple(n->n, StaticLength(N₁′))..., ntuple(n->N₁′+N₂′+1-n, StaticLength(N₂′)))
#     p1 = TupleTools.getindices(p, ntuple(n->n, StaticLength(N₁)))
#     p2 = reverse(TupleTools.getindices(p, ntuple(n->N₁+n, StaticLength(N₂))))
#     pdata = (p1..., p2...)
#
#     if sectortype(S) == Trivial
#         TensorOperations.add!(1, tsrc[], Val{:N}, 0, tdst[], pdata)
#     else
#         fill!(tdst, 0)
#         for (f1,f2) in fusiontrees(t)
#             for ((f1′,f2′), coeff) in repartition(f1, f2, StaticLength(N₁))
#                 TensorOperations.add!(coeff, tsrc[f1,f2], Val{:N}, 1, tdst[f1′,f2′], pdata)
#             end
#         end
#     end
#     return tdst
# end
