# Index manipulations
#---------------------
"""
    permuteind(tsrc::AbstractTensorMap{S}, p1::NTuple{N₁,Int}, p2::NTuple{N₂,Int} = ()) -> tdst::TensorMap{S,N₁,N₂}

Permute the indices of `tsrc::AbstractTensorMap{S}` such that a new tensor
`tdst::TensorMap{S,N₁,N₂}` is obtained, with indices in `p1` playing the role of
the codomain or range of the map, and indices in `p2` indicating the domain.

To permute into an existing `tdst`, use `permuteind!(tdst, tsrc, p1, p2)`.
"""
function permuteind(t::TensorMap{S}, p1::IndexTuple{N₁},  p2::IndexTuple{N₂}=(); copy::Bool = false) where {S,N₁,N₂}
    cod = ProductSpace{S,N₁}(map(n->space(t, n), p1))
    dom = ProductSpace{S,N₂}(map(n->dual(space(t, n)), p2))

    if !copy
        # share data if possible
        if p1 === codomainind(t) && p2 === domainind(t)
            return t
        elseif isa(t, TensorMap) && sectortype(S) == Trivial
            stridet = i->stride(t[], i)
            sizet = i->size(t[], i)
            canfuse1, d1, s1 = TensorOperations._canfuse(sizet.(p1), stridet.(p1))
            canfuse2, d2, s2 = TensorOperations._canfuse(sizet.(p2), stridet.(p2))
            if canfuse1 && canfuse2 && s1 == 1 && (d2 == 1 || s2 == d1)
                return TensorMap(reshape(t.data, dim(cod), dim(dom)), cod, dom)
            end
        end
    end
    # general case
    @inbounds begin
        return permuteind!(similar(t, cod←dom), t, p1, p2)
    end
end

function permuteind(t::AdjointTensorMap{S}, p1::IndexTuple{N₁},  p2::IndexTuple{N₂}=(); copy::Bool = false) where {S,N₁,N₂}
    p1′ = map(n->adjointtensorindex(t, p2))
    p2′ = map(n->adjointtensorindex(t, p1))
    adjoint(permuteind(adjoint(t), p1′, p2′); copy = copy)
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
