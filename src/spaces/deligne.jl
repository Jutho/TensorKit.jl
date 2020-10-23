"""
    ⊠(V₁::VectorSpace, V₂::VectorSpace)

Given two vector spaces `V₁` and `V₂` (`ElementarySpace` or `ProductSpace`), or thus,
objects of corresponding fusion categories ``C₁`` and ``C₂``, ``V₁ ⊠ V₂`` constructs the
Deligne tensor product, an object in ``C₁ ⊠ C₂`` which is the natural tensor product of
those categories. In particular, the corresponding type of sectors (simple objects) is
given by `sectortype(V₁ ⊠ V₂) == sectortype(V₁) ⊠ sectortype(V₂)` and can be thought of as
a tuple of the individual sectors.

The Deligne tensor product also works in the type domain and for sectors and tensors. For
group representations, we have `Rep[G₁] ⊠ Rep[G₂] == Rep[G₁ × G₂]`, i.e. these are the
natural representation spaces of the direct product of two groups.
"""
⊠(V1::VectorSpace, V2::VectorSpace) = (V1 ⊠ one(V2)) ⊗ (one(V1) ⊠ V2)

# define deligne products with empty tensor product: just add a trivial sector of the type of the empty space to each of the sectors in the non-empty space
function ⊠(V::GradedSpace, P0::ProductSpace{<:EuclideanSpace{ℂ},0})
    I1 = sectortype(V)
    I2 = sectortype(P0)
    return Vect[I1 ⊠ I2](ifelse(isdual(V), dual(c), c) ⊠ one(I2) => dim(V, c) for c in sectors(V); dual = isdual(V))
end

function ⊠(P0::ProductSpace{<:EuclideanSpace{ℂ},0}, V::GradedSpace)
    I1 = sectortype(P0)
    I2 = sectortype(V)
    return Vect[I1 ⊠ I2](one(I1) ⊠ ifelse(isdual(V), dual(c), c) => dim(V, c) for c in sectors(V); dual = isdual(V))
end

function ⊠(V::ComplexSpace, P0::ProductSpace{<:EuclideanSpace{ℂ},0})
    I2 = sectortype(P0)
    return Vect[I2](one(I2) => dim(V); dual = isdual(V))
end

function ⊠(P0::ProductSpace{<:EuclideanSpace{ℂ},0}, V::ComplexSpace)
    I1 = sectortype(P0)
    return Vect[I1](one(I1) => dim(V); dual = isdual(V))
end

function ⊠(P::ProductSpace{<:EuclideanSpace{ℂ},0}, P0::ProductSpace{<:EuclideanSpace{ℂ},0})
    I1 = sectortype(P)
    I2 = sectortype(P0)
    return one(Vect[I1 ⊠ I2])
end

function ⊠(P::ProductSpace{<:EuclideanSpace{ℂ}}, P0::ProductSpace{<:EuclideanSpace{ℂ},0})
    I1 = sectortype(P)
    I2 = sectortype(P0)
    S = Vect[I1 ⊠ I2]
    N = length(P)
    return ProductSpace{S,N}(map(V->V ⊠ P0, tuple(P...)))
end

function ⊠(P0::ProductSpace{<:EuclideanSpace{ℂ},0}, P::ProductSpace{<:EuclideanSpace{ℂ}})
    I1 = sectortype(P0)
    I2 = sectortype(P)
    S = Vect[I1 ⊠ I2]
    N = length(P)
    return ProductSpace{S,N}(map(V->P0 ⊠ V, tuple(P...)))
end
