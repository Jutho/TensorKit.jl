⊠(V1::VectorSpace, V2::VectorSpace) = (V1 ⊠ one(V2)) ⊗ (one(V1) ⊠ V2)

# define deligne products with empty tensor product: just add a trivial sector of the type of the empty space to each of the sectors in the non-empty space
function ⊠(V::GradedSpace, P0::ProductSpace{<:EuclideanSpace{ℂ},0})
    I1 = sectortype(V)
    I2 = sectortype(P0)
    return GradedSpace[I1 ⊠ I2](ifelse(isdual(V), dual(c), c) ⊠ one(I2) => dim(V, c) for c in sectors(V); dual = isdual(V))
end

function ⊠(P0::ProductSpace{<:EuclideanSpace{ℂ},0}, V::GradedSpace)
    I1 = sectortype(P0)
    I2 = sectortype(V)
    return GradedSpace[I1 ⊠ I2](one(I1) ⊠ ifelse(isdual(V), dual(c), c) => dim(V, c) for c in sectors(V); dual = isdual(V))
end

function ⊠(V::ComplexSpace, P0::ProductSpace{<:EuclideanSpace{ℂ},0})
    I2 = sectortype(P0)
    return GradedSpace[I2](one(I2) => dim(V); dual = isdual(V))
end

function ⊠(P0::ProductSpace{<:EuclideanSpace{ℂ},0}, V::ComplexSpace)
    I1 = sectortype(P0)
    return GradedSpace[I1](one(I1) => dim(V); dual = isdual(V))
end

function ⊠(P::ProductSpace{<:EuclideanSpace{ℂ},0}, P0::ProductSpace{<:EuclideanSpace{ℂ},0})
    I1 = sectortype(P)
    I2 = sectortype(P0)
    return one(GradedSpace[I1 ⊠ I2])
end

function ⊠(P::ProductSpace{<:EuclideanSpace{ℂ}}, P0::ProductSpace{<:EuclideanSpace{ℂ},0})
    I1 = sectortype(P)
    I2 = sectortype(P0)
    S = GradedSpace[I1 ⊠ I2]
    N = length(P)
    return ProductSpace{S,N}(map(V->V ⊠ P0, tuple(P...)))
end

function ⊠(P0::ProductSpace{<:EuclideanSpace{ℂ},0}, P::ProductSpace{<:EuclideanSpace{ℂ}})
    I1 = sectortype(P0)
    I2 = sectortype(P)
    S = GradedSpace[I1 ⊠ I2]
    N = length(P)
    return ProductSpace{S,N}(map(V->P0 ⊠ V, tuple(P...)))
end
