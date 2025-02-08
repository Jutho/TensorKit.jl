module TensorKitFiniteDifferencesExt

using TensorKit
using TensorKit: sqrtdim, invsqrtdim
using VectorInterface: scale!
using FiniteDifferences

function FiniteDifferences.to_vec(t::AbstractTensorMap)
    # convert to vector of vectors to make use of existing functionality
    vec_of_vecs = [b * sqrtdim(c) for (c, b) in blocks(t)]
    vec, back = FiniteDifferences.to_vec(vec_of_vecs)

    function from_vec(x)
        t′ = similar(t)
        xvec_of_vecs = back(x)
        for (i, (c, b)) in enumerate(blocks(t′))
            scale!(b, xvec_of_vecs[i], invsqrtdim(c))
        end
        return t′
    end

    return vec, from_vec
end
FiniteDifferences.to_vec(t::TensorKit.AdjointTensorMap) = to_vec(copy(t))

function FiniteDifferences.to_vec(t::DiagonalTensorMap)
    x_vec, back = to_vec(TensorMap(t))
    function DiagonalTensorMap_from_vec(x_vec)
        return DiagonalTensorMap(back(x_vec))
    end
    return x_vec, DiagonalTensorMap_from_vec
end

end

# TODO: Investigate why the approach below doesn't work
# module TensorKitFiniteDifferencesExt

# using TensorKit
# using TensorKit: sqrtdim, invsqrtdim
# using VectorInterface: scale!
# using FiniteDifferences

# function FiniteDifferences.to_vec(t::AbstractTensorMap{T}) where {T}
#     # convert to vector of vectors to make use of existing functionality
#     structure = TensorKit.fusionblockstructure(t)
#     vec = storagetype(t)(undef, structure.totaldim)
#     for (c, ((d₁, d₂), r)) in structure.blockstructure
#         scale!(reshape(view(vec, r), (d₁, d₂)), block(t, c), sqrtdim(c))
#     end

#     function from_vec(x)
#         y = T <: Complex ? reinterpret(T, x) : x
#         t′ = similar(t)
#         for (c, ((d₁, d₂), r)) in structure.blockstructure
#             scale!(block(t′, c), reshape(view(y, r), (d₁, d₂)), invsqrtdim(c))
#         end
#         return t′
#     end

#     return T <: Complex ? reinterpret(real(T), vec) : vec, from_vec
# end

# end
