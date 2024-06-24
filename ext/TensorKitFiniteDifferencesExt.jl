module TensorKitFiniteDifferencesExt

using TensorKit
using TensorKit: sqrtdim, isqrtdim
using VectorInterface: scale!
using FiniteDifferences

function FiniteDifferences.to_vec(t::T) where {T<:TensorKit.TrivialTensorMap}
    vec, from_vec = to_vec(t.data)
    return vec, x -> T(from_vec(x), codomain(t), domain(t))
end
function FiniteDifferences.to_vec(t::AbstractTensorMap)
    # convert to vector of vectors to make use of existing functionality
    vec_of_vecs = [b * sqrtdim(c) for (c, b) in blocks(t)]
    vec, back = FiniteDifferences.to_vec(vec_of_vecs)

    function from_vec(x)
        t′ = similar(t)
        xvec_of_vecs = back(x)
        for (i, (c, b)) in enumerate(blocks(t′))
            scale!(b, xvec_of_vecs[i], isqrtdim(c))
        end
        return t′
    end

    return vec, from_vec
end
FiniteDifferences.to_vec(t::TensorKit.AdjointTensorMap) = to_vec(copy(t))

end
