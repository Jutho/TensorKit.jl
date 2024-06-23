module TensorKitFiniteDifferencesExt

using TensorKit
using FiniteDifferences

function FiniteDifferences.to_vec(t::T) where {T<:TensorKit.TrivialTensorMap}
    vec, from_vec = to_vec(t.data)
    return vec, x -> T(from_vec(x), codomain(t), domain(t))
end
function FiniteDifferences.to_vec(t::AbstractTensorMap)
    vec = mapreduce(vcat, blocks(t); init=scalartype(t)[]) do (c, b)
        return reshape(b, :) .* sqrt(dim(c))
    end
    vec_real = scalartype(t) <: Real ? vec : collect(reinterpret(real(scalartype(t)), vec))

    function from_vec(x_real)
        x = scalartype(t) <: Real ? x_real : reinterpret(scalartype(t), x_real)
        t′ = similar(t)
        ctr = 0
        for (c, b) in blocks(t′)
            n = length(b)
            copyto!(b, reshape(view(x, ctr .+ (1:n)), size(b)) ./ sqrt(dim(c)))
            ctr += n
        end
        return t′
    end
    return vec_real, from_vec
end
FiniteDifferences.to_vec(t::TensorKit.AdjointTensorMap) = to_vec(copy(t))

end
