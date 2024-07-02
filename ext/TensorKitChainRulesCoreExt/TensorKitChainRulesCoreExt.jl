module TensorKitChainRulesCoreExt

using TensorOperations
using VectorInterface
using TensorKit
using ChainRulesCore
using LinearAlgebra
using TupleTools

import TensorOperations as TO
using TensorOperations: Backend, promote_contract
using VectorInterface: promote_scale, promote_add

ext = @static if isdefined(Base, :get_extension)
    Base.get_extension(TensorOperations, :TensorOperationsChainRulesCoreExt)
else
    TensorOperations.TensorOperationsChainRulesCoreExt
end
const _conj = ext._conj
const trivtuple = ext.trivtuple

include("utility.jl")
include("constructors.jl")
include("linalg.jl")
include("tensoroperations.jl")
include("factorizations.jl")

end
