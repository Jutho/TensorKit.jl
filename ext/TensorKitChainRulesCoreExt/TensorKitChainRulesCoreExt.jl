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

_conj(conjA::Symbol) = conjA == :C ? :N : :C
trivtuple(N) = ntuple(identity, N)

include("utility.jl")
include("constructors.jl")
include("linalg.jl")
include("tensoroperations.jl")
include("factorizations.jl")

end
