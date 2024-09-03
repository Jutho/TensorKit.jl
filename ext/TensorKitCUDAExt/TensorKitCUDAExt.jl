module TensorKitCUDAExt

using CUDA
using cuTENSOR: cuTENSOR

using TensorKit
using TensorKit: TrivialTensor
using TensorKit: SectorDict
using Random: Random

include("cutensormap.jl")
include("cuda_fixes.jl")

end
