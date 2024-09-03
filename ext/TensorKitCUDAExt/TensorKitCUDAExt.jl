module TensorKitCUDAExt

using CUDA
using cuTENSOR: cuTENSOR

using TensorKit
using TensorKit: TrivialTensor
using TensorKit: SectorDict
using TensorKit: OrthogonalFactorizationAlgorithm,
                 QL, QLpos, QR, QRpos, LQ, LQpos, RQ, RQpos, SVD, SDD, Polar

using LinearAlgebra: LinearAlgebra, Adjoint, Diagonal, BlasFloat
using Random: Random

include("cutensormap.jl")
include("cumatrixalgebra.jl")
include("cuda_fixes.jl")

end
