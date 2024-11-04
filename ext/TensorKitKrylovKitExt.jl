module TensorKitKrylovKitExt

using TensorKit, KrylovKit

# AbstractTensorMap as KrylovKit operator
KrylovKit.apply(A::AbstractTensorMap, x::AbstractTensorMap) = A * x
KrylovKit.apply_normal(A::AbstractTensorMap, x::AbstractTensorMap) = A * x
KrylovKit.apply_adjoint(A::AbstractTensorMap, x::AbstractTensorMap) = A' * x

end