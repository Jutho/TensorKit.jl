import Base: transpose

#! format: off
Base.@deprecate(permute(t::AbstractTensorMap, p1::IndexTuple, p2::IndexTuple; copy::Bool=false),
                permute(t, (p1, p2); copy=copy))
Base.@deprecate(transpose(t::AbstractTensorMap, p1::IndexTuple, p2::IndexTuple; copy::Bool=false),
                transpose(t, (p1, p2); copy=copy))
Base.@deprecate(braid(t::AbstractTensorMap, p1::IndexTuple, p2::IndexTuple, levels; copy::Bool=false),
                braid(t, (p1, p2), levels; copy=copy))

Base.@deprecate(tsvd(t::AbstractTensorMap, p₁::IndexTuple, p₂::IndexTuple; kwargs...),
                tsvd(t, (p₁, p₂); kwargs...))
Base.@deprecate(leftorth(t::AbstractTensorMap, p₁::IndexTuple, p₂::IndexTuple; kwargs...),
                leftorth(t, (p₁, p₂); kwargs...))
Base.@deprecate(rightorth(t::AbstractTensorMap, p₁::IndexTuple, p₂::IndexTuple; kwargs...),
                rightorth(t, (p₁, p₂); kwargs...))
Base.@deprecate(leftnull(t::AbstractTensorMap, p₁::IndexTuple, p₂::IndexTuple; kwargs...),
                leftnull(t, (p₁, p₂); kwargs...))
Base.@deprecate(rightnull(t::AbstractTensorMap, p₁::IndexTuple, p₂::IndexTuple; kwargs...),
                rightnull(t, (p₁, p₂); kwargs...))
Base.@deprecate(LinearAlgebra.eigen(t::AbstractTensorMap, p₁::IndexTuple, p₂::IndexTuple; kwargs...),
                LinearAlgebra.eigen(t, (p₁, p₂); kwargs...), false)
Base.@deprecate(eig(t::AbstractTensorMap, p₁::IndexTuple, p₂::IndexTuple; kwargs...),
                eig(t, (p₁, p₂); kwargs...))
Base.@deprecate(eigh(t::AbstractTensorMap, p₁::IndexTuple, p₂::IndexTuple; kwargs...),
                eigh(t, (p₁, p₂); kwargs...))

for f in (:rand, :randn, :zeros, :ones)
    @eval begin
        Base.@deprecate TensorMap(::typeof($f), T::Type, P::HomSpace) $f(T, P)
        Base.@deprecate TensorMap(::typeof($f), P::HomSpace) $f(P)
        Base.@deprecate TensorMap(::typeof($f), T::Type, cod::TensorSpace, dom::TensorSpace) $f(T, P, cod, dom)
        Base.@deprecate TensorMap(::typeof($f), cod::TensorSpace, dom::TensorSpace) $f(cod, dom)
        Base.@deprecate Tensor(::typeof($f), T::Type, space::TensorSpace) $f(T, space)
        Base.@deprecate Tensor(::typeof($f), space::TensorSpace) $f(space)
    end
end

Base.@deprecate EuclideanProduct() EuclideanInnerProduct()

#! format: on
