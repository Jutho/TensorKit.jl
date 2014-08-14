# Composite vector spaces
#-------------------------
# Obtained as (some subspace of) the tensor product elementary vector spaces
abstract CompositeSpace{S<:ElementarySpace} <: VectorSpace

Base.eltype{S<:ElementarySpace}(V::CompositeSpace{S}) = eltype(S)
Base.eltype{S<:ElementarySpace}(::Type{CompositeSpace{S}}) = eltype(S)
Base.eltype{S<:CompositeSpace}(::Type{S}) = eltype(super(S))

# Composite spaces with finite number of elements N, defining spaces of tensors
abstract TensorSpace{S<:ElementarySpace,N} <: CompositeSpace{S}
Base.isfinite(::TensorSpace) = true

# Explicit realizations:
#------------------------
# ProductSpace: type and methods for tensor products of ElementarySpace objects
include("spaces/productspace.jl")

# InvariantSpace: invariant subspace of tensor product of UnitaryRepresentationSpace objects
include("spaces/invariantspace.jl")

# Other examples might include:
# braidedspace and fermionspace
# symmetric and antisymmetric subspace of a tensor product of identical vector spaces
# ...
