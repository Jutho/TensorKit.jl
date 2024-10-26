# Planned changes for v1.0

Features that are planned to be implemented before the release of v1.0.0, in no particular order.

- [x] Separate `Sectors` module
- [x] Make `TrivialTensorMap` and `TensorMap` be the same
- [x] Simplify `TensorMap` type to hide `rowr` and `colr`
- [x] Change block order in `rowr` / `colr` to speed up particular contractions
- [x] Make `AdjointTensorMap` generic
- [ ] Rewrite planar operations in order to be AD-compatible
- [x] Fix rrules for fermionic tensors
- [ ] Fix GPU support
- [ ] Proper threading support
- [ ] Rewrite documentation

# Changelog

### `AbstractTensorMap{E,S,N₁,N₂}`

This adds the scalar type as a parameter to the `AbstractTensorMap` type. This is useful in
the contexts where different types of tensors are used (`AdjointTensor`, `BraidingTensor`,
...), which still have the same scalartype. Additionally, this removes many specializations
for methods in order to reduce ambiguity errors.

### `copy(BraidingTensor)`

This PR changes the behaviour of `copy` as an instantiator for creating a `TensorMap` from a
`BraidingTensor`. The rationale is that while this does sometimes happen in Julia `Base`,
this is always in the context of lazy wrapper types, for which it makes sense to not copy
the parent and then create a new wrapper. `BraidingTensor` does not wrap anything, so this
definition makes less sense.

### Refactor tensormap constructors

This PR refactors the constructors for `TensorMap` to be more in line with `Array`
constructors in Julia. In particular, this means that the default way to create
uninitialized tensors is now `TensorMap{E}(undef, codomain ← domain)`, reminiscent of
`Array{E}(undef, dims)`. Several convenience constructors are also added: `ones`, `zeros`,
`rand` and `randn` construct tensors when `dims` is replaced by `domain ← codomain`.

### TensorOperations v5

This PR bumps the compatibility of `TensorOperations` to v5. This is a breaking change
as there are some changes in the API.

### TensorKitSectors

This promotes TensorKitSectors to its own package, in order to make the dependencies
lighter and to separate the concerns of the two packages.

### FusionTree vertices

In order to simplify the `FusionTree` struct, we removed the usage of anything other than
`Int` as a vertex-label when working with `GenericFusion` sectors.

### TensorStructure

This PR changed the data structure of `TensorMap` to consist of a single vector of data,
where all blocks and subblocks are various views into this object. This entails major
simplifications for the outwards-facing interface, as now the `TensorMap` type requires less
parameters, and the fusion-tree structure is more clearly considered as an implementation
detail that can be left hidden to the user. Various other improvements and documentation
work was also carried out.