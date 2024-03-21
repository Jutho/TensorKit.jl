# Planned changes for v1.0

Features that are planned to be implemented before the release of v1.0.0, in no particular order.

- [ ] Separate `Sectors` module
- [ ] Make `TrivialTensorMap` and `TensorMap` be the same
- [ ] Simplify `TensorMap` type to hide `rowr` and `colr`
- [ ] Make `AdjointTensorMap` generic
- [ ] Rewrite planar operations in order to be AD-compatible
- [ ] Fix rrules for fermionic tensors
- [ ] Fix GPU support
- [ ] Proper threading support
- [ ] Rewrite documentation

# Changelog

### `AbstractTensorMap{E,S,N₁,N₂}`

This adds the scalar type as a parameter to the `AbstractTensorMap` type. This is useful in
the contexts where different types of tensors are used (`AdjointTensor`, `BraidingTensor`,
...), which still have the same scalartype. Additionally, this removes many specializations
for methods in order to reduce ambiguity errors.
