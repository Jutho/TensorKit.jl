# Tensors

```@meta
CurrentModule = TensorKit
```

## Type hierarchy

```@docs
AbstractTensorMap
AbstractTensor
TensorMap
AdjointTensorMap
```

## Specific `TensorMap` constructors

```@docs
id
isomorphism
unitary
isometry
```

## `TensorMap` operations

```@docs
permute(t::TensorMap{S}, p1::IndexTuple, p2::IndexTuple) where {S}
permute!
braid
braid!
twist
twist!
add!
trace!
contract!
```


## `TensorMap` factorizations

```@docs
leftorth
rightorth
leftnull
rightnull
tsvd
eigh
eig
TensorKit.eigen
```
