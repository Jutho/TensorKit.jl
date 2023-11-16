# Tensors

```@meta
CurrentModule = TensorKit
```

## Type hierarchy

The type hierarchy of tensors is as follows:

```@docs
AbstractTensorMap
AbstractTensor
TensorMap
AdjointTensorMap
BraidingTensor
```

Some aliases are provided for convenience:

```@docs
AbstractTensor
Tensor
TrivialTensorMap
TrivialTensor
```

## Specific `TensorMap` constructors

```@docs
id
isomorphism
unitary
isometry
```

Additionally, several special-purpose methods exist to generate data according to specific distributions:

```@docs
randuniform
randnormal
randisometry
```

## Accessing properties and data

The following methods exist to obtain type information:

```@docs
spacetype
sectortype
storagetype
tensormaptype
```

To obtain information about the indices, you can use:
```@docs
domain
codomain
space
numin
numout
numind
codomainind
domainind
allind
```

To obtain information about the data, the following methods exist:
```@docs
blocksectors
blockdim
block
blocks
fusiontrees
hasblock
```

## `TensorMap` operations

The operations that can be performed on a `TensorMap` can be organized into *index
manipulations*, *(planar) traces* and *(planar) contractions*.

### Index manipulations

A general index manipulation of a `TensorMap` object can be built up by considering some
transformation of the fusion trees, along with a permutation of the stored data. They come
in three flavours, which are either of the type `transform(!)` which are exported, or of the
type `add_transform!`, for additional expert-mode options that allows for addition and
scaling, as well as the selection of a custom backend.

```@docs
permute(t::AbstractTensorMap{S}, (p₁, p₂)::Index2Tuple{N₁,N₂}; copy::Bool=false) where {S,N₁,N₂}
braid(t::AbstractTensorMap{S}, (p₁, p₂)::Index2Tuple, levels::IndexTuple; copy::Bool=false) where {S}
transpose
twist
```
```@docs
permute!(tdst::AbstractTensorMap{S,N₁,N₂}, tsrc::AbstractTensorMap{S}, p::Index2Tuple{N₁,N₂}) where {S,N₁,N₂}
braid!
transpose!
twist!
```
```@docs
add_permute!
add_braid!
add_transpose!
add_transform!
```

### Traces and contractions

```@docs
trace_permute!
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
