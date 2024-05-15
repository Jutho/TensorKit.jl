# Tensors

```@meta
CurrentModule = TensorKit
```

## Type hierarchy

The type hierarchy of tensors is as follows:

```@docs
AbstractTensorMap
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

## `TensorMap` constructors

### General constructors

A general `TensorMap` can be constructed by specifying its data, codmain and domain in one
of the following ways:
```@docs
TensorMap(::AbstractDict{<:Sector,<:DenseMatrix}, ::ProductSpace{S,N₁},
                   ::ProductSpace{S,N₂}) where {S<:IndexSpace,N₁,N₂}
TensorMap(::Any, ::Type{T}, codom::ProductSpace{S},
                   dom::ProductSpace{S}) where {S<:IndexSpace,T<:Number}
TensorMap(::DenseArray, ::ProductSpace{S,N₁}, ::ProductSpace{S,N₂};
                   tol) where {S<:IndexSpace,N₁,N₂}
```

Several special-purpose methods exist to generate data according to specific distributions:
```@docs
randuniform
randnormal
randisometry
```

### Specific constructors

Additionally, several special-purpose constructors exist to generate data according to specific distributions:
```@docs
id
isomorphism
unitary
isometry
```

## Accessing properties and data

The following methods exist to obtain type information:

```@docs
spacetype
sectortype(::Type{<:AbstractTensorMap{S}}) where {S<:IndexSpace}
storagetype
tensormaptype
```

To obtain information about the indices, you can use:
```@docs
domain
codomain
space(::AbstractTensorMap)
numin
numout
numind
codomainind
domainind
allind
```

To obtain information about the data, the following methods exist:
```@docs
blocksectors(::AbstractTensorMap)
blockdim(::AbstractTensorMap, ::Sector)
block
blocks
fusiontrees(::AbstractTensorMap)
hasblock
```

For `TensorMap`s with `Trivial` `sectortype`, the data can be directly accessed and
manipulated in a straightforward way:
```@docs
Base.getindex(t::TrivialTensorMap)
Base.getindex(t::TrivialTensorMap, indices::Vararg{Int})
Base.setindex!(t::TrivialTensorMap, ::Any, indices::Vararg{Int})
```

For general `TensorMap`s, this can be done using custom `getindex` and `setindex!` methods:
```@docs
Base.getindex(t::TensorMap{<:IndexSpace,N₁,N₂,I},
              sectors::Tuple{Vararg{I}}) where {N₁,N₂,I<:Sector}
Base.getindex(t::TensorMap{<:IndexSpace,N₁,N₂,I},
              f₁::FusionTree{I,N₁},
              f₂::FusionTree{I,N₂}) where {N₁,N₂,I<:Sector}
Base.setindex!(::TensorMap{<:IndexSpace,N₁,N₂,I}, ::Any, ::FusionTree{I,N₁}, ::FusionTree{I,N₂}) where {N₁,N₂,I<:Sector}
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
transpose(::AbstractTensorMap, ::Index2Tuple)
repartition(::AbstractTensorMap, ::Int, ::Int)
twist(::AbstractTensorMap, ::Int)
```
```@docs
permute!(tdst::AbstractTensorMap{S,N₁,N₂}, tsrc::AbstractTensorMap{S}, p::Index2Tuple{N₁,N₂}) where {S,N₁,N₂}
braid!
transpose!
repartition!(::AbstractTensorMap{S}, ::AbstractTensorMap{S}) where {S}
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
