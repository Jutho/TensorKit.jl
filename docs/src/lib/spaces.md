# Vector spaces

```@meta
CurrentModule = TensorKit
```

## Type hierarchy

```@docs
Field
VectorSpace
ElementarySpace
GeneralSpace
InnerProductSpace
EuclideanSpace
CartesianSpace
ComplexSpace
GradedSpace
CompositeSpace
ProductSpace
```

## Methods
Methods often apply similar to e.g. spaces and corresponding tensors or tensor maps, e.g.:
```@docs
field
sectortype
sectors
hassector
dim
dims
blocksectors
blockdim
space
```

The following methods act specifically on `ElementarySpace` spaces:
```@docs
isdual
dual
conj
flip
:⊕
oneunit
supremum
infimum
```
while the following also work on both `ElementarySpace` and `ProductSpace`

```@docs
fuse
:⊗
:⊠
one
ismonomorphic
isepimorphic
isisomorphic
insertunit
```
