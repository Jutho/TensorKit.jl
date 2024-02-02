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
CartesianSpace
ComplexSpace
GradedSpace
CompositeSpace
ProductSpace
```

## Useful constants
```@docs
Vect
Rep
ZNSpace{N}
Z2Space
Z3Space
Z4Space
U1Space
SU2Space
CU1Space
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
blocksectors(::ProductSpace)
blocksectors(::HomSpace)
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
⊠(::VectorSpace, ::VectorSpace)
one
ismonomorphic
isepimorphic
isisomorphic
insertunit
```
