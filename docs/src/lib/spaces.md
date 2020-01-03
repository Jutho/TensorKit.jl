# Vector spaces

```@meta
CurrentModule = TensorKit
```

The type hierarchy for representing vector spaces
```@docs
VectorSpace
ElementarySpace
GeneralSpace
InnerProductSpace
EuclideanSpace
CartesianSpace
ComplexSpace
RepresentationSpace
GenericRepresentationSpace
FiniteRepresentationSpace
CompositeSpace
ProductSpace
```

The type hierarchy for representing sectors
```@docs
Sector
FusionStyle
BraidingStyle
Irrep
AbelianIrrep
ZNIrrep
U1Irrep
SU2Irrep
CU1Irrep
```

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
```
The following methods act specifically on `ElementarySpace` spaces:
```@docs
isdual
dual
conj
flip
:⊕
oneunit
```
or also on `ProductSpace`

```@docs
fuse
:⊗
one
min
max
```
