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
ZNSpace
CompositeSpace
ProductSpace
```

The type hierarchy for representing sectors
```@docs
Sector
AbelianIrrep
ZNIrrep{N}
U1Irrep
SU2Irrep
CU1Irrep
FusionStyle
Abelian
NonAbelian
SimpleNonAbelian
DegenerateNonAbelian
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
The following methods act specifically `ElementarySpace` spaces
```
isdual
dual
conj
flip
:⊕
oneunit
```
or also on `ProductSpace`

```
fuse
:⊗
one
```
