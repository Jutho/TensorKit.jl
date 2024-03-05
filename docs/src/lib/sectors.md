# Symmetry sectors and fusion trees

```@meta
CurrentModule = TensorKit
```

## Type hierarchy
```@docs
Sector
SectorValues
FusionStyle
BraidingStyle
AbstractIrrep
Trivial
ZNIrrep
U1Irrep
SU2Irrep
CU1Irrep
ProductSector
FermionParity
FermionNumber
FermionSpin
FibonacciAnyon
IsingAnyon
FusionTree
```

## Useful constants
```@docs
Irrep
```

## Methods for defining and characterizing `Sector` subtypes
```@docs
Base.one(::Sector)
dual(::Sector)
Nsymbol
Fsymbol
Rsymbol
Bsymbol
dim(::Sector)
frobeniusschur
twist(::Sector)
Base.isreal(::Type{<:Sector})
TensorKit.vertex_labeltype
TensorKit.vertex_ind2label
⊠(::Sector, ::Sector)
fusiontrees(uncoupled::NTuple{N,I}, coupled::I,
                     isdual::NTuple{N,Bool}) where {N,I<:Sector}
```

## Methods for manipulating fusion trees

For manipulating single fusion trees, the following internal methods are defined:
```@docs
insertat
split
merge
elementary_trace
planar_trace(f::FusionTree{I,N}, q1::IndexTuple{N₃}, q2::IndexTuple{N₃}) where {I<:Sector,N,N₃}
artin_braid
braid(f::FusionTree{I,N}, levels::NTuple{N,Int}, p::NTuple{N,Int}) where {I<:Sector,N}
permute(f::FusionTree{I,N}, p::NTuple{N,Int}) where {I<:Sector,N}
```

These can be composed to manipulate fusion-splitting tree pairs, for which the following
methods are defined:

```@docs
bendright
bendleft
foldright
foldleft
cycleclockwise
cycleanticlockwise
repartition
transpose(f₁::FusionTree{I}, f₂::FusionTree{I},
                        p1::IndexTuple{N₁}, p2::IndexTuple{N₂}) where {I<:Sector,N₁,N₂}
braid(f₁::FusionTree{I}, f₂::FusionTree{I}, levels1::IndexTuple, levels2::IndexTuple, p1::IndexTuple{N₁}, p2::IndexTuple{N₂}) where {I<:Sector,N₁,N₂}
permute(f₁::FusionTree{I}, f₂::FusionTree{I}, p1::IndexTuple{N₁}, p2::IndexTuple{N₂}) where {I<:Sector,N₁,N₂}
```
