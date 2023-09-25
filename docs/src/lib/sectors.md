# Symmetry sectors an fusion trees

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
ZNIrrep
U1Irrep
SU2Irrep
CU1Irrep
FermionParity
FibonacciAnyon
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
twist
Base.isreal(::Type{<:Sector})
TensorKit.vertex_labeltype
TensorKit.vertex_ind2label
⊠(::Sector, ::Sector)
```

## Methods for manipulating fusion trees or pairs of fusion-splitting trees
The main method for manipulating a fusion-splitting tree pair is
```@docs
braid(f1::FusionTree{G}, f2::FusionTree{G}, levels1::IndexTuple, levels2::IndexTuple,
        p1::IndexTuple{N₁}, p2::IndexTuple{N₂}) where {G<:Sector,N₁,N₂}
```
which, for `FusionStyle(G) isa SymmetricBraiding`, simplifies to
```@docs
permute(f1::FusionTree{G}, f2::FusionTree{G},
        p1::IndexTuple{N₁}, p2::IndexTuple{N₂}) where {G<:Sector,N₁,N₂}
```
These operations are implemented by composing the following more elementary manipulations
```@docs
braid(f::FusionTree{G,N}, levels::NTuple{N,Int}, p::NTuple{N,Int}) where {G<:Sector, N}
permute(f::FusionTree{G,N}, p::NTuple{N,Int}) where {G<:Sector, N}
TensorKit.repartition
TensorKit.artin_braid
```
Finally, there are some additional manipulations for internal use
```@docs
TensorKit.insertat
TensorKit.split
TensorKit.merge
```
