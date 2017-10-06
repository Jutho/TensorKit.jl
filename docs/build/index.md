
<a id='TensorKit.jl-1'></a>

# TensorKit.jl

- [TensorKit.jl](index.md#TensorKit.jl-1)

<a id='TensorKit.CartesianSpace' href='#TensorKit.CartesianSpace'>#</a>
**`TensorKit.CartesianSpace`** &mdash; *Type*.



```
struct immutable CartesianSpace <: EuclideanSpace{ℝ}
```

A CartesianSpace is a real euclidean space `ℝ^d` and therefore self-dual. It has no additonal structure and is completely characterised by its dimension `d`. This is the vector space that is implicitly assumed in most of matrix algebra.

<a id='TensorKit.ComplexSpace' href='#TensorKit.ComplexSpace'>#</a>
**`TensorKit.ComplexSpace`** &mdash; *Type*.



`immutable ComplexSpace <: EuclideanSpace{ℂ}`

A ComplexSpace is a standard complex vector space ℂ^d with Euclidean inner product and no additional structure. It is completely characterised by its dimension and whether its the normal space or its dual (which is canonically isomorphic to the conjugate space).

<a id='TensorKit.Field' href='#TensorKit.Field'>#</a>
**`TensorKit.Field`** &mdash; *Type*.



```
abstract type Field end
```

abstract type at the top of the type hierarchy for denoting fields over which vector spaces can be defined. Two common fields are `ℝ` and `ℂ`, representing the field of real or complex numbers respectively.

<a id='TensorKit.Sector' href='#TensorKit.Sector'>#</a>
**`TensorKit.Sector`** &mdash; *Type*.



```
abstract type Sector end
```

Abstract type for representing the label sets of graded vector spaces, which should correspond to (unitary) fusion categories.

Every new `G<:Sector` should implement the following methods:

  * `one(::Type{G})` -> unit element of `G`
  * `conj(a::G)` -> a̅: conjugate or dual label of a
  * `⊗(a::G, b::G)` -> iterable with fusion outputs of `a ⊗ b`
  * `Nsymbol(a::G, b::G, c::G)` -> number of times `c` appears in `a ⊗ b`
  * `fusiontype(::Type{G})` -> `Abelian`, `SimpleNonAbelian` or `DegenerateNonAbelian`
  * `braidingtype(::Type{G})` -> `Bosonic`, `Fermionic`, `Anyonic`, ...

and, if `fusiontype(G) <: NonAbelian`,

  * `Fsymbol(a::G, b::G, c::G, d::G, e::G, f::G)` -> F-symbol: scalar (in case of `SimpleNonAbelian`) or matrix (in case of `DegenerateNonAbelian`)
  * ... can all other information (quantum dimension, cups and caps) be extracted from `F`?

and if `braidingtype(G) == Fermionic`

  * `fermionparity(a::G)` -> `Bool` representing the fermion parity of sector `a`

and optionally, if if `fusiontype(G) == DegenerateNonAbelian`

  * `vertex_ind2label(i::Int, a::G, b::G, c::G)` -> a custom label for the `i`th copy of `c` appearing in `a ⊗ b`

<a id='TensorKit.VectorSpace' href='#TensorKit.VectorSpace'>#</a>
**`TensorKit.VectorSpace`** &mdash; *Type*.



```
abstract type VectorSpace end
```

abstract type at the top of the type hierarchy for denoting vector spaces

<a id='TensorKit.:⊗-Tuple{TensorKit.Trivial,TensorKit.Trivial}' href='#TensorKit.:⊗-Tuple{TensorKit.Trivial,TensorKit.Trivial}'>#</a>
**`TensorKit.:⊗`** &mdash; *Method*.



```
function ⊗(a::G, b::G) where {G<:Sector}
```

Returns an iterable of elements of `c::G` that appear in the fusion product `a ⊗ b`. Note that every element `c` should appear at most once, fusion degeneracies (if `fusiontype(G) == DegenerateNonAbelian`) should be accessed via `Nsymbol(a,b,c)`.

<a id='TensorKit.Bsymbol-Union{Tuple{G}, Tuple{G,G,G}} where G<:TensorKit.Sector' href='#TensorKit.Bsymbol-Union{Tuple{G}, Tuple{G,G,G}} where G<:TensorKit.Sector'>#</a>
**`TensorKit.Bsymbol`** &mdash; *Method*.



```
function Bsymbol(a::G, b::G, c::G) where {G<:Sector}
```

Returns the value of B^{a,b}_c which appears in transforming a splitting vertex into a fusion vertex using the transformation

```
a -<-μ-<- c                                 a -<-ν-<- c
     ∨          -> Bsymbol(a,b,c)[μ,ν]           ∧
     b                                         dual(b)
```

If `fusiontype(G)` is `Abelian` or `SimpleNonAbelian`, the B-symbol is a number. Otherwise it is a square matrix with row and column size `Nsymbol(a, b, c) == Nsymbol(c, dual(b), a)`.

<a id='TensorKit.Fsymbol' href='#TensorKit.Fsymbol'>#</a>
**`TensorKit.Fsymbol`** &mdash; *Function*.



```
function Fsymbol(a::G, b::G, c::G, d::G, e::G, f::G) where {G<:Sector}
```

Returns the F-symbol F^{a,b,c}_d that associates the two different fusion orders of sectors `a`, `b` and `c` into an ouput sector `d`, using either an intermediate sector `a ⊗ b → e` or `b ⊗ c → f`:

```
a-<-μ-<-e-<-ν-<-d                                     a-<-λ-<-d
    ∨       ∨       -> Fsymbol(a,b,c,d,e,f)[μ,ν,κ,λ]      ∨
    b       c                                         b-<-κ
                                                          ∨
                                                          c
```

If `fusiontype(G)` is `Abelian` or `SimpleNonAbelian`, the F-symbol is a number. Otherwise it is a rank 4 array of size `(Nsymbol(a,b,e), Nsymbol(e,c,d), Nsymbol(b,c,f), Nsymbol(a,f,d))`.

<a id='TensorKit.Nsymbol-Tuple{TensorKit.Trivial,TensorKit.Trivial,TensorKit.Trivial}' href='#TensorKit.Nsymbol-Tuple{TensorKit.Trivial,TensorKit.Trivial,TensorKit.Trivial}'>#</a>
**`TensorKit.Nsymbol`** &mdash; *Method*.



```
function Nsymbol(a::G, b::G, c::G) where {G<:Sector} -> Integer
```

Returns an `Integer` representing the number of times `c` appears in the fusion product `a ⊗ b`. Could be a `Bool` if `fusiontype(G) = Abelian` or `SimpleNonAbelian`.

<a id='TensorKit.Rsymbol' href='#TensorKit.Rsymbol'>#</a>
**`TensorKit.Rsymbol`** &mdash; *Function*.



```
function Rsymbol(a::G, b::G, c::G) where {G<:Sector}
```

Returns the R-symbol R^{c}_{a,b} that maps between `a ⊗ b → c` and `b ⊗ a → c` as in

```
a -<-μ-<- c                                 b -<-ν-<- c
     ∨          -> Rsymbol(a,b,c)[μ,ν]           ∧
     b                                           a
```

If `fusiontype(G)` is `Abelian` or `SimpleNonAbelian`, the R-symbol is a number. Otherwise it is a square matrix with row and column size `Nsymbol(a,b,c) == Nsymbol(b,a,c)`.

<a id='TensorKit.braidingtype-Tuple{Type{TensorKit.Trivial}}' href='#TensorKit.braidingtype-Tuple{Type{TensorKit.Trivial}}'>#</a>
**`TensorKit.braidingtype`** &mdash; *Method*.



```
function braidingtype(G::Type{<:Sector}) -> Type{<:Braiding}
```

Returns the type of braiding behavior of sectors of type G, which can be either

  * `Bosonic`: trivial exchange
  * `Fermionic`: fermionic exchange depending on `fermionparity`
  * `Anyonic`: requires general R_(a,b)^c phase or matrix (depending on `SimpleNonAbelian` or `DegenerateNonAbelian` fusion)

Note that `Bosonic` and `Fermionic` are subtypes of `SymmetricBraiding`, which means that braids are in fact equivalent to crossings (i.e. braiding twice is an identity: `Rsymbol(b,a,c)*Rsymbol(a,b,c) = I`) and permutations are uniquely defined.

<a id='TensorKit.dim' href='#TensorKit.dim'>#</a>
**`TensorKit.dim`** &mdash; *Function*.



```
dim(V::VectorSpace) -> Int
```

Returns the total dimension of the vector space `V` as an Int.

<a id='TensorKit.dim-Tuple{TensorKit.Sector}' href='#TensorKit.dim-Tuple{TensorKit.Sector}'>#</a>
**`TensorKit.dim`** &mdash; *Method*.



```
function dim(a::Sector)
```

Returns the (quantum) dimension of the sector `a`.

<a id='TensorKit.dual' href='#TensorKit.dual'>#</a>
**`TensorKit.dual`** &mdash; *Function*.



```
dual(V::VectorSpace) -> VectorSpace
```

Returns the dual space of `V`; also obtained via `V'`. It is assumed that `typeof(V) == typeof(V')`.

<a id='TensorKit.dual-Tuple{TensorKit.Sector}' href='#TensorKit.dual-Tuple{TensorKit.Sector}'>#</a>
**`TensorKit.dual`** &mdash; *Method*.



```
function dual(a::Sector) -> Sector
```

Returns the conjugate label `conj(a)`

<a id='TensorKit.fieldtype' href='#TensorKit.fieldtype'>#</a>
**`TensorKit.fieldtype`** &mdash; *Function*.



```
function fieldtype(V::VectorSpace) -> Field
```

Returns the field type over which a vector space is defined.

<a id='TensorKit.frobeniusschur-Tuple{TensorKit.Sector}' href='#TensorKit.frobeniusschur-Tuple{TensorKit.Sector}'>#</a>
**`TensorKit.frobeniusschur`** &mdash; *Method*.



```
function frobeniusschur(a::Sector)
```

Returns the Frobenius-Schur indicator of a sector `a`.

<a id='TensorKit.fusiontype-Tuple{Type{TensorKit.Trivial}}' href='#TensorKit.fusiontype-Tuple{Type{TensorKit.Trivial}}'>#</a>
**`TensorKit.fusiontype`** &mdash; *Method*.



```
function fusiontype(G::Type{<:Sector}) -> Type{<:Fusion}
```

Returns the type of fusion behavior of sectors of type G, which can be either

  * `Abelian`: single fusion output when fusing two sectors;
  * `SimpleNonAbelian`: multiple outputs, but every output occurs at most one, also known as multiplicity free (e.g. irreps of SU(2));
  * `DegenerateNonAbelian`: multiple outputs that can occur more than once (e.g. irreps of SU(3)).

<a id='TensorKit.leftnull-Tuple{TensorKit.AbstractTensorMap,Tuple{Vararg{Int64,N}} where N,Tuple{Vararg{Int64,N}} where N}' href='#TensorKit.leftnull-Tuple{TensorKit.AbstractTensorMap,Tuple{Vararg{Int64,N}} where N,Tuple{Vararg{Int64,N}} where N}'>#</a>
**`TensorKit.leftnull`** &mdash; *Method*.



```
leftnull(t::AbstractTensor, leftind::Tuple, rightind::Tuple) -> N
```

Create orthonormal basis for the orthogonal complement of the support of the indices in `leftind`, such that `N' * permute(t, leftind, rightind) = 0`.

If leftind and rightind are not specified, the current partition of left and right indices of `t` is used. In that case, less memory is allocated if one allows the data in `t` to be destroyed/overwritten, by using `leftnull!(t)`.

<a id='TensorKit.leftorth-Tuple{TensorKit.AbstractTensorMap,Tuple{Vararg{Int64,N}} where N,Tuple{Vararg{Int64,N}} where N}' href='#TensorKit.leftorth-Tuple{TensorKit.AbstractTensorMap,Tuple{Vararg{Int64,N}} where N,Tuple{Vararg{Int64,N}} where N}'>#</a>
**`TensorKit.leftorth`** &mdash; *Method*.



```
leftorth(t::AbstractTensorMap, leftind::Tuple, rightind::Tuple, truncation::TruncationScheme = notrunc()) -> Q, R
```

Create orthonormal basis `Q` for indices in `leftind`, and remainder `R` such that `permute(t,leftind,rightind) = Q*R`.

If leftind and rightind are not specified, the current partition of left and right indices of `t` is used. In that case, less memory is allocated if one allows the data in `t` to be destroyed/overwritten, by using `leftorth!(t)`.

This decomposition should be unique, such that it always returns the same result for the same input tensor `t`. This uses a QR decomposition with correction for making the diagonal elements of R positive.

<a id='TensorKit.rightnull-Tuple{TensorKit.AbstractTensorMap,Tuple{Vararg{Int64,N}} where N,Tuple{Vararg{Int64,N}} where N}' href='#TensorKit.rightnull-Tuple{TensorKit.AbstractTensorMap,Tuple{Vararg{Int64,N}} where N,Tuple{Vararg{Int64,N}} where N}'>#</a>
**`TensorKit.rightnull`** &mdash; *Method*.



```
rightnull(t::AbstractTensor, leftind::Tuple, rightind::Tuple) -> N
```

Create orthonormal basis for the orthogonal complement of the support of the indices in `rightind`, such that `permute(t, leftind, rightind)*N' = 0`.

If leftind and rightind are not specified, the current partition of left and right indices of `t` is used. In that case, less memory is allocated if one allows the data in `t` to be destroyed/overwritten, by using `rightnull!(t)`.

<a id='TensorKit.rightorth-Tuple{TensorKit.AbstractTensorMap,Tuple{Vararg{Int64,N}} where N,Tuple{Vararg{Int64,N}} where N}' href='#TensorKit.rightorth-Tuple{TensorKit.AbstractTensorMap,Tuple{Vararg{Int64,N}} where N,Tuple{Vararg{Int64,N}} where N}'>#</a>
**`TensorKit.rightorth`** &mdash; *Method*.



```
rightorth(t::AbstractTensorMap, leftind::Tuple, rightind::Tuple, truncation::TruncationScheme = notrunc()) -> L, Q
```

Create orthonormal basis `Q` for indices in `leftind`, and remainder `R` such that `permute(t,leftind,rightind) = L*Q`.

If leftind and rightind are not specified, the current partition of left and right indices of `t` is used. In that case, less memory is allocated if one allows the data in `t` to be destroyed/overwritten, by using `rightorth!(t)`.

This decomposition should be unique, such that it always returns the same result for the same input tensor `t`. This uses an LQ decomposition with correction for making the diagonal elements of R positive.

<a id='TensorKit.sectors-Tuple{TensorKit.ElementarySpace}' href='#TensorKit.sectors-Tuple{TensorKit.ElementarySpace}'>#</a>
**`TensorKit.sectors`** &mdash; *Method*.



```
function sectors(a)

Returns the different sectors of object `a`( e.g. a representation space or
an invariant tensor).
```

<a id='TensorKit.sectors-Tuple{TensorKit.RepresentationSpace}' href='#TensorKit.sectors-Tuple{TensorKit.RepresentationSpace}'>#</a>
**`TensorKit.sectors`** &mdash; *Method*.



```
sectors(V::ElementarySpace) -> sectortype(V)
sectors(V::ProductSpace{S,N}) -> NTuple{N,sectortype{V}}
```

Iterate over the different sectors in the vector space.

<a id='TensorKit.sectortype-Tuple{TensorKit.VectorSpace}' href='#TensorKit.sectortype-Tuple{TensorKit.VectorSpace}'>#</a>
**`TensorKit.sectortype`** &mdash; *Method*.



```
function sectortype(a) -> Sector
```

Returns the type of sector over which object `a` (e.g. a representation space or an invariant tensor) is defined. Also works in type domain.

<a id='TensorKit.space' href='#TensorKit.space'>#</a>
**`TensorKit.space`** &mdash; *Function*.



```
space(a) -> VectorSpace
```

Returns the vector space associated to object `a`.

<a id='Base.LinAlg.eig-Tuple{TensorKit.AbstractTensorMap,Tuple{Vararg{Int64,N}} where N,Tuple{Vararg{Int64,N}} where N}' href='#Base.LinAlg.eig-Tuple{TensorKit.AbstractTensorMap,Tuple{Vararg{Int64,N}} where N,Tuple{Vararg{Int64,N}} where N}'>#</a>
**`Base.LinAlg.eig`** &mdash; *Method*.



```
eig(t::AbstractTensor, leftind::Tuple, rightind::Tuple) -> D, V
```

Create orthonormal basis for the orthogonal complement of the support of the indices in `rightind`, such that `permute(t, leftind, rightind)*N' = 0`.

If leftind and rightind are not specified, the current partition of left and right indices of `t` is used. In that case, less memory is allocated if one allows the data in `t` to be destroyed/overwritten, by using `rightnull!(t)`.

<a id='Base.LinAlg.svd' href='#Base.LinAlg.svd'>#</a>
**`Base.LinAlg.svd`** &mdash; *Function*.



```
svd(t::AbstractTensorMap, leftind::Tuple, rightind::Tuple, truncation::TruncationScheme = notrunc()) -> U,S,V'
```

Performs the singular value decomposition such that tensor `permute(t,leftind,rightind) = U * S *V`.

If leftind and rightind are not specified, the current partition of left and right indices of `t` is used. In that case, less memory is allocated if one allows the data in `t` to be destroyed/overwritten, by using `svd!(t, truncation = notrun())`.

A truncation parameter can be specified for the new internal dimension, in which case a singular value decomposition will be performed. Choices are:

  * `notrunc()`: no truncation (default)
  * `truncerr(ϵ, p)`: truncates such that the p-norm of the truncated singular values is smaller than `ϵ`
  * `truncdim(χ)`: truncates such that the equivalent total dimension of the internal vector space is no larger than `χ`
  * `truncspace(V)`: truncates such that the dimension of the internal vector space is smaller than that of `V` in any sector

<a id='Base.SparseArrays.permute-Union{Tuple{N}, Tuple{G}, Tuple{TensorKit.FusionTree{G,N,M,L,T} where T where L where M,Tuple{Vararg{Int64,N}}}} where N where G<:TensorKit.Sector' href='#Base.SparseArrays.permute-Union{Tuple{N}, Tuple{G}, Tuple{TensorKit.FusionTree{G,N,M,L,T} where T where L where M,Tuple{Vararg{Int64,N}}}} where N where G<:TensorKit.Sector'>#</a>
**`Base.SparseArrays.permute`** &mdash; *Method*.



```
function permute(t::FusionTree{<:Sector,N}, i) where {N} -> (Immutable)Dict{typeof(t),<:Number}
```

Performs a permutation of the outgoing indices of the fusion tree `t` and returns the result as a `Dict` (or `ImmutableDict`) of output trees and corresponding coefficients.

<a id='Base.SparseArrays.permute-Union{Tuple{N₂}, Tuple{N₁}, Tuple{G}, Tuple{TensorKit.FusionTree{G,N,M,L,T} where T where L where M where N,TensorKit.FusionTree{G,N,M,L,T} where T where L where M where N,Tuple{Vararg{Int64,N₁}},Tuple{Vararg{Int64,N₂}}}} where N₂ where N₁ where G<:TensorKit.Sector' href='#Base.SparseArrays.permute-Union{Tuple{N₂}, Tuple{N₁}, Tuple{G}, Tuple{TensorKit.FusionTree{G,N,M,L,T} where T where L where M where N,TensorKit.FusionTree{G,N,M,L,T} where T where L where M where N,Tuple{Vararg{Int64,N₁}},Tuple{Vararg{Int64,N₂}}}} where N₂ where N₁ where G<:TensorKit.Sector'>#</a>
**`Base.SparseArrays.permute`** &mdash; *Method*.



```
function permute(t1::FusionTree{G}, t2::FusionTree{G}, p1::NTuple{N₁,Int}, p2::NTuple{N₂,Int}) where {G,N₁,N₂} -> (Immutable)Dict{Tuple{FusionTree{G,N₁}, FusionTree{G,N₂}},<:Number}
```

Input is a double fusion tree that describes the fusion of a set of incoming charges to a set of outgoing charges, represented using the individual trees of outgoing (`t1`) and incoming charges (`t2`) respectively (with `t1.incoming==t2.incoming`). Computes new trees and corresponding coefficients obtained from repartitioning and permuting the tree such that charges `p1` become outgoing and charges `p2` become incoming.

<a id='Base.conj-Tuple{TensorKit.ElementarySpace{ℝ}}' href='#Base.conj-Tuple{TensorKit.ElementarySpace{ℝ}}'>#</a>
**`Base.conj`** &mdash; *Method*.



```
conj(V::ElementarySpace) -> ElementarySpace
```

Returns the conjugate space of `V`. For `fieldtype(V)==ℝ`, `conj(V) == V` It is assumed that `typeof(V) == typeof(conj(V))`.

<a id='Base.one-Tuple{TensorKit.Sector}' href='#Base.one-Tuple{TensorKit.Sector}'>#</a>
**`Base.one`** &mdash; *Method*.



```
function one(::Sector) -> Sector
function one(::Type{<:Sector}) -> Sector
```

Returns the unit element within this type of sector

<a id='TensorKit.braid-Union{Tuple{N}, Tuple{G}, Tuple{TensorKit.FusionTree{G,N,M,L,T} where T where L where M,Any}} where N where G<:TensorKit.Sector' href='#TensorKit.braid-Union{Tuple{N}, Tuple{G}, Tuple{TensorKit.FusionTree{G,N,M,L,T} where T where L where M,Any}} where N where G<:TensorKit.Sector'>#</a>
**`TensorKit.braid`** &mdash; *Method*.



```
function braid(t::FusionTree{<:Sector,N}, i) where {N} -> ImmutableDict{typeof(t),<:Number}
```

Performs a braid of neighbouring outgoing indices `i` and `i+1` on a fusion tree `t`, and returns the result as a linked list of output trees and corresponding coefficients.

<a id='TensorKit.fermionparity-Tuple{fℤ₂}' href='#TensorKit.fermionparity-Tuple{fℤ₂}'>#</a>
**`TensorKit.fermionparity`** &mdash; *Method*.



```
function fermionparity(s::Fermion) -> Bool
```

Returns the fermion parity of a sector `s` that is a subtype of `Fermion`, as a `Bool` being true if odd and false if even.

<a id='TensorKit.repartition-Union{Tuple{N}, Tuple{N₂}, Tuple{N₁}, Tuple{G}, Tuple{TensorKit.FusionTree{G,N₁,M,L,T} where T where L where M,TensorKit.FusionTree{G,N₂,M,L,T} where T where L where M,Val{N}}} where N where N₂ where N₁ where G<:TensorKit.Sector' href='#TensorKit.repartition-Union{Tuple{N}, Tuple{N₂}, Tuple{N₁}, Tuple{G}, Tuple{TensorKit.FusionTree{G,N₁,M,L,T} where T where L where M,TensorKit.FusionTree{G,N₂,M,L,T} where T where L where M,Val{N}}} where N where N₂ where N₁ where G<:TensorKit.Sector'>#</a>
**`TensorKit.repartition`** &mdash; *Method*.



```
function repartition(t1::FusionTree{G,N₁}, t2::FusionTree{G,N₂}, ::Val{N}) where {G,N₁,N₂,N} -> (Immutable)Dict{Tuple{FusionTree{G,N}, FusionTree{G,N₁+N₂-N}},<:Number}
```

Input is a double fusion tree that describes the fusion of a set of `N₂` incoming charges to a set of `N₁` outgoing charges, represented using the individual trees of outgoing (`t1`) and incoming charges (`t2`) respectively (with `t1.incoming==t2.incoming`). Computes new trees an corresponding coefficients obtained from repartitioning the tree by bending incoming to outgoing charges (or vice versa) in order to have `N` outgoing charges.

<a id='TensorKit.vertex_ind2label-Union{Tuple{G}, Tuple{Int64,G,G,G}} where G<:TensorKit.Sector' href='#TensorKit.vertex_ind2label-Union{Tuple{G}, Tuple{Int64,G,G,G}} where G<:TensorKit.Sector'>#</a>
**`TensorKit.vertex_ind2label`** &mdash; *Method*.



```
function vertex_ind2label(i::Int, a::G, b::G, c::G) where {G<:Sector}
```

Convert the index i of the fusion vertex (a,b)->c into a label. For `fusiontype(G)==Abelian` or `fusiontype(G)==NonAbelian`, where every fusion output occurs only once and `i===1`, the default is to suppress vertex labels by setting them equal to `nothing`. For `fusiontype(G)==DegenerateNonAbelian`, the default is to just use `i`, unless a specialized method is provided.

<a id='TensorKit.vertex_labeltype-Tuple{Type{#s13} where #s13<:TensorKit.Sector}' href='#TensorKit.vertex_labeltype-Tuple{Type{#s13} where #s13<:TensorKit.Sector}'>#</a>
**`TensorKit.vertex_labeltype`** &mdash; *Method*.



```
function vertex_labeltype(G::Type{<:Sector}) -> Type
```

Returns the type of labels for the fusion vertices of sectors of type `G`.

