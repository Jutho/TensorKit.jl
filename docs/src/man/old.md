# TensorKit.jl

A Julia tensor package with a hint of category theory.
***

## Introduction

`TensorKit.jl` is a Julia package for working with tensors, which are consistently treated as [the elements of a tensor product of vector spaces](http://en.m.wikipedia.org/wiki/Tensor#Using_tensor_products) or, more generally, as linear maps between pairs of such tensor product spaces. While tensors can typically be represented as multidimensional arrays with respect to a chosen basis, they have a richer mathematical structure depending on the type of vector spaces used in the tensor product construction.

## What is a tensor?



In abstract sense, tensors (and tensor maps; see below) correspond to the morphisms in the category ``Vect``, i.e. the category with vector spaces as objects, or some subcategory thereof.



some subspace thereof. While tensors can typically be represented as multidimensional arrays with respect to a chosen basis, they have a richer mathematical structure depending on the type of vector spaces used in the tensor product construction.
Henceforth, we represent a tensor using [index notation](http://en.m.wikipedia.org/wiki/Abstract_index_notation) and refer to the different "dimensons" as **indices**. The tensor ``T^{i_1\;\;\;\overline{\imath}_4}_{\;i_2\overline{\imath}_3}`` represents a an element of the tensor product space `V_1 \otimes V_2^{\ast} \otimes \overline{V}_3 \otimes \overline{V}_4^{\ast}` where general complex vector spaces `V` can appear in four different ways:

* contravariant index ``i_1`` : normal vector space ``V_1``
* contravariant index ``i_2`` : [dual space](http://en.wikipedia.org/wiki/Dual_space) $V_2^{\ast}$
* barred or dotted covariant index $\overline{\imath}_3$ : [complex conjugate space](http://en.wikipedia.org/wiki/Complex_conjugate_vector_space) $\overline{V}_3$
* barred or dotted contravariant index $\overline{\imath}_4$ : complex conjugate of the dual space $\overline{V}_4^{\ast}$ , which is equivalent to the dual of the complex conjugate space

The four different vector spaces $V$ , $V^{\ast}$ , $\overline{V}$ and $\overline{V}^{\ast}$ correspond to the representation spaces of respectively the fundamental, dual or contragredient, complex conjugate and dual complex conjugate representation of the general linear group $\mathsf{GL}(V)$ [^tung]. Simplifications will occur for certain types of vector spaces.

***

## Vector spaces

### Elementary vector spaces

Tensors in TensorToolbox.jl are treated as elements of a tensor product of a homogeneous family of elementary vector spaces, which we also refer to as index spaces and can be user defined. We thus define a type hierarchy for representing a hierarchy of common vector spaces:

```julia
abstract VectorSpace
abstract ElementarySpace{F} <: VectorSpace
const IndexSpace = ElementarySpace
```
where the parameter `F` can be used to represent a field over which the vector space is defined. In particular, we define a unicode shorthand for two common fields, which we take from the Julia `Number` type hierarchy:

```julia
const ℝ=Real
const ℂ=Complex{Real}
```

Every `ElementarySpace` should implement the following methods

* `dim(::ElementarySpace) -> ::Int`  
  returns the dimension of the space as an `Int`

* `dual{S<:ElementarySpace}(::S) -> ::S`  
  returns the [dual space](http://en.wikipedia.org/wiki/Dual_space) `dual(V)`, preferably using an instance of the same concrete type (i.e. not via type parameters) to combine well with the way tensors are defined; this should satisfy `dual(dual(V)==V`

* `conj{S<:ElementarySpace}(::S) -> ::S`  
  returns the [complex conjugate space](http://en.wikipedia.org/wiki/Complex_conjugate_vector_space) `conj(V)`, preferably using an instance of the same concrete type (i.e. not via type parameters) to combine well with the way tensors are defined; this should satisfy `conj(conj(V))==V` and we automatically have `conj{F<:Real}(V::ElementarySpace{F}) = V`.

In particular, there is concrete type `GeneralSpace` which is completely characterized by its field `F`, its dimension and whether its the dual and/or complex conjugate of $\mathbb{F}^d$ .

We furthermore define the abstract type

```julia
abstract InnerProductSpace{F} <: ElementarySpace{F}
```

to contain all vector spaces `V` which have an inner product and thus a natural mapping from `dual(V)` to `V` (for ` F<:Real`) or from `dual(V)` to `conj(V)` (for ` F<:Complex`). This mapping is provided by the metric, but no further support for working with metrics is currently implemented.

Finally there is

```julia
abstract EuclideanSpace{F} <: InnerProductSpace{F}
```

to contain all spaces `V` with a standard Euclidean inner product (i.e. where the metric is the identity). These spaces have the natural isomorphisms `dual(V)==V` (for ` F<:Real`) or `dual(V)==conj(V)` (for ` F<:Complex`). In particular, we have two concrete types

```julia
immutable CartesianSpace <: EuclideanSpace{ℝ}
    d::Int
end
immutable ComplexSpace <: EuclideanSpace{ℂ}
  d::Int
  dual::Bool
end
```
to represent the Euclidean spaces $ℝ^d$ or $ℂ^d$ without further inner structure. They can be created using the syntax `ℝ^d` and `ℂ^d`.

#### WIP: Graded spaces, superselection sectors and braiding

If tensors are used to describe system with a certain symmetry corresponding to a group $\mathsf{G}$ , this implies that the vector spaces $V$ involved carry a corresponding representation $\rho_V: \mathsf{G} \to V$ . For compact groups and an otherwise complex euclidean space $V$ , these representations can be chosen unitary and it makes sense to decompose $V$ according to the irreducible representations of $\mathsf{G}$ as

$V=\bigoplus_{\lambda} R_{\lambda} \otimes \mathbb{C}^{d_{\lambda}}$

$V$ thus becomes a [graded vector space](http://en.m.wikipedia.org/wiki/Graded_vector_space) with the different sectors labelled by the irreducible representations $\lambda$ of $\mathsf{G}$ , and where every sector decomposes as the tensor product of the irrep space $R_\lambda$ and a part $\mathbb{C}^{d_\lambda}$ that transforms trivially. More generally, a vector space can be graded by the representations of a [Hopf algebra](http://en.m.wikipedia.org/wiki/Hopf_algebra), corresponding to a set of labels constituting a unitary [fusion category](http://ncatlab.org/nlab/show/fusion+category).

The implementation of graded vector spaces is currently limited to those cases where $V$ represents a complex Euclidean space, which would be the typical case for unitary representations of groups. This amounts to the definition

```julia
abstract UnitaryRepresentationSpace{G<:Sector} <: EuclideanSpace{ℂ}
```
Here, `Sector` is an abstract type. A subtype of `Sector` corresponds to a particular fusion category and the possible objects correspond to the different labels, i.e. the different charges or superselection sectors. `Sector` objects should support the functionality to map objects (labels) to the corresponding conjugate label (anticharge), to create the trivial object (identity, zero charge) and to determine the outcome of the fusion product. So far, only abelian categories are implemented, corresponding to the representations of abelian groups:

```julia
abstract Sector
abstract Abelian <: Sector
abstract NonAbelian <: Sector
immutable Parity <: Abelian
    charge::Bool
end
immutable ZNCharge{N} <: Abelian
    charge::Int
    ZNCharge(charge::Int)=new(mod(charge,N))
end
immutable U1Charge <: Abelian
    charge::Int
end
```

***TODO: Braiding***

When the different superselection sectors correspond to e.g. different fermion or anyon occupation numbers, a natural action will arise when changing the order of the corresponding vectors in a tensor product. The graded vector space thus becomes a [braided vector space](http://en.m.wikipedia.org/wiki/Braided_vector_space). The simplest example is that of a vector space `V` graded by fermion parity. An element of `V1⊗V2` can be mapped to one of `V2⊗V1` by permuting the two tensor indices and adding a phase `-1` in the sector where both indices have an odd fermion number. More generally, a complete braiding tensor $R_{\alpha,\beta}^{\gamma}$ needs to be specified.

#### ElementarySpace methods

### Composite spaces
Composite spaces are built out of elementary vector spaces of a homogeneous type `S`. The most relevant case is the abstract family `TensorSpace{S,N}` used to denote certain subspaces in the tensor product space of `N` vector spaces of type `S`. These spaces will be used to define rank-`N` tensors, where the different tensor indices $i_j$ correspond to the elements of a basis in $V_j$ for $j=1,2,...,N$ . We start with the definitions

```julia
abstract CompositeSpace{S<:ElementarySpace} <: VectorSpace
abstract TensorSpace{S<:ElementarySpace,N} <: CompositeSpace{S}
```

The homogenity restriction is the only sensible way of defining tensor product spaces, since there is no point in defining i.e. a tensor with a group action on some indices and not on other indices and it is even impossible to define the tensor product space of vector spaces over different fields. It is thus not possible to construct tensor product spaces of e.g. $\mathbb{R}^{d_1}$ and $\mathbb{C}^{d_2}$ , but it is possible to construct tensor product spaces of $\mathbb{C}^{d_1}$ and $\mathbb{C}^{d_2}$ , or even the dual of the latter. Therefore, for new vector spaces, it is important that any related vector space (e.g. the dual or conjugated space) with which one wants to construct tensor product spaces are of the same concrete type (e.g. no type parameters to denote dual spaces).

#### ProductSpace
 The complete tensor product space is represented by the concrete type `ProductSpace{S,N}`. This corresponds to the definitions

```julia
immutable ProductSpace{S<:ElementarySpace,N} <: TensorSpace{S,N}
    spaces::NTuple{N, S}
end
```
The `ProductSpace` of a set of elementary spaces `V1`, `V2`, ... of type `S` can be created as  `V1 ⊗ V2 ⊗ ...`. Product spaces can be iterated over and indexed in order to extract the elementary spaces, or the tensor product of a subset of them. The dual and conjugate spaces are defined by mapping these actions to the respective elementary vector spaces `V1`, `V2`, ... For convience, we also define the `transpose` of a `ProductSpace` by reversing the factors `V1` to `VN`, and the `ctranspose` by reversing the conjugated spaces. While there is no such thing as the transpose of a vector space, this definition is convenient because it is compatible with the way `(c)transpose` is defined for tensors. Finally, the `dim` of a `ProductSpace` is given by the product of the `dim` of its constituents.

#### WIP: InvariantSpace
A `InvariantSpace` corresponds to the subspace of the tensor product of some `UnitaryRepresentationSpaces` that fuses to the identity (i.e. total 'charge' zero). In the case of irreducible representations of groups, it corresponds to the invariant subspace, i.e. the subspace of the tensor product that couples to the trivial representation. The different sectors of an `InvariantSpace` are labelled not only by the set of sectors of the individual elementary spaces (under the constraint that they have a fusion channel to trivial charge), but also by the intermediate fusion sectors. This gives rise to the concept of a fusion tree.

In order to describe and manipulate the trivial sector in the tensor product of `UnitaryRepresentationSpace`s, one thus needs to be able to store and manipulate fusion trees using recouplings (F-moves) or braidings (R-moves). So far, this has only been implemented for spaces with Abelian fusion rules and trivial braiding.

* TODO: develop interface to work with fusion trees and braidings

* TODO: implement some non-trivial cases (SU(2) symmetry, fermions, ...)

References:

* [General theory of anyons and unitary fusion categoies](http://thesis.library.caltech.edu/2447/2/thesis.pdf)
* [General treatment of symmetries in tensors](http://arxiv.org/abs/0907.2994.pdf)
* [U1 symmetric tensors](http://arxiv.org/abs/1008.4774.pdf)
* [SU2 symmetric tensors](http://arxiv.org/abs/1208.3919.pdf)
* [Anyonic tensors 1](http://arxiv.org/abs/1006.3532.pdf) , [Anyonic tensors 2](http://arxiv.org/abs/1311.0967.pdf)

#### TODO: Symmetric and antisymmetric vector spaces (Fock space)

For the tensor product of `N` identical copies of a given vector space `V`, we can also consider the symmetric or antisymmetric subspace of $V^{\otimes N}$ , corresponding to e.g. the `N` particle boson or fermion Fock space corresponding to a single particle Hilbert Space `V`. This has of course also other applications and can be extended to tensors with (anti)symmetry in subsets of indices.

***

## Tensors
The most important elements in `TensorToolbox.jl` are of course tensors. A rank `N` tensor is interpreted as the element of (a subspace of) the tensor product of some `N` elementary vector spaces, represented as a `TensorSpace{S,N}` object `V`. A tensor needs to store its components as a list of numbers of type `T<:Number`. The following observations are in order:

* The element type `T` must not be the same as the field of the vector space, i.e. a tensor in a (tensor product) of `ComplexSpace`s can have real components, but a tensor in the product space of `CartesianSpace`s should not have complex entries. However, this is not strictly enforced.
* The components represent the tensor with respect to a canonical choice of basis in `V`; so far there is no support to represent different basis choices and the transformations between them. This might change in the future.
* The number of (independent) components of a tensor is given by `dim V`. When `V` is a proper subspace of `V1 ⊗ V2 ⊗ ... ⊗ VN`, then `dim V` is not just the product of the dimensions of the elementary spaces and the independent components cannot simply be represent as a `N`-dimensional array.

### Different tensor types
The only difference between tensors (so far) is how their independent components are stored. All other characteristics are encoded in the type of vector space.

#### DenseTensor
We start the type hierarchy with an abstract type and currently have a single concrete tensor type, `DenseTensor`, that stores its components using a `Vector{T}`, corresponding to the following definitions
```julia
abstract AbstractTensor{S,T,N}
immutable DenseTensor{S,T,N,P} <: AbstractTensor{S,T,N}
    data::Vector{T}
    space::P
end
```
Here, we should have `P<:TensorSpace{S,N}`. With the current Julia type system, this cannot be enforced in the type but only in its constructor (which also checks that `length(data)=dim(space)`. This might change with the type system redesign.

We can then define some useful type aliasses for e.g. the standard tensor living in the full tensor product space
```julia
typealias Tensor{S,T,N} DenseTensor{S,T,N,ProductSpace{S,N}}
typealias CartesianTensor{T,N} Tensor{CartesianSpace,T,N}
typealias ComplexTensor{T,N} Tensor{ComplexSpace,T,N}
typealias InvariantTensor{S,T,N} DenseTensor{S,T,N,InvariantSpace{S,N}}
typealias U1Tensor{T,N} ...
```
and so on.

A tensor can be created from a set of components as
```julia
tensor(data, space)
```
where `data` can be an arbitrary `Vector{T}`.

#### TODO: DiagonalTensor
For the specific case of a rank `N=2` tensor in `V ⊗ dual(V)`, it is often useful to have an explicit diagonal representation, e.g. to store the eigenvalues or singular values corresponding to a given tensor factorization (see below).

#### Other tensors ?
Should there be sparse tensors?

### Tensor properties
The basic tensor methods allow to construct tensors and query their characteristics

* `space(t)` returns the vector space of a tensor `t`.
* `eltype(t)` returns the element type `T` of the coefficient vector.
* `numind(t)=order(t)` returns the number of tensor indices `N`, i.e. the number of elementary vector spaces in `space(t)`.
* `in(t,V)` can be used to check if `space(t)` is a subspace of `V`.

 * `vec(t)` returns the coefficient vector `data` which allows to modify the tensor components
* `full(t)` returns an `Array{T,N}` representation of a rank `N` tensor. Only when `space(t)` is a `ProductSpace` is this isomorphic to the vector of coefficients, otherwise zeros or repeated coefficients might appear. Therefore, `full(t)` does not share data with the tensor and cannot be used to modified its contens.

### Constructing and converting tensors
* `tensor(data,V)` can be used to construct a `DenseTensor`. Here, `data` represents an arbitrary `Array{T,N}`. If the vector space `V` is provided, the multidimensional characteristics of `data` are ignored. Only `vec(data)` is used and the only requirement is that `length(data)` equals `dim(V)`. If `V` is absent, then `tensor(data)` creates a `CartesianTensor` if `T<:Real` with `V=ProductSpace(map(CartesianSpace,size(data)))`. If `T<:Complex`, a `ComplexTensor` is constructed, even though there it is already ambiguous whether the normal complex Euclidean space or the dual space should be constructed for every index.
* `zeros(T,V)` creates a tensor in `V` filled with zero coefficients, which is equivalent to the zero vector. If `T` is omitted, it is given the default value `T=Float64`.
* `rand(T,V)` creates a tensor in `V` filled with random coefficients. A default value of `T=Float64` is asssumed when `T` is omitted.
* `similar(t,T,V)` constructs an unitialized tensor similar to `t`, but with element type `T` and for different space `V` (of the same type of `space(t)`).

* `complex(t)` converts `t` to a tensor with complex-valued coefficients; it does nothing if  `eltype(t)<:Complex`.
* `real(t)` and `imag(t)` returns a tensor with the real and imaginary parts of the coefficients; this is a basis-dependent operations and refers to the canonical basis with respect to which the coefficients are stored.
* `float32`, `float64`, `complex64` and `complex128` can be used to cast the tensor coefficients into a specific format.

### Basic linear algebra methods
The following methods allow to  perform basic linear algebra (corresponding to their interpretation as elements in a vector space):

* arithmetic: tensors in the same vector space can be added, subtracted en multiplied with scalars. There are also mutating methods such as `scale!` and `axpy!`.
* `conj(t)` conjugates the tensor in the canonical basis. Note that this also maps the tensor to the space `conj(V)` which is different from `V`. Therefore, `conj!` is not an inplace operation but can be used to store the result of conjugating the tensor `src` in a preallocated tensor `dst` in `conj(V)` using `conj!(dst,src)`.
* `transpose(t)` implements an isomorphism from `V=V1 ⊗ V2 ⊗ ... ⊗ VN` to `reverse(V) = VN ⊗ ... ⊗ V2 ⊗ V1`, i.e. it reverses the order of the indices. For a tensor with `N=1`, this has no effect. For a tensor with `N=2`, this corresponds to the most general definition of the [transpose of a linear map](http://en.wikipedia.org/wiki/Transpose#Transpose_of_a_linear_map). A linear map $f:V\to W$ can be identified with a tensor in `W ⊗ dual(V)`. The transpose of this tensor lives in `dual(V) ⊗ W`, which can be identified with a linear map from `dual(W)` to `dual(V)`, in accordance with the aforementioned definition. Only for real Euclidean vector spaces is `dual(V) == V` and does this correspond to a map from `W` to `V`.  For `N>2`, there is no standard definition of transpose, but reversing all indices corresponds to the convention used in the [Penrose graphical notation](http://en.wikipedia.org/wiki/Penrose_graphical_notation), where transposing corresponds to mirroring the diagrammatic representation of the tensor. There is again a mutating version `transpose!(dst,src)` that allows to store the result of transposing `src` in the preallocated tensor `dst`.
* `ctranspose(t)` is equivalent to `conj(transpose(t))` but performs this operation in a single step. In particular, for `N=2`, it maps a tensor in `W ⊗ dual(V)` to a tensor in `dual(conj(V)) ⊗ conj(W)`. For complex Euclidean spaces (where `dual(V)=conj(V)`) or real Euclidean spaces (where `dual(V)=V` and `conj(V)=V`), the conjugate transpose of a tensor in `W ⊗ dual(V)` is a tensor in `V ⊗ dual(W)`, which can be interpreted as a linear map from `W` to `V`, according to the definition of the adjoint map. As before, `ctranpose!(dst,src)` stores the result in the preallocated destination tensor.
* `dot(t1,t2)` computers the inner product between two tensors `t1` and `t2`. This is only possible if `space(t1)==space(t2)` and if this space is the tensor product of elementary vector spaces with an inner product, i.e. `S<:InnerProductSpace`. However, the interface for specifying general inner products still needs to be developed, and thus so far `dot(t1,t2)` only works if `S<:EuclideanSpace`. We choose the canonical basis of euclidean spaces orthonormal, such that `dot(t1,t2) = dot(vec(t1),vec(t2))`, i.e. the inner product corresponds to the normal scalar product of the coefficient vectors.
* `vecnorm(t)` computes the norm of tensor `t`; it is essentially equivalent to `sqrt(dot(t,t))` and is therefore subject to the same restrictions (`S<:EuclideanSpace`) and satisfies `vecnorm(t)=norm(vec(t))`.

Currently, `conj`, `transpose` and `ctranspose` allocate new tensors for storing the result. This might change in the future such that they return a simple view over the same data, although this is not entirely trivial for tensors which do not live in a simple `ProductSpace{S,N}`.

### TODO: Indexing

### Tensor operations
* `scalar(t)` can be applied to a rank `N=0` tensor to construct the single scalar component, since in that case `space(t)` is an empty tensor product space and thus equivalent to the corresponding number field.


### Tensor factorizations

***

## Tensor Maps
Linear maps between tensor spaces with possible efficient implementation.

***

## Tensor Networks

***

## Bibliography
[^tung]:  Wu-Ki Tung. Group Theory in Physics: Introduction to Symmetry Principles, Group Representations, and Special Functions in Classical and Quantum Physics. World Scientific Publishing Company, 1985.  
