# Vector spaces

```@setup tensorkit
using TensorKit
```

From the [Introduction](@ref), it should be clear that an important aspect in the definition
of a tensor (map) is specifying the vector spaces and their structure in the domain and
codomain of the map. The starting point is an abstract type `VectorSpace`
```julia
abstract type VectorSpace end
```
which serves in a sense as the category ``\mathbf{Vect}``. All instances of subtypes of
`VectorSpace` will represent vector spaces. In particular, we define two abstract subtypes
```julia
abstract type ElementarySpace{𝕜} <: VectorSpace end
const IndexSpace = ElementarySpace

abstract type CompositeSpace{S<:ElementarySpace} <: VectorSpace end
```
Here, `ElementarySpace` is a super type for all vector spaces that can be associated with
the individual indices of a tensor, as hinted to by its alias `IndexSpace`. It is
parametrically dependent on `𝕜`, the field of scalars (see the next section on
[Fields](@ref)).

On the other hand, subtypes of `CompositeSpace{S}` where `S<:ElementarySpace` are composed
of a number of elementary spaces of type `S`. So far, there is a single concrete type
`ProductSpace{S,N}` that represents the homogeneous tensor product of `N` vector spaces of
type `S`. Its properties are discussed in the section on [Composite spaces](@ref), together
with possible extensions for the future.

## Fields

Vector spaces are defined over a field of scalars. We define a type hierarchy to specify the
scalar field, but so far only support real and complex numbers, via
```julia
abstract type Field end

struct RealNumbers <: Field end
struct ComplexNumbers <: Field end

const ℝ = RealNumbers()
const ℂ = ComplexNumbers()
```
Note that `ℝ` and `ℂ` can be typed as `\bbR`+TAB and `\bbC`+TAB. One reason for defining
this new type hierarchy instead of recycling the types from Julia's `Number` hierarchy is
to introduce some syntactic suggar without commiting type piracy. In particular, we now have
```@repl tensorkit
3 ∈ ℝ
5.0 ∈ ℂ
5.0+1.0*im ∈ ℝ
Float64 ⊆ ℝ
ComplexF64 ⊆ ℂ
ℝ ⊆ ℂ
ℂ ⊆ ℝ
```
and furthermore ––probably more usefully–– `ℝ^n` and `ℂ^n` create specific elementary vector
spaces as described in the next section. The underlying field of a vector space or tensor
`a` can be obtained with `field(a)`.

## Elementary vector spaces

As mentioned at the beginning of this section, vector spaces that are associated with the
individual indices of a tensor should be implemented as subtypes of `ElementarySpace`. As
the domain and codomain of a tensor map will be the tensor product of such objects which all
have the same type, it is important that related vector spaces, e.g. the dual space, are
objects of the same concrete type (i.e. with the same type parameters in case of a
parametric type). In particular, every `ElementarySpace` should implement the following
methods

*   `dim(::ElementarySpace) -> ::Int` returns the dimension of the space as an `Int`

*   `dual{S<:ElementarySpace}(::S) -> ::S` returns the
    [dual space](http://en.wikipedia.org/wiki/Dual_space) `dual(V)`, using an instance of
    the same concrete type (i.e. not via type parameters); this should satisfy
    `dual(dual(V)==V`

*   `conj{S<:ElementarySpace}(::S) -> ::S` returns the
    [complex conjugate space](http://en.wikipedia.org/wiki/Complex_conjugate_vector_space)
    `conj(V)`, using an instance of the same concrete type (i.e. not via type parameters);
    this should satisfy `conj(conj(V))==V` and we automatically have
    `conj(V::ElementarySpace{ℝ}) = V`.

For convenience, the dual of a space `V` can also be obtained as `V'`.

There is concrete type `GeneralSpace` which is completely characterized by its field `𝕜`,
its dimension and whether its the dual and/or complex conjugate of $𝕜^d$.
```julia
struct GeneralSpace{𝕜} <: ElementarySpace{𝕜}
    d::Int
    dual::Bool
    conj::Bool
end
```

We furthermore define the abstract type
```julia
abstract InnerProductSpace{𝕜} <: ElementarySpace{𝕜}
```
to contain all vector spaces `V` which have an inner product and thus a canonical mapping
from `dual(V)` to `V` (for `𝕜 ⊆ ℝ`) or from `dual(V)` to `conj(V)` (otherwise). This
mapping is provided by the metric, but no further support for working with metrics is
currently implemented.

Finally there is
```julia
abstract EuclideanSpace{𝕜} <: InnerProductSpace{𝕜}
```
to contain all spaces `V` with a standard Euclidean inner product (i.e. where the metric is
the identity). These spaces have the natural isomorphisms `dual(V) == V` (for `𝕜 == ℝ`)
or `dual(V) == conj(V)` (for ` 𝕜 == ℂ`). In particular, we have two concrete types
```julia
immutable CartesianSpace <: EuclideanSpace{ℝ}
    d::Int
end
immutable ComplexSpace <: EuclideanSpace{ℂ}
  d::Int
  dual::Bool
end
```
to represent the Euclidean spaces $ℝ^d$ or $ℂ^d$ without further inner structure. They can
be created using the syntax `ℝ^d` and `ℂ^d`, or `(ℂ^d)'`for the dual space of the latter.
Note that the brackets are required because of the precedence rules, since `d' == d` for
`d::Integer`. Some examples:
```@repl tensorkit
dim(ℝ^10)
(ℝ^10)' == ℝ^10
isdual((ℂ^5))
isdual((ℂ^5)')
isdual((ℝ^5)')
dual(ℂ^5) == (ℂ^5)' == conj(ℂ^5)
```
We refer to the next section on [Sectors, representation spaces and fusion trees](@ref) for
further information about `RepresentationSpace`, which is a subtype of `EuclideanSpace{ℂ}` with
an inner structure corresponding to the irreducible representations of a group.

## Composite spaces

Composite spaces are vector spaces that are built up out of individual elementary vector
spaces. The most prominent (and currently only) example is a tensor product of `N` elementary spaces of the same type `S`, which is implemented as
```julia
struct ProductSpace{S<:ElementarySpace, N} <: CompositeSpace{S}
    spaces::NTuple{N, S}
end
```
Given some `V1::S`, `V2::S`, `V3::S` of the same type `S<:ElementarySpace`, we can easily
construct `ProductSpace{S,3}((V1,V2,V3))` as `ProductSpace(V1,V2,V3)` or using
`V1 ⊗ V2 ⊗ V3`, where `⊗` is simply obtained by typing `\otimes`+TAB. In fact, for
convenience, also the regular multiplication operator `*` acts as tensor product between
vector spaces, and as a consequence so does raising a vector space to a positive integer
power, i.e.
```@repl tensorkit
V1 = ℂ^2
V2 = ℂ^3
V1 ⊗ V2 ⊗ V1' == V1 * V2 * V1' == ProductSpace(V1,V2,V1') == ProductSpace(V1,V2) ⊗ V1'
V1^3
dim(V1 ⊗ V2)
dims(V1 ⊗ V2)
dual(V1 ⊗ V2)
```
Here, the new function `dims` maps `dim` to the individual spaces in a `ProductSpace` and
returns the result as a tuple. Note that the rationale for the last result was explained in
the subsection [Duals](@ref) of [Properties of monoidal categories](@ref).

Following Julia's Base library, the function `one` applied to a `ProductSpace{S,N}` returns
the multiplicative identity, which is `ProductSpace{S,0}`. The same result is obtained when
acting on an instance `V` of `S::ElementarySpace` directly, however note that `V ⊗ one(V)`
will yield a `ProductSpace{S,1}(V)` and not `V` itself. Similar to Julia Base, `one` also
works in the type domain.

In the future, other `CompositeSpace` types could be added. For example, the wave function
of an `N`-particle quantum system in first quantization would require the introduction of a
`SymmetricSpace{S,N}` or a `AntiSymmetricSpace{S,N}` for bosons or fermions respectively,
which correspond to the symmetric (permutation invariant) or antisymmetric subspace of
`V^N`, where `V::S` represents the Hilbert space of the single particle system. Other
domains, like general relativity, might also benefit from tensors living in a subspace with
certain symmetries under specific index permutations.

## Some more functionality
Some more convenience functions are provided for the euclidean spaces
[`CartesianSpace`](@ref) and [`ComplexSpace`](@ref), as well as for
[`RepresentationSpace`](@ref) discussed in the next section. All functions below that act
on more than a single elementary space, are only defined when the different spaces are of
the same concrete subtype `S<:ElementarySpace`

The function `fuse(V1, V2, ...)` or `fuse(V1 ⊗ V2 ⊗ ...)` returns an elementary space that
is isomorphic to `V1 ⊗ V2 ⊗ ...`, in the sense that a unitary tensor map can be constructed
between those spaces, e.g. from `W = V1 ⊗ V2 ⊗ ...` to `V = fuse(V1 ⊗ V2 ⊗ ...)`. The
function `flip(V1)` returns a space that is isomorphic to `V1` but has
`isdual(flip(V1)) == isdual(V1')`, i.e. if `V1` is a normal space than `flip(V1)` is a dual
space. Again, isomorphism here implies that a unitary map (but there is no canonical
choice) can be constructed between both spaces. `flip(V1)` is different from `dual(V1)` in
the case of [`RepresentationSpace`](@ref). It is useful to flip a tensor index from a ket
to a bra (or vice versa), by contracting that index with a unitary map from `V1` to
`flip(V1)`. We refer to `[Index operations](@ref)` for further information. Some examples:
```@repl tensorkit
fuse(ℝ^5, ℝ^3)
fuse(ℂ^3, (ℂ^5)', ℂ^2)
flip(ℂ^4)
```

We also define the direct sum `V1` and `V2` as `V1 ⊕ V2`, where `⊕` is obtained by typing
`\oplus`+TAB. This is possible only if `isdual(V1) == isdual(V2)`. With a little pun on
Julia Base, `oneunit` applied to an elementary space (in the value or type domain) returns
the one-dimensional space, which is isomorphic to the scalar field of the spaceitself. Some
examples illustrate this better
```@repl tensorkit
ℝ^5 ⊕ ℝ^3
ℂ^5 ⊕ ℂ^3
ℂ^5 ⊕ (ℂ^3)'
oneunit(ℝ^3)
ℂ^5 ⊕ oneunit(ComplexSpace)
oneunit((ℂ^3)')
(ℂ^5) ⊕ oneunit((ℂ^5))
(ℂ^5)' ⊕ oneunit((ℂ^5)')
```

For two spaces `V1` and `V2`, `min(V1,V2)` returns the space with the smallest dimension,
whereas `max(V1,V2)` returns the space with the largest dimension, as illustrated by
```@repl tensorkit
min(ℝ^5, ℝ^3)
max(ℂ^5, ℂ^3)
max(ℂ^5, (ℂ^3)')
```
Again, we impose `isdual(V1) == isdual(V2)`. Again, the use of these methods is to construct
unitary or isometric tensors that map between different spaces, which will be elaborated
upon in the section on Tensors.
