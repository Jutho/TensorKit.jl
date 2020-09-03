# [Vector spaces](@id s_spaces)

```@setup tensorkit
using TensorKit
```

From the [Introduction](@ref s_intro), it should be clear that an important aspect in the
definition of a tensor (map) is specifying the vector spaces and their structure in the domain and codomain of the map. The starting point is an abstract type `VectorSpace`
```julia
abstract type VectorSpace end
```
which is actually a too restricted name. All instances of subtypes of `VectorSpace` will
represent objects in ``𝕜``-linear monoidal categories, but this can go beyond normal
vector spaces (i.e. objects in the category ``\mathbf{Vect}``) and even beyond objects of
``\mathbf{SVect}``. However, in order not to make the remaining discussion to abstract
or complicated, we will simply refer to subtypes of `VectorSpace` instead of specific
categories, and to spaces (i.e. `VectorSpace` instances) instead of objects from these
categories. In particular, we define two abstract subtypes
```julia
abstract type ElementarySpace{𝕜} <: VectorSpace end
const IndexSpace = ElementarySpace

abstract type CompositeSpace{S<:ElementarySpace} <: VectorSpace end
```
Here, `ElementarySpace` is a super type for all vector spaces (objects) that can be
associated with the individual indices of a tensor, as hinted to by its alias `IndexSpace`.
It is parametrically dependent on `𝕜`, the field of scalars (see the next section on
[Fields](@ref ss_fields)).

On the other hand, subtypes of `CompositeSpace{S}` where `S<:ElementarySpace` are composed
of a number of elementary spaces of type `S`. So far, there is a single concrete type
`ProductSpace{S,N}` that represents the homogeneous tensor product of `N` vector spaces of
type `S`. Its properties are discussed in the section on
[Composite spaces](@ref ss_compositespaces), together with possible extensions for the
future.

Throughout TensorKit.jl, the function `spacetype` returns the type of `ElementarySpace`
associated with e.g. a composite space or a tensor. It works both on instances and in the
type domain. Its use will be illustrated below.

## [Fields](@id ss_fields)

Vector spaces (linear categories) are defined over a field of scalars ``𝕜``. We define a
type hierarchy to specify the scalar field, but so far only support real and complex
numbers, via
```julia
abstract type Field end

struct RealNumbers <: Field end
struct ComplexNumbers <: Field end

const ℝ = RealNumbers()
const ℂ = ComplexNumbers()
```
Note that `ℝ` and `ℂ` can be typed as `\bbR`+TAB and `\bbC`+TAB. One reason for defining
this new type hierarchy instead of recycling the types from Julia's `Number` hierarchy is
to introduce some syntactic sugar without committing type piracy. In particular, we now have
```@repl tensorkit
3 ∈ ℝ
5.0 ∈ ℂ
5.0+1.0*im ∈ ℝ
Float64 ⊆ ℝ
ComplexF64 ⊆ ℂ
ℝ ⊆ ℂ
ℂ ⊆ ℝ
```
and furthermore —probably more usefully— `ℝ^n` and `ℂ^n` create specific elementary vector
spaces as described in the next section. The underlying field of a vector space or tensor
`a` can be obtained with `field(a)`.

## [Elementary spaces](@id ss_elementaryspaces)

As mentioned at the beginning of this section, vector spaces that are associated with the
individual indices of a tensor should be implemented as subtypes of `ElementarySpace`. As
the domain and codomain of a tensor map will be the tensor product of such objects which all
have the same type, it is important that related vector spaces, e.g. the dual space, are
objects of the same concrete type (i.e. with the same type parameters in case of a
parametric type). In particular, every `ElementarySpace` should implement the following
methods

*   `dim(::ElementarySpace) -> ::Int` returns the dimension of the space as an `Int`

*   `dual(::S) where {S<:ElementarySpace} -> ::S` returns the
    [dual space](http://en.wikipedia.org/wiki/Dual_space) `dual(V)`, using an instance of
    the same concrete type (i.e. not via type parameters); this should satisfy
    `dual(dual(V))==V`

*   `conj(::S) where {S<:ElementarySpace} -> ::S` returns the
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
abstract type InnerProductSpace{𝕜} <: ElementarySpace{𝕜} end
```
to contain all vector spaces `V` which have an inner product and thus a canonical mapping
from `dual(V)` to `V` (for `𝕜 ⊆ ℝ`) or from `dual(V)` to `conj(V)` (otherwise). This
mapping is provided by the metric, but no further support for working with metrics is
currently implemented.

Finally there is
```julia
abstract type EuclideanSpace{𝕜} <: InnerProductSpace{𝕜} end
```
to contain all spaces `V` with a standard Euclidean inner product (i.e. where the metric is
the identity). These spaces have the natural isomorphisms `dual(V) == V` (for `𝕜 == ℝ`)
or `dual(V) == conj(V)` (for ` 𝕜 == ℂ`). In the language of the previous section on
[categories](@ref s_categories), this subtype represents
[dagger or unitary categories](@ref ss_adjoints), and support an `adjoint` operation. In
particular, we have two concrete types
```julia
struct CartesianSpace <: EuclideanSpace{ℝ}
    d::Int
end
struct ComplexSpace <: EuclideanSpace{ℂ}
  d::Int
  dual::Bool
end
```
to represent the Euclidean spaces $ℝ^d$ or $ℂ^d$ without further inner structure. They can
be created using the syntax `CartesianSpace(d) == ℝ^d == ℝ[d]` and
`ComplexSpace(d) == ℂ^d == ℂ[d]`, or
`ComplexSpace(d, true) == ComplexSpace(d; dual = true) == (ℂ^d)' == ℂ[d]'` for the
dual space of the latter. Note that the brackets are required because of the precedence
rules, since `d' == d` for `d::Integer`.

Some examples:
```@repl tensorkit
dim(ℝ^10)
(ℝ^10)' == ℝ^10 == ℝ[10] == ℝ[](10)
isdual((ℂ^5))
isdual((ℂ^5)')
isdual((ℝ^5)')
dual(ℂ^5) == (ℂ^5)' == conj(ℂ^5) == ComplexSpace(5; dual = true)
typeof(ℝ^3)
spacetype(ℝ^3)
spacetype(ℝ[])
```
Note that `ℝ[]` and `ℂ[]` are synonyms for `CartesianSpace` and `ComplexSpace` respectively,
such that yet another syntax is e.g. `ℂ[](d)`. This is not very useful in itself, and is
motivated by its generalization to `GradedSpace`. We refer to the subsection on
[graded spaces](@ref s_rep) on the [next page](@ref s_sectorsrepfusion) for further
information about `GradedSpace`, which is another subtype of `EuclideanSpace{ℂ}`
with an inner structure corresponding to the irreducible representations of a group, or more
generally, the simple objects of a fusion category.

!!! note
    For `ℂ^n` the dual space is equal (or naturally isomorphic) to the conjugate space, but
    not to the space itself. This means that even for `ℂ^n`, arrows matter in the
    diagrammatic notation for categories or for tensors, and in particular that a
    contraction between two tensor indices will check that one is living in the space and
    the other in the dual space. This is in contrast with several other software packages,
    especially in the context of tensor networks, where arrows are only introduced when
    discussing symmetries. We believe that our more purist approach can be useful to detect
    errors (e.g. unintended contractions). Only with `ℝ^n` will their be no distinction
    between a space and its dual. When creating tensors with indices in `ℝ^n` that have
    complex data, a one-time warning will be printed, but most operations should continue
    to work nonetheless.

## [Composite spaces](@id ss_compositespaces)

Composite spaces are vector spaces that are built up out of individual elementary vector
spaces of the same type. The most prominent (and currently only) example is a tensor
product of `N` elementary spaces of the same type `S`, which is implemented as
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
spacetype(V1 ⊗ V2)
spacetype(ProductSpace{ComplexSpace,3})
```
Here, the new function `dims` maps `dim` to the individual spaces in a `ProductSpace` and
returns the result as a tuple. Note that the rationale for the last result was explained in
the subsection on [duality](@ref ss_dual) in the introduction to
[category theory](@ref s_categories).

Following Julia's Base library, the function `one` applied to a `ProductSpace{S,N}` returns
the multiplicative identity, which is `ProductSpace{S,0}(())`. The same result is obtained
when acting on an instance `V` of `S::ElementarySpace` directly, however note that `V ⊗
one(V)` will yield a `ProductSpace{S,1}(V)` and not `V` itself. The same result can be
obtained with `⊗(V)`. Similar to Julia Base, `one` also works in the type domain.

In the future, other `CompositeSpace` types could be added. For example, the wave function
of an `N`-particle quantum system in first quantization would require the introduction of a
`SymmetricSpace{S,N}` or a `AntiSymmetricSpace{S,N}` for bosons or fermions respectively,
which correspond to the symmetric (permutation invariant) or antisymmetric subspace of
`V^N`, where `V::S` represents the Hilbert space of the single particle system. Other
domains, like general relativity, might also benefit from tensors living in a subspace with
certain symmetries under specific index permutations.

## [Space of morphisms](@id ss_homspaces)
Given that we define tensor maps as morphisms in a ``𝕜``-linear monoidal category, i.e.
linear maps, we also define a type to denote the corresponding space. Indeed, in a
``𝕜``-linear category ``C``, the set of morphisms ``\mathrm{Hom}(W,V)`` for ``V,W ∈ C`` is
always an actual vector space, irrespective of whether or not ``C`` is a subcategory of
``\mathbf{(S)Vect}``.

We introduce the type
```julia
struct HomSpace{S<:ElementarySpace, P1<:CompositeSpace{S}, P2<:CompositeSpace{S}}
    codomain::P1
    domain::P2
end
```
and can create it as either `domain → codomain` or `codomain ← domain` (where the arrows
are obtained as `\to+TAB` or `\leftarrow+TAB`, and as `\rightarrow+TAB` respectively). The
reason for first listing the codomain and than the domain will become clear in the
[section on tensor maps](@ref s_tensors).

Note that `HomSpace` is not a subtype of `VectorSpace`, i.e. we restrict the latter to
denote certain categories and their objects, and keep `HomSpace` distinct. However,
`HomSpace` has a number of properties defined, which we illustrate via examples
```@repl tensorkit
W = ℂ^2 ⊗ ℂ^3 → ℂ^3 ⊗ dual(ℂ^4)
field(W)
dual(W)
adjoint(W)
spacetype(W)
spacetype(typeof(W))
W[1]
W[2]
W[3]
W[4]
dim(W)
```
Note that indexing `W` yields first the spaces in the codomain, followed by the dual of the
spaces in the domain. This particular convention is useful in combination with the
instances of type [`TensorMap`](@ref), which represent morphisms living in such a
`HomSpace`. Also note that `dim(W)` here seems to be the product of the dimensions of the
individual spaces, but that this is no longer true once symmetries are involved. At any
time will `dim(::HomSpace)` represent the number of linearly independent morphisms in this
space.

## Partial order among vector spaces

Vector spaces of the same `spacetype` can be given a partial order, based on whether there
exist injective morphisms (a.k.a *monomorphisms*) or surjective morphisms (a.k.a.
*epimorphisms*) between them. In particular, we define `ismonomorphic(V1, V2)`, with
Unicode synonym `V1 ≾ V2` (obtained as `\precsim+TAB`), to express whether there exist
monomorphisms in `V1→V2`. Similarly, we define `isepimorphic(V1, V2)`, with Unicode
synonym `V1 ≿ V2` (obtained as `\succsim+TAB`), to express whether there exist
epimorphisms in `V1→V2`. Finally, we define `isisomorphic(V1, V2)`, with Unicode
alternative `V1 ≅ V2` (obtained as `\cong+TAB`), to express whether there exist
isomorphism in `V1→V2`. In particular `V1 ≅ V2` if and only if `V1 ≾ V2 && V1 ≿ V2`.

For completeness, we also export the strict comparison operators `≺` and `≻` (`\prec+TAB` and `\succ+TAB`), with definitions
```julia
≺(V1::VectorSpace, V2::VectorSpace) = V1 ≾ V2 && !(V1 ≿ V2)
≻(V1::VectorSpace, V2::VectorSpace) = V1 ≿ V2 && !(V1 ≾ V2)
```
However, as we expect these to be less commonly used, no ASCII alternative is provided.

In the context of `spacetype(V) <: EuclideanSpace`, `V1 ≾ V2` implies that there exists
isometries ``W:V1 → V2`` such that ``W^† ∘ W = \mathrm{id}_{V1}``, while `V1 ≅ V2` implies
that there exist unitaries ``U:V1→V2`` such that ``U^† ∘ U = \mathrm{id}_{V1}`` and
``U ∘ U^† = \mathrm{id}_{V2}``.

Note that spaces that are isomorphic are not necessarily equal. One can be a dual space,
and the other a normal space, or one can be an instance of `ProductSpace`, while the other
is an `ElementarySpace`. There will exist (infinitely) many isomorphisms between the
corresponding spaces, but in general none of those will be canonical.

There are also a number of convenience functions to create isomorphic spaces. The function
`fuse(V1, V2, ...)` or `fuse(V1 ⊗ V2 ⊗ ...)` returns an elementary space that is isomorphic
to `V1 ⊗ V2 ⊗ ...`. The function `flip(V::ElementarySpace)` returns a space that is
isomorphic to `V` but has `isdual(flip(V)) == isdual(V')`, i.e. if `V` is a normal space
than `flip(V)` is a dual space. `flip(V)` is different from `dual(V)` in the case of
[`GradedSpace`](@ref). It is useful to flip a tensor index from a ket to a bra (or
vice versa), by contracting that index with a unitary map from `V1` to `flip(V1)`. We refer
to [Index operations](@ref) for further information. Some examples:
```@repl tensorkit
ℝ^3 ≾ ℝ^5
ℂ^3 ≾ (ℂ^5)'
(ℂ^5) ≅ (ℂ^5)'
fuse(ℝ^5, ℝ^3)
fuse(ℂ^3, (ℂ^5)' ⊗ ℂ^2)
fuse(ℂ^3, (ℂ^5)') ⊗ ℂ^2 ≅ fuse(ℂ^3, (ℂ^5)', ℂ^2) ≅ ℂ^3 ⊗ (ℂ^5)' ⊗ ℂ^2
flip(ℂ^4)
flip(ℂ^4) ≅ ℂ^4
flip(ℂ^4) == ℂ^4
```

We also define the direct sum `V1` and `V2` as `V1 ⊕ V2`, where `⊕` is obtained by typing
`\oplus`+TAB. This is possible only if `isdual(V1) == isdual(V2)`. With a little pun on
Julia Base, `oneunit` applied to an elementary space (in the value or type domain) returns
the one-dimensional space, which is isomorphic to the scalar field of the space itself. Some
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

Finally, while spaces have a partial order, there is no unique infimum or supremum of a two
or more spaces. However, if `V1` and `V2` are two `ElementarySpace` instances with
`isdual(V1) == isdual(V2)`, then we can define a unique infimum `V::ElementarySpace` with
the same value of `isdual` that satisfies `V ≾ V1` and `V ≾ V2`, as well as a unique
supremum `W::ElementarySpace` with the same value of `isdual` that satisfies `W ≿ V1`
and `W ≿ V2`. For `CartesianSpace` and `ComplexSpace`, this simply amounts to the
space with minimal or maximal dimension, i.e.
```@repl tensorkit
infimum(ℝ^5, ℝ^3)
supremum(ℂ^5, ℂ^3)
supremum(ℂ^5, (ℂ^3)')
```
The names `infimum` and `supremum` are especially suited in the case of
[`GradedSpace`](@ref), as the infimum of two spaces might be different from either
of those two spaces, and similar for the supremum.
