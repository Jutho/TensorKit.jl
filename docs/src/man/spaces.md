# Vector spaces

From the [Introduction](@ref), it should be clear that an important aspect in the definition
of a tensor (map) is specifying the vector spaces and their structure in the domain and codomain
of the map. The starting point is an abstract type `VectorSpace`

```julia
abstract type VectorSpace end
```

which serves in a sense as the category ``Vect``. All instances of subtypes of `VectorSpace`
will represent vector spaces. In particular, we define two abstract subtypes

```julia
abstract type ElementarySpace{k} <: VectorSpace end
const IndexSpace = ElementarySpace

abstract type CompositeSpace{S<:ElementarySpace} <: VectorSpace end
```

Here, `ElementarySpace` is a super type for all vector spaces that can be associated with the
individual indices of a tensor, as hinted to by its alias `IndexSpace`. It is parametrically
dependent on `k`, the field of scalars (see the next section on [Fields](@ref)).

On the other hand, subtypes of `CompositeSpace{S}` where `S<:ElementarySpace` are composed of
a number of elementary spaces of type `S`. So far, there is a single concrete type `ProductSpace{S,N}`
that represents the homogeneous tensor product of `N` vector spaces of type `S`. Its properties
are discussed in the section on [Composite spaces](@ref), together with possible extensions
for the future.

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

Note that `ℝ` and `ℂ` can be typed as `\bbR`+TAB and `\bbC`+TAB. One reason for defining this
new type hierarchy instead of recycling the types from Julia's `Number` hierarchy is to introduce
some syntactic suggar without commiting type piracy. In particular, we now have
```@setup tensorkit
using TensorKit
```

```@repl tensorkit
3 ∈ ℝ
5.0 ∈ ℂ
5.0+1.0*im ∈ ℝ
Float64 ⊂ ℝ
ComplexF64 ⊂ ℂ
ℝ ⊂ ℂ
```
and furthermore ––probably more usefully–– `ℝ^n` and `ℂ^n` create specific elementary vector
spaces as described in the next section.

## Elementary vector spaces

```@docs
dim
dual
conj
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
