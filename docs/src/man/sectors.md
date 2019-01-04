# Sectors, representation spaces and fusion trees

```@setup tensorkit
using TensorKit
```

Symmetries in a physical system often result in tensors which are invariant under the action
of the symmetry group, where this group acts as a tensor product of group actions on every
tensor index separately. The group action on a single index, or thus, on the corresponding
vector space, can be decomposed into irreducible representations (irreps). Here, we
restrict to unitary representations, such that the corresponding vector spaces also have a
natural Euclidean inner product. In particular, the Euclidean inner product between two
vectors is invariant under the group action and thus transformas according to the trivial
representation of the group.

The corresponding vector spaces will be canonically represented as
``V = ⨁_a ℂ^{n_a} ⊗ R_{a}``, where ``a`` labels the different irreps, ``n_a`` is the number
of times irrep ``a`` appears and ``R_a`` is the vector space associated with irrep ``a``.
Irreps are also known as spin sectors (in the case of ``\mathsf{SU}_2``) or charge sectors
(in the case of `\mathsf{U}_1`), and we henceforth refer to `a` as a sector. As is briefly
discussed below, the approach we follow does in fact go beyond the case of irreps of groups,
and sectors would more generally correspond to simple objects in a (ribbon) fusion
category. Nonetheless, every step can be appreciated by using the representation theory of
``\mathsf{SU}_2`` or ``\mathsf{SU}_3`` as example. The vector space ``V`` is completely
specified by the values of ``n_a``.

The gain in efficiency (both in memory occupation and computation time) obtained from using
symmetric tensor maps is that, by Schur's lemma, they are block diagonal in the basis of
coupled sectors. To exploit this block diagonal form, it is however essential that we know
the basis transform from the individual (uncoupled) sectors appearing in the tensor product
form of the domain and codomain, to the totally coupled sectors that label the different
blocks. We refer to the latter as block sectors. The transformation from the uncoupled
sectors in the domain (or codomain) of the tensor map to the block sector is encoded in a
fusion tree (or splitting tree). Essentially, it is a sequential application of pairwise
fusion as described by the group's
[Clebsch-Gordan (CG) coefficients](https://en.wikipedia.org/wiki/Clebsch–Gordan_coefficients).
However, it turns out that we do not need the actual CG coefficients, but only how they
transform under transformations such as interchanging the order of the incoming irreps or
interchanging incoming and outgoing irreps. This information is known as the topological
data of the group, i.e. mainly the F-symbols, which are also known as recoupling
coefficients or [6j-symbols](https://en.wikipedia.org/wiki/6-j_symbol) (more accurately,
it's actually [Racah's W-coefficients](https://en.wikipedia.org/wiki/Racah_W-coefficient))
in the case of ``\mathsf{SU}_2``.

Below, we describe how to specify a certain type of sector what information about them
needs to be implemented. Then, we describe how to build a space `V` composed of a direct sum
of different sectors. In the last section, we explain the details of fusion trees, i.e.
their construction and manipulation. But first, we provide a quick theoretical overview of
the required data of the representation theory of a group.

## Representation theory and unitary fusion categories

Let the different irreps or sectors be labeled as ``a``, ``b``, ``c``, … First and foremost,
we need to specify the *fusion rules* ``a ⊗ b = ⨁ N_{a,b}^{c} c`` with ``N_{a,b}^c`` some
non-negative integers. There should always exists a unique trivial sector ``u`` such that
``a ⊗ u = a = u ⊗ a``. Furthermore, there should exist a unique sector ``\overline{a}``
such that ``N_{a,\overline{a}}^{u} = 1``, whereas for all ``b ≂̸ \overline{a}``,
``N_{a,b}^{u} = 0``. For example, for the representations of ``\mathsf{SU}_2``, all irreps
are self-dual (i.e. ``a = \overline{a}``) and the trivial sector corresponds to spin zero.

The meaning of the fusion rules is that the space of transformations ``R_a ⊗ R_b → R_c``
(or vice versa) has dimension ``N_{a,b}^c``. In particular, we assume the existence of a
basis consisting of unitary tensor maps ``X_{a,b}^{c,μ} : R_c → R_a ⊗ R_b`` with
``μ = 1, …, N_{a,b}^c`` such that

``(X_{a,b}^{c,μ})^† X_{a,b}^{c,μ} = \mathrm{id}_{R_c}``

and

``\sum_{c} \sum_{μ = 1}^{N_{a,b}^c} X_{a,b}^{c,μ} (X_{a,b}^{c,μ})^\dagger = \mathrm{id}_{R_a ⊗ R_b}``

The tensors ``X_{a,b}^{c,μ}`` are the splitting tensors, their hermitian conjugate are the
fusion tensors. They are only determined up to a unitary basis transform within the space,
i.e. acting on the multiplicity label ``μ = 1, …, N_{a,b}^c``. For ``\mathsf{SU}_2``, where
``N_{a,b}^c`` is zero or one and the multiplicity labels are absent, the entries of
``X_{a,b}^{c}`` are precisely given by the CG coefficients. The point is that we do not
need to know the tensors ``X_{a,b}^{c,μ}``, the topological data of (the representation
category of) the group describes the following transformation:

*   F-move or recoupling: the transformation between ``(a ⊗ b) ⊗ c`` to ``a ⊗ (b ⊗ c)``:

    ``(X_{a,b}^{e,μ} ⊗ \mathrm{id}_c) ∘ X_{e,c}^{d,ν} = ∑_{f,κ,λ} [F^{a,b,c}_{d}]^{e,μν}_{f,κλ} (X_{a,b}^{e,μ} ⊗ \mathrm{id}_c) X_{e,c}^{d,ν} (\mathrm{id}_a ⊗ X_{b,c}^{f,κ}) ∘ X_{a,f}^{d,λ}``

*   [Braiding](@ref) or permuting as defined by ``σ_{a,b}: R_a ⊗ R_b → R_b ⊗ R_a``:

    ``σ_{R_a,R_b} ∘ X_{a,b}^{c,μ} = ∑_{ν} [R_{a,b}^c]^μ_ν X_{b,a}^{c,ν}``

The dimensions of the spaces ``R_a`` on which representation ``a`` acts are denoted as
``d_a`` and referred to as quantum dimensions. In particular ``d_u = 1`` and
``d_a = d_{\overline{a}}``. This information is also encoded in the F-symbol as
``d_a = | [F^{a \overline{a} a}_a]^u_u |^{-1}``. Note that there are no multiplicity labels
in that particular F-symbol as ``N_{a,\overline{a}}^u = 1``.

If, for every ``a`` and ``b``, there is a unique ``c`` such that ``a ⊗ b = c`` (i.e.
``N_{a,b}^{c} = 1`` and ``N_{a,b}^{c′} = 0`` for all other ``c′``), the category is abelian.
Indeed, the representations of a group have this property if and only if the group
multiplication law is commutative. In that case, all spaces ``R_{a}`` associated with the
representation are one-dimensional and thus trivial. In all other cases, the category is
nonabelian. We find it useful to further finegrain between categories which have all
``N_{a,b}^c`` equal to zero or one (such that no multiplicity labels are needed), e.g. the
representations of ``\mathsf{SU}_2``, and those where some ``N_{a,b}^c`` are larger than
one, e.g. the representations of ``\mathsf{SU}_3``.

## Sectors

We introduce a new abstract type to represent different possible sectors
```julia
abstract type Sector end
```
Any concrete subtype of `Sector` should be such that its instances represent a consistent
set of sectors, corresponding to the irreps of some group, or, more generally, the simple
objects of a (unitary) fusion category. We refer to Appendix E of [^kitaev] for a good
reference.

The minimal data to completely specify a type of sector are
*   the fusion rules, i.e. `` a ⊗ b = ⨁ N_{a,b}^{c} c ``; this is implemented by a function
    [`Nsymbol(a,b,c)`](@ref)
*   the list of fusion outputs from ``a ⊗ b``; while this information is contained in
    ``N_{a,b}^c``, it might be costly or impossible to iterate over all possible values of
    `c` and test `Nsymbol(a,b,c)`; instead we implement for `a ⊗ b` to return an iterable
    object (e.g. tuple, array or a custom Julia type that listens to `Base.iterate`) and
    which generates all `c` for which ``N_{a,b}^c ≂̸ 0``
*   the identity object `u`, such that ``a ⊗ u = a = u ⊗ a``; this is implemented by the
    function `one(a)` (and also in type domain) from Julia Base
*   the dual or conjugate representation ``\overline{a}`` for which
    `N_{a,\overline{a}}^{u} = 1`; this is implemented by `conj(a)` from Julia Base;
    `dual(a)` also works as alias, but `conj(a)` is the method that should be defined
*   the F-symbol or recoupling coefficients ``[F^{a,b,c}_{d}]^e_f``, implemented as the
    function [`Fsymbol(a,b,c,d,e,f)`](@ref)
*   the R-symbol ``R_{a,b}^c``, implemented as the function [`Rsymbol(a,b,c)`](@ref)
*   for practical reasons: a hash function `hash(a, h)`, because sectors and objects
    created from them are used as keys in lookup tables (i.e. dictionaries), and a
    canonical order of sectors via `isless(a,b)`, in order to unambiguously represent
    representation spaces ``V = ⨁_a ℂ^{n_a} ⊗ R_{a}``.

Further information, such as the quantum dimensions ``d_a`` and Frobenius-Schur indicator
``χ_a`` (only if ``a == \overline{a}``) are encoded in the F-symbol. They are obtained as
`dim(a)` and [`frobeniusschur(a)`](@ref). These functions have default definitions which
extract the requested data from `Fsymbol(a,conj(a),a,a,one(a),one(a))`, but they can be
overloaded in case the value can be computed more efficiently.

It is useful to distinguish between three cases with respect to the fusion rules. For irreps
of Abelian groups, we have that for every ``a`` and ``b``, there exists a unique ``c`` such
that ``a ⊗ b = c``, i.e. there is only a single fusion channel. This follows simply from the
fact that all irreps are one-dimensional. All other cases are referred to as non-abelian,
i.e. the irreps of a non-abelian group or some more general fusion category. We still
distinguish between the case where all entries of ``N_{a,b}^c ≦ 1``, i.e. they are zero or
one. In that case, ``[F^{a,b,c}_{d}]^e_f`` and ``R_{a,b}^c`` are scalars. If some
``N_{a,b}^c > 1``, it means that the same sector ``c`` can appear more than once in the
fusion product of ``a`` and ``b``, and we need to introduce some multiplicity label ``μ``
for the different copies. We implement a "trait" (similar to `IndexStyle` for
`AbstractArray`s in Julia Base), i.e. a type hierarchy
```julia
abstract type FusionStyle end
struct Abelian <: FusionStyle
end
abstract type NonAbelian <: FusionStyle end
struct SimpleNonAbelian <: NonAbelian # non-abelian fusion but multiplicity free
end
struct DegenerateNonAbelian <: NonAbelian # non-abelian fusion with multiplicities
end
```
New sector types `G<:Sector` should then indicate which fusion style they have by defining
`FusionStyle(::Type{G})`.

In the representation and manipulation of symmetric tensors, it will be important to couple
or fuse different sectors together into a single block sector. The section on
[Fusion trees](@ref) describes the details of this process, which consists of pairwise
fusing two sectors into a single coupled sector, which is then fused with the next
uncoupled sector. For this, we assume the existence of a basis of unitary tensor maps
``X_{a,b}^{c,μ} : R_c → R_a ⊗ R_b`` such that

*   ``(X_{a,b}^{c,μ})^† X_{a,b}^{c,μ} = \mathrm{id}_{R_c}`` and

*   ``\sum_{c} \sum_{μ = 1}^{N_{a,b}^c} X_{a,b}^{c,μ} (X_{a,b}^{c,μ})^\dagger = \mathrm{id}_{R_a ⊗ R_b}``

The tensors ``X_{a,b}^{c,μ}`` are the splitting tensors, their hermitian conjugate are the
fusion tensors. For ``\mathsf{SU}_2``, their entries are precisely given by the CG
coefficients. The point is that we do not need to know the tensors ``X_{a,b}^{c,μ}``, the
topological data of (the representation category of) the group describes the following
transformation:

*   F-move or recoupling: the transformation between ``(a ⊗ b) ⊗ c`` to ``a ⊗ (b ⊗ c)``:

    ``(X_{a,b}^{e,μ} ⊗ \mathrm{id}_c) ∘ X_{e,c}^{d,ν} = ∑_{f,κ,λ} [F^{a,b,c}_{d}]^{e,μν}_{f,κλ} (X_{a,b}^{e,μ} ⊗ \mathrm{id}_c) X_{e,c}^{d,ν} (\mathrm{id}_a ⊗ X_{b,c}^{f,κ}) ∘ X_{a,f}^{d,λ}``

*   [Braiding](@ref) or permuting as defined by ``σ_{a,b}: R_a ⊗ R_b → R_b ⊗ R_a``:

    ``σ_{a,b} ∘ X_{a,b}^{c,μ} = ∑_{ν} [R_{a,b}^c]^μ_ν X_{b,a}^{c,ν}

Furthermore, there is a relation between splitting vertices and fusion vertices given by the
B-symbol, but we refer to the section on [Fusion trees](@ref) for the precise definition and
further information. The required data is completely encoded in the the F-symbol, and
corresponding Julia function `Bsymbol(a,b,c)` is implemented as
```julia
function Bsymbol(a::G, b::G, c::G) where {G<:Sector}
    if FusionStyle(G) isa Abelian || FusionStyle(G) isa SimpleNonAbelian
        Fsymbol(a, b, dual(b), a, c, one(a))
    else
        reshape(Fsymbol(a,b,dual(b),a,c,one(a)), (Nsymbol(a,b,c), Nsymbol(c,dual(b),a)))
    end
end
```
but a more efficient implementation may be provided.

Before discussing in more detail how a new sector type should be implemented, let us study
the cases which have already been implemented. Currently, they all correspond to the irreps
of groups.

### Existing group representations
The first sector type is called `Trivial`, and corresponds to the case where there is
actually no symmetry, or thus, the symmetry is the trivial group with only an identity
operation and a trivial representation. Its representation theory is particularly simple:
```julia
struct Trivial <: Sector
end
Base.one(a::Sector) = one(typeof(a))
Base.one(::Type{Trivial}) = Trivial()
Base.conj(::Trivial) = Trivial()
⊗(::Trivial, ::Trivial) = (Trivial(),)
Nsymbol(::Trivial, ::Trivial, ::Trivial) = true
FusionStyle(::Type{Trivial}) = Abelian()
Fsymbol(::Trivial, ::Trivial, ::Trivial, ::Trivial, ::Trivial, ::Trivial) = 1
Rsymbol(::Trivial, ::Trivial, ::Trivial) = 1
```
The `Trivial` sector type is special cased in the construction of tensors, so that most of
these definitions are not actually used.

For all abelian groups, we gather a number of common definitions
```julia
abstract type AbelianIrrep <: Sector end

Base.@pure FusionStyle(::Type{<:AbelianIrrep}) = Abelian()
Base.@pure BraidingStyle(::Type{<:AbelianIrrep}) = Bosonic()

Nsymbol(a::G, b::G, c::G) where {G<:AbelianIrrep} = c == first(a ⊗ b)
Fsymbol(a::G, b::G, c::G, d::G, e::G, f::G) where {G<:AbelianIrrep} =
    Int(Nsymbol(a,b,e)*Nsymbol(e,c,d)*Nsymbol(b,c,f)*Nsymbol(a,f,d))
frobeniusschur(a::AbelianIrrep) = 1
Bsymbol(a::G, b::G, c::G) where {G<:AbelianIrrep} = Float64(Nsymbol(a, b, c))
Rsymbol(a::G, b::G, c::G) where {G<:AbelianIrrep} = Float64(Nsymbol(a, b, c))
```
With these common definition, we implement the representation theory of the two most common
Abelian groups
```julia
struct ZNIrrep{N} <: AbelianIrrep
    n::Int8
    function ZNIrrep{N}(n::Integer) where {N}
        new{N}(mod(n, N))
    end
end
Base.one(::Type{ZNIrrep{N}}) where {N} =ZNIrrep{N}(0)
Base.conj(c::ZNIrrep{N}) where {N} = ZNIrrep{N}(-c.n)
⊗(c1::ZNIrrep{N}, c2::ZNIrrep{N}) where {N} = (ZNIrrep{N}(c1.n+c2.n),)

struct U1Irrep <: AbelianIrrep
    charge::HalfInteger
end
Base.one(::Type{U1Irrep}) = U1Irrep(0)
Base.conj(c::U1Irrep) = U1Irrep(-c.charge)
⊗(c1::U1Irrep, c2::U1Irrep) = (U1Irrep(c1.charge+c2.charge),)
```
together with some abbreviated Unicode aliases
```julia
const ℤ₂ = ZNIrrep{2}
const ℤ₃ = ZNIrrep{3}
const ℤ₄ = ZNIrrep{4}
const U₁ = U1Irrep
```
In the definition of `U1Irrep`, `HalfInteger<:Number` is a Julia type defined in [WignerSymbols.jl](https://github.com/Jutho/WignerSymbols.jl),
which is also used for `SU2Irrep` below, that stores integer or half integer numbers using
twice their value. Strictly speaking, the linear representations of `U₁` can only have
integer charges, and fractional charges lead to a projective representation. It can be
useful to allow half integers in order to describe spin 1/2 systems with an axis rotation
symmetry. As a user, you should not worry about the details of `HalfInteger`, and
additional methods for automatic conversion and pretty printing are provided, as
illustrated by the following example
```@repl tensorkit
U₁(0.5)
U₁(0.4)
U₁(1) ⊗ U₁(1//2)
u = first(U₁(1) ⊗ U₁(1//2))
Nsymbol(u, conj(u), one(u))
z = ℤ₃(1)
z ⊗ z
conj(z)
one(z)
```
For `ZNIrrep{N}`, we use an `Int8` for compact storage, assuming that this type will not be
used with `N>64` (we need `2*(N-1) <= 127` in order for `a ⊗ b` to work correctly).

As a further remark, even in the abelian case where `a ⊗ b` is equivalent to a single new
label `c`, we return it as an iterable container, in this case a one-element tuple `(c,)`.

As mentioned above, we also provide the following definitions
```julia
Base.hash(c::ZNIrrep{N}, h::UInt) where {N} = hash(c.n, h)
Base.isless(c1::ZNIrrep{N}, c2::ZNIrrep{N}) where {N} = isless(c1.n, c2.n)
Base.hash(c::U1Irrep, h::UInt) = hash(c.charge, h)
Base.isless(c1::U1Irrep, c2::U1Irrep) where {N} =
    isless(abs(c1.charge), abs(c2.charge)) || zero(HalfInteger) < c1.charge == -c2.charge
```
Since sectors or objects made out of tuples of sectors (see the section on
[Fusion Trees](@ref) below) are often used as keys in look-up tables (i.e. subtypes of
`AbstractDictionary` in Julia), it is important that they can be hashed efficiently. We
just hash the sectors above based on their numerical value. Note that hashes will only be
used to compare sectors of the same type. The `isless` function provides a canonical order
for sectors of a given type `G<:Sector`, which is useful to uniquely and unambiguously
specify a representation space ``V = ⨁_a ℂ^{n_a} ⊗ R_{a}``, as described in the section on
[Representation spaces](@ref) below.

The first example of a non-abelian representation category is that of ``\mathsf{SU}_2``, the
implementation of which is summarized by
```julia
struct SU2Irrep <: Sector
    j::HalfInteger
end
Base.one(::Type{SU2Irrep}) = SU2Irrep(zero(HalfInteger))
Base.conj(s::SU2Irrep) = s
⊗(s1::SU2Irrep, s2::SU2Irrep) =
    SectorSet{SU2Irrep}(HalfInteger, abs(s1.j.num-s2.j.num):2:(s1.j.num+s2.j.num) )
dim(s::SU2Irrep) = s.j.num+1
Base.@pure FusionStyle(::Type{SU2Irrep}) = SimpleNonAbelian()
Nsymbol(sa::SU2Irrep, sb::SU2Irrep, sc::SU2Irrep) = WignerSymbols.δ(sa.j, sb.j, sc.j)
Fsymbol(s1::SU2Irrep, s2::SU2Irrep, s3::SU2Irrep,
        s4::SU2Irrep, s5::SU2Irrep, s6::SU2Irrep) =
    WignerSymbols.racahW(s1.j, s2.j, s4.j, s3.j, s5.j, s6.j)*sqrt(dim(s5)*dim(s6))
function Rsymbol(sa::SU2Irrep, sb::SU2Irrep, sc::SU2Irrep)
    Nsymbol(sa, sb, sc) || return 0.
    iseven(convert(Int, sa.j+sb.j-sc.j)) ? 1.0 : -1.0
end
Base.hash(s::SU2Irrep, h::UInt) = hash(s.j, h)
Base.isless(s1::SU2Irrep, s2::SU2Irrep) = isless(s1.j, s2.j)
const SU₂ = SU2Irrep
```
and some methods for pretty printing and converting from real numbers to irrep labels. As
one can notice, the topological data (i.e. `Nsymbol` and `Fsymbol`) are provided by the
package [WignerSymbols.jl](https://github.com/Jutho/WignerSymbols.jl). The iterable `a ⊗ b`
is a custom type, that the user does not need to care about. Some examples
```@repl tensorkit
s = SU₂(3//2)
conj(s)
dim(s)
collect(s ⊗ s)
for s′ in s ⊗ s
    @show Nsymbol(s, s, s′)
    @show Rsymbol(s, s, s′)
end
```

A final non-abelian representation theory is that of the semidirect product
``\mathsf{U}₁ ⋉ ℤ_2``, where in the context of quantum systems, this occurs in the case of
systems with particle hole symmetry and the non-trivial element of ``ℤ_2`` acts as charge
conjugation `C`. It has the effect of interchaning ``\mathsf{U}_1`` irreps ``n`` and
``-n``, and turns them together in a joint 2-dimensional index, except for the case
``n=0``. Irreps are therefore labeled by integers `n ≧ 0`, however for `n=0` the ``ℤ₂``
symmetry can be realized trivially or non-trivially, resulting in an even and odd one-
dimensional irrep with ``\mathsf{U})_1`` charge ``0``. Given
``\mathsf{U}_1 ≂ \mathsf{SO}_2``, this group is also simply known as ``\mathsf{O}_2``, and
the two representations with `` n = 0`` are the scalar and pseudo-scalar, respectively.
However, because we also allow for half integer representations, we refer to it as `CU₁` or
`CU1Irrep` in full.
```julia
struct CU1Irrep <: Sector
    j::HalfInteger # value of the U1 charge
    s::Int # rep of charge conjugation:
    # if j == 0, s = 0 (trivial) or s = 1 (non-trivial),
    # else s = 2 (two-dimensional representation)
    # Let constructor take the actual half integer value j
    function CU1Irrep(j::HalfInteger, s::Int = ifelse(j>0, 2, 0))
        if ((j > 0 && s == 2) || (j == 0 && (s == 0 || s == 1)))
            new(j, s)
        else
            error("Not a valid CU₁ irrep")
        end
    end
end
Base.one(::Type{CU1Irrep}) = CU1Irrep(zero(HalfInteger), 0)
Base.conj(c::CU1Irrep) = c
dim(c::CU1Irrep) = ifelse(c.j == zero(HalfInteger), 1, 2)
Base.@pure FusionStyle(::Type{CU1Irrep}) = SimpleNonAbelian()
...
const CU₁ = CU1Irrep
```
The rest of the implementation can be read in the source code, but is rather long due to all
the different cases for the arguments of `Fsymbol`.

So far, no sectors have been implemented with `FusionStyle(G) == DegenerateNonAbelian()`,
though an example would be the representation theory of ``\mathsf{SU}_3``. Such sectors are
not yet fully supported; certain operations remain to be implemented.

### Combining different sectors
It is also possible to define two or more different types of symmetries, e.g. when the total
symmetry group is a direct product of individual simple groups. Such sectors are obtained
using the binary operator `×`, which can be entered as `\times`+TAB. Some examples
```@repl tensorkit
a = ℤ₃(1) × U₁(1)
typeof(a)
conj(a)
one(a)
dim(a)
collect(a ⊗ a)
FusionStyle(a)
b = ℤ₃(1) × SU₂(3//2)
typeof(b)
conj(b)
one(b)
dim(b)
collect(b ⊗ b)
FusionStyle(c)
c = SU₂(1) × SU₂(3//2)
typeof(c)
conj(c)
one(c)
dim(c)
collect(c ⊗ c)
FusionStyle(c)
```
We refer to the source file of [`ProductSector`](@ref) for implementation details.

### Defining a new type of sector

By know, it should be clear how to implement a new `Sector` subtype. Ideally, a new
`G<:Sector` type is a `struct G ... end` (immutable) that has `isbitstype(G) == true` (see
Julia's manual), and implements the following minimal set of methods
```julia
Base.one(::Type{G}) = G(...)
Base.conj(a::G) = G(...)
TensorKit.FusionStyle(::Type{G}) = ...
    # choose one: Abelian(), SimpleNonAbelian(), DegenerateNonAbelian()
TensorKit.Nsymbol(a::G, b::G, c::G) = ...
    # Bool or Integer if FusionStyle(G) == DegenerateNonAbelian()
Base.:⊗(a::G, b::G) = ... # some iterable object that generates all possible fusion outputs
TensorKit.Fsymbol(a::G, b::G, c::G, d::G, e::G, f::G)
TensorKit.Rsymbol(a::G, b::G, c::G)
Base.hash(a::G, h::UInt)
Base.isless(a::G, b::G)
```

Additionally, suitable definitions can be given for
```julia
TensorKit.dim(a::G) = ...
TensorKit.frobeniusschur(a::G) = ...
TensorKit.Bsymbol(a::G, b::G, c::G) = ...
```

If `FusionStyle(G) == DegenerateNonAbelian()`, then the multiple outputs `c` in the tensor
product of `a` and `b` will be labeled as `i=1`, `2`, …, `Nsymbol(a,b,c)`. Optionally, a
different label can be provided by defining
```julia
TensorKit.vertex_ind2label(i::Int, a::G, b::G, c::G) = ...
# some label, e.g. a `Char` or `Symbol`
```
The following function will then automatically determine the corresponding label type (which
should not vary, i.e. `vertex_ind2label` should be type stable)
```julia
Base.@pure vertex_labeltype(G::Type{<:Sector}) =
    typeof(vertex_ind2label(1, one(G), one(G), one(G)))
```

The following type, which already appeared in the implementation of `SU2Irrep` above, can be
useful for providing the return type of `a ⊗ b`
```julia
struct SectorSet{G<:Sector,F,S}
    f::F
    set::S
end
...
function Base.iterate(s::SectorSet{G}, args...) where {G<:Sector}
    next = iterate(s.set, args...)
    next === nothing && return nothing
    val, state = next
    return convert(G, s.f(val)), state
end
```
That is, `SectorSet(f, set)` behaves as an iterator that applies `x->convert(G, f(x))` on
the elements of `set`; if `f` is not provided it is just taken as the function `identity`.

### Generalizations

As mentioned before, the framework for sectors outlined above depends is in one-to-one
correspondence to the topological data for specifying a unitary
[fusion category](https://en.wikipedia.org/wiki/Fusion_category).
In fact, because we also need a braiding (corresponding to `Rsymbol(a,b,c)`) it is a so-
called ribbon fusion category. However, the category does not need to be modular.  The
category of representations of a finite group[^1] corresponds to a typical example (which is
not modular and which have a symmetric braiding). Other examples are the representation of
quasi-triangular Hopf algebras, which are typically known as anyon theories in the physics
literature, e.g. Fibonicci anyons, Ising anyons, … In those cases, quantum dimensions
``d_a`` are non-integer, and there is no vector space interpretation to objects ``R_a``
(which we can identify with just ``a``) in the decomposition ``V = ⨁_a ℂ^{n_a} ⊗ R_{a}``.
The different sectors ``a``, … just represent abstract objects. However, there is still a
vector space associated with the homomorphisms ``a ⊗ b → c``, whose dimension is
``N_{a,b}^c``. The objects ``X_{a,b}^{c,μ}`` for ``μ = 1,…,N_{a,b}^c`` serve as an abstract
basis for this space and from there on the discussion is completely equivalent.

So far, none of these cases have been implemented, but it is a simple exercise to do so.

## Representation spaces
We have introduced `Sector` subtypes as a way to label the irreps or sectors in the
decomposition ``V = ⨁_a ℂ^{n_a} ⊗ R_{a}``. To actually represent such spaces, we now also
introduce a corresponding type `RepresentationSpace`, which is a subtype of
`EuclideanSpace{ℂ}`, i.e.
```julia
abstract type RepresentationSpace{G<:Sector} <: EuclideanSpace{ℂ} end
```
Note that this is still an abstract type, nonetheless it will be the type name that the user
calls to create specific instances.

### Types
The actual implementation comes in two flavors
```julia
struct GenericRepresentationSpace{G<:Sector} <: RepresentationSpace{G}
    dims::SectorDict{G,Int}
    dual::Bool
end
struct ZNSpace{N} <: RepresentationSpace{ZNIrrep{N}}
    dims::NTuple{N,Int}
    dual::Bool
end
```
The `GenericRepresentationSpace` is the default implementation and stores the different
sectors ``a`` and their corresponding degeneracy ``n_a`` as key value pairs in an
`Associative` array, i.e. a dictionary `dims::SectorDict`. `SectorDict` is a constant type
alias for a specific dictionary implementation, either Julia's default `Dict` or the type
`SortedVectorDict` implemented in TensorKit.jl. Note that only sectors ``a`` with non-zero
``n_a`` are stored. The second implementation `ZNSpace{N}` is a dedicated implementation for
`ZNIrrep{N}` symmetries, and just stores all `N` different values ``n_a`` in a tuple.

As mentioned, creating instances of these types goes via `RepresentationSpace`, using a list
of pairs `a=>n_a`, i.e. `V = RepresentationSpace(a=>n_a, b=>n_b, c=>n_c)`. In this case, the
sector type `G` is inferred from the sectors. However, it is often more convenient to
specify the sector type explicitly, since then the sectors are automatically converted to
the correct type, i.e. compare
```@repl tensorkit
RepresentationSpace{U1Irrep}(0=>3, 1=>2, -1=>1) ==
    RepresentationSpace(U1Irrep(0)=>3, U1Irrep(1)=>2, U1Irrep(-1)=>1)
```
or using Unicode
```@repl tensorkit
RepresentationSpace{U₁}(0=>3, 1=>2, -1=>1) ==
    RepresentationSpace(U₁(0)=>3, U₁(-1)=>1, U₁(1)=>2)
```
However, both are still to long for the most common cases. Therefore, we provide a number of
type aliases, both in plain ASCII and in Unicode
```julia
const ℤ₂Space = ZNSpace{2}
const ℤ₃Space = ZNSpace{3}
const ℤ₄Space = ZNSpace{4}
const U₁Space = GenericRepresentationSpace{U₁}
const CU₁Space = GenericRepresentationSpace{CU₁}
const SU₂Space = GenericRepresentationSpace{SU₂}

# non-Unicode alternatives
const Z2Space = ℤ₂Space
const Z3Space = ℤ₃Space
const Z4Space = ℤ₄Space
const U1Space = U₁Space
const CU1Space = CU₁Space
const SU2Space = SU₂Space
```

### Methods
There are a number of methods to work with instances `V` of `RepresentationSpace`. The
function [`sectortype`](@ref) returns the type of the sector labels. It also works on other
vector spaces, in which case it returns [`Trivial`](@ref). The function [`sectors`](@ref)
returns an iterator over the different sectors `a` with non-zero `n_a`, for other
`ElementarySpace` types it returns `(Trivial,)`. The degeneracy dimensions `n_a` can be
extracted as `dim(V, a)`, it properly returns `0` if sector `a` is not present in the
decomposition of `V`. With `hassector(V, a)` one can check if `V` contains a sector `a` with
`dim(V,a)>0`. Finally, `dim(V)` returns the total dimension of the space `V`, i.e.
``∑_a n_a d_a`` or thus `dim(V) = sum(dim(V,a) * dim(a) for a in sectors(V))`.

Other methods for `ElementarySpace`, such as [`dual`](@ref), [`fuse`](@ref) and
[`flip`](@ref) also work. In fact, `RepresentationSpace` is the reason `flip` exists, cause
in this case it is different then `dual`. The existence of flip originates from the
non-trivial isomorphism between ``R_{\overline{a}}`` and ``R_{a}^*``, i.e. the
representation space of the dual ``\overline{a}`` of sector ``a`` and the dual of the
representation space of sector ``a``.

In order for `flip(V)` to be isomorphic to `V`, it is such that, if
`V = RepresentationSpace(a=>n_a,...)` then
`flip(V) = dual(RepresentationSpace(dual(a)=>n_a,....))`.
Furthermore, for two spaces `V1 = RepresentationSpace(a=>n1_a, ...)` and
`V2 = RepresentationSpace(a=>n2_a, ...)`, we have
`min(V1,V2) = RepresentationSpace(a=>min(n1_a,n2_a), ....)` and similarly for `max`,
i.e. they act on the degeneracy dimensions of every sector separately. Therefore, it can be
that the return value of `min(V1,V2)` or `max(V1,V2)` is neither equal to `V1` or `V2`.

For `W` a `ProductSpace{<:RepresentationSpace{G},N}`, `sectors(W)` returns an iterator that
generates all possible combinations of sectors `as` represented as `NTuple{G,N}`. The
function `dims(W, as)` returns the corresponding tuple with degeneracy dimensions, while
`dim(W, as)` returns the product of these dimensions. `hassector(W, as)` is equivalent to
`dim(W, as)>0`. Finally, there is the function [`blocksectors(W)`](@ref) which returns a
list (of type `Vector`) with all possible "block sectors" or total/coupled sectors that can
result from fusing the individual uncoupled sectors in `W`. Correspondingly,
[`blockdim(W, a)`](@ref) counts the total dimension of coupled sector `a` in `W`. The
machinery for computing this is the topic of the next section on [Fusion trees](@ref), but
first, it's time for some examples.

### Examples
Let's start with an example involving ``\mathsf{U}_1``:
```@repl tensorkit
V1 = RepresentationSpace{U₁}(0=>3, 1=>2, -1=>1)
V1 == U1Space(0=>3, 1=>2, -1=>1) == U₁Space(0=>3, 1=>2, -1=>1)
(sectors(V1)...,)
dim(V1, U₁(1))
dim(V1)
hassector(V1, U₁(1))
hassector(V1, U₁(2))
dual(V1)
flip(V1)
V2 = U1Space(0=>2, 1=>1, -1=>1, 2=>1, -2=>1)
min(V1,V2)
max(V1,V2)
⊕(V1,V2)
W = ⊗(V1,V2)
(sectors(W)...,)
dims(W, (U₁(0), U₁(0)))
dim(W, (U₁(0), U₁(0)))
hassector(W, (U₁(0), U₁(0)))
hassector(W, (U₁(2), U₁(0)))
fuse(W)
(blocksectors(W)...,)
blockdim(W, U₁(0))
```
and then with ``\mathsf{SU}_2``:
```@repl tensorkit
V1 = RepresentationSpace{SU₂}(0=>3, 1//2=>2, 1=>1)
V1 == SU2Space(0=>3, 1//2=>2, 1=>1) == SU₂Space(0=>3, 1//2=>2, 1=>1)
(sectors(V1)...,)
dim(V1, SU₂(1))
dim(V1)
hassector(V1, SU₂(1))
hassector(V1, SU₂(2))
dual(V1)
flip(V1)
V2 = SU2Space(0=>2, 1//2=>1, 1=>1, 3//2=>1, 2=>1)
min(V1,V2)
max(V1,V2)
⊕(V1,V2)
W = ⊗(V1,V2)
(sectors(W)...,)
dims(W, (SU₂(0), SU₂(0)))
dim(W, (SU₂(0), SU₂(0)))
hassector(W, (SU₂(0), SU₂(0)))
hassector(W, (SU₂(2), SU₂(0)))
fuse(W)
(blocksectors(W)...,)
blockdim(W, SU₂(0))
```

## Fusion trees

**Work in progress**

The gain in efficiency (both in memory occupation and computation time) obtained from using
symmetric tensor maps is that, by Schur's lemma, they are block diagonal in the basis of
coupled sectors. To exploit this block diagonal form, it is however essential that we know
the basis transform from the individual (uncoupled) sectors appearing in the tensor product
form of the domain and codomain, to the totally coupled sectors that label the different blocks.
We refer to the latter as block sectors, as we already encountered in the previous section
[`blocksectors`](@ref) and [`blockdim`](@ref) defined on the type [`ProductSpace`](@ref).

To couple or fuse the different sectors together into a single block sector, we sequentially
fuse together two sectors into a single coupled sector, which is then fused with the next
uncoupled sector. For this, we assume the existence of unitary tensor maps
``X_{a,b}^{c,μ} : R_c → R_a ⊗ R_b`` introduced in the section [Sectors](@ref).


such that ``(X_{a,b}^{c,μ})^† X_{a,b}^{c,μ} = \mathrm{id}_{R_c}`` and

``\sum_{c} \sum_{μ = 1}^{N_{a,b}^c} X_{a,b}^{c,μ} (X_{a,b}^{c,μ})^\dagger = \mathrm{id}_{R_a ⊗ R_b}``

The tensors ``X_{a,b}^{c,μ}`` are the splitting tensors, their hermitian conjugate are the
fusion tensors. For ``\mathsf{SU}_2``, their entries are given by the Clebsch-Gordan
coefficients

### Canonical representation

TODO

### Possible manipulations

TODO

## Fermions

TODO

## Bibliography
[^kitaev]:
    Kitaev, A. (2006). Anyons in an exactly solved model and beyond.
    Annals of Physics, 321(1), 2-111.

[^1]:
    Strictly speaking the number of sectors, i.e. simple objects, in a fusion category needs to
    be finite, so that ``Rep\{\mathsf{G}\}`` is only a fusion category for a finite group ``\mathsf{G}``.
    It is clear our formalism also works for compact Lie groups with an infinite number of irreps,
    since any finite-dimensional vector space will only have a finite number of all possible
    irreps in its decomposition.
