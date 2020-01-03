# [Introduction](@id s_intro)

Before providing a typical "user guide" and discussing the implementation of TensorKit.jl
on the next pages, let us discuss some of the rationale behind this package.

## [What is a tensor?](@id ss_tensor)

At the very start we should ponder about the most suitable and sufficiently general
definition of a tensor. A good starting point is the following:

*   A tensor ``t`` is an element from the
    [tensor product](https://en.wikipedia.org/wiki/Tensor_product) of ``N`` vector spaces
    ``V_1 , V_2, …, V_N``, where ``N`` is referred to as the *rank* or *order* of the
    tensor, i.e.

    ``t ∈ V_1 ⊗ V_2 ⊗ … ⊗ V_N.``

If you think of a tensor as an object with indices, a rank ``N`` tensor has ``N`` indices
where every index is associated with the corresponding vector space in that it labels a
particular basis in that space. We will return to index notation at the very end of this
manual.

As the tensor product of vector spaces is itself a vector space, this implies that a tensor
behaves as a vector, i.e. tensors from the same tensor product space can be added and
multiplied by scalars. The tensor product is only defined for vector spaces over the same
field of scalars, e.g. there is no meaning in ``ℝ^5 ⊗ ℂ^3``. When all the vector spaces in
the tensor product have an inner product, this also implies an inner product for the tensor
product space. It is hence clear that the different vector spaces in the tensor product
should have some form of homogeneity in their structure, yet they do not need to be all
equal and can e.g. have different dimensions. It goes without saying that defining the
vector spaces and their properties will be an important part of the definition of a tensor.
As a consequence, this also constitutes a significant part of the implementation, and is
discussed in the section on [Vector spaces](@ref).

Aside from the interpretation of a tensor as a vector, we also want to interpret it as a
matrix (or more correctly, a linear map) in order to decompose tensors using linear algebra
factorisations (e.g. eigenvalue or singular value decomposition). Henceforth, we use the
term "tensor map" as follows:

*   A tensor map ``t`` is a linear map from a source or *domain*
    ``W_1 ⊗ W_2 ⊗ … ⊗ W_{N_2}`` to a target or *codomain* ``V_1 ⊗ V_2 ⊗ … ⊗ V_{N_1}``, i.e.

    ``t:W_1 ⊗ W_2 ⊗ … ⊗ W_{N_2} → V_1 ⊗ V_2 ⊗ … ⊗ V_{N_1}.``

A *tensor* of rank ``N`` is then just a special case of a tensor map with ``N_1 = N`` and
``N_2 = 0``. A contraction between two tensors is just a composition of linear maps (i.e.
matrix multiplication), where the contracted indices correspond to the domain of the first
tensor and the codomain of the second tensor.

In order to allow for arbitrary tensor contractions or decompositions, we need to be able to
reorganise which vector spaces appear in the domain and the codomain of the tensor map, and
in which order. This amounts to defining canonical isomorphisms between the different ways
to order and partition the tensor indices (i.e. the vector spaces). For example, a linear
map ``W → V`` is often denoted as a rank 2 tensor in ``V ⊗ W^*``, where ``W^*`` corresponds
to the dual space of ``W``. This simple example introduces two new concepts.

1.  Typical vector spaces can appear in the domain and codomain in different related forms,
    e.g. as normal space or dual space. In fact, the most generic case is that every vector
    space ``V`` has associated with it
    a [dual space](https://en.wikipedia.org/wiki/Dual_space) ``V^*``,
    a [conjugate space](https://en.wikipedia.org/wiki/Complex_conjugate_vector_space)
    ``\overline{V}`` and a conjugate dual space ``\overline{V}^*``. The four different
    vector spaces ``V``, ``V^*``, ``\overline{V}`` and ``\overline{V}^*`` correspond to the
    representation spaces of respectively the fundamental, dual or contragredient, complex
    conjugate and dual complex conjugate representation of the general linear group
    ``\mathsf{GL}(V)`` [^tung]. In index notation these spaces are denoted with
    respectively contravariant (upper), covariant (lower), dotted contravariant
    and dotted covariant indices.

    For real vector spaces, the conjugate (dual) space is identical to the normal (dual)
    space and we only have upper and lower indices, i.e. this is the setting of e.g.
    general relativity. For (complex) vector spaces with a sesquilinear inner product
    ``\overline{V} ⊗ V → ℂ``, the inner product allows to define an isomorphism from the
    conjugate space to the dual space (known as
    [Riesz representation theorem](https://en.wikipedia.org/wiki/Riesz_representation_theorem)
    in the more general context of Hilbert spaces).

    In particular, in spaces with a Euclidean inner product (the setting of e.g. quantum
    mechanics), the conjugate and dual space are naturally isomorphic (because the dual and
    conjugate representation of the unitary group are the same). Again we only need upper
    and lower indices (or kets and bras).

    Finally, in ``ℝ^d`` with a Euclidean inner product, these four different spaces are all
    equivalent and we only need one type of index. The space is completely characterized by
    its dimension ``d``. This is the setting of much of classical mechanics and we refer to
    such tensors as cartesian tensors and the corresponding space as cartesian space. These
    are the tensors that can equally well be represented as multidimensional arrays (i.e.
    using some `AbstractArray{<:Real,N}` in Julia) without loss of structure.

    The implementation of all of this is discussed in [Vector spaces](@ref).

2.  In the generic case, the identification between maps ``W → V`` and tensors in
    ``V ⊗ W^*`` is not an equivalence but an isomorphism, which needs to be defined.
    Similarly, there is an isomorphism between between ``V ⊗ W`` and ``W ⊗ V`` that can be
    non-trivial (e.g. in the case of fermions / super vector spaces). The correct formalism
    here is provided by theory of monoidal categories. Nonetheless, we try to hide these
    canonical isomorphisms from the user wherever possible.

This brings us to our final (yet formal) definition

*   A tensor (map) is a homorphism between two objects from the category ``\mathbf{Vect}``
    (or some subcategory thereof). In practice, this will be ``\mathbf{FinVect}``, the
    category of finite dimensional vector spaces. More generally even, our concept of a
    tensor makes sense, in principle, for any linear (a.k.a. ``\mathbf{Vect}``-enriched)
    monoidal category. We refer to the section
    "[Monoidal categories and their properties](@ref ss_categories)".

## [Symmetries and block sparsity](@id ss_symmetries)

Physical problems often have some symmetry, i.e. the setup is invariant under the action of
a group ``\mathsf{G}`` which acts on the vector spaces ``V`` in the problem according to a
certain representation. Having quantum mechanics in mind, TensorKit.jl is so far restricted
to unitary representations. A general representation space ``V`` can be specified as the
number of times every irreducible representation (irrep) ``a`` of ``\mathsf{G}`` appears,
i.e.

``V = \bigoplus_{a} ℂ^{n_a} ⊗ R_a``

with ``R_a`` the space associated with irrep ``a`` of ``\mathsf{G}``, which itself has
dimension ``d_a`` (often called the quantum dimension), and ``n_a`` the number of times
this irrep appears in ``V``. If the unitary irrep ``a`` for ``g ∈ \mathsf{G}`` is given by
``u_a(g)``, then the group action of ``\mathsf{G}`` on ``V`` is given by the unitary
representation

``u(g) = \bigoplus_{a}  𝟙_{n_a} ⊗ u_a(g)``

with ``𝟙_{n_a}`` the ``n_a × n_a`` identity matrix. The total dimension of ``V`` is given
by ``∑_a n_a d_a``.

The reason for implementing symmetries is to exploit the computation and memory gains
resulting from restricting to tensor maps ``t:W_1 ⊗ W_2 ⊗ … ⊗ W_{N_2} → V_1 ⊗ V_2 ⊗ … ⊗
V_{N_1}`` that are invariant under the symmetry, i.e. that act as
[intertwiners](https://en.wikipedia.org/wiki/Equivariant_map#Representation_theory)
between the symmetry action on the domain and the codomain. Indeed, such tensors should be
block diagonal because of [Schur's lemma](https://en.wikipedia.org/wiki/Schur%27s_lemma),
but only after we couple the individual irreps in the spaces ``W_i`` to a joint irrep,
which is then again split into the individual irreps of the spaces ``V_i``. The basis
change from the tensor product of irreps in the (co)domain to the joint irrep is implemented
by a sequence of Clebsch–Gordan coefficients, also known as a fusion (or splitting) tree.
We implement the necessary machinery to manipulate these fusion trees under index
permutations and repartitions for arbitrary groups ``\mathsf{G}``. In particular, this fits
with the formalism of monoidal categories, and more specifically fusion categories,
discussed below and only requires the *topological* data of the group, i.e. the fusion
rules of the irreps, their quantum dimensions and the F-symbol (6j-symbol or more precisely
Racah's W-symbol in the case of ``\mathsf{SU}_2``). In particular, we don't actually need
the Clebsch–Gordan coefficients themselves (but they can be useful for checking purposes).

Further details are provided in
[Sectors, representation spaces and fusion trees](@ref s_sectorsrepfusion).

## [Monoidal categories and their properties (optional)](@id ss_categories)

The purpose of this final introductory section (which can safely be skipped), is to explain
how certain concepts and terminology from the theory of monoidal categories apply in the
context of tensors.  In the end, the goal of identifying tensor manipulations in
TensorKit.jl with concepts from category theory is to put the diagrammatic formulation of
tensor networks in the most general context on a firmer footing. The following definitions
are mostly based on [^selinger], [^kitaev], [^kassel], [^turaev] and [``n``Lab](https://ncatlab.org/),
to which  we refer for further information. Furthermore, we recommend the nice introduction
of [Beer et al.](^beer)

To start, a category ``C`` consists of
*   a class ``|C|`` of objects ``V``, ``W``, …
*   for each pair of objects ``V`` and ``W``, a set ``hom(W,V)`` of morphisms ``f:W→V``;
    for a given map ``f``, ``W`` is called the *domain* or *source*, and ``V`` the
    *codomain* or *target*.
*   composition of morphisms ``f:W→V`` and ``g:X→W`` into ``(f ∘ g):X→V`` that is
    associative, such that for ``h:Y→X`` we have ``f ∘ (g ∘ h) = (f ∘ g) ∘ h``
*   for each object ``V``, an identity morphism ``\mathrm{id}_V:V→V`` such that
    ``f ∘ \mathrm{id}_W = f = \mathrm{id}_V ∘ f``.

In our case, i.e. the category ``\mathbf{Vect}`` (or some subcategory thereof), the objects
are vector spaces, and the morphisms are linear maps between these vector spaces with
"matrix multiplication" as composition. We refer to these morphisms as tensor maps exactly
because there is a binary operation ``⊗``, the tensor product, that allows to combine
objects into new objects. This makes ``\mathbf{Vect}`` into a **tensor category**, a.k.a
a *monoidal category*, which has
*   a binary operation on objects ``⊗: |C| × |C| → |C|``
*   a binary operation on morphisms, also denoted as ``⊗``, such that
    ``⊗: hom(W_1,V_1) × hom(W_2,V_2) → hom(W_1 ⊗ W_2, V_1 ⊗ V_2)``
*   an identity or unit object ``I``
*   three families of natural isomorphisms:
    *   ``∀ V ∈ |C|``, a left unitor ``λ_V: I ⊗ V → V``
    *   ``∀ V ∈ |C|``, a right unitor ``ρ_V: V ⊗ I → V``
    *   ``∀ V_1, V_2, V_3 ∈ |C|``, an associator
        ``α_{V_1,V_2,V_3}:(V_1 ⊗ V_2) ⊗ V_3 → V_1 ⊗ (V_2 ⊗ V_3)``
    that satisfy certain consistency conditions (coherence axioms), which are known as the
    *triangle equation* and *pentagon equation*.
In abstract terms, ``⊗`` is a (bi)functor from the product category ``C × C`` to ``C``.

For the category ``\mathbf{Vect}``, the identity object ``I`` is just the scalar field
``𝕜`` over which the vector spaces are defined, and which can be identified with a one-
dimensional vector space. Every monoidal category is equivalent to a strict tensor
category, where the left and right unitor and associator act as the identity and their
domain and codomain are truly identical. Nonetheless, for tensor maps, we do actually
discriminate between ``V``, ``I ⊗ V`` and ``V ⊗ I`` because this amounts to adding or
removing an extra factor `I` to the tensor product structure of the (co)domain, i.e. the
left and right unitor are analogous to removing extra dimensions of size 1 from an array,
and an actual operation is required to do so (this has in fact led to some controversy in
several programming languages that provide native support for multidimensional arrays). For
what concerns the associator, the distinction between ``(V_1 ⊗ V_2) ⊗ V_3`` and
``V_1 ⊗ (V_2 ⊗ V_3)`` is typically absent for simple tensors or multidimensional arrays.
However, this grouping can be taken to indicate how to build the fusion tree for coupling
irreps to a joint irrep in the case of symmetric tensors. As such, going from one to the
other requires a recoupling (F-move) which has a non-trivial action on the reduced blocks.
We return to this in [the section on fusion trees](@ref s_sectorsrepfusion). However, we
can already note that we will always represent tensor products using a canonical order
``(…((V_1 ⊗ V_2) ⊗ V_3) … ⊗ V_N)``. A similar approach can be followed to map any tensor
category into a strict tensor category (see Section XI.5 of [^kassel]).

With these definitions, we have the minimal requirements for defining tensor maps. In
principle, we could use a more general definition and define tensor maps as morphism of any
tensor category where the hom-sets are themselves vector spaces, such that we can add
morphisms and multiply them with scalars. Furthermore, the composition of morphisms and the
tensor product of morphisms are bilinear operations. Such categories are called linear or
``\mathbf{Vect}``-enriched.

In order to make tensor (maps) useful and to define operations with them, we can now
introduce additional structure or quantifiers to the tensor category for which they are
the morphisms.

### [Braiding](@id sss_braiding)

To reorder tensor indices, or, equivalently, to reorder objects in the tensor product
``V_1 ⊗ V_2 ⁠⊗ … V_N``, we need at the very least a **braided tensor category** which has,
``∀ V, W ∈ |C|``, a braiding ``σ_{V,W}: V⊗W → W⊗V``. A valid braiding needs to satisfy
consistency condition with the associator ``α`` known as the *hexagon equation*.

However, for general braidings, there is no unique choice to identify a tensor in ``V⊗W``
and ``W⊗V``, as any of the maps ``σ_{V,W}``, ``σ_{W,V}^{-1}``,
``σ_{V,W} ∘ σ_{W,V} ∘ σ_{V,W}``, …  mapping from ``V⊗W`` to ``W⊗V`` are all different. In
order for there to be a unique map from ``V_1 ⊗ V_2 ⁠⊗ … V_N`` to any permutation of the
objects in this tensor product, the braiding needs to be *symmetric*, i.e.
``σ_{V,W} = σ_{W,V}^{-1}`` or, equivalently ``σ_{W,V} ∘ σ_{V,W} = \mathrm{id}_{V⊗W}``. The
resulting category is then referred to as a **symmetric tensor category**. In a graphical
representation, it means that there is no distinction between over- and under-
crossings and, as such, lines can just cross.

For a simple cartesian tensor, permuting the tensor indices is equivalent to applying
Julia's function `permutedims` on the underlying data. Less trivial braiding
implementations arise in the context of tensors with symmetries (where the fusion tree
needs to be reordered) or in the case of fermions (described using so-called super vector
spaces where the braiding is given by the Koszul sign rule).

### [Duals and pivotal structure](@id sss_dual)

For tensor maps, the braiding structure only allows to reorder the objects within the domain
or within the codomain separately. An **autonomous** or **rigid** monoidal category is one
where objects have duals, defined via an exact pairing, i.e. two families of canonical maps,
the unit ``η_V: I → V ⊗ V^*`` and the co-unit ``ϵ_V: V^* ⊗ V → I`` that satisfy the "snake
rules":

``ρ_V ∘ (\mathrm{id}_V ⊗ ϵ_V) ∘ (η_V ⊗ \mathrm{id}_V) ∘ λ_V^{-1} = \mathrm{id}_V``

``λ_{V^*}^{-1} ∘ (ϵ_V ⊗ \mathrm{id}_{V^*}) ∘ (\mathrm{id}_{V^*} ⊗ η_V) ∘ ρ_{V^*}^{-1} = \mathrm{id}_{V^*}``

Given a morphism ``t:W→V``, we can now identify it with ``(t ⊗ \mathrm{id}_{W^*}) ∘ η_W``
to obtain a morphism ``I→V⊗W^*``. For the category ``\mathbf{Vect}``, this is the
identification between linear maps ``W→V`` and tensors in ``V⊗W^*``. In particular, for
complex vector spaces, using a bra-ket notation and a generic basis ``{|n⟩}`` for ``V`` and
dual basis ``{⟨m|}`` for ``V^*`` (such that ``⟨m|n⟩ = δ_{m,n}``), the unit is
``η_V:ℂ → V ⊗ V^*:α → α ∑_n |n⟩ ⊗ ⟨n|`` and the co-unit is
``⁠ϵ_V:V^* ⊗ V → ℂ: ⟨m| ⊗ |n⟩ → δ_{m,n}``. Note that this does not require an inner
product, i.e. no mapping from ``|n⟩`` to ``⟨n|`` was defined.

For a general tensor map ``t:W_1 ⊗ W_2 ⊗ … ⊗ W_{N_2} → V_1 ⊗ V_2 ⊗ … ⊗ V_{N_1}``, by
successively applying ``η_{W_{N_2}}``, ``η_{W_{N_2-1}}``, …, ``η_{W_{1}}`` (and the left or
right unitor) but no braiding, we obtain a tensor in
``V_1 ⊗ V_2 ⊗ … ⊗ V_{N_1} ⊗ W_{N_2}^* ⊗ … ⊗ W_{1}^*``.
It does makes sense to define or identify
``(W_1 ⊗ W_2 ⊗ … ⊗ W_{N_2})^* = W_{N_2}^* ⊗ … ⊗ W_{1}^*``. Indeed, it can be shown that an
exact pairing between ``V ⊗ W`` and ``W^* ⊗ V^*`` can be constructed out of the unit and
counit of ``V`` and ``W``.

These exact pairings are known as the right unit and co-unit, and ``V^*`` is the
right dual of ``V``. Likewise, then, ``V`` is a left dual of ``V^*``, and we can also
define a left dual ``^*V`` of ``V`` and associated pairings, the left unit
``η'_V: I → {^*V} ⊗ V`` and the left co-unit ``ϵ'_V: V ⊗ {^*V} → I``. An autonomous category
``\mathbf{C}`` is one where every object ``V`` has both a left and right dual. If we have a
natural isomorphism between both, typically expressed via a pivotal structure
``δ_V : {^*V} → V^*`` which satisfies ``δ_{V ⊗ W} = δ_W ⊗ δ_V``, we do not have to
distinguish between both. The resulting category is known as a *pivotal category*. Indeed,
in TensorKit.jl we assume to be working with **pivotal categories** and simply refer to
`dual(V)` for the dual of a vector space.

For a pivotal category, there is a well defined notion of a transpose ``f^*:V^* → W^*``
(also called adjoint mate) of a morphism ``f:W→V``, namely as as

``f^* = λ_{W^*} ∘ (ϵ_V ⊗ \mathrm{id}_{W^*}) ∘ (\mathrm{id}_{V^*} ⊗ f ⊗ \mathrm{id}_{W^*}) ∘ (\mathrm{id}_{V^*} ⊗ η_{W}) ∘ ρ_{V^*}^{-1}``

``{^*f} = ρ_{W^*} ∘ (\mathrm{id}_{W^*} ⊗ ϵ_{V^*}) ∘ (\mathrm{id}_{V^*} ⊗ f ⊗ \mathrm{id}_{W^*}) ∘ (η_{W^*} ⊗ \mathrm{id}_{V^*}) ∘ λ_{V^*}^{-1}``

and both definitions coincide (which is not the case if the category is not pivotal). In a
graphical representation, this means that boxes (representing tensor maps or morphisms more
generally) can be rotated. The transpose corresponds to a 180˚ rotation (either way).

Furthermore, in a pivotal category, we can define a map from endomorphisms of an object
``V``, i.e. a morphism ``f:V→V`` to endomorphisms of the identity object ``I``, i.e.
scalars, known as the trace of ``f``. In fact, we can define both a left trace as

``tr(f) = ϵ'_V ∘ (f ⊗ \mathrm{id}_{V^*}) ∘ η_V``

and a right trace as

``tr'(f) = ϵ_V ∘ (\mathrm{id}_{V^*} ⊗ f) ∘ η'_V``

In a **spherical** category, both definitions coincide for all ``V`` and we simply refer to
the trace of an endomorphism. The particular value ``d_V = tr(\mathrm{id}_V)`` is known as
the (quantum) dimension of the object ``V``, referred to as `dim(V)` in TensorKit.jl.

### [Twists and ribbons](@id sss_twists)

The braiding of a space and a dual space also follows naturally, it is given by
``σ_{V^*,W} = λ_{W ⊗ V^*} ∘ (ϵ_V ⊗ \mathrm{id}_{W ⊗ V^*}) ∘ (\mathrm{id}_{V^*} ⊗ σ_{V,W}^{-1} ⊗ \mathrm{id}_{V^*}) ∘ (\mathrm{id}_{V^*⊗ W} ⊗ η_V) ∘ ρ_{V^* ⊗ W}^{-1}``

Furthermore, in a braided pivotal category, we can define a family of natural isomorphisms
``θ_V:V→V`` (i.e. for ``f:W→V``, ``θ_V ∘ f = f ∘ θ_W``) as

``θ_V = ρ_V ∘ (\mathrm{id}_V ⊗ ϵ'_V) ∘ (σ_{V,V} ⊗ \mathrm{id}_{V^*}) ∘ (\mathrm{id}_V ⊗ η_V) ∘ ρ_V^{-1}``

which satisfy

``θ_{V⊗W} = σ_{W,V} ∘ (θ_W ⊗ θ_V) ∘ σ_{V,W} = (θ_V ⊗ θ_W) ∘ σ_{W,V} ∘ σ_{V,W}``

A family of natural isomorphisms satisfying this relation is called a **twist**, and the resulting category is called a **balanced** monoidal category.  Here, we defined the twist
via the exact pairings, and ultimately via the pivotal structure, i.e. the
``\mathrm{id}_{V^*}`` in the definition of ``θ_V`` should have been a ``δ_V^{-1}``. The
interaction between the twist and the braiding is consistent with the graphical rules of a
ribbon. However, for the graphical rules of ribbons to also be compatible with the exact
pairing, we furthermore need the condition ``θ_{V^*} = θ_V^*`` (i.e. the transpose), in
which case the category is said to be **tortile** or also a **ribbon category**.

Alternatively, we can start with a balanced and autonomous category and use the twist to
define the pivotal structure. In particular, we can express the left unit and counit in
terms of the right unit and counit, the braiding and the twist, as

``η'_V = (\mathrm{id}_{V^*} ⊗ θ_V) ∘ σ_{V,V^*} ∘ η_V``

``ϵ'_V = ϵ_V ∘ σ_{V,V^*} ∘ (θ_V ⊗ \mathrm{id}_{V^*})``

The trace of an endomorphism ``f:V→V`` is then given by

``tr(f) = ϵ_V ∘ σ_{V,V^*} ∘ (( θ_V ∘ f) ⊗ \mathrm{id}_{V^*}) ∘ η_V``

and it can be verified using the naturality of the braiding that the resulting category is
spherical, i.e. that this is equal to

``tr(f) = ϵ_V ∘ (\mathrm{id}_{V^*} ⊗ (f ∘ θ_V)) ∘ σ_{V,V^*} ∘ η_V``

Note finally, that a ribbon category where the braiding is symmetric, is known as a
**compact closed category**. For a symmetric braiding, the trivial twist
``θ_V = \mathrm{id}_V`` is always a valid choice, but it might not be the choice that one
necessarily want to use. This brings us to the final paragraph.

### [Adjoints](@id sss_adjoints)

A final aspect of categories as they are relevant to physics, and in particular quantum
physics, is the notion of an adjoint or dagger. A **dagger category** ``C`` is a category
together with an involutive functor ``†:C→C`` that acts as the identity on objects, whereas
on morphisms ``f:W→V`` it defines a morphism ``f^†:V→W`` such that
* ``\mathrm{id}_V^† = \mathrm{id}_V``
* ``(g ∘ f)^† = f^† ∘ g^†``
* ``(f^†)^† = f``

In a dagger category, a morphism ``f:W→V`` is said to be unitary if it is an isomorphism
and ``f^{-1} = f^†``. Furthermore, an endomorphism ``f:V→V`` is hermitian or self-adjoint if
``f^† = f``.

A dagger monoidal category is one in which the associator and left and right unitor are
unitary morphisms. Similarly, a dagger braided category also has a unitary braiding, and a
dagger balanced category in addition has a unitary twist.

There is more to be said about the interplay between the dagger and duals. Given a right
unit ``η_V: I → V ⊗ V^*`` and co-unit ``ϵ_V: V^* ⊗ V → I``, we can define a left unit and
co-unit ``η'_V = (ϵ_V)^†`` and ``ϵ'_V = (η_V)^†``, and from this, a unitary pivotal
structure. Hence, right autonomous dagger categories are automatically pivotal dagger
categories.

The twist defined via the pivotal structure now becomes

``θ_V = ρ_V ∘ (\mathrm{id}_V ⊗ (η_V)^†) ∘ (σ_{V,V} ⊗ \mathrm{id}_{V^*}) ∘ (\mathrm{id}_V ⊗ η_V) ∘ ρ_V^{-1}``

and is itself unitary. Even for a symmetric category, the twist defined as such must not be
the identity. We will return to this in the discussion of fermions.

## Bibliography

[^tung]:        Tung, W. K. (1985). Group theory in physics: an introduction to symmetry
                principles, group representations, and special functions in classical and
                quantum physics.
                World Scientific Publishing Company.

[^selinger]:    Selinger, P. (2010). A survey of graphical languages for monoidal
                categories.
                In New structures for physics (pp. 289-355). Springer, Berlin, Heidelberg.

[^kitaev]:      Kitaev, A. (2006). Anyons in an exactly solved model and beyond.
                Annals of Physics, 321(1), 2-111.

[^kassel]:      Kassel, C. (2012). Quantum groups (Vol. 155).
                Springer Science & Business Media.

[^turaev]:      Turaev, V. G., & Virelizier, A. (2017). Monoidal categories and topological
                field theory (Vol. 322).
                Birkhäuser.

[^beer]:        From categories to anyons: a travelogue
                Kerstin Beer, Dmytro Bondarenko, Alexander Hahn, Maria Kalabakov, Nicole
                Knust, Laura Niermann, Tobias J. Osborne, Christin Schridde, Stefan
                Seckmeyer, Deniz E. Stiegemann, and Ramona Wolf
                [https://arxiv.org/pdf/1811.06670.pdf](https://arxiv.org/pdf/1811.06670.pdf)
