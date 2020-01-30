# [Tensors and the `TensorMap` type](@id s_tensors)

This last page explains how to create and manipulate tensors in TensorKit.jl. As this is
probably the most important part of the manual, we will also focus more strongly on the
usage and interface, and less so on the underlying implementation. The only aspect of the
implementation that we will address is the storage of the tensor data, as this is important
to know how to create and initialize a tensor, but will in fact also shed light on how some
of the methods work.

As mentioned, all tensors in TensorKit.jl are interpreted as linear maps (morphisms) from a
domain (a `ProductSpace{S,N₂}`) to a domain (another `ProductSpace{S,N₁}`), with the same
`S<:ElementarySpace` that labels the type of spaces associated with the individual tensor
indices. The overall type for all such tensor maps is `AbstractTensorMap{S, N₁, N₂}`. Note
that we place information about the codomain before that of the domain. Indeed, we have
already encountered the constructor for the concrete parametric type `TensorMap` in the
form `TensorMap(..., codomain, domain)`. This convention is opposite to the mathematical
notation, e.g. ``\mathrm{Hom}(W,V)`` or ``f:W→V``, but originates from the fact that a
normal matrix is also denoted as having size `m × n` or is constructed in Julia as
`Array(..., (m, n))`, where the first integer `m` refers to the codomain being `m`-
dimensional, and the seond integer `n` to the domain being `n`-dimensional. This also
explains why we have consistently used the symbol ``W`` for spaces in the domain and ``V``
for spaces in the codomain. A tensor map ``t:(W₁ ⊗ … ⊗ W_{N₂}) → (V₁ ⊗ … ⊗ V_{N₁})`` will
be created in Julia as `TensorMap(..., V1 ⊗ ... ⊗ VN₁, W1 ⊗ ... ⊗ WN2)`.

Furthermore, the abstract type `AbstractTensor{S,N}` is just a synonym for
`AbstractTensorMap{S,N,0}`, i.e. for tensor maps with an empty domain, which is equivalent
to the unit of the monoidal category, or thus, the field of scalars ``𝕜``.

Currently, `AbstractTensorMap` has two subtypes. `TensorMap` provides the actual
implementation, where the data of the tensor is stored in a `DenseArray` (more specifically
a `DenseMatrix` as will be explained below). `AdjointTensorMap` is a simple wrapper type to
denote the adjoint of an existing `TensorMap` object. In the future, additional types could
be defined, to deal with sparse data, static data, diagonal data, etc...

## Storage of tensor data

Before discussion how to construct and initalize a `TensorMap{S}`, let us discuss what is
meant by 'tensor data' and how it can efficiently and compactly be stored. Let us first
discuss the case `sectortype(S) == Trivial` sector, i.e. the case of no symmetries. In that
case the data of a tensor `t = TensorMap(..., V1 ⊗ ... ⊗ VN₁, W1 ⊗ ... ⊗ WN2)` can just be
represented as a multidimensional array of size

`(dim(V1), dim(V2), …, dim(VN₁), dim(W1), …, dim(WN₂))`

which can also be reshaped into matrix of size

`(dim(V1)*dim(V2)*…*dim(VN₁), dim(W1)*dim(W2)*…*dim(WN₂))`

and is really the matrix representation of the linear map that the tensor represents. In
particular, given a second tensor `t2` whose domain matches with the codomain of `t`,
function composition amounts to multiplication of their corresponding data matrices.
Similarly, tensor factorizations such as the singular value decomposition, which we discuss
below, can act directly on this matrix representation.

!!! note
    One might wonder if it would not have been more natural to represent the tensor data as
    `(dim(V1), dim(V2), …, dim(VN₁), dim(WN₂), …, dim(W1))` given how employing the duality
    naturally reverses the tensor product, as encountered with the interface of
    [`repartition`](@ref) for [fusion trees](@ref ss_fusiontrees). However, such a
    representation, when plainly `reshape`d to a matrix, would not have the above
    properties and would thus not constitute the matrix representation of the tensor in a
    compatible basis.

Now consider the case where `sectortype(S) == G` for some `G` which has
`FusionStyle(G) == Abelian()`, i.e. the representations of an Abelian group, e.g. `G == ℤ₂`
or `G == U₁`. In this case, the tensor data is associated with sectors
`(a1, a2, …, aN₁) ∈ sectors(V1 ⊗ V2 ⊗ … ⊗ VN₁)` and `(b1, …, bN₂) ∈ sectors(W1 ⊗ … ⊗ WN₂)`
such that they fuse to a same common charge, i.e.
`(c = first(⊗(a1, …, aN₁))) == first(⊗(b1, …, bN₂))`. The data associated with this takes
the form of a multidimensional array with size
`(dim(V1, a1), …, dim(VN₁, aN₁), dim(W1, b1), …, dim(WN₂, bN₂))`, or equivalently, a
matrix of with row size `dim(V1, a1)*…*dim(VN₁, aN₁) == dim(codomain, (a1, …, aN₁))` and
column size `dim(W1, b1)*…*dim(WN₂, aN₂) == dim(domain, (b1, …, bN₂))`.

However, there are multiple combinations of `(a1, …, aN₁)` giving rise to the same `c`, and
so there is data associated with all of these, as well as all possible combinations of
`(b1, …, bN₂)`. Stacking all matrices for different `(a1,…)` and a fixed value of `(b1,…)`
underneath each other, and for fixed value of `(a1,…)` and different values of `(b1,…)` next
to each other, gives rise to a larger block matrix of all data associated with the central
sector `c`. The size of this matrix is exactly
`(blockdim(codomain, c), blockdim(domain, c))` and these matrices are exactly the diagonal
blocks whose existence is guaranteed by Schur's lemma, and which are labeled by the coupled
sector `c`. Indeed, if we would represent the tensor map `t` as a matrix without explicitly
using the symmetries, we could reorder the rows and columns to group data corresponding to
sectors that fuse to the same `c`, and the resulting block diagonal representation would
emerge. This basis transform is thus a permutation, which is a unitary operation, that will
cancel or go through trivially for linear algebra operations such as composing tensor maps
(matrix multiplication) or tensor factorizations such as a singular value decomposition. For
such linear algebra operations, we can thus directly act on these large matrices, which
correspond to the diagonal blocks that emerge after a basis transform, provided that the
partition of the tensor indices in domain and codomain of the tensor are in line with our
needs. For example, composing two tensor maps amounts to multiplying the matrices
corresponding to the same `c` (provided that its subblocks labeled by the different
combinations of sectors are ordered in the same way, which we guarantee by associating a
canonical order with sectors). Henceforth, we refer to the `blocks` of a tensor map as those
diagonal blocks, the existence of which is provided by Schur's lemma and which are labeled
by the coupled sectors `c`. We directly store these blocks as `DenseMatrix` and gather them
as values in a dictionary, together with the corresponding coupled sector `c` as key. For a
given tensor `t`, we can access a specific block as `block(t, c)`, whereas `blocks(t)`
yields an iterator over pairs `c=>block(t,c)`.

The subblocks corresponding to a particular combination of sectors then correspond to a
particular view for some range of the rows and some range of the colums, i.e.
`view(block(t, c), m₁:m₂, n₁:n₂)` where the ranges `m₁:m₂` associated with `(a1, …, aN₁)`
and `n₁:n₂` associated with `(b₁, …, bN₂)` are stored within the fields of the instance `t`
of type `TensorMap`. This `view` can then lazily be reshaped to a multidimensional array,
for which we rely on the package [Strided.jl](https://github.com/Jutho/Strided.jl). Indeed,
the data in this `view` is not contiguous, because the stride between the different columns
is larger than the length of the columns. Nonetheless, this does not pose a problem and even
as multidimensional array there is still a definite stride associated with each dimension.

When `FusionStyle(G) isa NonAbelian`, things become slightly more complicated. Not only do
`(a1, …, aN₁)` give rise to different coupled sectors `c`, there can be multiply ways in
which they fuse to `c`. These different possibilities are enumerated by the iterator
`fusiontrees((a1, …, aN₁), c)` and `fusiontrees((b1, …, bN₂), c)`, and with each of those,
there is tensor data that takes the form of a multidimensional array, or, after reshaping,
a matrix of size `(dim(codomain, (a1, …, aN₁)), dim(domain, (b1, …, bN₂))))`. Again, we can
stack all such matrices with the same value of `f₁ ∈ fusiontrees((a1, …, aN₁), c)`
horizontally (as they all have the same number of rows), and with the same value of
`f₂ ∈ fusiontrees((b1, …, bN₂), c)` vertically (as they have the same number of columns).
What emerges is a large matrix of size `(blockdim(codomain, c), blockdim(domain, c))`
containing all the tensor data associated with the coupled sector `c`, where
`blockdim(P, c) = sum(dim(P, s)*length(fusiontrees(s, c)) for s in sectors(P))` for some
instance `P` of `ProductSpace`.

The tensor implementation does not distinguish between abelian or non-abelian sectors and
still stores these matrices as a `DenseMatrix`, accessible via `block(t, c)`. At first
sight, it might now be less clear what the relevance of this block is in relation to the
full matrix representation of the tensor map, where the symmetry is not exploited. The
essential interpretation is still the same. Schur's lemma now tells that there is a unitary
basis transform which makes this matrix representation block diagonal, more specifically, of
the form ``⨁_{c} B_c ⊗ 𝟙_{c}``, where ``B_c`` denotes `block(t,c)` and ``𝟙_{c}`` is an
identity matrix of size `(dim(c), dim(c))`. The reason for this extra identity is that the
group representation is recoupled to act as ``⨁_{c} 𝟙 ⊗ u_c(g)`` for all
``g ∈ \mathsf{G}``, with ``u_c(g)`` the matrix representation of group element ``g``
according to the irrep ``c``. In the abelian case, `dim(c) == 1`, i.e. all irreducible
representations are one-dimensional and Schur's lemma only dictates that all off-diagonal
blocks are zero. However, in this case the basis transform to the block diagonal
representation is not simply a permutation matrix, but a more general unitary matrix
composed of the different fusion trees. Indeed, let us denote the fusion trees
`f₁ ∈ fusiontrees((a1, …, aN₁), c)` as ``X^{a_1, …, a_{N₁}}_{c,α}`` where
``α = (e_1, …, e_{N_1-2}; μ₁, …, μ_{N_1-1})`` is a collective label for the internal sectors `e` and the vertex degeneracy labels `μ` of a generic fusion tree, as discussed in the [corresponding section](@ref ss_fusiontrees). The
tensor is then represented as

![tensor storage](img/tensor-storage.svg)

In this diagram, we have indicated how the tensor map can be rewritten in terms of a block
diagonal matrix with a unitary matrix on its left and another unitary matrix (if domain and
codomain are different) on its right. So the left and right matrices should actually have
been drawn as squares. They represent the unitary basis transform. In this picture, red and
white regions are zero. The center matrix is most easy to interpret. It is the block
diagonal matrix ``⨁_{c} B_c ⊗ 𝟙_{c}`` with diagonal blocks labeled by the coupled charge
`c`, in this case it takes two values. Every single small square corresponds to a single
element of ``B_c``, tensored with the identity of size ``d_c``. Instead of ``B_c``, a more
accurate labelling is ``t^c_{(a_1 … a_{N₁})α, (b_1 … b_{N₂}β)}`` where ``α`` labels
different fusion trees from ``(a_1 … a_{N₁})`` to ``c``. The dashed lines indicate regions
corresponding to different fusion trees, either because of different sectors
``(a_1 … a_{N₁})`` or different labels ``α`` within the same sector.

To understand this better, we need to understand the basis transform, e.g. on the left
(codomain) side. In more detail, it is given by

![tensor unitary](img/tensor-unitary.svg)


Indeed, remembering that ``V_i = ⨁_{a_i} R_{a_i} ⊗ ℂ^{n_{a_i}}`` with ``R_{a_i}`` the
representation space on which irrep ``a_i`` acts (with dimension ``\mathrm{dim}(a_i)``), we
find
``V_1 ⊗ … ⊗ V_{N_1} = ⨁_{a_1, …, a_{N₁}} (R_{a_1} ⊗ … ⊗ R_{a_{N_1}}) ⊗ ℂ^{n_{a_1} × … n_{a_{N_1}}}``.
In the diagram above, the wiggly lines correspond to the direct sum over the different
sectors ``(a_1, …, a_{N₁})``, there depicted taking three possible values ``(a…)``,
``(a…)′`` and ``(a…)′′``. The tensor product
``(R_{a_1} ⊗ … ⊗ R_{a_{N_1}}) ⊗ ℂ^{n_{a_1} × … n_{a_{N_1}}}`` is depicted as
``(R_{a_1} ⊗ … ⊗ R_{a_{N_1}})^{⊕ n_{a_1} × … n_{a_{N_1}}}``, i.e. as a direct sum of the
spaces ``R_{(a…)} = (R_{a_1} ⊗ … ⊗ R_{a_{N_1}})`` according to the dotted horizontal lines,
which repeat ``n_{(a…)} = n_{a_1} × … n_{a_{N_1}}`` times. In this particular example,
``n_{(a…)}=2``, ``n_{(a…)'}=3`` and ``n_{(a…)''}=5``. The thick vertical line represents the
separation between the two different coupled sectors, denoted as ``c`` and ``c'``. Dashed
vertical lines represent different ways of reaching the coupled sector, corresponding to
different `α`. In this example, the first sector ``(a…)`` has one fusion tree to ``c``,
labeled by ``c,α``, and two fusion trees to ``c'``, labeled by ``c',α`` and ``c',α'``. The
second sector has only a fusion tree to ``c``, labeled by ``c,α'``. The third sector only
has a fusion tree to ``c'``, labeld by ``c', α''``. Finally then, because the fusion trees
do not act on the spaces ``ℂ^{n_{a_1} × … n_{a_{N_1}}}``, the dotted lines which represent
the different ``n_{(a…)} = n_{a_1} × … n_{a_{N_1}}`` dimensions are also drawn vertically.
In particular, for a given sector ``(a…)`` and a specific fusion tree
``X^{(a…)}_{c,α}: R_{(a…)}→R_c``, the action is ``X^{(a…)}_{c,α} ⊗ 𝟙_{n_{(a…)}}``, which
corresponds to the diagonal blocks in this drawing where the same matrix ``X^{(a…)}_{c,α}``
(the fusion tree) is repeated along the diagonal. A similar interpretation can be given to
the basis transform on the right, by taking its adjoint. In this particular example, it has
two different combinations of sectors ``(b…)`` and ``(b…)'``, where both have a single
fusion tree to `c` as well as to `c'`, and ``n_{(b…)}=2``, ``n_{(b…)'}=3``.


## Defining and constructing tensor maps


The most convenient set of constructors are those that construct tensors or tensor maps
with random or uninitialized data. They take the form

`TensorMap(f, codomain, domain)`

`TensorMap(f, eltype::Type{<:Number}, codomain, domain)`

`TensorMap(undef, codomain, domain)`

`TensorMap(undef, eltype::Type{<:Number}, codomain, domain)`

Here, in the first form, `f` can be any function or object that is called with an argument
of type `Dims{2} = Tuple{Int,Int}` and is such that `f((m,n))` creates a `DenseMatrix`
instance with `size(f(m,n)) == (m,n)`. In the second form, `f` is called as
`f(eltype,(m,n))`. Possibilities for `f` are `randn` and `rand` from Julia Base.
TensorKit.jl provides `randnormal` and `randuniform` as an synonym for `randn` and `rand`,
as well as the new function  `randisometry`, alternatively called `randhaar`, that creates
a random isometric `m × n` matrix `w` satisfying `w'*w ≈ I` distributed according to the
Haar measure (this requires `m>= n`). The third and fourth calling syntax use the
`UndefInitializer` from Julia Base and generates a `TensorMap` with unitialized data, which
could thus contain `NaN`s.


## Storage


TODO

## Linear algebra operations

TODO

## Index manipulations

TODO

## Tensor contractions and tensor networks

TODO
