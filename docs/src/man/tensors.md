# [Tensors and the `TensorMap` type](@id s_tensors)

```@setup tensors
using TensorKit
using LinearAlgebra
```

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
instance `P` of `ProductSpace`. The tensor implementation does not distinguish between
abelian or non-abelian sectors and still stores these matrices as a `DenseMatrix`,
accessible via `block(t, c)`.

At first sight, it might now be less clear what the relevance of this block is in relation
to the full matrix representation of the tensor map, where the symmetry is not exploited.
The essential interpretation is still the same. Schur's lemma now tells that there is a
unitary basis transform which makes this matrix representation block diagonal, more
specifically, of the form ``⨁_{c} B_c ⊗ 𝟙_{c}``, where ``B_c`` denotes `block(t,c)` and
``𝟙_{c}`` is an identity matrix of size `(dim(c), dim(c))`. The reason for this extra
identity is that the group representation is recoupled to act as ``⨁_{c} 𝟙 ⊗ u_c(g)`` for
all ``g ∈ \mathsf{G}``, with ``u_c(g)`` the matrix representation of group element ``g``
according to the irrep ``c``. In the abelian case, `dim(c) == 1`, i.e. all irreducible
representations are one-dimensional and Schur's lemma only dictates that all off-diagonal
blocks are zero. However, in this case the basis transform to the block diagonal
representation is not simply a permutation matrix, but a more general unitary matrix
composed of the different fusion trees. Indeed, let us denote the fusion trees `f₁ ∈
fusiontrees((a1, …, aN₁), c)` as ``X^{a_1, …, a_{N₁}}_{c,α}`` where
``α = (e_1, …, e_{N_1-2}; μ₁, …, μ_{N_1-1})`` is a collective label for the internal sectors
`e` and the vertex degeneracy labels `μ` of a generic fusion tree, as discussed in the
[corresponding section](@ref ss_fusiontrees). The tensor is then represented as

![tensor storage](img/tensor-storage.svg)

In this diagram, we have indicated how the tensor map can be rewritten in terms of a block
diagonal matrix with a unitary matrix on its left and another unitary matrix (if domain and
codomain are different) on its right. So the left and right matrices should actually have
been drawn as squares. They represent the unitary basis transform. In this picture, red and
white regions are zero. The center matrix is most easy to interpret. It is the block
diagonal matrix ``⨁_{c} B_c ⊗ 𝟙_{c}`` with diagonal blocks labeled by the coupled charge
`c`, in this case it takes two values. Every single small square in between the dotted or
dashed lines has size ``d_c × d_c`` and corresponds to a single element of ``B_c``,
tensored with the identity ``\mathrm{id}_c``. Instead of ``B_c``, a more accurate labelling
is ``t^c_{(a_1 … a_{N₁})α, (b_1 … b_{N₂})β}`` where ``α`` labels different fusion trees from
``(a_1 … a_{N₁})`` to ``c``. The dashed horizontal lines indicate regions corresponding to
different fusion (actually splitting) trees, either because of different sectors
``(a_1 … a_{N₁})`` or different labels ``α`` within the same sector. Similarly, the dashed
vertical lines define the border between regions of different fusion trees from the domain
to `c`, either because of different sectors ``(b_1 … b_{N₂})`` or a different label ``β``.

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
corresponds to the diagonal green blocks in this drawing where the same matrix
``X^{(a…)}_{c,α}`` (the fusion tree) is repeated along the diagonal. Note that the fusion
tree is not a vector or single column, but a matrix with number of rows equal to
``\mathrm{dim}(R_{(a\ldots)}) = d_{a_1} d_{a_2} … d_{a_{N_1}} `` and number of columns
equal to ``d_c``. A similar interpretation can be given to the basis transform on the
right, by taking its adjoint. In this particular example, it has two different combinations
of sectors ``(b…)`` and ``(b…)'``, where both have a single fusion tree to ``c`` as well as
to ``c'``, and ``n_{(b…)}=2``, ``n_{(b…)'}=3``.

Note that we never explicitly store or act with the basis transforms on the left and the
right. For composing tensor maps (i.e. multiplying them), these basis transforms just
cancel, whereas for tensor factorizations they just go through trivially. They transform
non-trivially when reshuffling the tensor indices, both within or in between the domain and
codomain. For this, however, we can completely rely on the manipulations of fusion trees to
implicitly compute the effect of the basis transform and construct the new blocks ``B_c``
that result with respect to the new basis.

Hence, as before, we only store the diagonal blocks ``B_c`` of size
`(blockdim(codomain(t), c), blockdim(domain(t), c))` as a `DenseMatrix`, accessible via
`block(t, c)`. Within this matrix, there are regions of the form
`view(block(t, c), m₁:m₂, n₁:n₂)` that correspond to the data
``t^c_{(a_1 … a_{N₁})α, (b_1 … b_{N₂})β}`` associated with a pair of fusion trees
``X^{(a_1 … a_{N₁}}_{c,α}`` and ``X^{(b_1 … b_{N₂})}_{c,β}``, henceforth again denoted as
`f₁` and `f₂`, with `f₁.coupled == f₂.coupled == c`. The ranges where this subblock is
living are managed within the tensor implementation, and these subblocks can be accessed
via `t[f₁,f₂]`, and is returned as a `StridedArray` of size
``n_{a_1} × n_{a_2} × … × n_{a_{N_1}} × n_{b_1} × … n_{b_{N₂}}``, or in code,
`(dim(V1, a1), dim(V2, a2), …, dim(VN₁, aN₁), dim(W1, b1), …, dim(WN₂, bN₂))`. While the
implementation does not distinguish between `FusionStyle isa Abelian` or
`FusionStyle isa NonAbelian`, in the former case the fusion tree is completely
characterized by the uncoupled sectors, and so the subblocks can also be accessed as
`t[(a1, …, aN₁), (b1, …, bN₂)]`. When there is no symmetry at all, i.e.
`sectortype(t) == Trivial`, `t[]` returns the raw tensor data as a `StridedArray` of size
`(dim(V1), …, dim(VN₁), dim(W1), …, dim(WN₂))`, whereas `block(t, Trivial())` returns the
same data as a `DenseMatrix` of size `(dim(V1) * … * dim(VN₁), dim(W1) * … * dim(WN₂))`.

## Constructing tensor maps and accessing tensor data

Having learned how a tensor is represented and stored, we can now discuss how to create
tensors and tensor maps. From hereon, we focus purely on the interface rather than the
implementation.

The most convenient set of constructors are those that construct  tensors or tensor maps
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

In all of these constructors, the last two arguments can be replaced by `domain→codomain`
or `codomain←domain`, where the arrows are obtained as `\rightarrow+TAB` and
`\leftarrow+TAB`. These arrows just create a Julia `Pair`, i.e. also `domain => codomain`
can be used, provided that `domain` and `codomain` are of type `ProductSpace`. The
advantage of the unicode arrows is that they will also convert a single instance of type
`S<:ElementarySpace` to a corresponding `ProductSpace{S,1}`. Some examples are perhaps in
order

```@repl tensors
t1 = TensorMap(randnormal, ℂ^2 ⊗ ℂ^3, ℂ^2)
t2 = TensorMap(randisometry, Float32, ℂ^2 ⊗ ℂ^3 ← ℂ^2)
t3 = TensorMap(undef, ℂ^2 → ℂ^2 ⊗ ℂ^3)
t4failed = TensorMap(undef, ComplexF64, ℂ^2 => ℂ^2 ⊗ ℂ^3)
t4 = TensorMap(undef, ComplexF64, ProductSpace(ℂ^2) => ℂ^2 ⊗ ℂ^3)
domain(t1) == domain(t2) == domain(t3) == domain(t4)
codomain(t1) == codomain(t2) == codomain(t3) == codomain(t4)
disp(x) = show(IOContext(Core.stdout, :compact=>false), "text/plain", trunc.(x; digits = 3));
t1[] |> disp
block(t1, Trivial()) |> disp
reshape(t1[], dim(codomain(t1)), dim(domain(t1))) |> disp
```

Finally, all constructors can also be replaced by `Tensor(..., codomain)`, in which case
the domain is assumed to be the empty `ProductSpace{S,0}()`, which can easily be obtained
as `one(codomain)`. Indeed, the empty product space is the unit object of the monoidal
category, equivalent to the field of scalars `𝕜`, and thus the multiplicative identity
(especially since `*` also acts as tensor product on vector spaces).

The matrices created by `f` are the matrices ``B_c`` discussed above, i.e. those returned
by `block(t, c)`. Only numerical matrices of type `DenseMatrix` are accepted, which in
practice just means Julia's intrinsic `Matrix{T}` for some `T<:Number`. In the future, we
will add support for `CuMatrix` from [CuArrays.jl](https://github.com/JuliaGPU/CuArrays.jl)
to harness GPU computing power, and maybe `SharedArray` from the Julia's `SharedArrays`
standard library.

Support for static or sparse data is currently not available, and if it would be
implemented, it would lead to new subtypes of `AbstractTensorMap` which are distinct from
`TensorMap`.

Let's conclude this section with some examples with `RepresentationSpace`.
```@repl tensors
V1 = ℤ₂Space(0=>3,1=>2)
V2 = ℤ₂Space(0=>2,1=>1)
t = TensorMap(randn, V1 ⊗ V1, V2 ⊗ V2')
(array = convert(Array, t)) |> disp
d1 = dim(codomain(t))
d2 = dim(domain(t))
(matrix = reshape(array, d1, d2)) |> disp
(u = reshape(convert(Array, TensorMap(I, codomain(t), fuse(codomain(t)))), d1, d1)) |> disp
(v = reshape(convert(Array, TensorMap(I, domain(t), fuse(domain(t)))), d2, d2)) |> disp
u'*u ≈ I ≈ v'*v
(u'*matrix*v) |> disp
# compare with:
block(t, ℤ₂(0)) |> disp
block(t, ℤ₂(1)) |> disp
```
Here, we illustrated some additional concepts. We constructed a `TensorMap` where the
blocks are initialized with the identity matrix using `I::UniformScaling` from Julia's
`LinearAlgebra` standard library. This works even if the blocks are not square, in this
case zero rows or columns (depending on the shape of the block) will be added. Creating a
`TensorMap` with `I` is a useful way to construct a fixed unitary or isometry between two
spaces. The operation `fuse(V)` creates an `ElementarySpace` which is isomorphic to a given
space `V` (of type `ProductSpace` or `ElementarySpace`). Constructing a `TensorMap` between
`V` and `fuse(V)` using the `I` constructor definitely results in a unitary, in particular
it is the unitary which implements the basis change from the product basis to the coupled
basis. In this case, for a group `G` with `FusionStyle(G) isa Abelian`, it is a permutation
matrix. Specifically choosing `V` equal to the codomain and domain of `t`, we can construct
the explicit basis transforms that bring `t` into block diagonal form.

Let's repeat the same exercise for `G = SU₂`, which has `FusionStyle(G) isa NonAbelian`.
```@repl tensors
V1 = SU₂Space(0=>2,1=>1)
V2 = SU₂Space(0=>1,1=>1)
t = TensorMap(randn, V1 ⊗ V1, V2 ⊗ V2')
(array = convert(Array, t)) |> disp
d1 = dim(codomain(t))
d2 = dim(domain(t))
(matrix = reshape(array, d1, d2)) |> disp
(u = reshape(convert(Array, TensorMap(I, codomain(t), fuse(codomain(t)))), d1, d1)) |> disp
(v = reshape(convert(Array, TensorMap(I, domain(t), fuse(domain(t)))), d2, d2)) |> disp
u'*u ≈ I ≈ v'*v
(u'*matrix*v) |> disp
# compare with:
block(t, SU₂(0)) |> disp
block(t, SU₂(1)) |> disp
block(t, SU₂(2)) |> disp
```
Note that the basis transforms `u` and `v` are no longer permutation matrices, but are
still unitary. Furthermore, note that they render the tensor block diagonal, but that now
every element of the diagonal blocks labeled by `c` comes itself in a tensor product with
an identity matrix of size `dim(c)`, i.e. `dim(SU₂(1)) = 3` and `dim(SU₂(2)) = 5`.

To create a `TensorMap` with existing data, one can use the aforementioned form but with
the function `f` replaced with the actual data, i.e. `TensorMap(data, codomain, domain)` or
any of its equivalents. For the specific form of `data`, we distinguish between the case
without and with symmetry. In the former case, one can just pass a `DenseArray`, either of
rank `N₁+N₂` and with matching size `(dims(codomain)..., dims(domain)...)`, or just as a
`DenseMatrix` with size `(dim(codomain), dim(domain))`. In the case of symmetry, `data`
needs to be specified as a dictionary (some subtype of `AbstractDict`) with the
blocksectors `c::G <: Sector` as keys and the corresponding matrix blocks as value, i.e.
`data[c]` is some `DenseMatrix` of size `(blockdim(codomain, c), blockdim(domain, c))`.

```@repl tensors
data = randn(3,3,3)
t = TensorMap(data, ℂ^3 ⊗ ℂ^3, ℂ^3)
t ≈ TensorMap(reshape(data, (9, 3)), ℂ^3 ⊗ ℂ^3, ℂ^3)
V = ℤ₂Space(0=>2, 1=>2)
data = Dict(ℤ₂(0)=>randn(8,2), ℤ₂(1)=>randn(8,2))
t2 = TensorMap(data, V*V, V)
for (c,b) in blocks(t2)
    println("Data for block $c :")
    b |> disp
    println()
end
```

## Vector space and linear algebra operations

`AbstractTensorMap` instances `t` represent linear maps, i.e. homomorphisms in a `𝕜`-linear
category, just like matrices. To a large extent, they follow the interface of `Matrix` in
Julia's `LinearAlgebra` standard library. Many methods from `LinearAlgebra` are (re)exported
by TensorKit.jl, and can then us be used without `using LinearAlgebra` explicitly. In all
of the following methods, the implementation acts directly on the underlying matrix blocks
(typically using the same method) and never needs to perform any basis transforms.

In particular, `AbstractTensorMap` instances can be composed, provided the domain of the
first object coincides with the codomain of the second. Composing tensor maps uses the
regular multiplication symbol as in `t = t1*t2`, which is also used for matrix
multiplication. TensorKit.jl also supports (and exports) the mutating method
`mul!(t, t1, t2)`.

`AbstractTensorMap` instances behave themselves as vectors (i.e. they are `𝕜`-linear) and
so they can be multiplied by scalars and, if they live in the same space, i.e. have the same
domain and codomain, they can be added to each other. There is also a `zero(t)`, the
additive identity, which produces a zero tensor with the same domain and codomain as `t`. In
addition, `TensorMap` supports basic Julia methods such as `fill!` and `copyto!`. Aside from
basic `+` and `*` operations, TensorKit.jl reexports a number of efficient in-place methods
from `LinearAlgebra`, such as `axpy!` (for `y ← α * x + y`), `axpby!` (for
`y ← α * x + β * y`), `lmul!` and `rmul!` (for `y ← α*y` and `y ← y*α`, which is typically
the same) and `mul!`, which can be used for out-of-place scalar multiplication `y ← α*x`.

For `t::AbstractTensorMap{S}` where `S<:EuclideanSpace`, henceforth referred to as
a `(Abstract)EuclideanTensorMap`, we can compute `norm(t)`, and for two such instances, the
inner product `dot(t1, t2)`, provided `t1` and `t2` have the same domain and codomain.
Furthermore, there is `normalize(t)` and `normalize!(t)` to return a scaled version of `t`
with unit norm.

With instances `t::AbstractEuclideanTensorMap` there is associated an adjoint operation,
given by `adjoint(t)` or simply `t'`, such that `domain(t') == codomain(t)` and
`codomain(t') == domain(t)`. Note that for an instance `t::TensorMap{S,N₁,N₂}`, `t'` is
simply stored in a wrapper called `AdjointTensorMap{S,N₂,N₁}`, which is another subtype of
`AbstractTensorMap`. This should be mostly unvisible to the user, as all methods should work
for this type as well. It can be hard to reason about the index order of `t'`, i.e. index
`i` of `t` appears in `t'` at index position `j = TensorKit.adjointtensorindex(t, i)`,
where the latter method is typically not necessary and hence unexported. There is also a
plural `TensorKit.adjointtensorindices` to convert multiple indices at once. Note that,
because the adjoint interchanges domain and codomain, we have
`space(t', j) == space(t, i)'`.

`AbstractTensorMap` instances can furthermore be tested for exact (`t1 == t2`) or
approximate (`t1 ≈ t2`) equality, though the latter requires `norm` can be computed.

Finally, when tensor map instances endomorphisms, i.e. they have the same domain and
codomain, there is a multiplicative identity which can be obtained as `one(t)` or `one!(t)`,
where the latter overwrites the contents of `t`. We can then also try to invert them using
`inv(t)`, or, in case their inverse is composed with another tensor `t2`, `t1\t2` or
`t2/t1`. The latter syntax also accepts instances `t1` whose domain and codomain are not
the same, and then amounts to `pinv(t1)`, the Moore-Penrose pseudoinverse. This, however,
is only really justified for `AbstractEuclideanTensorMap` instances. Returning to
endomorphisms, we can compute their trace via `tr(t)` and exponentiate them using `exp(t)`,
or if the contents of `t` can be destroyed in the process, `exp!(t)`. Furthermore, there are
a number of tensor factorizations for both endomorphisms and general homomorphism that we
discuss below.

## Tensor factorizations

As tensors are linear maps, they have various kinds of factorizations. Endomorphism, i.e.
tensor maps `t` with `codomain(t) == domain(t)`, have an eigenvalue decomposition. For
this, we overload both `LinearAlgebra.eigen(t; kwargs...)` and
`LinearAlgebra.eigen!(t; kwargs...)`, where the latter destroys `t` in the process. The
keyword arguments are the same that are accepted by `LinearAlgebra.eigen(!)` for matrices.
The result is returned as `D, V = eigen(t)`, such that `t*V ≈ V*D`. For given
`t::TensorMap{S,N,N}`, `V` is a `TensorMap{S,N,1}`, whose codomain corresponds to that of
`t`, but whose domain is a single space `S` (or more correctly a `ProductSpace{S,1}`), that
corresponds to `fuse(codomain(t))`. The eigenvalues are encoded in `D`, a
`TensorMap{S,1,1}`, whose domain and codomain correspond to the domain of `V`. Indeed, we
cannot reasonably associate a tensor product structure with the different eigenvalues. Note
that `D` stores the eigenvalues on the diagonal of a (collection of) `DenseMatrix`
instance(s), as there is currently no dedicated `DiagonalTensorMap` or diagonal storage
support.

We also define `LinearAlgebra.ishermitian(t)`, which can only return true for instances of
`TensorMap{<:EuclideanSpace}`, henceforth referred to as `EuclideanTensorMap`. In all other
cases, as the inner product is not defined, there is no notion of hermiticity (i.e. we are
not working in a `†`-category). For instances of `EuclideanTensorMap`, we also define and
export the routines `eigh` and `eigh!`, which compute the eigenvalue decomposition under
the guarantee (not checked) that the map is hermitian. Hence, eigenvalues will be real and
`V` will be unitary. We also define and export `eig` and `eig!`, which similarly assume
that the `TensorMap` is not hermitian (hence this does not require `EuclideanTensorMap`),
and always returns complexed values eigenvalues and eigenvectors. Like for matrices,
`LinearAlgebra.eigen` is type unstable and checks hermiticity at run-time, then falling
back to either `eig` or `eigh`.

Other factorizations that are provided by TensorKit.jl are orthogonal or unitary in nature,
and thus always require a `EuclideanTensorMap`. However, they don't require equal domain
and codomain. Let us first discuss the *singular value decomposition*, for which we define
and export the methods `tsvd` and `tsvd!` (where as always, the latter destroys the input)

`U, Σ, Vʰ, ϵ = tsvd(t; truncation = notrunc(), p::Real = 2, alg::OrthogonalFactorizationAlgorithm = SDD())`

This computes a (possibly truncated) singular value decomposition of
`t::TensorMap{S,N₁,N₂}` (with `S<:EuclideanSpace`), such that
`norm(t - U*Σ*Vʰ) ≈ ϵ`, where `U::TensorMap{S,N₁,1}`, `S::TensorMap{S,1,1}`,
`Vʰ::TensorMap{S,1,N₂}` and `ϵ::Real`. `U` is an isometry, i.e. `U'*U` approximates the
identity, whereas `U*U'` is an idempotent (squares to itself). The same holds for
`adjoint(Vʰ)`. The domain of `U` equals the domain and codomain of `Σ` and the codomain of
`Vʰ`. In the case of `truncation = notrunc()` (default value, see below) is given by
`min(fuse(codomain(t)), fuse(domain(t)))`. The singular values are contained in `Σ` and are
stored on the diagonal of a (collection of) `DenseMatrix` instance(s), similar to the
eigenvalues before.

The keyword argument `truncation` provides a way to control the truncation, and is
connected to the keyword argument `p`. The default value `notrunc()` implies no truncation,
and thus `ϵ = 0`. Other valid options are

* `truncerr(η::Real)`: truncates such that the `p`-norm of the truncated singular values is
  smaller than `η` times the `p`-norm of all singular values;

* `truncdim(χ::Integer)`: truncates such that the equivalent total dimension of the
  internal vector space is no larger than `χ`;

* `truncspace(W)`: truncates such that the dimension of the internal vector space is
  smaller than that of `W` in any sector, i.e. with
  `W₀ = min(fuse(codomain(t)), fuse(domain(t)))` this option will result in
  `domain(U) == domain(Σ) == codomain(Σ) == codomain(Vᵈ) == min(W, W₀)`;

* `trunbelow(η::Real)`: truncates such that every singular value is larger then `η`; this
  is different from `truncerr(η)` with `p = Inf` because it works in absolute rather than
  relative values.

Furthermore, the `alg` keyword can be either `SVD()` or `SDD()` (default), which
corresponds to two different algorithms in LAPACK to compute singular value decompositions.
The default value `SDD()` uses a divide-and-conquer algorithms and is typically the
fastest, but can loose some accuracy. The `SVD()` method uses a QR-iteration scheme and can
be more accurate, but is typically slower. Since Julia 1.3, these two algorithms are also
available in the `LinearAlgebra` standard library, where they are specified as
`LinearAlgebra.DivideAndConquer()` and `LinearAlgebra.QRIteration()`.

Note that we defined the new method `tsvd` (truncated or tensor singular value
decomposition), rather than overloading `LinearAlgebra.svd`. We (will) also support
`LinearAlgebra.svd(t)` as alternative for `tsvd(t; truncation = notrunc())`, but note that
the return values are then given by `U, Σ, V = svd(t)` with `V = adjoint(Vʰ)`.

We also define the following pair of orthogonal factorization algorithms, which are useful
when one is not interested in truncating a tensor or knowing the singular values, but only
in its image or coimage.

*   `Q, R = leftorth(t; alg::OrthogonalFactorizationAlgorithm = QRpos(), kwargs...)`:
    this produces an isometry `Q::TensorMap{S,N₁,1}` (i.e. `Q'*Q` approximates the identity,
    `Q*Q'` is an idempotent, i.e. squares to itself) and a general tensor map
    `R::TensorMap{1,N₂}`, such that `t ≈ Q*R`. Here, the domain of `Q` and thus codomain of
    `R` is a single vector space of type `S` that is typically given by
    `min(fuse(codomain(t)), fuse(domain(t)))`.

    The underlying algorithm used to compute this decomposition can be chosen among `QR()`,
    `QRpos()`, QL(), QLpos(), `SVD()`, `SDD()`, `Polar()`. `QR()` uses the underlying `qr`
    decomposition from `LinearAlgebra`, while `QRpos()` (the default) adds a correction to
    that to make sure that the diagonal elements of `R` are positive. Both result in block
    matrices in `R` which are upper triangular. `QL()` and `QLpos()` similarly result in a
    lower triangular block matrices in `R`, but only work if all block matrices are tall,
    i.e. `blockdim(codomain(t), c) >= blockdim(domain(t), c)` for all `c ∈ blocksectors(t)`.
    All of these methods assume `t` has full rank.

    If this is not the case, one can also use `alg = SVD()` or `alg = SDD()`, with extra
    keywords to control the absolute (`atol`) or relative (`rtol`) tolerance. We then set
    `Q=U` and `R=Σ*Vʰ` from the corresponding singular value decomposition, where only
    these singular values `σ > max(atol, norm(t)*rtol)` (and corresponding singular vectors
    in `U`) are kept. More finegrained control on the chosen singular values can be
    obtained with `tsvd` and its `truncation` keyword.

    Finally, `Polar()` sets `Q=U*Vʰ` and `R = (Vʰ)'*Σ*Vʰ`, such that `R` is positive
    definite; in this case `SDD()` is used to actually compute the singular value
    decomposition and no `atol` or `rtol` can be provided.

*   `L, Q = leftorth(t; alg::OrthogonalFactorizationAlgorithm = QRpos())`: this produces a
    general tensor map `L::TensorMap{S,N₁,1}` and the adjoint of an isometry
    `Q::TensorMap{S,1,N₂}`, such that `t ≈ L*Q`. Here, the domain of `L` and thus codomain
    of `Q` is a single vector space of type `S` that is typically given by
    `min(fuse(codomain(t)), fuse(domain(t)))`.

    The underlying algorithm used to compute this decomposition can be chosen among `LQ()`,
    `LQpos()`, `RQ()`, `RQpos()`, `SVD()`, `SDD()`, `Polar()`. `LQ()` uses the underlying
    `qr` decomposition from `LinearAlgebra` on the transposed data, and leads to lower
    triangular block matrices in `L`; `LQpos()` makes sure the diagonal elements are
    positive. `RQ()` and `RQpos()` similarly result in upper triangular block matrices in
    `L`, but only works for wide matrices, i.e. `blockdim(codomain(t), c) <=
    blockdim(domain(t), c)` for all `c ∈ blocksectors(t)`. All of these methods assume `t`
    has full rank.

    If this is not the case, one can also use `alg = SVD()` or `alg = SDD()`, with extra
    keywords to control the absolute (`atol`) or relative (`rtol`) tolerance. We then set
    `L=U*Σ` and `Q=Vʰ` from the corresponding singular value decomposition, where only these
    singular values `σ > max(atol, norm(t)*rtol)` (and corresponding singular vectors in
    `Vʰ`) are kept. More finegrained control on the chosen singular values can be obtained
    with `tsvd` and its `truncation` keyword.

    Finally, `Polar()` sets `L = U*Σ*U'` and `Q=U*Vʰ`, such that `L` is positive definite;
    in this case `SDD()` is used to actually compute the singular value decomposition and no
    `atol` or `rtol` can be provided.

Furthermore, we can compute an orthonormal basis for the orthogonal complement of the image
and of the co-image (i.e. the kernel) with the following methods:

*   `N = leftnull(t; alg::OrthogonalFactorizationAlgorithm = QR(), kwargs...)`:
    returns an isometric `TensorMap{S,N₁,1}` (i.e. `N'*N` approximates the identity) such
    that `N'*t` is approximately zero.

    Here, `alg` can be `QR()` (or `QRpos()`, there is actually no distinction), which
    assumes that `t` is full rank in all of its blocks and only returns an orthonormal basis
    for the missing columns.

    If this is not the case, one can also use `alg = SVD()` or `alg = SDD()`, with extra
    keywords to control the absolute (`atol`) or relative (`rtol`) tolerance. We then
    construct `N` from the left singular vectors corresponding to singular values
    `σ < max(atol, norm(t)*rtol)`.

*   `N = rightnull(t; alg::OrthogonalFactorizationAlgorithm = QR(), kwargs...)`:
    returns a `TensorMap{S,1,N₂}` with isometric adjoint (i.e. `N*N'` approximates the
    identity) such that `t*N'` is approximately zero.

    Here, `alg` can be `LQ()` (or `LQpos()`, there is actually no distinction), which
    assumes that `t` is full rank in all of its blocks and only returns an orthonormal
    basis for the missing rows.

    If this is not the case, one can also use `alg = SVD()` or `alg = SDD()`, with extra
    keywords to control the absolute (`atol`) or relative (`rtol`) tolerance. We then
    construct `N` from the right singular vectors corresponding to singular values
    `σ < max(atol, norm(t)*rtol)`.

Note that the methods `leftorth`, `rightorth`, `leftnull` and `rightnull` also come in a
form with exclamation mark, i.e. `leftorth!`, `rightorth!`, `leftnull!` and `rightnull!`,
which destroy the input tensor `t`.

Finally, note that each of the factorizations take a single argument, the tensor map `t`,
and a number of keyword arguments. They perform the factorization according to the given
codomain and domain of the tensor map. In many cases, we want to perform the factorization
according to a different bipartition of the indices. When `BraidingStyle(sectortype(t)) isa
Symmetric`, we can immediately specify an alternative bipartition of the indices of `t` in
all of these methods, in the form

```factorize(t, pleft, pright; kwargs...)```

where `pleft` will be the indices in the codomain of the new tensor map, and `pright` the
indices of the domain. Here, `factorize` is any of the methods `LinearAlgebra.eigen`, `eig`,
`eigh`, `tsvd`, `LinearAlgebra.svd`, `leftorth`, `rightorth`, `leftnull` and `rightnull`.
This signature does not allow for the exclamation mark, because it amounts to

```factorize!(permuteind(t, pleft, pright); kwargs...)

where `permuteind` is introduced and discussed in the next section.

## Index manipulations


## Tensor contractions and tensor networks

TODO
