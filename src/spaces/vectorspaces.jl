# FIELDS:
#==============================================================================#
"""
    abstract type Field end

Abstract type at the top of the type hierarchy for denoting fields over which
vector spaces can be defined. Two common fields are `ℝ` and `ℂ`, representing the
field of real or complex numbers respectively.
"""
abstract type Field end

struct RealNumbers <: Field end
struct ComplexNumbers <: Field end

const ℝ = RealNumbers()
const ℂ = ComplexNumbers()

Base.show(io::IO, ::RealNumbers) = print(io, "ℝ")
Base.show(io::IO, ::ComplexNumbers) = print(io, "ℂ")

Base.in(::Any, ::Field) = false
Base.in(::Real, ::RealNumbers) = true
Base.in(::Number, ::ComplexNumbers) = true

Base.@pure Base.issubset(::Type, ::Field) = false
Base.@pure Base.issubset(::Type{<:Real}, ::RealNumbers) = true
Base.@pure Base.issubset(::Type{<:Number}, ::ComplexNumbers) = true
Base.@pure Base.issubset(::RealNumbers, ::ComplexNumbers) = true

# VECTOR SPACES:
#==============================================================================#
"""
    abstract type VectorSpace end

Abstract type at the top of the type hierarchy for denoting vector spaces.
"""
abstract type VectorSpace end

"""
    function fieldtype(V::VectorSpace) -> Field

Returns the field type over which a vector space is defined.
"""
function fieldtype end
fieldtype(V::VectorSpace) = fieldtype(typeof(V))

# Basic vector space methods
#----------------------------
"""
    space(a) -> VectorSpace

Returns the vector space associated to object `a`.
"""
function space end

"""
    dim(V::VectorSpace) -> Int
Returns the total dimension of the vector space `V` as an Int.
"""
function dim end

"""
    dual(V::VectorSpace) -> VectorSpace
Returns the dual space of `V`; also obtained via `V'`. It is assumed that
`typeof(V) == typeof(V')`.
"""
function dual end

"""
    isdual(V::ElementarySpace) -> Bool
Returns wether an ElementarySpace `V` is normal or rather a dual space. Always
returns `false` for spaces where `V==dual(V)`.
"""
function isdual end

# convenience definitions:
Base.adjoint(V::VectorSpace) = dual(V)
Base.:*(V1::VectorSpace, V2::VectorSpace) = ⊗(V1, V2)

# Hierarchy of elementary vector spaces
#---------------------------------------
"""
    abstract type ElementarySpace{k} <: VectorSpace end

Elementary finite-dimensional vector space over a field `k` that can be used as the
index space corresponding to the indices of a tensor. Every elementary vector space
should respond to the methods `conj` and `dual`, returning the complex conjugate space
and the dual space respectively. The complex conjugate of the dual space is obtained
as `dual(conj(V)) === conj(dual(V))`. These different spaces should be of the same type,
so that a tensor can be defined as an element of a homogeneous tensor product
of these spaces.
"""
abstract type ElementarySpace{k} <: VectorSpace end
const IndexSpace = ElementarySpace

fieldtype(::Type{<:ElementarySpace{k}}) where {k} = k


"""
    oneunit(V::S) where {S<:ElementarySpace} -> S

Returns the corresponding vector space of type `S` that represents the trivial
space, i.e. the space that is isomorphic to the corresponding field. Note that
this is different from `one(V::S)`, which returns the empty product space
`ProductSpace{S,0}(())`.
"""
Base.oneunit(V::ElementarySpace) = oneunit(typeof(V))


"""
    ⊕(V1::S, V2::S, V3::S...) where {S<:ElementarySpace} -> S

Returns the corresponding vector space of type `S` that represents the direct sum
sum of the spaces `V1`, `V2`, ... Note that all the individual spaces should have
the same value for `isdual`, as otherwise the direct sum is not defined.
"""
function ⊕ end
⊕(V1, V2, V3, V4...) = ⊕(⊕(V1, V2), V3, V4...)


"""
    ⊗(V1::S, V2::S, V3::S...) where {S<:ElementarySpace} -> S

Creates a `ProductSpace{S}(V1, V2, V3...)` representing the tensor product of
several elementary vector spaces. The tensor product structure is preserved, see
`fuse(V1, V2, V3...) = fuse(V1 ⊗ V2 ⊗ V3...)` for returning a single elementary
space of type `S` that is isomorphic to this tensor product.
"""
function ⊗ end
⊗(V1, V2, V3, V4...) = ⊗(⊗(V1, V2), V3, V4...)

"""
    fuse(V1::S, V2::S, V3::S...) where {S<:ElementarySpace} -> S
    fuse(P::ProductSpace{S}) where {S<:ElementarySpace} -> S

Returns a single vector space of type `S` that is isomorphic to the fusion product
of the individual spaces `V1`, `V2`, ..., or the spaces contained in `P`.
"""
function fuse end
fuse(V1, V2, V3, V4...) = fuse(fuse(V1, V2), V3, V4...)

"""
    flip(V::S) where {S<:ElementarySpace} -> S

Returns a single vector space of type `S` that has the same value of `isdual` as
`dual(V)`, but yet is isomorphic to `V` rather than to `dual(V)`. The spaces `flip(V)`
and `dual(V)` only differ in the case of `RepresentationSpace{G}`.
"""
function flip end

"""
    conj(V::S) where {S<:ElementarySpace} -> S
Returns the conjugate space of `V`. For `fieldtype(V)==ℝ`, `conj(V) == V`
It is assumed that `typeof(V) == typeof(conj(V))`.
"""
Base.conj(V::ElementarySpace{ℝ}) = V

"""
    abstract type InnerProductSpace{k} <: ElementarySpace{k} end

Abstract type for denoting vector with an inner product and a corresponding
metric, which can be used to raise or lower indices of tensors.
"""
abstract type InnerProductSpace{k} <: ElementarySpace{k} end

"""
    abstract type EuclideanSpace{k<:Union{ℝ,ℂ}} <: InnerProductSpace{k} end

Abstract type for denoting real or complex spaces with a standard (Euclidean)
inner product (i.e. orthonormal basis), such that the dual space is naturally
isomorphic to the conjugate space (in the complex case) or even to the
space itself (in the real case), also known as the category of finite-dimensional
Hilbert spaces FdHilb.
"""
abstract type EuclideanSpace{k} <: InnerProductSpace{k} end # k should be ℝ or ℂ

dual(V::EuclideanSpace) = conj(V)
isdual(V::EuclideanSpace{ℝ}) = false
# dual space is naturally isomorphic to conjugate space for inner product spaces

# representation spaces: we restrict to complex Euclidean space supporting unitary representations
"""
    abstract type RepresentationSpace{G<:Sector} <: EuclideanSpace{ℂ} end

Complex Euclidean space with a direct sum structure corresponding to different
superselection sectors of type `G<:Sector`, e.g. the elements or irreps of a
compact or finite group, or the labels of a unitary fusion category.
"""
abstract type RepresentationSpace{G<:Sector} <: EuclideanSpace{ℂ} end

"""
    function sectortype(a) -> Sector

Returns the type of sector over which object `a` (e.g. a representation space or
an invariant tensor) is defined. Also works in type domain.
"""
sectortype(V::VectorSpace) = sectortype(typeof(V))
sectortype(::Type{<:ElementarySpace}) = Trivial
sectortype(::Type{<:RepresentationSpace{G}}) where {G} = G

checksectors(::ElementarySpace, ::Trivial) = true
Base.axes(V::ElementarySpace, ::Trivial) = axes(V)

"""
    sectors(V::ElementarySpace) -> sectortype(V)
    sectors(V::ProductSpace{S,N}) -> NTuple{N,sectortype{V}}

Returns the different sectors of object `a`( e.g. a representation space or an
invariant tensor).
"""
sectors(::ElementarySpace) = (Trivial(),)
dim(V::ElementarySpace, ::Trivial) = sectortype(V) == Trivial ? dim(V) : throw(SectorMismatch())

# Composite vector spaces
#-------------------------
"""
    abstract type CompositeSpace{S<:ElementarySpace} <: VectorSpace end

Abstract type for composite spaces that are defined in terms of a number of
elementary vector spaces of a homogeneous type `S<:ElementarySpace{k}`.
"""
abstract type CompositeSpace{S<:ElementarySpace} <: VectorSpace end

fieldtype(::Type{<:CompositeSpace{S}}) where {S<:ElementarySpace} = fieldtype(S)
sectortype(::Type{<:CompositeSpace{S}}) where {S<:ElementarySpace} = sectortype(S)

# Specific realizations of ElementarySpace types
#------------------------------------------------
# spaces without internal structure
include("cartesianspace.jl")
include("complexspace.jl")
include("generalspace.jl")
include("representationspace.jl")

# # Specific realizations of CompositeSpace types
# #-----------------------------------------------
include("productspace.jl")

# Other examples might include:
# braidedspace and fermionspace
# symmetric and antisymmetric subspace of a tensor product of identical vector spaces
# ...
