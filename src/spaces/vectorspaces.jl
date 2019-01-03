# FIELDS:
#==============================================================================#
"""
    abstract type Field end

Abstract type at the top of the type hierarchy for denoting fields over which vector spaces
can be defined. Two common fields are `ℝ` and `ℂ`, representing the field of real or complex
numbers respectively.
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
Base.@pure Base.issubset(::RealNumbers, ::RealNumbers) = true
Base.@pure Base.issubset(::RealNumbers, ::ComplexNumbers) = true
Base.@pure Base.issubset(::ComplexNumbers, ::RealNumbers) = false

# VECTOR SPACES:
#==============================================================================#
"""
    abstract type VectorSpace end

Abstract type at the top of the type hierarchy for denoting vector spaces.
"""
abstract type VectorSpace end

"""
    function field(V::VectorSpace) -> Field

Return the field type over which a vector space is defined.
"""
function field end
field(V::VectorSpace) = field(typeof(V))

# Basic vector space methods
#----------------------------
"""
    space(a) -> VectorSpace

Return the vector space associated to object `a`.
"""
function space end

"""
    dim(V::VectorSpace) -> Int

Return the total dimension of the vector space `V` as an Int.
"""
function dim end

"""
    dual(V::VectorSpace) -> VectorSpace

Return the dual space of `V`; also obtained via `V'`. It is assumed that
`typeof(V) == typeof(V')`.
"""
function dual end

"""
    isdual(V::ElementarySpace) -> Bool

Return wether an ElementarySpace `V` is normal or rather a dual space. Always returns
`false` for spaces where `V == dual(V)`.
"""
function isdual end

# convenience definitions:
Base.adjoint(V::VectorSpace) = dual(V)
Base.:*(V1::VectorSpace, V2::VectorSpace) = ⊗(V1, V2)

# Hierarchy of elementary vector spaces
#---------------------------------------
"""
    abstract type ElementarySpace{𝕜} <: VectorSpace end

Elementary finite-dimensional vector space over a field `𝕜` that can be used as the index
space corresponding to the indices of a tensor.

Every elementary vector space should respond to the methods [`conj`](@ref) and
[`dual`](@ref), returning the complex conjugate space and the dual space respectively. The
complex conjugate of the dual space is obtained as `dual(conj(V)) === conj(dual(V))`. These
different spaces should be of the same type, so that a tensor can be defined as an element
of a homogeneous tensor product of these spaces.
"""
abstract type ElementarySpace{𝕜} <: VectorSpace end
const IndexSpace = ElementarySpace

field(::Type{<:ElementarySpace{𝕜}}) where {𝕜} = 𝕜


"""
    oneunit(V::S) where {S<:ElementarySpace} -> S

Return the corresponding vector space of type `S` that represents the trivial
one-dimensional space, i.e. the space that is isomorphic to the corresponding field. Note
that this is different from `one(V::S)`, which returns the empty product space
`ProductSpace{S,0}(())`.
"""
Base.oneunit(V::ElementarySpace) = oneunit(typeof(V))


"""
    ⊕(V1::S, V2::S, V3::S...) where {S<:ElementarySpace} -> S

Return the corresponding vector space of type `S` that represents the direct sum sum of the
spaces `V1`, `V2`, ... Note that all the individual spaces should have the same value for
[`isdual`](@ref), as otherwise the direct sum is not defined.
"""
function ⊕ end
⊕(V1, V2, V3, V4...) = ⊕(⊕(V1, V2), V3, V4...)


"""
    ⊗(V1::S, V2::S, V3::S...) where {S<:ElementarySpace} -> S

Create a [`ProductSpace{S}(V1, V2, V3...)`](@ref) representing the tensor product of several
elementary vector spaces. For convience, Julia's regular multiplication operator `*` applied
to vector spaces has the same effect.

The tensor product structure is preserved, see [`fuse`](@ref) for returning a single
elementary space of type `S` that is isomorphic to this tensor product.
"""
function ⊗ end
⊗(V1, V2, V3, V4...) = ⊗(⊗(V1, V2), V3, V4...)

"""
    fuse(V1::S, V2::S, V3::S...) where {S<:ElementarySpace} -> S
    fuse(P::ProductSpace{S}) where {S<:ElementarySpace} -> S

Return a single vector space of type `S` that is isomorphic to the fusion product of the
individual spaces `V1`, `V2`, ..., or the spaces contained in `P`.
"""
function fuse end
fuse(V1, V2, V3, V4...) = fuse(fuse(V1, V2), V3, V4...)

"""
    flip(V::S) where {S<:ElementarySpace} -> S

Return a single vector space of type `S` that has the same value of [`isdual`](@ref) as
`dual(V)`, but yet is isomorphic to `V` rather than to `dual(V)`. The spaces `flip(V)` and
`dual(V)` only differ in the case of [`RepresentationSpace{G}`](@ref).
"""
function flip end

"""
    conj(V::S) where {S<:ElementarySpace} -> S

Return the conjugate space of `V`.

For `field(V)==ℝ`, `conj(V) == V`. It is assumed that `typeof(V) == typeof(conj(V))`.
"""
Base.conj(V::ElementarySpace{ℝ}) = V

"""
    abstract type InnerProductSpace{𝕜} <: ElementarySpace{𝕜} end

Abstract type for denoting vector with an inner product and a corresponding metric, which
can be used to raise or lower indices of tensors.
"""
abstract type InnerProductSpace{𝕜} <: ElementarySpace{𝕜} end

"""
    abstract type EuclideanSpace{𝕜} <: InnerProductSpace{𝕜} end

Abstract type for denoting real or complex spaces with a standard (Euclidean) inner product
(i.e. orthonormal basis), such that the dual space is naturally isomorphic to the conjugate
space (in the complex case) or even to the space itself (in the real case), also known as
the category of finite-dimensional Hilbert spaces ``FdHilb``.
"""
abstract type EuclideanSpace{𝕜} <: InnerProductSpace{𝕜} end # 𝕜 should be ℝ or ℂ

dual(V::EuclideanSpace) = conj(V)
isdual(V::EuclideanSpace{ℝ}) = false
# dual space is naturally isomorphic to conjugate space for inner product spaces

# representation spaces: we restrict to complex Euclidean space supporting unitary representations
"""
    abstract type RepresentationSpace{G<:Sector} <: EuclideanSpace{ℂ} end

Complex Euclidean space with a direct sum structure corresponding to different
superselection sectors of type `G<:Sector`, e.g. the elements or irreps of a compact or
finite group, or the labels of a unitary fusion category.
"""
abstract type RepresentationSpace{G<:Sector} <: EuclideanSpace{ℂ} end
const Rep{G<:Sector} = RepresentationSpace{G}

"""
    function sectortype(a) -> Sector

Return the type of sector over which object `a` (e.g. a representation space or a tensor) is
defined. Also works in type domain.
"""
sectortype(V::VectorSpace) = sectortype(typeof(V))
sectortype(::Type{<:ElementarySpace}) = Trivial
sectortype(::Type{<:RepresentationSpace{G}}) where {G} = G

hassector(::ElementarySpace, ::Trivial) = true
Base.axes(V::ElementarySpace, ::Trivial) = axes(V)

"""
    sectors(V::ElementarySpace) -> sectortype(V)
    sectors(V::ProductSpace{S,N}) -> NTuple{N,sectortype{V}}

Return the different sectors of object `a` (e.g. a representation space or a tensor).
"""
sectors(::ElementarySpace) = (Trivial(),)
dim(V::ElementarySpace, ::Trivial) =
    sectortype(V) == Trivial ? dim(V) : throw(SectorMismatch())

# Composite vector spaces
#-------------------------
"""
    abstract type CompositeSpace{S<:ElementarySpace} <: VectorSpace end

Abstract type for composite spaces that are defined in terms of a number of elementary
vector spaces of a homogeneous type `S<:ElementarySpace{𝕜}`.
"""
abstract type CompositeSpace{S<:ElementarySpace} <: VectorSpace end

field(::Type{<:CompositeSpace{S}}) where {S<:ElementarySpace} = field(S)
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
