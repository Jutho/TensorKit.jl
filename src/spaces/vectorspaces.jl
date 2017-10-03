# FIELDS:
#==============================================================================#
"""
    abstract type Field end

abstract type at the top of the type hierarchy for denoting fields over which
vector spaces can be defined. Two common fields are `ℝ` and `ℂ`, representing the
field of real or complex numbers respectively.
"""
abstract type Field end

struct RealNumbers <: Field end
struct ComplexNumbers <: Field end

const ℝ = RealNumbers()
const ℂ = ComplexNumbers()

Base.show(io::IO, ::RealNumbers) = print(io,"ℝ")
Base.show(io::IO, ::ComplexNumbers) = print(io,"ℂ")

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

abstract type at the top of the type hierarchy for denoting vector spaces
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

# convenience definitions:
adjoint(V::VectorSpace) = dual(V)
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
    conj(V::ElementarySpace) -> ElementarySpace
Returns the conjugate space of `V`. For `fieldtype(V)==ℝ`, `conj(V) == V`
It is assumed that `typeof(V) == typeof(conj(V))`.
"""
Base.conj(V::ElementarySpace{ℝ}) = V

"""
    abstract type InnerProductSpace{k} <: ElementarySpace{k} end

abstract type for denoting vector with an inner product and a corresponding
metric, which can be used to raise or lower indices of tensors
"""
abstract type InnerProductSpace{k} <: ElementarySpace{k} end

"""
    abstract type EuclideanSpace{k<:Union{ℝ,ℂ}} <: InnerProductSpace{k} end

abstract type for denoting real or complex spaces with a standard (Euclidean)
inner product (i.e. orthonormal basis), such that the dual space is naturally
isomorphic to the conjugate space (in the complex case) or even to the
space itself (in the real case)
"""
abstract type EuclideanSpace{k} <: InnerProductSpace{k} end # k should be ℝ or ℂ

dual(V::EuclideanSpace) = conj(V)
# dual space is naturally isomorphic to conjugate space for inner product spaces

# representation spaces: we restrict to complex Euclidean space supporting unitary representations
"""
    abstract type AbstractRepresentationSpace{G<:Sector} <: EuclideanSpace{ℂ} end

Complex Euclidean space with a direct sum structure corresponding to different
different superselection sectors of type `G<:Sector`, e.g. the elements or
irreps of a compact or finite group, or the labels of a unitary fusion category.
"""
abstract type AbstractRepresentationSpace{G<:Sector} <: EuclideanSpace{ℂ} end

"""
    function sectortype(a) -> Sector

Returns the type of sector over which object `a` (e.g. a representation space or
an invariant tensor) is defined. Also works in type domain.
"""
sectortype(V::VectorSpace) = sectortype(typeof(V))
sectortype(::Type{<:ElementarySpace}) = Trivial
sectortype(::Type{<:AbstractRepresentationSpace{G}}) where {G} = G

checksectors(::ElementarySpace, ::Trivial) = true

"""
    function sectors(a)

    Returns the different sectors of object `a`( e.g. a representation space or
    an invariant tensor).
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
# include("superspace.jl")
# include("invariantspace.jl")

# Other examples might include:
# braidedspace and fermionspace
# symmetric and antisymmetric subspace of a tensor product of identical vector spaces
# ...
