# Elementary vector spaces
#--------------------------
abstract ElementarySpace{F} <: VectorSpace
typealias IndexSpace ElementarySpace # index spaces for tensors
# Elementary finite-dimensional vector space over a field F that can be used as the index space corresponding to the indices
# of a tensor. Every elementary vector space should respond to the methods conj and dual, returning the complex conjugate space
# and the dual space respectively. The complex conjugate of the dual space is obtained as dual(conj(V)) === conj(dual(V)). These
# different spaces should be of the same type, so that a tensor can be defined as an element of a homogeneous tensor product
# of these spaces

Base.eltype{F}(V::ElementarySpace{F}) = F
Base.eltype{F}(::Type{ElementarySpace{F}}) = F
Base.eltype{S<:ElementarySpace}(::Type{S}) = eltype(super(S))

Base.isfinite(::ElementarySpace) = true

# The complex conjugate vector space of a real vector space is equal to itself
Base.conj{F<:Real}(V::ElementarySpace{F}) = V

abstract InnerProductSpace{F} <: ElementarySpace{F}
abstract HilbertSpace{F} <: InnerProductSpace{F}
# An inner product space, has the possibility to raise or lower indices of tensors using the metric

const ℝ=Real
const ℂ=Complex{Real}

abstract EuclideanSpace{F<:Union(ℝ,ℂ)} <: HilbertSpace{F}
# Elementary finite-dimensional space R^d or C^d with standard (Euclidean) inner product (i.e. orthonormal basis)

Base.conj(V::EuclideanSpace) = dual(V)
# The complex conjugate space of a Euclidean space is naturally isomorphic to its dual space

dual(V::EuclideanSpace{ℝ}) = V
# For a real Euclidean space, the dual space is naturally isomorphic to the vector space

# Functionality for extracting and iterating over elementary space
Base.length(V::ElementarySpace) = 1
Base.endof(V::ElementarySpace) = 1
Base.getindex(V::ElementarySpace, n::Integer) = (n == 1 ? V : throw(BoundsError()))

Base.start(V::ElementarySpace) = false
Base.next(V::ElementarySpace, state) = (V,true)
Base.done(V::ElementarySpace, state) = state

# CartesianSpace: standard real space R^N with euclidean structure (canonical orthonormal/cartesian basis)
include("spaces/cartesianspace.jl")

# ComplexSpace: standard complex space C^N with euclidean structure (canonical orthonormal basis)
include("spaces/complexspace.jl")

# GeneralSpace: a general elementary vector space (without inner product structure)
include("spaces/generalspace.jl")

# Graded elementary vector spaces
#---------------------------------
# Vector spaces which have a natural decomposition as a direct sum of several sectors

# Sector: hierarchy of types for labelling different sectors, e.g. irreps, quantum numbers, anyon types, ...
abstract Sector
# Abelian: sectors that have an abelian fusion structure
abstract Abelian <: Sector
*(s::Abelian)=s

# Explicit realizations
include("spaces/sectors.jl")

# UnitaryRepresentationSpace: Family of euclidean spaces that are graded corresponding to unitary representations
abstract UnitaryRepresentationSpace{G<:Sector} <: EuclideanSpace{ℂ}

sectortype{G}(V::UnitaryRepresentationSpace{G}) = G
sectortype{G}(::Type{UnitaryRepresentationSpace{G}}) = G
sectortype{S<:UnitaryRepresentationSpace}(::Type{S}) = sectortype(super(S))

# AbelianSpace: general space that is graded by abelian sectors (i.e. one-dimensional representations)
include("spaces/abelianspace.jl")

# Auxiliary functionality:
_dim{G<:Sector,N}(spaces::NTuple{N,UnitaryRepresentationSpace{G}},s::NTuple{N,G})=(d=1;for n=1:N;d*=dim(spaces[n],s[n]);end;return d)

@ngenerate N Vector{NTuple{N,G}} function _sectors{G<:Sector,N}(spaces::NTuple{N,UnitaryRepresentationSpace{G}})
    numsectors=1
    @nexprs N i->(s_i=collect(sectors(spaces[i]));numsectors*=length(s_i))
    sectorlist=Array(NTuple{N,G},numsectors)
    counter=0
    @nloops N i d->1:length(s_d) begin
        counter+=1
        sectorlist[counter]=@ntuple N k->s_k[i_k]
    end
    return sectorlist
end

@ngenerate N Vector{NTuple{N,G}} function _invariantsectors{G<:Abelian,N}(spaces::NTuple{N,AbelianSpace{G}})
    @nexprs N i->(s_i=collect(sectors(spaces[i])))
    sectorlist=Array(NTuple{N,G},0)
    @nloops N i d->1:length(s_d) begin
        sector=@ntuple N k->s_k[i_k]
        c=one(G)
        for n=1:N
            c*=sector[n]
        end
        if c==one(G)
            push!(sectorlist,sector)
        end
    end
    return sectorlist
end


