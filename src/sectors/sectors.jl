# Superselection sectors (quantum numbers):
# for defining graded vector spaces and invariant subspaces of tensor products
#==============================================================================#
"""
    abstract type Sector end

Abstract type for representing the (isomorphism classes of) simple objects in (unitary
and pivotal) (pre-)fusion categories, e.g. the irreducible representations of a finite or
compact group. Subtypes `I<:Sector` as the set of labels of a `GradedSpace`.

Every new `I<:Sector` should implement the following methods:
*   `one(::Type{I})`: unit element of `I`
*   `conj(a::I)`: ``a̅``, conjugate or dual label of ``a``
*   `⊗(a::I, b::I)`: iterable with unique fusion outputs of ``a ⊗ b``
    (i.e. don't repeat in case of multiplicities)
*   `Nsymbol(a::I, b::I, c::I)`: number of times `c` appears in `a ⊗ b`, i.e. the
    multiplicity
*   `FusionStyle(::Type{I})`: `Abelian()`, `SimpleNonAbelian()` or
    `DegenerateNonAbelian()`
*   `BraidingStyle(::Type{I})`: `Bosonic()`, `Fermionic()`, `Anyonic()`, ...
*   `Fsymbol(a::I, b::I, c::I, d::I, e::I, f::I)`: F-symbol: scalar (in case of
    `Abelian`/`SimpleNonAbelian`) or matrix (in case of `DegenerateNonAbelian`)
*   `Rsymbol(a::I, b::I, c::I)`: R-symbol: scalar (in case of
    `Abelian`/`SimpleNonAbelian`) or matrix (in case of `DegenerateNonAbelian`)
and optionally
*   `dim(a::I)`: quantum dimension of sector `a`
*   `frobeniusschur(a::I)`: Frobenius-Schur indicator of `a`
*   `Bsymbol(a::I, b::I, c::I)`: B-symbol: scalar (in case of
    `Abelian`/`SimpleNonAbelian`) or matrix (in case of `DegenerateNonAbelian`)
*   `twist(a::I)` -> twist of sector `a`
and optionally, if `FusionStyle(I) isa DegenerateNonAbelian`
*   `vertex_ind2label(i::Int, a::I, b::I, c::I)` -> a custom label for the `i`th copy of
    `c` appearing in `a ⊗ b`

Furthermore, `iterate` and `Base.IteratorSize` should be made to work for the singleton type
[`SectorValues{I}`](@ref).
"""
abstract type Sector end

# iterator over the values (i.e., elements of representative set of simple objects)
# in the sector
"""
    struct SectorValues{I<:Sector}

Singleton type to represent an iterator over the possible values of type `I`, whose
instance is obtained as `values(I)`. For a new `I::Sector`, the following should be defined
*   `Base.iterate(::SectorValues{I}[, state])`: iterate over the values
*   `Base.IteratorSize(::Type{SectorValues{I}})`: `HasLenght()`, `SizeUnkown()`
    or `IsInfinite()` depending on whether the number of values of type `I` is finite
    (and sufficiently small) or infinite; for a large number of values, `SizeUnknown()` is
    recommend because this will trigger the use of `GenericGradedSpace`.
If `IteratorSize(I) == HasLength()`, also the following must be implemented:
*   `Base.length(::SectorValues{I})`: the number of different values
*   `Base.getindex(::SectorValues{I}, i::Int)`: a mapping between an index `i` and an
    instance of `I`
*   `findindex(::SectorValues{I}, c::I)`: reverse mapping between a value `c::I` and an
    index `i::Integer ∈ 1:length(values(I))`
"""
struct SectorValues{I<:Sector} end
Base.IteratorEltype(::Type{<:SectorValues}) = HasEltype()
Base.eltype(::Type{SectorValues{I}}) where {I<:Sector} = I
Base.values(::Type{I}) where {I<:Sector} = SectorValues{I}()

# Define a sector for ungraded vector spaces
struct Trivial <: Sector
end
Base.show(io::IO, ::Trivial) = print(io, "Trivial()")
Base.show(io::IO, ::Type{Trivial}) = print(io, "Trivial")

"""
    one(::Sector) -> Sector
    one(::Type{<:Sector}) -> Sector

Return the unit element within this type of sector.
"""
Base.one(a::Sector) = one(typeof(a))
Base.one(::Type{Trivial}) = Trivial()

"""
    dual(a::Sector) -> Sector

Return the conjugate label `conj(a)`.
"""
dual(a::Sector) = conj(a)
Base.conj(::Trivial) = Trivial()

"""
    isreal(::Type{<:Sector}) -> Bool

Return whether the topological data (Fsymbol, Rsymbol) of the sector is real or not (in
which case it is complex).
"""
Base.@pure function Base.isreal(I::Type{<:Sector})
    u = one(I)
    return (eltype(Fsymbol(u, u, u, u, u, u))<:Real) && (eltype(Rsymbol(u, u, u))<:Real)
end
Base.@pure Base.isreal(::Type{Trivial}) = true

Base.isless(::Trivial, ::Trivial) = false

# FusionStyle: the most important aspect of Sector
#---------------------------------------------
"""
    ⊗(a::I, b::I) where {I<:Sector}

Return an iterable of elements of `c::I` that appear in the fusion product `a ⊗ b`.

Note that every element `c` should appear at most once, fusion degeneracies (if
`FusionStyle(I) == DegenerateNonAbelian()`) should be accessed via `Nsymbol(a, b, c)`.
"""
⊗(::Trivial, ::Trivial) = (Trivial(),)

"""
    Nsymbol(a::I, b::I, c::I) where {I<:Sector} -> Integer

Return an `Integer` representing the number of times `c` appears in the fusion product
`a ⊗ b`. Could be a `Bool` if `FusionStyle(I) == Abelian()` or `SimpleNonAbelian()`.
"""
function Nsymbol end
Nsymbol(::Trivial, ::Trivial, ::Trivial) = true

# trait to describe the fusion of superselection sectors
abstract type FusionStyle end
struct Abelian <: FusionStyle
end
abstract type NonAbelian <: FusionStyle end
struct SimpleNonAbelian <: NonAbelian # non-abelian fusion but multiplicity free
end
struct DegenerateNonAbelian <: NonAbelian # non-abelian fusion with multiplicities
end

"""
    FusionStyle(a::Sector) -> ::FusionStyle
    FusionStyle(I::Type{<:Sector}) -> ::FusionStyle

Return the type of fusion behavior of sectors of type I, which can be either
*   `Abelian()`: single fusion output when fusing two sectors;
*   `SimpleNonAbelian()`: multiple outputs, but every output occurs at most one,
    also known as multiplicity free (e.g. irreps of ``SU(2)``);
*   `DegenerateNonAbelian()`: multiple outputs that can occur more than once (e.g. irreps
    of ``SU(3)``).
There is an abstract supertype `NonAbelian` of which both `SimpleNonAbelian` and
`DegenerateNonAbelian` are subtypes.
"""
FusionStyle(a::Sector) = FusionStyle(typeof(a))
FusionStyle(::Type{Trivial}) = Abelian()

# NOTE: the following inline is extremely important for performance, especially
# in the case of Abelian, because ⊗(...) is computed very often
@inline function ⊗(a::I, b::I, c::I, rest::Vararg{I}) where {I<:Sector}
    if FusionStyle(I) isa Abelian
        return a ⊗ first(⊗(b, c, rest...))
    else
        s = Set{I}()
        for d in ⊗(b, c, rest...)
            for e in a ⊗ d
                push!(s, e)
            end
        end
        return s
    end
end

"""
    Fsymbol(a::I, b::I, c::I, d::I, e::I, f::I) where {I<:Sector}

Return the F-symbol ``F^{abc}_d`` that associates the two different fusion orders of sectors
`a`, `b` and `c` into an ouput sector `d`, using either an intermediate sector ``a ⊗ b → e``
or ``b ⊗ c → f``:
```
a-<-μ-<-e-<-ν-<-d                                     a-<-λ-<-d
    ∨       ∨       -> Fsymbol(a,b,c,d,e,f)[μ,ν,κ,λ]      ∨
    b       c                                             f
                                                          v
                                                      b-<-κ
                                                          ∨
                                                          c
```
If `FusionStyle(I)` is `Abelian` or `SimpleNonAbelian`, the F-symbol is a number. Otherwise
it is a rank 4 array of size
`(Nsymbol(a, b, e), Nsymbol(e, c, d), Nsymbol(b, c, f), Nsymbol(a, f, d))`.
"""
function Fsymbol end
Fsymbol(::Trivial, ::Trivial, ::Trivial, ::Trivial, ::Trivial, ::Trivial) = 1

"""
    Rsymbol(a::I, b::I, c::I) where {I<:Sector}

Returns the R-symbol ``R^{ab}_c`` that maps between ``c → a ⊗ b`` and ``c → b ⊗ a`` as in
```
a -<-μ-<- c                                 b -<-ν-<- c
     ∨          -> Rsymbol(a,b,c)[μ,ν]           v
     b                                           a
```
If `FusionStyle(I)` is `Abelian()` or `SimpleNonAbelian()`, the R-symbol is a number.
Otherwise it is a square matrix with row and column size `Nsymbol(a,b,c) == Nsymbol(b,a,c)`.
"""
function Rsymbol end
Rsymbol(::Trivial, ::Trivial, ::Trivial) = 1

# If a I::Sector with `fusion(I) == DegenerateNonAbelian` fusion wants to have custom vertex
# labels, a specialized method for `vertindex2label` should be added
"""
    vertex_ind2label(k::Int, a::I, b::I, c::I) where {I<:Sector}

Convert the index `k` of the fusion vertex (a,b)->c into a label. For
`FusionStyle(I) == Abelian()` or `FusionStyle(I) == NonAbelian()`, where every fusion
output occurs only once and `k == 1`, the default is to suppress vertex labels by setting
them equal to `nothing`. For `FusionStyle(I) == DegenerateNonAbelian()`, the default is to
just use `k`, unless a specialized method is provided.
"""
vertex_ind2label(k::Int, a::I, b::I, c::I) where {I<:Sector}=
    _ind2label(FusionStyle(I), k::Int, a::I, b::I, c::I)
_ind2label(::Abelian, k, a, b, c) = nothing
_ind2label(::SimpleNonAbelian, k, a, b, c) = nothing
_ind2label(::DegenerateNonAbelian, k, a, b, c) = k

"""
    vertex_labeltype(I::Type{<:Sector}) -> Type

Return the type of labels for the fusion vertices of sectors of type `I`.
"""
Base.@pure vertex_labeltype(I::Type{<:Sector}) =
    typeof(vertex_ind2label(1, one(I), one(I), one(I)))

# combine fusion properties of tensor products of sectors
Base.:&(f::F, ::F) where {F<:FusionStyle} = f
Base.:&(f1::FusionStyle, f2::FusionStyle) = f2 & f1

Base.:&(::SimpleNonAbelian, ::Abelian) = SimpleNonAbelian()
Base.:&(::DegenerateNonAbelian, ::Abelian) = DegenerateNonAbelian()
Base.:&(::DegenerateNonAbelian, ::SimpleNonAbelian) = DegenerateNonAbelian()

# properties that can be determined in terms of the F symbol
# TODO: find mechanism for returning these numbers with custom type T<:AbstractFloat
"""
    dim(a::Sector)

Return the (quantum) dimension of the sector `a`.
"""
function dim(a::Sector)
    if FusionStyle(a) isa Abelian
        1
    elseif FusionStyle(a) isa SimpleNonAbelian
        abs(1/Fsymbol(a, conj(a), a, a, one(a), one(a)))
    else
        abs(1/Fsymbol(a, conj(a), a, a, one(a), one(a))[1])
    end
end

"""
    frobeniusschur(a::Sector)

Return the Frobenius-Schur indicator of a sector `a`.
"""
function frobeniusschur(a::Sector)
    if FusionStyle(a) isa Abelian || FusionStyle(a) isa SimpleNonAbelian
        sign(Fsymbol(a, conj(a), a, a, one(a), one(a)))
    else
        sign(Fsymbol(a, conj(a), a, a, one(a), one(a))[1])
    end
end

"""
    twist(a::Sector)

Return the twist of a sector `a`
"""
function twist(a::Sector)
    if FusionStyle(a) isa Abelian || FusionStyle(a) isa SimpleNonAbelian
        θ = sum(dim(b)/dim(a)*Rsymbol(a, a, b) for b in a ⊗ a)
    else
        # TODO: is this correct?
        # θ = sum(dim(b)/dim(a)*tr(Rsymbol(a,a,b)) for b in a ⊗ a)
        throw(MethodError(twist, (a,)))
    end
    return θ
end


"""
    Bsymbol(a::I, b::I, c::I) where {I<:Sector}

Return the value of ``B^{ab}_c`` which appears in transforming a splitting vertex
into a fusion vertex using the transformation
```
a -<-μ-<- c                                                    a -<-ν-<- c
     ∨          -> √(dim(c)/dim(a)) * Bsymbol(a,b,c)[μ,ν]           ∧
     b                                                            dual(b)
```
If `FusionStyle(I)` is `Abelian()` or `SimpleNonAbelian()`, the B-symbol is a number.
Otherwise it is a square matrix with row and column size
`Nsymbol(a, b, c) == Nsymbol(c, dual(b), a)`.
"""
function Bsymbol(a::I, b::I, c::I) where {I<:Sector}
    if FusionStyle(I) isa Abelian || FusionStyle(I) isa SimpleNonAbelian
        sqrt(dim(a)*dim(b)/dim(c))*Fsymbol(a, b, dual(b), a, c, one(a))
    else
        reshape(sqrt(dim(a)*dim(b)/dim(c))*Fsymbol(a, b, dual(b), a, c, one(a)),
            (Nsymbol(a, b, c), Nsymbol(c, dual(b), a)))
    end
end

# Not necessary
function Asymbol(a::I, b::I, c::I) where {I<:Sector}
    if FusionStyle(I) isa Abelian || FusionStyle(I) isa SimpleNonAbelian
        sqrt(dim(a)*dim(b)/dim(c))*
            conj(frobeniusschur(a)*Fsymbol(dual(a), a, b, b, one(a), c))
    else
        reshape(sqrt(dim(a)*dim(b)/dim(c))*
                    conj(frobeniusschur(a)*Fsymbol(dual(a), a, b, b, one(a), c)),
                (Nsymbol(a, b, c), Nsymbol(dual(a), c, b)))
    end
end

# Braiding:
#-------------------------------------------------
# trait to describe type to denote how the elementary spaces in a tensor product space
# interact under permutations or actions of the braid group
abstract type BraidingStyle end # generic braiding
abstract type SymmetricBraiding <: BraidingStyle end # symmetric braiding => actions of permutation group are well defined
struct Bosonic <: SymmetricBraiding end # trivial under permutations
struct Fermionic <: SymmetricBraiding end
struct Anyonic <: BraidingStyle end

Base.:&(b::B, ::B) where {B<:BraidingStyle} = b
Base.:&(B1::BraidingStyle, B2::BraidingStyle) = B2 & B1

Base.:&(::Bosonic, ::Fermionic) = Fermionic()
Base.:&(::Bosonic, ::Anyonic) = Anyonic()
Base.:&(::Fermionic, ::Anyonic) = Anyonic()

"""
    BraidingStyle(::Sector) -> ::BraidingStyle
    BraidingStyle(I::Type{<:Sector}) -> ::BraidingStyle

Return the type of braiding and twist behavior of sectors of type `I`, which can be either
*   `Bosonic()`: symmetric braiding with trivial twist (i.e. identity)
*   `Fermionic()`: symmetric braiding with non-trivial twist (squares to identity)
*   `Anyonic()`: general ``R_(a,b)^c`` phase or matrix (depending on `SimpleNonAbelian` or
    `DegenerateNonAbelian` fusion) and arbitrary twists

Note that `Bosonic` and `Fermionic` are subtypes of `SymmetricBraiding`, which means that
braids are in fact equivalent to crossings (i.e. braiding twice is an identity:
`isone(Rsymbol(b,a,c)*Rsymbol(a,b,c)) == true`) and permutations are uniquely defined.
"""
BraidingStyle(a::Sector) = BraidingStyle(typeof(a))
BraidingStyle(::Type{Trivial}) = Bosonic()

# SectorSet:
#-------------------------------------------------------------------------------
# Custum generator to represent sets of sectors with type inference
struct SectorSet{I<:Sector, F, S}
    f::F
    set::S
end
SectorSet{I}(::Type{F}, set::S) where {I<:Sector, F, S} = SectorSet{I, Type{F}, S}(F, set)
SectorSet{I}(f::F, set::S) where {I<:Sector, F, S} = SectorSet{I, F, S}(f, set)
SectorSet{I}(set) where {I<:Sector} = SectorSet{I}(identity, set)

Base.IteratorEltype(::Type{<:SectorSet}) = HasEltype()
Base.IteratorSize(::Type{SectorSet{I, F, S}}) where {I<:Sector, F, S} = Base.IteratorSize(S)

Base.eltype(::SectorSet{I}) where {I<:Sector} = I
Base.length(s::SectorSet) = length(s.set)
Base.size(s::SectorSet) = size(s.set)

function Base.iterate(s::SectorSet{I}, args...) where {I<:Sector}
    next = iterate(s.set, args...)
    next === nothing && return nothing
    val, state = next
    return convert(I, s.f(val)), state
end

# possible sectors
include("groups.jl")
include("irreps.jl") # irreps of symmetry groups, with bosonic braiding
# include("fermions.jl") # irreps with defined fermionparity and fermionic braiding
include("anyons.jl") # non-group sectors
include("product.jl") # direct product of different sectors
