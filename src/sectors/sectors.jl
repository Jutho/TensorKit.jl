# Superselection sectors (quantum numbers):
# for defining graded vector spaces and invariant subspaces of tensor products
#==============================================================================#
"""
    abstract type Sector end

Abstract type for representing the label sets of graded vector spaces, which should
correspond to (unitary) fusion categories.

Every new `G<:Sector` should implement the following methods:
*   `one(::Type{G})` -> unit element of `G`
*   `conj(a::G)` -> a̅: conjugate or dual label of a
*   `⊗(a::G, b::G)` -> iterable with unique fusion outputs of `a ⊗ b`
    (i.e. don't repeat in case of multiplicities)
*   `Nsymbol(a::G, b::G, c::G)` -> number of times `c` appears in `a ⊗ b`, i.e. the
    multiplicity
*   `FusionStyle(::Type{G})` -> `Abelian()`, `SimpleNonAbelian()` or
    `DegenerateNonAbelian()`
*   `BraidingStyle(::Type{G})` -> `Bosonic()`, `Fermionic()`, `Anyonic()`, ...
and, if `FusionStyle(G) == NonAbelian()`,
*   `Fsymbol(a::G, b::G, c::G, d::G, e::G, f::G)` -> F-symbol: scalar (in case of
    `SimpleNonAbelian`) or matrix (in case of `DegenerateNonAbelian`)
*   ... can all other information (quantum dimension, cups and caps) be extracted from `F`?
and if `BraidingStyle(G) == Fermionic()`
*   `fermionparity(a::G)` -> `Bool` representing the fermion parity of sector `a`
    and optionally, if if `FusionStyle(G) isa DegenerateNonAbelian`
*   `vertex_ind2label(i::Int, a::G, b::G, c::G)` -> a custom label for the `i`th copy of `c` appearing in `a ⊗ b`
"""
abstract type Sector end

# Define a sector for ungraded vector spaces
struct Trivial <: Sector
end
Base.show(io::IO, ::Trivial) = print(io, "Trivial()")
Base.show(io::IO, ::Type{Trivial}) = print(io, "Trivial")

"""
    function one(::Sector) -> Sector
    function one(::Type{<:Sector}) -> Sector

Return the unit element within this type of sector.
"""
Base.one(a::Sector) = one(typeof(a))
Base.one(::Type{Trivial}) = Trivial()

"""
    function dual(a::Sector) -> Sector

Return the conjugate label `conj(a)`.
"""
dual(a::Sector) = conj(a)
Base.conj(::Trivial) = Trivial()

# FusionStyle: the most important aspect of Sector
#---------------------------------------------
"""
    function ⊗(a::G, b::G) where {G<:Sector}

Return an iterable of elements of `c::G` that appear in the fusion product `a ⊗ b`.

Note that every element `c` should appear at most once, fusion degeneracies (if
`FusionStyle(G) == DegenerateNonAbelian()`) should be accessed via `Nsymbol(a,b,c)`.
"""
⊗(::Trivial, ::Trivial) = (Trivial(),)

"""
    function Nsymbol(a::G, b::G, c::G) where {G<:Sector} -> Integer

Return an `Integer` representing the number of times `c` appears in the fusion product
`a ⊗ b`. Could be a `Bool` if `FusionStyle(G) == Abelian()` or `SimpleNonAbelian()`.
"""
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
    function FusionStyle(G::Type{<:Sector}) -> ::FusionStyle

Return the type of fusion behavior of sectors of type G, which can be either
*   `Abelian()`: single fusion output when fusing two sectors;
*   `SimpleNonAbelian()`: multiple outputs, but every output occurs at most one,
    also known as multiplicity free (e.g. irreps of SU(2));
*   `DegenerateNonAbelian()`: multiple outputs that can occur more than once (e.g. irreps
    of SU(3)).
"""
FusionStyle(::Type{Trivial}) = Abelian()
FusionStyle(a::Sector) = FusionStyle(typeof(a))

# NOTE: the following inline is extremely important for performance, especially
# in the case of Abelian, because ⊗(...) is computed very often
@inline function ⊗(a::G, b::G, c::G, rest::Vararg{G}) where {G<:Sector}
    if FusionStyle(G) isa Abelian
        return a ⊗ first(⊗(b, c, rest...))
    else
        s = Set{G}()
        for d in ⊗(b, c, rest...)
            for e in a ⊗ d
                push!(s, e)
            end
        end
        return s
    end
end

"""
    function Fsymbol(a::G, b::G, c::G, d::G, e::G, f::G) where {G<:Sector}

Return the F-symbol F^{a,b,c}_d that associates the two different fusion orders of sectors
`a`, `b` and `c` into an ouput sector `d`, using either an intermediate sector `a ⊗ b → e`
or `b ⊗ c → f`:
```
a-<-μ-<-e-<-ν-<-d                                     a-<-λ-<-d
    ∨       ∨       -> Fsymbol(a,b,c,d,e,f)[μ,ν,κ,λ]      ∨
    b       c                                         b-<-κ
                                                          ∨
                                                          c
```
If `FusionStyle(G)` is `Abelian` or `SimpleNonAbelian`, the F-symbol is a number. Otherwise
it is a rank 4 array of size
`(Nsymbol(a,b,e), Nsymbol(e,c,d), Nsymbol(b,c,f), Nsymbol(a,f,d))`.
"""
function Fsymbol end
Fsymbol(::Trivial, ::Trivial, ::Trivial, ::Trivial, ::Trivial, ::Trivial) = 1

"""
    function Rsymbol(a::G, b::G, c::G) where {G<:Sector}

Returns the R-symbol R^{c}_{a,b} that maps between `a ⊗ b → c` and `b ⊗ a → c` as in
```
a -<-μ-<- c                                 b -<-ν-<- c
     ∨          -> Rsymbol(a,b,c)[μ,ν]           ∧
     b                                           a
```
If `FusionStyle(G)` is `Abelian()` or `SimpleNonAbelian()`, the R-symbol is a number.
Otherwise it is a square matrix with row and column size `Nsymbol(a,b,c) == Nsymbol(b,a,c)`.
"""
function Rsymbol end
Rsymbol(::Trivial, ::Trivial, ::Trivial) = 1

# If a G::Sector with `fusion(G) == DegenerateNonAbelian` fusion wants to have custom vertex
# labels, a specialized method for `vertindex2label` should be added
"""
    function vertex_ind2label(i::Int, a::G, b::G, c::G) where {G<:Sector}

Convert the index i of the fusion vertex (a,b)->c into a label. For
`FusionStyle(G) == Abelian()` or `FusionStyle(G) == NonAbelian()`, where every fusion
output occurs only once and `i == 1`, the default is to suppress vertex labels by setting
them equal to `nothing`. For `FusionStyle(G) == DegenerateNonAbelian()`, the default is to
just use `i`, unless a specialized method is provided.
"""
vertex_ind2label(i::Int, s1::G, s2::G, sout::G) where {G<:Sector}=
    _ind2label(FusionStyle(G), i::Int, s1::G, s2::G, sout::G)
_ind2label(::Abelian, i, s1, s2, sout) = nothing
_ind2label(::SimpleNonAbelian, i, s1, s2, sout) = nothing
_ind2label(::DegenerateNonAbelian, i, s1, s2, sout) = i

"""
    function vertex_labeltype(G::Type{<:Sector}) -> Type

Return the type of labels for the fusion vertices of sectors of type `G`.
"""
Base.@pure vertex_labeltype(G::Type{<:Sector}) =
    typeof(vertex_ind2label(1, one(G), one(G), one(G)))

# combine fusion properties of tensor products of sectors
Base.:&(f::F, ::F) where {F<:FusionStyle} = f
Base.:&(f1::FusionStyle, f2::FusionStyle) = f2 & f1

Base.:&(::SimpleNonAbelian, ::Abelian) = SimpleNonAbelian()
Base.:&(::DegenerateNonAbelian, ::Abelian) = DegenerateNonAbelian()
Base.:&(::DegenerateNonAbelian, ::SimpleNonAbelian) = DegenerateNonAbelian()

# properties that can be determined in terms of the F symbol
# TODO: find mechanism for returning these numbers with custom type T<:AbstractFloat
"""
    function dim(a::Sector)

Return the (quantum) dimension of the sector `a`.
"""
function dim(a::Sector)
    if FusionStyle(a) isa Abelian
        1
    elseif FusionStyle(a) isa SimpleNonAbelian
        abs(1/Fsymbol(a,conj(a),a,a,one(a),one(a)))
    else
        abs(1/Fsymbol(a,conj(a),a,a,one(a),one(a))[1])
    end
end

"""
    function frobeniusschur(a::Sector)

Return the Frobenius-Schur indicator of a sector `a`.
"""
function frobeniusschur(a::Sector)
    if FusionStyle(a) isa Abelian || FusionStyle(a) isa SimpleNonAbelian
        sign(Fsymbol(a,conj(a),a,a,one(a),one(a)))
    else
        sign(Fsymbol(a,conj(a),a,a,one(a),one(a))[1])
    end
end

"""
    function Bsymbol(a::G, b::G, c::G) where {G<:Sector}

Return the value of B^{a,b}_c which appears in transforming a splitting vertex
into a fusion vertex using the transformation
```
a -<-μ-<- c                                 a -<-ν-<- c
     ∨          -> Bsymbol(a,b,c)[μ,ν]           ∧
     b                                         dual(b)
```
If `FusionStyle(G)` is `Abelian()` or `SimpleNonAbelian()`, the B-symbol is a number.
Otherwise it is a square matrix with row and column size
`Nsymbol(a, b, c) == Nsymbol(c, dual(b), a)`.
"""
function Bsymbol(a::G, b::G, c::G) where {G<:Sector}
    if FusionStyle(G) isa Abelian || FusionStyle(G) isa SimpleNonAbelian
        Fsymbol(a, b, dual(b), a, c, one(a))
    else
        reshape(Fsymbol(a,b,dual(b),a,c,one(a)), (Nsymbol(a,b,c), Nsymbol(c,dual(b),a)))
    end
end
# isotopic normalization convention
# _Bsymbol(a,b,c, ::Type{SimpleNonAbelian}) =
#     sign(sqrt(dim(a)*dim(b)/dim(c))*Fsymbol(a, b, dual(b), a, c, one(a)))
#     # sign enforces this to be a pure phase (abs=1)
# _Bsymbol(a,b,c, ::Type{DegenerateNonAbelian}) =
#     sqrt(dim(a)*dim(b)/dim(c))*
#     reshape(Fsymbol(a,b,dual(b),a,c,one(a)), (Nsymbol(a,b,c), Nsymbol(c,dual(b),a)))

# Not necessary
function Asymbol(a::G, b::G, c::G) where {G<:Sector}
    if FusionStyle(G) isa Abelian || FusionStyle(G) isa SimpleNonAbelian
        conj(frobeniusschur(a)*Fsymbol(dual(a),a,b,b,one(a),c))
    else
        reshape(conj(frobeniusschur(a)*Fsymbol(dual(a),a,b,b,one(a),c)),
                (Nsymbol(a,b,c), Nsymbol(dual(a),c,b)))
    end
end
# isotopic normalization convention
# _Asymbol(a,b,c, ::Type{SimpleNonAbelian}) =
#     sqrt(dim(a)*dim(b)/dim(c))*conj(frobeniusschur(a)*Fsymbol(dual(a),a,b,b,one(a),c))
# _Asymbol(a,b,c, ::Type{DegenerateNonAbelian}) =
#     sqrt(dim(a)*dim(b)/dim(c))*
#     reshape(conj(frobeniusschur(a)*Fsymbol(dual(a),a,b,b,one(a),c)),
#             (Nsymbol(a,b,c), Nsymbol(dual(a),c,b)))

# Braiding:
#-------------------------------------------------
# trait to describe type to denote how the elementary spaces in a tensor product space
# interact under permutations or actions of the braid group
abstract type BraidingStyle end # generic braiding
abstract type SymmetricBraiding <: BraidingStyle end # symmetric braiding => actions of permutation group are well defined
struct Bosonic <: SymmetricBraiding end # trivial under permutations
struct Fermionic <: SymmetricBraiding end
struct Anyonic <: BraidingStyle end

Base.:&(b::B,::B) where {B<:BraidingStyle} = b
Base.:&(B1::BraidingStyle, B2::BraidingStyle) = B2 & B1

Base.:&(::Bosonic,::Fermionic) = Fermionic()
Base.:&(::Bosonic,::Anyonic) = Anyonic()
Base.:&(::Fermionic,::Anyonic) = Anyonic()

"""
    function BraidingStyle(G::Type{<:Sector}) -> ::BraidingStyle

Return the type of braiding behavior of sectors of type G, which can be either
*   `Bosonic()`: trivial exchange
*   `Fermionic()`: fermionic exchange depending on `fermionparity`
*   `Anyonic()`: requires general R_(a,b)^c phase or matrix (depending on `SimpleNonAbelian` or `DegenerateNonAbelian` fusion)

Note that `Bosonic` and `Fermionic` are subtypes of `SymmetricBraiding`, which means that
braids are in fact equivalent to crossings (i.e. braiding twice is an identity:
`Rsymbol(b,a,c)*Rsymbol(a,b,c) = I`) and permutations are uniquely defined.
"""
BraidingStyle(::Type{Trivial}) = Bosonic()
BraidingStyle(a::Sector) = BraidingStyle(typeof(a))

# SectorSet:
#-------------------------------------------------------------------------------
# Custum generator to represent sets of sectors with type inference
struct SectorSet{G<:Sector,F,S}
    f::F
    set::S
end
SectorSet{G}(::Type{F}, set::S) where {G<:Sector,F,S} = SectorSet{G,Type{F},S}(F, set)
SectorSet{G}(f::F, set::S) where {G<:Sector,F,S} = SectorSet{G,F,S}(f, set)
SectorSet{G}(set) where {G<:Sector} = SectorSet{G}(identity, set)

Base.IteratorEltype(::Type{<:SectorSet}) = HasEltype()
Base.IteratorSize(::Type{SectorSet{G,F,S}}) where {G<:Sector,F,S} = Base.IteratorSize(S)

Base.eltype(::SectorSet{G}) where {G<:Sector} = G
Base.length(s::SectorSet) = length(s.set)
Base.size(s::SectorSet) = size(s.set)

function Base.iterate(s::SectorSet{G}, args...) where {G<:Sector}
    next = iterate(s.set, args...)
    next === nothing && return nothing
    val, state = next
    return convert(G, s.f(val)), state
end

# possible sectors
include("irreps.jl") # irreps of symmetry groups, with bosonic braiding
# include("fermions.jl") # irreps with defined fermionparity and fermionic braiding
include("product.jl") # direct product of different sectors
