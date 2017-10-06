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
*   `⊗(a::G, b::G)` -> iterable with fusion outputs of `a ⊗ b`
*   `Nsymbol(a::G, b::G, c::G)` -> number of times `c` appears in `a ⊗ b`
*   `fusiontype(::Type{G})` -> `Abelian`, `SimpleNonAbelian` or `DegenerateNonAbelian`
*   `braidingtype(::Type{G})` -> `Bosonic`, `Fermionic`, `Anyonic`, ...
and, if `fusiontype(G) <: NonAbelian`,
*   `Fsymbol(a::G, b::G, c::G, d::G, e::G, f::G)` -> F-symbol: scalar (in case of `SimpleNonAbelian`) or matrix (in case of `DegenerateNonAbelian`)
*   ... can all other information (quantum dimension, cups and caps) be extracted from `F`?
and if `braidingtype(G) == Fermionic`
*   `fermionparity(a::G)` -> `Bool` representing the fermion parity of sector `a`
and optionally, if if `fusiontype(G) == DegenerateNonAbelian`
*   `vertex_ind2label(i::Int, a::G, b::G, c::G)` -> a custom label for the `i`th copy of `c` appearing in `a ⊗ b`
"""
abstract type Sector end

# Define a sector for ungraded vector spaces
struct Trivial <: Sector
end

"""
    function one(::Sector) -> Sector
    function one(::Type{<:Sector}) -> Sector

Returns the unit element within this type of sector
"""
Base.one(a::Sector) = one(typeof(a))
Base.one(::Type{Trivial}) = Trivial()

"""
    function dual(a::Sector) -> Sector

Returns the conjugate label `conj(a)`
"""
dual(a::Sector) = conj(a)
dual(::Trivial) = Trivial()

# Fusion: the most important aspect of Sector
#---------------------------------------------
"""
    function ⊗(a::G, b::G) where {G<:Sector}

Returns an iterable of elements of `c::G` that appear in the fusion product `a ⊗ b`.
Note that every element `c` should appear at most once, fusion degeneracies (if
`fusiontype(G) == DegenerateNonAbelian`) should be accessed via `Nsymbol(a,b,c)`.
"""
⊗(::Trivial, ::Trivial) = (Trivial(),)

"""
    function Nsymbol(a::G, b::G, c::G) where {G<:Sector} -> Integer

Returns an `Integer` representing the number of times `c` appears in the fusion
product `a ⊗ b`. Could be a `Bool` if `fusiontype(G) = Abelian` or `SimpleNonAbelian`.
"""
Nsymbol(::Trivial, ::Trivial, ::Trivial) = true

# trait to describe the fusion of superselection sectors
abstract type Fusion end
struct Abelian <: Fusion
end
abstract type NonAbelian <: Fusion end
struct SimpleNonAbelian <: NonAbelian # non-abelian fusion but multiplicity free
end
struct DegenerateNonAbelian <: NonAbelian # non-abelian fusion with multiplicities
end

"""
    function fusiontype(G::Type{<:Sector}) -> Type{<:Fusion}

Returns the type of fusion behavior of sectors of type G, which can be either
*   `Abelian`: single fusion output when fusing two sectors;
*   `SimpleNonAbelian`: multiple outputs, but every output occurs at most one, also known as multiplicity free (e.g. irreps of SU(2));
*   `DegenerateNonAbelian`: multiple outputs that can occur more than once (e.g. irreps of SU(3)).
"""
fusiontype(::Type{Trivial}) = Abelian
fusiontype(a::Sector) = fusiontype(typeof(a))

function ⊗(a::G, b::G, c::G, rest::Vararg{G}) where {G<:Sector}
    if fusiontype(G) == Abelian
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

Returns the F-symbol F^{a,b,c}_d that associates the two different fusion orders
of sectors `a`, `b` and `c` into an ouput sector `d`, using either an intermediate
sector `a ⊗ b → e` or `b ⊗ c → f`:
```
a-<-μ-<-e-<-ν-<-d                                     a-<-λ-<-d
    ∨       ∨       -> Fsymbol(a,b,c,d,e,f)[μ,ν,κ,λ]      ∨
    b       c                                         b-<-κ
                                                          ∨
                                                          c
```
If `fusiontype(G)` is `Abelian` or `SimpleNonAbelian`, the F-symbol is a number.
Otherwise it is a rank 4 array of size `(Nsymbol(a,b,e), Nsymbol(e,c,d), Nsymbol(b,c,f), Nsymbol(a,f,d))`.
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
If `fusiontype(G)` is `Abelian` or `SimpleNonAbelian`, the R-symbol is a number.
Otherwise it is a square matrix with row and column size `Nsymbol(a,b,c) == Nsymbol(b,a,c)`.
"""
function Rsymbol end
Rsymbol(::Trivial, ::Trivial, ::Trivial) = 1

# If a G::Sector with `fusion(G) == DegenerateNonAbelian` fusion wants to have custom vertex
# labels, a specialized method for `vertindex2label` should be added
"""
    function vertex_ind2label(i::Int, a::G, b::G, c::G) where {G<:Sector}

Convert the index i of the fusion vertex (a,b)->c into a label. For `fusiontype(G)==Abelian`
or `fusiontype(G)==NonAbelian`, where every fusion output occurs only once and `i===1`, the
default is to suppress vertex labels by setting them equal to `nothing`. For
`fusiontype(G)==DegenerateNonAbelian`, the default is to just use `i`, unless a specialized
method is provided.
"""
vertex_ind2label(i::Int, s1::G, s2::G, sout::G) where {G<:Sector}= _ind2label(fusiontype(G), i::Int, s1::G, s2::G, sout::G)
_ind2label(::Type{Abelian}, i, s1, s2, sout) = nothing
_ind2label(::Type{SimpleNonAbelian}, i, s1, s2, sout) = nothing
_ind2label(::Type{DegenerateNonAbelian}, i, s1, s2, sout) = i

"""
    function vertex_labeltype(G::Type{<:Sector}) -> Type

Returns the type of labels for the fusion vertices of sectors of type `G`.
"""
Base.@pure vertex_labeltype(G::Type{<:Sector}) = typeof(vertex_ind2label(1, one(G), one(G), one(G)))

# combine fusion properties of tensor products of sectors
Base.:&(::Type{F},::Type{F}) where {F<:Fusion} = F
Base.:&(F1::Type{<:Fusion},F2::Type{<:Fusion}) = F2 & F1

Base.:&(::Type{SimpleNonAbelian},::Type{Abelian}) = SimpleNonAbelian
Base.:&(::Type{DegenerateNonAbelian},::Type{Abelian}) = DegenerateNonAbelian
Base.:&(::Type{DegenerateNonAbelian},::Type{SimpleNonAbelian}) = DegenerateNonAbelian

# properties that can be determined in terms of the F symbol
# TODO: find mechanism for returning these numbers with custom type T<:AbstractFloat
"""
    function dim(a::Sector)

Returns the (quantum) dimension of the sector `a`.
"""
function dim(a::Sector)
    if fusiontype(a) == Abelian
        1
    elseif fusiontype(a) == SimpleNonAbelian
        abs(1/Fsymbol(a,conj(a),a,a,one(a),one(a)))
    else
        abs(1/Fsymbol(a,conj(a),a,a,one(a),one(a))[1])
    end
end

"""
    function frobeniusschur(a::Sector)

Returns the Frobenius-Schur indicator of a sector `a`.
"""
function frobeniusschur(a::Sector)
    if fusiontype(a) == Abelian || fusiontype(a) == SimpleNonAbelian
        sign(Fsymbol(a,conj(a),a,a,one(a),one(a)))
    else
        sign(Fsymbol(a,conj(a),a,a,one(a),one(a))[1])
    end
end

"""
    function Bsymbol(a::G, b::G, c::G) where {G<:Sector}

Returns the value of B^{a,b}_c which appears in transforming a splitting vertex
into a fusion vertex using the transformation
```
a -<-μ-<- c                                 a -<-ν-<- c
     ∨          -> Bsymbol(a,b,c)[μ,ν]           ∧
     b                                         dual(b)
```
If `fusiontype(G)` is `Abelian` or `SimpleNonAbelian`, the B-symbol is a number.
Otherwise it is a square matrix with row and column size
`Nsymbol(a, b, c) == Nsymbol(c, dual(b), a)`.
"""
function Bsymbol(a::G, b::G, c::G) where {G<:Sector}
    if fusiontype(G) == Abelian || fusiontype(G) == SimpleNonAbelian
        Fsymbol(a, b, dual(b), a, c, one(a))
    else
        reshape(Fsymbol(a,b,dual(b),a,c,one(a)), (Nsymbol(a,b,c), Nsymbol(c,dual(b),a)))
    end
end
# isotopic normalization convention
# _Bsymbol(a,b,c, ::Type{SimpleNonAbelian}) = sign(sqrt(dim(a)*dim(b)/dim(c))*Fsymbol(a, b, dual(b), a, c, one(a))) # sign enforces this to be a pure phase (abs=1)
# _Bsymbol(a,b,c, ::Type{DegenerateNonAbelian}) = sqrt(dim(a)*dim(b)/dim(c))*reshape(Fsymbol(a,b,dual(b),a,c,one(a)), (Nsymbol(a,b,c), Nsymbol(c,dual(b),a)))


# Not necessary
function Asymbol(a::G, b::G, c::G) where {G<:Sector}
    if fusiontype(G) == Abelian || fusiontype(G) == SimpleNonAbelian
        conj(frobeniusschur(a)*Fsymbol(dual(a),a,b,b,one(a),c))
    else
        reshape(conj(frobeniusschur(a)*Fsymbol(dual(a),a,b,b,one(a),c)), (Nsymbol(a,b,c), Nsymbol(dual(a),c,b)))
    end
end
# isotopic normalization convention
# _Asymbol(a,b,c, ::Type{SimpleNonAbelian}) = sqrt(dim(a)*dim(b)/dim(c))*conj(frobeniusschur(a)*Fsymbol(dual(a),a,b,b,one(a),c))
# _Asymbol(a,b,c, ::Type{DegenerateNonAbelian}) = sqrt(dim(a)*dim(b)/dim(c))*reshape(conj(frobeniusschur(a)*Fsymbol(dual(a),a,b,b,one(a),c)), (Nsymbol(a,b,c), Nsymbol(dual(a),c,b)))

# Braiding:
#-------------------------------------------------
# trait to describe type to denote how the elementary spaces in a tensor product space
# interact under permutations or actions of the braid group
abstract type Braiding end # generic braiding
abstract type SymmetricBraiding <: Braiding end # symmetric braiding => actions of permutation group are well defined
struct Bosonic <: SymmetricBraiding end # trivial under permutations
struct Fermionic <: SymmetricBraiding end
struct Anyonic <: Braiding end

Base.:&(::Type{B},::Type{B}) where {B<:Braiding} = B
Base.:&(B1::Type{<:Braiding},B2::Type{<:Braiding}) = B2 & B1

Base.:&(::Type{Bosonic},::Type{Fermionic}) = Fermionic
Base.:&(::Type{Bosonic},::Type{Anyonic}) = Anyonic
Base.:&(::Type{Fermionic},::Type{Anyonic}) = Anyonic

"""
    function braidingtype(G::Type{<:Sector}) -> Type{<:Braiding}

Returns the type of braiding behavior of sectors of type G, which can be either
*   `Bosonic`: trivial exchange
*   `Fermionic`: fermionic exchange depending on `fermionparity`
*   `Anyonic`: requires general R_(a,b)^c phase or matrix (depending on `SimpleNonAbelian` or `DegenerateNonAbelian` fusion)

Note that `Bosonic` and `Fermionic` are subtypes of `SymmetricBraiding`, which means that
braids are in fact equivalent to crossings (i.e. braiding twice is an identity:
`Rsymbol(b,a,c)*Rsymbol(a,b,c) = I`) and permutations are uniquely defined.
"""
braidingtype(::Type{Trivial}) = Bosonic
braidingtype(a::Sector) = braidingtype(typeof(a))

# SectorSet:
#-------------------------------------------------------------------------------
# wrapper type to represent a subset of sectors, e.g. as output of a fusion of
# two sectors. It is used as iterator but should be a set, i.e. unique elements
struct SectorSet{G<:Sector,S}
    set::S
end
SectorSet{T}(set::S) where {T<:Sector,S} = SectorSet{T,S}(set)

Base.iteratoreltype(::Type{<:SectorSet}) = Base.HasEltype()
Base.eltype(::SectorSet{G}) where {G<:Sector} = G
Base.iteratorsize(::Type{SectorSet{G,S}}) where {G<:Sector,S} = Base.iteratorsize(S)
Base.length(s::SectorSet) = length(s.set)
Base.size(s::SectorSet) = size(s.set)

Base.start(s::SectorSet) = start(s.set)
function Base.next(s::SectorSet{G}, state) where {G<:Sector}
    v, nextstate = next(s.set, state)
    return G(v), nextstate
end
Base.done(s::SectorSet, state) = done(s.set, state)

# possible sectors
include("irreps.jl") # irreps of symmetry groups, with bosonic braiding
include("fermions.jl") # irreps of symmetry groups, with defined fermionparity and fermionic braiding
include("product.jl") # direct product of different sectors
