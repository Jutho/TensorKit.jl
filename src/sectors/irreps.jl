# Sectors corresponding to irreducible representations of compact groups
#==============================================================================#
# Irreps of groups
#------------------------------------------------------------------------------#
"""
    abstract type AbstractIrrep{G<:Group} <: Sector end

Abstract supertype for sectors which corresponds to irreps (irreducible representations) of
a group `G`. As we assume unitary representations, these would be finite groups or compact
Lie groups. Note that this could also include projective rather than linear representations.

Actual concrete implementations of those irreps can be obtained as `Irrep[G]`, or via their
actual name, which generically takes the form `(asciiG)Irrep`, i.e. the ASCII spelling of
the group name followed by `Irrep`.

All irreps have [`BraidingStyle`](@ref) equal to `Bosonic()` and thus trivial twists.
"""
abstract type AbstractIrrep{G<:Group} <: Sector end # irreps have integer quantum dimensions
BraidingStyle(::Type{<:AbstractIrrep}) = Bosonic()

struct IrrepTable end
"""
    const Irrep

A constant of a singleton type used as `Irrep[G]` with `G<:Group` a type of group, to
construct or obtain a concrete subtype of `AbstractIrrep{G}` that implements the data
structure used to represent irreducible representations of the group `G`.
"""
const Irrep = IrrepTable()

type_repr(::Type{<:AbstractIrrep{G}}) where {G<:Group} = "Irrep[" * type_repr(G) * "]"
function Base.show(io::IO, c::AbstractIrrep)
    I = typeof(c)
    if get(io, :typeinfo, nothing) !== I
        print(io, type_repr(I), "(")
        for k in 1:fieldcount(I)
            k > 1 && print(io, ", ")
            print(io, getfield(c, k))
        end
        print(io, ")")
    else
        fieldcount(I) > 1 && print(io, "(")
        for k in 1:fieldcount(I)
            k > 1 && print(io, ", ")
            print(io, getfield(c, k))
        end
        fieldcount(I) > 1 && print(io, ")")
    end
end

const AbelianIrrep{G} = AbstractIrrep{G} where {G<:AbelianGroup}
FusionStyle(::Type{<:AbelianIrrep}) = UniqueFusion()
Base.isreal(::Type{<:AbelianIrrep}) = true

Nsymbol(a::I, b::I, c::I) where {I<:AbelianIrrep} = c == first(a ⊗ b)
function Fsymbol(a::I, b::I, c::I, d::I, e::I, f::I) where {I<:AbelianIrrep}
    return Int(Nsymbol(a, b, e) * Nsymbol(e, c, d) * Nsymbol(b, c, f) * Nsymbol(a, f, d))
end
frobeniusschur(a::AbelianIrrep) = 1
Asymbol(a::I, b::I, c::I) where {I<:AbelianIrrep} = Int(Nsymbol(a, b, c))
Bsymbol(a::I, b::I, c::I) where {I<:AbelianIrrep} = Int(Nsymbol(a, b, c))
Rsymbol(a::I, b::I, c::I) where {I<:AbelianIrrep} = Int(Nsymbol(a, b, c))

function fusiontensor(a::I, b::I, c::I) where {I<:AbelianIrrep}
    return fill(Int(Nsymbol(a, b, c)), (1, 1, 1, 1))
end

# ZNIrrep: irreps of Z_N are labelled by integers mod N; do we ever want N > 64?
"""
    struct ZNIrrep{N} <: AbstractIrrep{ℤ{N}}
    ZNIrrep{N}(n::Integer)
    Irrep[ℤ{N}](n::Integer)

Represents irreps of the group ``ℤ_N`` for some value of `N<64`. (We need 2*(N-1) <= 127 in
order for a ⊗ b to work correctly.) For `N` equals `2`, `3` or `4`, `ℤ{N}` can be replaced
by `ℤ₂`, `ℤ₃`, `ℤ₄`. An arbitrary `Integer` `n` can be provided to the constructor, but only
the value `mod(n, N)` is relevant.

## Fields
- `n::Int8`: the integer label of the irrep, modulo `N`.
"""
struct ZNIrrep{N} <: AbstractIrrep{ℤ{N}}
    n::Int8
    function ZNIrrep{N}(n::Integer) where {N}
        @assert N < 64
        return new{N}(mod(n, N))
    end
end
Base.getindex(::IrrepTable, ::Type{ℤ{N}}) where {N} = ZNIrrep{N}
Base.convert(Z::Type{<:ZNIrrep}, n::Real) = Z(n)
const Z2Irrep = ZNIrrep{2}
const Z3Irrep = ZNIrrep{3}
const Z4Irrep = ZNIrrep{4}

Base.one(::Type{ZNIrrep{N}}) where {N} = ZNIrrep{N}(0)
Base.conj(c::ZNIrrep{N}) where {N} = ZNIrrep{N}(-c.n)
⊗(c1::ZNIrrep{N}, c2::ZNIrrep{N}) where {N} = (ZNIrrep{N}(c1.n + c2.n),)

Base.IteratorSize(::Type{SectorValues{ZNIrrep{N}}}) where {N} = HasLength()
Base.length(::SectorValues{ZNIrrep{N}}) where {N} = N
function Base.iterate(::SectorValues{ZNIrrep{N}}, i=0) where {N}
    return i == N ? nothing : (ZNIrrep{N}(i), i + 1)
end
function Base.getindex(::SectorValues{ZNIrrep{N}}, i::Int) where {N}
    return 1 <= i <= N ? ZNIrrep{N}(i - 1) : throw(BoundsError(values(ZNIrrep{N}), i))
end
findindex(::SectorValues{ZNIrrep{N}}, c::ZNIrrep{N}) where {N} = c.n + 1

Base.hash(c::ZNIrrep{N}, h::UInt) where {N} = hash(c.n, h)
Base.isless(c1::ZNIrrep{N}, c2::ZNIrrep{N}) where {N} = isless(c1.n, c2.n)

# U1Irrep: irreps of U1 are labelled by integers
"""
    struct U1Irrep <: AbstractIrrep{U₁}
    U1Irrep(charge::Real)
    Irrep[U₁](charge::Real)

Represents irreps of the group ``U₁``. The irrep is labelled by a charge, which should be
an integer for a linear representation. However, it is often useful to allow half integers
to represent irreps of ``U₁`` subgroups of ``SU₂``, such as the Sz of spin-1/2 system.
Hence, the charge is stored as a `HalfInt` from the package HalfIntegers.jl, but can be
entered as arbitrary `Real`. The sequence of the charges is: 0, 1/2, -1/2, 1, -1, ...

## Fields
- `charge::HalfInt`: the label of the irrep, which can be any half integer.
"""
struct U1Irrep <: AbstractIrrep{U₁}
    charge::HalfInt
end
Base.getindex(::IrrepTable, ::Type{U₁}) = U1Irrep
Base.convert(::Type{U1Irrep}, c::Real) = U1Irrep(c)

Base.one(::Type{U1Irrep}) = U1Irrep(0)
Base.conj(c::U1Irrep) = U1Irrep(-c.charge)
⊗(c1::U1Irrep, c2::U1Irrep) = (U1Irrep(c1.charge + c2.charge),)

Base.IteratorSize(::Type{SectorValues{U1Irrep}}) = IsInfinite()
function Base.iterate(::SectorValues{U1Irrep}, i=0)
    return i <= 0 ? (U1Irrep(half(i)), (-i + 1)) : (U1Irrep(half(i)), -i)
end
function Base.getindex(::SectorValues{U1Irrep}, i::Int)
    i < 1 && throw(BoundsError(values(U1Irrep), i))
    return U1Irrep(iseven(i) ? half(i >> 1) : -half(i >> 1))
end
function findindex(::SectorValues{U1Irrep}, c::U1Irrep)
    return (n = twice(c.charge); 2 * abs(n) + (n <= 0))
end

Base.hash(c::U1Irrep, h::UInt) = hash(c.charge, h)
@inline function Base.isless(c1::U1Irrep, c2::U1Irrep)
    return isless(abs(c1.charge), abs(c2.charge)) || zero(HalfInt) < c1.charge == -c2.charge
end

# Non-abelian groups
#------------------------------------------------------------------------------#
# SU2Irrep: irreps of SU2 are labelled by half integers j
struct SU2IrrepException <: Exception end
function Base.show(io::IO, ::SU2IrrepException)
    return print(io,
                 "Irreps of (bosonic or fermionic) `SU₂` should be labelled by non-negative half integers, i.e. elements of `Rational{Int}` with denominator 1 or 2")
end

"""
    struct SU2Irrep <: AbstractIrrep{SU₂}
    SU2Irrep(j::Real)
    Irrep[SU₂](j::Real)

Represents irreps of the group ``SU₂``. The irrep is labelled by a half integer `j` which
can be entered as an abitrary `Real`, but is stored as a `HalfInt` from the HalfIntegers.jl
package.

## Fields
- `j::HalfInt`: the label of the irrep, which can be any non-negative half integer.
"""
struct SU2Irrep <: AbstractIrrep{SU₂}
    j::HalfInt
    function SU2Irrep(j)
        j >= zero(j) || error("Not a valid SU₂ irrep")
        return new(j)
    end
end
Base.getindex(::IrrepTable, ::Type{SU₂}) = SU2Irrep
Base.convert(::Type{SU2Irrep}, j::Real) = SU2Irrep(j)

const _su2one = SU2Irrep(zero(HalfInt))
Base.one(::Type{SU2Irrep}) = _su2one
Base.conj(s::SU2Irrep) = s
⊗(s1::SU2Irrep, s2::SU2Irrep) = SectorSet{SU2Irrep}(abs(s1.j - s2.j):(s1.j + s2.j))

Base.IteratorSize(::Type{SectorValues{SU2Irrep}}) = IsInfinite()
Base.iterate(::SectorValues{SU2Irrep}, i=0) = (SU2Irrep(half(i)), i + 1)
function Base.getindex(::SectorValues{SU2Irrep}, i::Int)
    return 1 <= i ? SU2Irrep(half(i - 1)) : throw(BoundsError(values(SU2Irrep), i))
end
findindex(::SectorValues{SU2Irrep}, s::SU2Irrep) = twice(s.j) + 1

dim(s::SU2Irrep) = twice(s.j) + 1

FusionStyle(::Type{SU2Irrep}) = SimpleFusion()
Base.isreal(::Type{SU2Irrep}) = true

Nsymbol(sa::SU2Irrep, sb::SU2Irrep, sc::SU2Irrep) = WignerSymbols.δ(sa.j, sb.j, sc.j)
function Fsymbol(s1::SU2Irrep, s2::SU2Irrep, s3::SU2Irrep,
                 s4::SU2Irrep, s5::SU2Irrep, s6::SU2Irrep)
    if all(==(_su2one), (s1, s2, s3, s4, s5, s6))
        return 1.0
    else
        return sqrtdim(s5) * sqrtdim(s6) *
               WignerSymbols.racahW(Float64, s1.j, s2.j,
                                    s4.j, s3.j, s5.j, s6.j)
    end
end
function Rsymbol(sa::SU2Irrep, sb::SU2Irrep, sc::SU2Irrep)
    Nsymbol(sa, sb, sc) || return 0.0
    return iseven(convert(Int, sa.j + sb.j - sc.j)) ? 1.0 : -1.0
end

function fusiontensor(a::SU2Irrep, b::SU2Irrep, c::SU2Irrep)
    C = Array{Float64}(undef, dim(a), dim(b), dim(c), 1)
    ja, jb, jc = a.j, b.j, c.j

    for kc in 1:dim(c), kb in 1:dim(b), ka in 1:dim(a)
        C[ka, kb, kc, 1] = WignerSymbols.clebschgordan(ja, ja + 1 - ka, jb, jb + 1 - kb, jc,
                                                       jc + 1 - kc)
    end
    return C
end

Base.hash(s::SU2Irrep, h::UInt) = hash(s.j, h)
Base.isless(s1::SU2Irrep, s2::SU2Irrep) = isless(s1.j, s2.j)

# U₁ ⋊ C (U₁ and charge conjugation)
"""
    struct CU1Irrep <: AbstractIrrep{CU₁}
    CU1Irrep(j, s = ifelse(j>zero(j), 2, 0))
    Irrep[CU₁](j, s = ifelse(j>zero(j), 2, 0))

Represents irreps of the group ``U₁ ⋊ C`` (``U₁`` and charge conjugation or reflection),
which is also known as just `O₂`. 

## Fields
- `j::HalfInt`: the value of the ``U₁`` charge.
- `s::Int`: the representation of charge conjugation.

They can take values:
*   if `j == 0`, `s = 0` (trivial charge conjugation) or
    `s = 1` (non-trivial charge conjugation)
*   if `j > 0`, `s = 2` (two-dimensional representation)
"""
struct CU1Irrep <: AbstractIrrep{CU₁}
    j::HalfInt # value of the U1 charge
    s::Int # rep of charge conjugation:
    # if j == 0, s = 0 (trivial) or s = 1 (non-trivial),
    # else s = 2 (two-dimensional representation)
    # Let constructor take the actual half integer value j
    function CU1Irrep(j::Real, s::Integer=ifelse(j > zero(j), 2, 0))
        if ((j > zero(j) && s == 2) || (j == zero(j) && (s == 0 || s == 1)))
            new(j, s)
        else
            error("Not a valid CU₁ irrep")
        end
    end
end
Base.getindex(::IrrepTable, ::Type{CU₁}) = CU1Irrep
Base.convert(::Type{CU1Irrep}, (j, s)::Tuple{Real,Integer}) = CU1Irrep(j, s)

Base.IteratorSize(::Type{SectorValues{CU1Irrep}}) = IsInfinite()
function Base.iterate(::SectorValues{CU1Irrep}, state=(0, 0))
    j, s = state
    if iszero(j) && s == 0
        return CU1Irrep(j, s), (j, 1)
    elseif iszero(j) && s == 1
        return CU1Irrep(j, s), (j + 1, 2)
    else
        return CU1Irrep(half(j), s), (j + 1, 2)
    end
end
function Base.getindex(::SectorValues{CU1Irrep}, i::Int)
    i < 1 && throw(BoundsError(values(CU1Irrep), i))
    if i == 1
        return CU1Irrep(0, 0)
    elseif i == 2
        return CU1Irrep(0, 1)
    else
        return CU1Irrep(half(i - 2), 2)
    end
end
findindex(::SectorValues{CU1Irrep}, c::CU1Irrep) = twice(c.j) + iszero(c.j) + c.s

Base.hash(c::CU1Irrep, h::UInt) = hash(c.s, hash(c.j, h))
function Base.isless(c1::CU1Irrep, c2::CU1Irrep)
    return isless(c1.j, c2.j) || (c1.j == c2.j == zero(HalfInt) && c1.s < c2.s)
end

# CU1Irrep(j::Real, s::Int = ifelse(j>0, 2, 0)) = CU1Irrep(convert(HalfInteger, j), s)

Base.convert(::Type{CU1Irrep}, j::Real) = CU1Irrep(j)
Base.convert(::Type{CU1Irrep}, js::Tuple{Real,Int}) = CU1Irrep(js...)

Base.one(::Type{CU1Irrep}) = CU1Irrep(zero(HalfInt), 0)
Base.conj(c::CU1Irrep) = c

struct CU1ProdIterator
    a::CU1Irrep
    b::CU1Irrep
end
function Base.iterate(p::CU1ProdIterator, s::Int=1)
    if s == 1
        if p.a.j == p.b.j == zero(HalfInt)
            return CU1Irrep(zero(HalfInt), xor(p.a.s, p.b.s)), 4
        elseif p.a.j == zero(HalfInt)
            return p.b, 4
        elseif p.b.j == zero(HalfInt)
            return p.a, 4
        elseif p.a == p.b # != zero
            return one(CU1Irrep), 2
        else
            return CU1Irrep(abs(p.a.j - p.b.j)), 3
        end
    elseif s == 2
        return CU1Irrep(zero(HalfInt), 1), 3
    elseif s == 3
        CU1Irrep(p.a.j + p.b.j), 4
    else
        return nothing
    end
end
function Base.length(p::CU1ProdIterator)
    if p.a.j == zero(HalfInt) || p.b.j == zero(HalfInt)
        return 1
    elseif p.a == p.b
        return 3
    else
        return 2
    end
end
Base.eltype(::Type{CU1ProdIterator}) = CU1Irrep

⊗(a::CU1Irrep, b::CU1Irrep) = CU1ProdIterator(a, b)

dim(c::CU1Irrep) = ifelse(c.j == zero(HalfInt), 1, 2)

FusionStyle(::Type{CU1Irrep}) = SimpleFusion()
Base.isreal(::Type{CU1Irrep}) = true

function Nsymbol(a::CU1Irrep, b::CU1Irrep, c::CU1Irrep)
    return ifelse(c.s == 0, (a.j == b.j) & ((a.s == b.s == 2) | (a.s == b.s)),
                  ifelse(c.s == 1, (a.j == b.j) & ((a.s == b.s == 2) | (a.s != b.s)),
                         (c.j == a.j + b.j) | (c.j == abs(a.j - b.j))))
end
function Fsymbol(a::CU1Irrep, b::CU1Irrep, c::CU1Irrep,
                 d::CU1Irrep, e::CU1Irrep, f::CU1Irrep)
    Nabe = convert(Int, Nsymbol(a, b, e))
    Necd = convert(Int, Nsymbol(e, c, d))
    Nbcf = convert(Int, Nsymbol(b, c, f))
    Nafd = convert(Int, Nsymbol(a, f, d))

    Nabe * Necd * Nbcf * Nafd == 0 && return 0.0

    op = CU1Irrep(0, 0)
    om = CU1Irrep(0, 1)

    if a == op || b == op || c == op
        return 1.0
    end
    if (a == b == om) || (a == c == om) || (b == c == om)
        return 1.0
    end
    if a == om
        if d.j == zero(HalfInt)
            return 1.0
        else
            return (d.j == c.j - b.j) ? -1.0 : 1.0
        end
    end
    if b == om
        return (d.j == abs(a.j - c.j)) ? -1.0 : 1.0
    end
    if c == om
        return (d.j == a.j - b.j) ? -1.0 : 1.0
    end
    # from here on, a, b, c are neither 0+ or 0-
    s = sqrt(2) / 2
    if a == b == c
        if d == a
            if e.j == 0
                if f.j == 0
                    return f.s == 1 ? -0.5 : 0.5
                else
                    return e.s == 1 ? -s : s
                end
            else
                return f.j == 0 ? s : 0.0
            end
        else
            return 1.0
        end
    end
    if a == b # != c
        if d == c
            if f.j == b.j + c.j
                return e.s == 1 ? -s : s
            else
                return s
            end
        else
            return 1.0
        end
    end
    if b == c
        if d == a
            if e.j == a.j + b.j
                return s
            else
                return f.s == 1 ? -s : s
            end
        else
            return 1.0
        end
    end
    if a == c
        if d == b
            if e.j == f.j
                return 0.0
            else
                return 1.0
            end
        else
            return d.s == 1 ? -1.0 : 1.0
        end
    end
    if d == om
        return b.j == a.j + c.j ? -1.0 : 1.0
    end
    return 1.0
end
function Rsymbol(a::CU1Irrep, b::CU1Irrep, c::CU1Irrep)
    R = convert(Float64, Nsymbol(a, b, c))
    return c.s == 1 && a.j > 0 ? -R : R
end

function fusiontensor(a::CU1Irrep, b::CU1Irrep, c::CU1Irrep)
    C = fill(0.0, dim(a), dim(b), dim(c), 1)
    !Nsymbol(a, b, c) && return C
    if c.j == 0
        if a.j == b.j == 0
            C[1, 1, 1, 1] = 1.0
        else
            if c.s == 0
                C[1, 2, 1, 1] = 1.0 / sqrt(2)
                C[2, 1, 1, 1] = 1.0 / sqrt(2)
            else
                C[1, 2, 1, 1] = 1.0 / sqrt(2)
                C[2, 1, 1, 1] = -1.0 / sqrt(2)
            end
        end
    elseif a.j == 0
        C[1, 1, 1, 1] = 1.0
        C[1, 2, 2, 1] = a.s == 1 ? -1.0 : 1.0
    elseif b.j == 0
        C[1, 1, 1, 1] = 1.0
        C[2, 1, 2, 1] = b.s == 1 ? -1.0 : 1.0
    elseif c.j == a.j + b.j
        C[1, 1, 1, 1] = 1.0
        C[2, 2, 2, 1] = 1.0
    elseif c.j == a.j - b.j
        C[1, 2, 1, 1] = 1.0
        C[2, 1, 2, 1] = 1.0
    elseif c.j == b.j - a.j
        C[2, 1, 1, 1] = 1.0
        C[1, 2, 2, 1] = 1.0
    end
    return C
end
frobeniusschur(::CU1Irrep) = 1
