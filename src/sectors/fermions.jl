"""
    FermionParity <: Sector

Represents sectors with fermion parity. The fermion parity is a ℤ₂ quantum number that
yields an additional sign when two odd fermions are exchanged.

See also: `FermionNumber`, `FermionSpin`
"""
struct FermionParity <: Sector
    isodd::Bool
end
const fℤ₂ = FermionParity
fermionparity(f::fℤ₂) = f.isodd

Base.convert(::Type{fℤ₂}, a::fℤ₂) = a
Base.convert(::Type{fℤ₂}, a) = fℤ₂(a)

Base.IteratorSize(::Type{SectorValues{fℤ₂}}) = HasLength()
Base.length(::SectorValues{fℤ₂}) = 2
function Base.iterate(::SectorValues{fℤ₂}, i=0)
    return i == 2 ? nothing : (fℤ₂(i), i + 1)
end
function Base.getindex(::SectorValues{fℤ₂}, i)
    return 1 <= i <= 2 ? fℤ₂(i - 1) : throw(BoundsError(values(fℤ₂), i))
end
findindex(::SectorValues{fℤ₂}, f::fℤ₂) = f.isodd ? 2 : 1

Base.one(::Type{fℤ₂}) = fℤ₂(false)
Base.conj(f::fℤ₂) = f
dim(f::fℤ₂) = 1

FusionStyle(::Type{fℤ₂}) = UniqueFusion()
BraidingStyle(::Type{fℤ₂}) = Fermionic()
Base.isreal(::Type{fℤ₂}) = true

⊗(a::fℤ₂, b::fℤ₂) = (fℤ₂(a.isodd ⊻ b.isodd),)

function Nsymbol(a::fℤ₂, b::fℤ₂, c::fℤ₂)
    return (a.isodd ⊻ b.isodd) == c.isodd
end
function Fsymbol(a::I, b::I, c::I, d::I, e::I, f::I) where {I<:fℤ₂}
    return Int(Nsymbol(a, b, e) * Nsymbol(e, c, d) * Nsymbol(b, c, f) * Nsymbol(a, f, d))
end
function Rsymbol(a::I, b::I, c::I) where {I<:fℤ₂}
    return a.isodd && b.isodd ? -Int(Nsymbol(a, b, c)) : Int(Nsymbol(a, b, c))
end
twist(a::fℤ₂) = a.isodd ? -1 : +1

function Base.show(io::IO, a::fℤ₂)
    if get(io, :typeinfo, nothing) === typeof(a)
        print(io, Int(a.isodd))
    else
        print(io, type_repr(typeof(a)), "(", Int(a.isodd), ")")
    end
end
type_repr(::Type{fℤ₂}) = "fℤ₂"

Base.hash(f::fℤ₂, h::UInt) = hash(f.isodd, h)
Base.isless(a::fℤ₂, b::fℤ₂) = isless(a.isodd, b.isodd)

# Common fermionic combinations
# -----------------------------

const FermionNumber = U1Irrep ⊠ fℤ₂
const fU₁ = FermionNumber
fU₁(a::Int) = U1Irrep(a) ⊠ fℤ₂(isodd(a))
type_repr(::Type{fU₁}) = "fU₁"

# convenience default converter -> allows Vect[fU₁](1 => 1)
Base.convert(::Type{fU₁}, a::Int) = fU₁(a)

const FermionSpin = SU2Irrep ⊠ fℤ₂
const fSU₂ = FermionSpin
fSU₂(a::Real) = (s = SU2Irrep(a);
                 s ⊠ fℤ₂(isodd(twice(s.j))))
type_repr(::Type{fSU₂}) = "fSU₂"

# convenience default converter -> allows Vect[fSU₂](1 => 1)
Base.convert(::Type{fSU₂}, a::Real) = fSU₂(a)