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
fermionparity(f::FermionParity) = f.isodd

Base.convert(::Type{FermionParity}, a::FermionParity) = a
Base.convert(::Type{FermionParity}, a) = FermionParity(a)

Base.IteratorSize(::Type{SectorValues{FermionParity}}) = HasLength()
Base.length(::SectorValues{FermionParity}) = 2
function Base.iterate(::SectorValues{FermionParity}, i=0)
    return i == 2 ? nothing : (FermionParity(i), i + 1)
end
function Base.getindex(::SectorValues{FermionParity}, i)
    return 1 <= i <= 2 ? FermionParity(i - 1) : throw(BoundsError(values(FermionParity), i))
end
findindex(::SectorValues{FermionParity}, f::FermionParity) = f.isodd ? 2 : 1

Base.one(::Type{FermionParity}) = FermionParity(false)
Base.conj(f::FermionParity) = f
dim(f::FermionParity) = 1

FusionStyle(::Type{FermionParity}) = UniqueFusion()
BraidingStyle(::Type{FermionParity}) = Fermionic()
Base.isreal(::Type{FermionParity}) = true

⊗(a::FermionParity, b::FermionParity) = (FermionParity(a.isodd ⊻ b.isodd),)

function Nsymbol(a::FermionParity, b::FermionParity, c::FermionParity)
    return (a.isodd ⊻ b.isodd) == c.isodd
end
function Fsymbol(a::I, b::I, c::I, d::I, e::I, f::I) where {I<:FermionParity}
    return Int(Nsymbol(a, b, e) * Nsymbol(e, c, d) * Nsymbol(b, c, f) * Nsymbol(a, f, d))
end

function Rsymbol(a::F, b::F, c::F) where {F<:FermionParity}
    return a.isodd && b.isodd ? -Int(Nsymbol(a, b, c)) : Int(Nsymbol(a, b, c))
end
twist(a::FermionParity) = a.isodd ? -1 : +1

function Base.show(io::IO, a::FermionParity)
    if get(io, :typeinfo, nothing) === FermionParity
        print(io, Int(a.isodd))
    else
        print(io, "FermionParity(", Int(a.isodd), ")")
    end
end
type_repr(::Type{FermionParity}) = "FermionParity"

Base.hash(f::FermionParity, h::UInt) = hash(f.isodd, h)
Base.isless(a::FermionParity, b::FermionParity) = isless(a.isodd, b.isodd)

# Common fermionic combinations
# -----------------------------

const FermionNumber = U1Irrep ⊠ FermionParity
const fU₁ = FermionNumber
type_repr(::Type{FermionNumber}) = "FermionNumber"

# convenience default converter -> allows Vect[FermionNumber](1 => 1)
function Base.convert(::Type{FermionNumber}, a::Int)
    return U1Irrep(a) ⊠ FermionParity(isodd(a))
end

const FermionSpin = SU2Irrep ⊠ FermionParity
const fSU₂ = FermionSpin
type_repr(::Type{FermionSpin}) = "FermionSpin"

# convenience default converter -> allows Vect[FermionSpin](1 => 1)
function Base.convert(::Type{FermionSpin}, a::Real)
    s = SU2Irrep(a)
    return s ⊠ FermionParity(isodd(twice(s.j)))
end
