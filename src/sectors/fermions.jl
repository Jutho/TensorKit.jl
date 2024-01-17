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
function Rsymbol(a::I, b::I, c::I) where {I<:FermionParity}
    return a.isodd && b.isodd ? -Int(Nsymbol(a, b, c)) : Int(Nsymbol(a, b, c))
end
twist(a::FermionParity) = a.isodd ? -1 : +1

function fusiontensor(a::I, b::I, c::I) where {I<:FermionParity}
    @warn "FermionParity Arrays do not preserve categorical properties." maxlog = 1
    return fill(Int(Nsymbol(a, b, c)), (1, 1, 1, 1))
end

function Base.show(io::IO, a::FermionParity)
    if get(io, :typeinfo, nothing) === typeof(a)
        print(io, Int(a.isodd))
    else
        print(io, type_repr(typeof(a)), "(", Int(a.isodd), ")")
    end
end
type_repr(::Type{FermionParity}) = "FermionParity"

Base.hash(f::FermionParity, h::UInt) = hash(f.isodd, h)
Base.isless(a::FermionParity, b::FermionParity) = isless(a.isodd, b.isodd)

# Common fermionic combinations
# -----------------------------

const FermionNumber = U1Irrep ⊠ FermionParity
const fU₁ = FermionNumber
fU₁(a::Int) = U1Irrep(a) ⊠ FermionParity(isodd(a))
type_repr(::Type{fU₁}) = "fU₁"

# convenience default converter -> allows Vect[fU₁](1 => 1)
Base.convert(::Type{fU₁}, a::Int) = fU₁(a)

const FermionSpin = SU2Irrep ⊠ FermionParity
const fSU₂ = FermionSpin
fSU₂(a::Real) = (s = SU2Irrep(a);
                 s ⊠ FermionParity(isodd(twice(s.j))))
type_repr(::Type{fSU₂}) = "fSU₂"

# convenience default converter -> allows Vect[fSU₂](1 => 1)
Base.convert(::Type{fSU₂}, a::Real) = fSU₂(a)
