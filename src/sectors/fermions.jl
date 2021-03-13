struct Fermion{P,I<:Sector} <: Sector
    sector::I
    function Fermion{P,I}(sector::I) where {P, I<:Sector}
        @assert BraidingStyle(I) isa Bosonic
        return new{P,I}(sector)
    end
end
Fermion{P}(sector::I) where {P, I<:Sector} = Fermion{P,I}(sector)
Fermion{P,I}(sector) where {P, I<:Sector} = Fermion{P,I}(convert(I, sector))
Base.convert(::Type{Fermion{P,I}}, a::Fermion{P,I}) where {P, I<:Sector} = a
Base.convert(::Type{Fermion{P,I}}, a) where {P, I<:Sector} = Fermion{P,I}(convert(I, a))

fermionparity(f::Fermion{P}) where P = P(f.sector)

Base.IteratorSize(::Type{SectorValues{Fermion{P,I}}}) where {P, I<:Sector} =
    Base.IteratorSize(SectorValues{I})
Base.length(::SectorValues{Fermion{P,I}}) where {P, I<:Sector} = length(values(I))
function Base.iterate(::SectorValues{Fermion{P, I}}) where {P, I<:Sector}
    next = iterate(values(I))
    @assert next !== nothing
    value, state = next
    return Fermion{P}(value), state
end
function Base.iterate(::SectorValues{Fermion{P, I}}, state) where {P, I<:Sector}
    next = iterate(values(I), state)
    if next === nothing
        return nothing
    else
        value, state = next
        return Fermion{P}(value), state
    end
end
Base.getindex(::SectorValues{Fermion{P, I}}, i) where {P, I<:Sector} =
    Fermion{P}(values(I)[i])
findindex(::SectorValues{Fermion{P, I}}, f::Fermion{P, I}) where {P, I<:Sector} =
    findindex(values(I), f.sector)

Base.one(::Type{Fermion{P, I}}) where {P, I<:Sector} = Fermion{P}(one(I))
Base.conj(f::Fermion{P}) where {P} = Fermion{P}(conj(f.sector))

dim(f::Fermion) = dim(f.sector)

FusionStyle(::Type{<:Fermion{<:Any,I}}) where {I<:Sector} = FusionStyle(I)
BraidingStyle(::Type{<:Fermion}) = Fermionic()
Base.isreal(::Type{Fermion{<:Any,I}}) where {I<:Sector} = isreal(I)

⊗(a::F, b::F) where {F<:Fermion} = SectorSet{F}(a.sector ⊗ b.sector)

Nsymbol(a::F, b::F, c::F) where {F<:Fermion} = Nsymbol(a.sector, b.sector, c.sector)

Fsymbol(a::F, b::F, c::F, d::F, e::F, f::F) where {F<:Fermion} =
    Fsymbol(a.sector, b.sector, c.sector, d.sector, e.sector, f.sector)

function Rsymbol(a::F, b::F, c::F) where {F<:Fermion}
    if fermionparity(a) && fermionparity(b)
        return -Rsymbol(a.sector, b.sector, c.sector)
    else
        return +Rsymbol(a.sector, b.sector, c.sector)
    end
end

twist(a::Fermion) = ifelse(fermionparity(a), -1, +1)*twist(a.sector)

type_repr(::Type{Fermion{P,I}}) where {P, I<:Sector} = "Fermion{$P, " * type_repr(I) * "}"

function Base.show(io::IO, a::Fermion{P, I}) where {P, I<:Sector}
    if get(io, :typeinfo, nothing) !== Fermion{P, I}
        print(io, type_repr(typeof(a)), "(")
    end
    print(IOContext(io, :typeinfo => I), a.sector)
    if get(io, :typeinfo, nothing) !== Fermion{P, I}
        print(io, ")")
    end
end

Base.hash(f::Fermion, h::UInt) = hash(f.sector, h)
Base.isless(a::F, b::F) where {F<:Fermion} = isless(a.sector, b.sector)

_fermionparity(a::Z2Irrep) = isodd(a.n)
_fermionnumber(a::U1Irrep) = isodd(convert(Int, a.charge))
_fermionspin(a::SU2Irrep) = isodd(twice(a.j))

const FermionParity = Fermion{_fermionparity, Z2Irrep}
const FermionNumber = Fermion{_fermionnumber, U1Irrep}
const FermionSpin = Fermion{_fermionspin, SU2Irrep}
const fℤ₂ = FermionParity
const fU₁ = FermionNumber
const fSU₂ = FermionSpin

type_repr(::Type{FermionParity}) = "FermionParity"
type_repr(::Type{FermionNumber}) = "FermionNumber"
type_repr(::Type{FermionSpin}) = "FermionSpin"
