struct FermionParity <: Sector
    sector::Z2Irrep
end
const fℤ₂ = FermionParity
fermionparity(f::FermionParity) = isodd(f.sector.n)

Base.convert(::Type{FermionParity}, a::FermionParity) = a
Base.convert(::Type{FermionParity}, a) = FermionParity(a)

Base.IteratorSize(::Type{SectorValues{FermionParity}}) =
    Base.IteratorSize(SectorValues{Z2Irrep})
Base.length(::SectorValues{FermionParity}) = length(values(Z2Irrep))
function Base.iterate(::SectorValues{FermionParity})
    next = iterate(values(Z2Irrep))
    @assert next !== nothing
    value, state = next
    return FermionParity(value), state
end

function Base.iterate(::SectorValues{FermionParity}, state)
    next = iterate(values(Z2Irrep), state)
    if next === nothing
        return nothing
    else
        value, state = next
        return FermionParity(value), state
    end
end
Base.getindex(::SectorValues{FermionParity}, i) = FermionParity(values(Z2Irrep)[i])
findindex(::SectorValues{FermionParity}, f::FermionParity) = findindex(values(Z2Irrep), f.sector)

Base.one(::Type{FermionParity}) = FermionParity(one(Z2Irrep))
Base.conj(f::FermionParity) = FermionParity(conj(f.sector))

dim(f::FermionParity) = dim(f.sector)

FusionStyle(::Type{FermionParity}) = FusionStyle(Z2Irrep)
BraidingStyle(::Type{FermionParity}) = Fermionic()
Base.isreal(::Type{FermionParity}) = isreal(Z2Irrep)

⊗(a::FermionParity, b::FermionParity) = SectorSet{FermionParity}(a.sector ⊗ b.sector)

Nsymbol(a::FermionParity, b::FermionParity, c::FermionParity) = Nsymbol(a.sector, b.sector, c.sector)

Fsymbol(a::F, b::F, c::F, d::F, e::F, f::F) where {F<:FermionParity} =
    Fsymbol(a.sector, b.sector, c.sector, d.sector, e.sector, f.sector)

function Rsymbol(a::F, b::F, c::F) where {F<:FermionParity}
    if fermionparity(a) && fermionparity(b)
        return -Rsymbol(a.sector, b.sector, c.sector)
    else
        return +Rsymbol(a.sector, b.sector, c.sector)
    end
end

twist(a::FermionParity) = ifelse(fermionparity(a), -1, +1) * twist(a.sector)

type_repr(::Type{FermionParity}) = "FermionParity"

function Base.show(io::IO, a::FermionParity)
    if get(io, :typeinfo, nothing) !== FermionParity
        print(io, type_repr(typeof(a)), "(")
    end
    print(IOContext(io, :typeinfo => Z2Irrep), a.sector)
    if get(io, :typeinfo, nothing) !== FermionParity
        print(io, ")")
    end
end

Base.hash(f::FermionParity, h::UInt) = hash(f.sector, h)
Base.isless(a::FermionParity, b::FermionParity) = isless(a.sector, b.sector)
