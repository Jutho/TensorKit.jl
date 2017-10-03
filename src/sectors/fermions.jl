# Sectors giving rise to super vector spaces, which have at least a Z2 grading
#==============================================================================#
abstract type Fermion <: Sector end

Base.@pure braidingtype(::Type{<:Fermion}) = Fermionic

# FermionParity: only total fermion parity is conserved
struct FermionParity <: Fermion
    parity::Bool
end
FermionParity(p::Int) = FermionParity(isodd(p))
Base.one(::Type{FermionParity}) = FermionParity(false)
Base.conj(p::FermionParity) = p
⊗(p1::FermionParity, p2::FermionParity) = (FermionParity(p1.parity != p2.parity),)

Base.@pure fusiontype(::Type{FermionParity}) = Abelian

const fℤ₂ = FermionParity

Base.show(io::IO, ::Type{FermionParity}) = print(io, "fℤ₂")
Base.show(io::IO, p::FermionParity) = get(io, :compact, false) ? print(io, Int(p.parity)) : print(io, "fℤ₂(", Int(p.parity), ")")

# FermionNumber: total number of fermions is conserved -> U1 symmetry
struct FermionNumber <: Fermion
    num::Int
end
Base.one(::Type{FermionNumber}) = FermionNumber(0)
Base.conj(n::FermionNumber) = FermionNumber(n.num)
⊗(n1::FermionNumber, n2::FermionNumber) = (FermionNumber(n1.num+n2.num),)

Base.@pure fusiontype(::Type{FermionNumber}) = Abelian

const fU₁ = FermionNumber

Base.show(io::IO, ::Type{FermionNumber}) = print(io, "fU₁")
Base.show(io::IO, n::FermionNumber) = get(io, :compact, false) ? print(io, n.num) : print(io, "fU₁(", n.num, ")")

const AbelianFermion = Union{FermionParity,FermionNumber}
Nsymbol(a::G, b::G, c::G) where {G<:AbelianFermion} = c == first(a ⊗ b)
Fsymbol(a::G, b::G, c::G, d::G, e::G, f::G) where {G<:AbelianFermion} =
    Int(Nsymbol(a,b,e)*Nsymbol(e,c,d)*Nsymbol(b,c,f)*Nsymbol(a,f,d))
Rsymbol(a::G, b::G, c::G) where {G<:AbelianFermion} = fermionparity(a) && fermionparity(b) ? -Int(Nsymbol(a, b, c)) : Int(Nsymbol(a, b, c))

"""
    function fermionparity(s::Fermion) -> Bool

Returns the fermion parity of a sector `s` that is a subtype of `Fermion`, as a
`Bool` being true if odd and false if even.
"""
fermionparity(p::FermionParity) = p.parity
fermionparity(n::FermionNumber) = isodd(n.num)

if VERSION > v"0.6.0"
    struct FermionSpin <: Fermion
        dim::Int
        FermionSpin(j::Int) = j >= 0 ? new(2*j+1) : throw(SU2IrrepException)
        function FermionSpin(j::Rational{Int})
            if j.den == 2
                new(j.num+1)
            elseif j.den == 1
                new(2*j.num+1)
            else
                throw(SU2IrrepException)
            end
        end
    end
    _getj(s::FermionSpin) = (s.dim-1)//2
    Base.one(::Type{FermionSpin}) = FermionSpin(0)
    Base.conj(s::FermionSpin) = s
    ⊗(s1::FermionSpin, s2::FermionSpin) = SectorSet{FermionSpin}( abs(_getj(s1)-_getj(s2)):(_getj(s1)+_getj(s2)) )

    Base.@pure fusiontype(::Type{FermionSpin}) = SimpleNonAbelian

    Nsymbol(s1::FermionSpin, s2::FermionSpin, s::FermionSpin) = (abs(s1.dim-s2.dim)+1) <= s.dim <= (s1.dim+s2.dim-1) && isodd(s1.dim+s2.dim - s.dim)

    # TODO: Fsymbol -> 6j-symbols of SU(2), compute, cache, ... -> Recycle those of SU2Irrep

    const fSU₂ = FermionSpin
    Base.show(io::IO, ::Type{FermionSpin}) = print(io, "fSU₂")
    Base.show(io::IO, s::FermionSpin) = get(io, :compact, false) ? print(io, _getj(s)) : print(io, "fSU₂(", _getj(s), ")")

    fermionparity(s::FermionSpin) = iseven(s.dim)
end
