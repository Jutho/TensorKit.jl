# Sectors corresponding to irreducible representations of compact groups
#==============================================================================#
# Irreps of Abelian groups
#------------------------------------------------------------------------------#
abstract type AbelianIrrep <: Sector end

Base.@pure fusiontype(::Type{<:AbelianIrrep}) = Abelian
Base.@pure braidingtype(::Type{<:AbelianIrrep}) = Bosonic

Nsymbol(a::G, b::G, c::G) where {G<:AbelianIrrep} = c == first(a ⊗ b)
Fsymbol(a::G, b::G, c::G, d::G, e::G, f::G) where {G<:AbelianIrrep} =
    Int(Nsymbol(a,b,e)*Nsymbol(e,c,d)*Nsymbol(b,c,f)*Nsymbol(a,f,d))
frobeniusschur(a::AbelianIrrep) = 1
Bsymbol(a::G, b::G, c::G) where {G<:AbelianIrrep} = Float64(Nsymbol(a, b, c))
Rsymbol(a::G, b::G, c::G) where {G<:AbelianIrrep} = Float64(Nsymbol(a, b, c))

# ZNIrrep: irreps of Z_N are labelled by integers mod N; do we ever want N > 127?
struct ZNIrrep{N} <: AbelianIrrep
    n::Int8
    function ZNIrrep{N}(n::Integer) where {N}
        new{N}(mod(n, N))
    end
end
Base.one(::Type{ZNIrrep{N}}) where {N} =ZNIrrep{N}(0)
Base.conj(c::ZNIrrep{N}) where {N} = ZNIrrep{N}(-c.n)
⊗(c1::ZNIrrep{N}, c2::ZNIrrep{N}) where {N} = (ZNIrrep{N}(c1.n+c2.n),)

Base.convert(Z::Type{<:ZNIrrep}, n::Real) = Z(convert(Int, n))

const ℤ₂ = ZNIrrep{2}
const ℤ₃ = ZNIrrep{3}
const ℤ₄ = ZNIrrep{4}
const Parity = ZNIrrep{2}
Base.show(io::IO, ::Type{ZNIrrep{2}}) = print(io, "ℤ₂")
Base.show(io::IO, ::Type{ZNIrrep{3}}) = print(io, "ℤ₃")
Base.show(io::IO, ::Type{ZNIrrep{4}}) = print(io, "ℤ₄")
Base.show(io::IO, ::Type{ZNIrrep{4}}) = print(io, "ℤ₄")
Base.show(io::IO, c::ZNIrrep{2}) = get(io, :compact, false) ? print(io, c.n) : print(io, "ℤ₂(", c.n, ")")
Base.show(io::IO, c::ZNIrrep{3}) = get(io, :compact, false) ? print(io, c.n) : print(io, "ℤ₃(", c.n, ")")
Base.show(io::IO, c::ZNIrrep{4}) = get(io, :compact, false) ? print(io, c.n) : print(io, "ℤ₄(", c.n, ")")
Base.show(io::IO, c::ZNIrrep{N}) where {N} = get(io, :compact, false) ? print(io, c.n) :
    print(io, "ZNIrrep{", N, "}(" , c.n, ")")

# U1Irrep: irreps of U1 are labelled by integers
struct U1Irrep <: AbelianIrrep
    charge::Int
end
Base.one(::Type{U1Irrep}) = U1Irrep(0)
Base.conj(c::U1Irrep) = U1Irrep(-c.charge)
⊗(c1::U1Irrep, c2::U1Irrep) = (U1Irrep(c1.charge+c2.charge),)

Base.convert(::Type{U1Irrep}, c::Real) = U1Irrep(convert(Int, c))

const U₁ = U1Irrep
Base.show(io::IO, ::Type{U1Irrep}) = print(io, "U₁")
Base.show(io::IO, c::U1Irrep) = get(io, :compact, false) ? print(io, c.charge) : print(io, "U₁(", c.charge, ")")

# NOTE: FractionalU1Charge?

if VERSION > v"0.6.0"
    # Nob-abelian groups
    #------------------------------------------------------------------------------#
    # SU2Irrep: irreps of SU2 are labelled by half integers j, internally we use the integer dimension 2j+1 instead
    import WignerSymbols

    struct SU2IrrepException <: Exception end
    Base.show(io::IO, ::SU2IrrepException) = print(io, "Irreps of (bosonic or fermionic) `SU₂` should be labelled by non-negative half integers, i.e. elements of `Rational{Int}` with denominator 1 or 2")

    struct SU2Irrep <: Sector
        dim::Int
        # Let constructor take the actual half integer value j
        SU2Irrep(j::Int) = j >= 0 ? new(2*j+1) : throw(SU2IrrepException)
        function SU2Irrep(j::Rational{Int})
            if j.den == 2
                new(j.num+1)
            elseif j.den == 1
                new(2*j.num+1)
            else
                throw(SU2IrrepException)
            end
        end
    end
    _getj(s::SU2Irrep) = (s.dim-1)//2

    Base.one(::Type{SU2Irrep}) = SU2Irrep(0)
    Base.conj(s::SU2Irrep) = s
    ⊗(s1::SU2Irrep, s2::SU2Irrep) = SectorSet{SU2Irrep}( abs(_getj(s1)-_getj(s2)):(_getj(s1)+_getj(s2)) )

    Base.convert(::Type{SU2Irrep}, s::Real) = SU2Irrep(convert(Int, 2*s)//2)

    dim(s::SU2Irrep) = s.dim

    Base.@pure fusiontype(::Type{SU2Irrep}) = SimpleNonAbelian
    Base.@pure braidingtype(::Type{SU2Irrep}) = Bosonic

    Nsymbol(sa::SU2Irrep, sb::SU2Irrep, sc::SU2Irrep) = WignerSymbols.δ(_getj(sa), _getj(sb), _getj(sc))
    Fsymbol(s1::SU2Irrep, s2::SU2Irrep, s3::SU2Irrep, s4::SU2Irrep, s5::SU2Irrep, s6::SU2Irrep) =
        WignerSymbols.racahW(map(_getj,(s1,s2,s4,s3,s5,s6))...)*sqrt(dim(s5)*dim(s6))
    function Rsymbol(sa::SU2Irrep, sb::SU2Irrep, sc::SU2Irrep)
        Nsymbol(sa, sb, sc) || return 0.
        iseven(convert(Int, _getj(sa)+_getj(sb)-_getj(sc))) ? 1.0 : -1.0
    end

    const SU₂ = SU2Irrep
    Base.show(io::IO, ::Type{SU2Irrep}) = print(io, "SU₂")
    Base.show(io::IO, s::SU2Irrep) = get(io, :compact, false) ? print(io, _getj(s)) : print(io, "SU₂(", _getj(s), ")")
end
