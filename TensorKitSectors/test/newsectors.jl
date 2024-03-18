# NewSU2Irrep
# Test of a bare-bones sector implementation, which is set to be `DegenerateNonAbelian`
# (even though it is not)
module NewSectors

export NewSU2Irrep

using HalfIntegers, WignerSymbols
using TensorKitSectors

import TensorKitSectors: FusionStyle, BraidingStyle, Nsymbol, Fsymbol, Rsymbol, dim,
                         fusiontensor, ⊗

struct NewSU2Irrep <: Sector
    j::HalfInt
    function NewSU2Irrep(j)
        j >= zero(j) || error("Not a valid SU₂ irrep")
        return new(j)
    end
end
Base.convert(::Type{NewSU2Irrep}, j::Real) = NewSU2Irrep(j)

const _su2one = NewSU2Irrep(zero(HalfInt))
Base.one(::Type{NewSU2Irrep}) = _su2one
Base.conj(s::NewSU2Irrep) = s
function ⊗(s1::NewSU2Irrep, s2::NewSU2Irrep)
    return SectorSet{NewSU2Irrep}(abs(s1.j - s2.j):(s1.j + s2.j))
end

Base.IteratorSize(::Type{SectorValues{NewSU2Irrep}}) = Base.IsInfinite()
Base.iterate(::SectorValues{NewSU2Irrep}, i=0) = (NewSU2Irrep(half(i)), i + 1)
# Base.getindex(::SectorValues{NewSU2Irrep}, i::Int) =
#     1 <= i ? NewSU2Irrep(half(i-1)) : throw(BoundsError(values(NewSU2Irrep), i))
# findindex(::SectorValues{NewSU2Irrep}, s::NewSU2Irrep) = twice(s.j)+1

# TensorKit.dim(s::NewSU2Irrep) = twice(s.j)+1
#
FusionStyle(::Type{NewSU2Irrep}) = GenericFusion()
BraidingStyle(::Type{NewSU2Irrep}) = Bosonic()
Base.isreal(::Type{NewSU2Irrep}) = true

function Nsymbol(sa::NewSU2Irrep, sb::NewSU2Irrep, sc::NewSU2Irrep)
    return convert(Int, WignerSymbols.δ(sa.j, sb.j, sc.j))
end

function Fsymbol(s1::NewSU2Irrep, s2::NewSU2Irrep, s3::NewSU2Irrep,
                 s4::NewSU2Irrep, s5::NewSU2Irrep, s6::NewSU2Irrep)
    n1 = Nsymbol(s1, s2, s5)
    n2 = Nsymbol(s5, s3, s4)
    n3 = Nsymbol(s2, s3, s6)
    n4 = Nsymbol(s1, s6, s4)
    f = all(==(_su2one), (s1, s2, s3, s4, s5, s6)) ? 1.0 :
        sqrt(dim(s5) * dim(s6)) * WignerSymbols.racahW(Float64, s1.j, s2.j,
                                                       s4.j, s3.j, s5.j, s6.j)
    return fill(f, (n1, n2, n3, n4))
end

function Rsymbol(sa::NewSU2Irrep, sb::NewSU2Irrep, sc::NewSU2Irrep)
    Nsymbol(sa, sb, sc) > 0 || return fill(0.0, (0, 0))
    return fill(iseven(convert(Int, sa.j + sb.j - sc.j)) ? 1.0 : -1.0, (1, 1))
end

dim(s::NewSU2Irrep) = twice(s.j) + 1

function fusiontensor(a::NewSU2Irrep, b::NewSU2Irrep, c::NewSU2Irrep)
    C = Array{Float64}(undef, dim(a), dim(b), dim(c), 1)
    ja, jb, jc = a.j, b.j, c.j

    for kc in 1:dim(c), kb in 1:dim(b), ka in 1:dim(a)
        C[ka, kb, kc, 1] = WignerSymbols.clebschgordan(ja, ja + 1 - ka, jb, jb + 1 - kb, jc,
                                                       jc + 1 - kc)
    end
    return C
end

Base.hash(s::NewSU2Irrep, h::UInt) = hash(s.j, h)
Base.isless(s1::NewSU2Irrep, s2::NewSU2Irrep) = isless(s1.j, s2.j)

end
