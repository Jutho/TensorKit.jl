# sectors.jl
# Explicit realization of sectors to be used for gradation of vector spaces.

# Abelian Sectors:
#-----------------
immutable Parity <: Abelian
    charge::Bool
end
Parity(d::Integer)=Parity(iseven(d))

*(c1::Parity,c2::Parity)=Parity(c1.charge != c2.charge)
Base.conj(c::Parity)=Parity(!c.charge)
Base.one(c::Parity)=Parity(false)
Base.one(::Type{Parity})=Parity(false)

immutable ZNCharge{N} <: Abelian
    charge::Int
    ZNCharge(charge::Int)=new(mod(charge,N))
end

*{N}(c1::ZNCharge{N},c2::ZNCharge{N})=ZNCharge{N}(c1.charge+c2.charge)
Base.conj{N}(c::ZNCharge{N})=ZNCharge{N}(-c.charge)
Base.one{N}(c::ZNCharge{N})=ZNCharge{N}(0)
Base.one{N}(::Type{ZNCharge{N}})=ZNCharge{N}(0)

immutable U1Charge <: Abelian
    charge::Int
end

*(c1::U1Charge,c2::U1Charge)=U1Charge(c1.charge+c2.charge)
Base.conj(c::U1Charge)=U1Charge(-c.charge)
Base.one(c::U1Charge)=U1Charge(0)
Base.one(::Type{U1Charge})=U1Charge(0)

# Nonabelian Sectors:
#---------------------
# to be done