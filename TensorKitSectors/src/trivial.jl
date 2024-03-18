"""
    Trivial

Singleton type to represent the trivial sector, i.e. the trivial representation of the
trivial group. This is equivalent to `Rep[ℤ₁]`, or the unit object of the category `Vect` of
ordinary vector spaces.
"""
struct Trivial <: Sector end

Base.show(io::IO, ::Trivial) = print(io, "Trivial()")

# iterators
Base.IteratorSize(::Type{SectorValues{Trivial}}) = HasLength()
Base.length(::SectorValues{Trivial}) = 1
Base.iterate(::SectorValues{Trivial}, i=false) = return i ? nothing : (Trivial(), true)
function Base.getindex(::SectorValues{Trivial}, i::Int)
    return i == 1 ? Trivial() : throw(BoundsError(values(Trivial), i))
end
findindex(::SectorValues{Trivial}, c::Trivial) = 1

# basic properties
Base.one(::Type{Trivial}) = Trivial()
Base.conj(::Trivial) = Trivial()

Base.isreal(::Type{Trivial}) = true
Base.isless(::Trivial, ::Trivial) = false

# fusion rules
⊗(::Trivial, ::Trivial) = (Trivial(),)
Nsymbol(::Trivial, ::Trivial, ::Trivial) = true
FusionStyle(::Type{Trivial}) = UniqueFusion()
Fsymbol(::Trivial, ::Trivial, ::Trivial, ::Trivial, ::Trivial, ::Trivial) = 1

# braiding rules
Rsymbol(::Trivial, ::Trivial, ::Trivial) = 1
BraidingStyle(::Type{Trivial}) = Bosonic()
