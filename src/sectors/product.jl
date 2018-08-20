# Direct product of different sectors
#==============================================================================#
const SectorTuple = Tuple{Vararg{Sector}}

struct ProductSector{T<:SectorTuple} <: Sector
    sectors::T
end
ProductSector{T}(args...) where {T<:SectorTuple} = ProductSector{T}(args)
Base.convert(::Type{ProductSector{T}}, t::Tuple) where {T<:SectorTuple} = ProductSector{T}(convert(T, t))

Base.one(::Type{ProductSector{T}}) where {G<:Sector, T<:Tuple{G}} = ProductSector((one(G),))
Base.one(::Type{ProductSector{T}}) where {G<:Sector, T<:Tuple{G,Vararg{Sector}}} = one(G) × one(ProductSector{Base.tuple_type_tail(T)})

Base.conj(p::ProductSector) = ProductSector(map(conj, p.sectors))
⊗(p1::P, p2::P) where {P<:ProductSector} = SectorSet{P}(product(map(⊗,p1.sectors,p2.sectors)...))

Nsymbol(a::P, b::P, c::P) where {P<:ProductSector} = prod(map(Nsymbol, a.sectors, b.sectors, c.sectors))
function Fsymbol(a::P, b::P, c::P, d::P, e::P, f::P) where {P<:ProductSector}
    if fusiontype(P) isa Abelian || fusiontype(P) isa SimpleNonAbelian
        return prod(map(Fsymbol, a.sectors, b.sectors, c.sectors, d.sectors, e.sectors, f.sectors))
    else
        # TODO: DegenerateNonAbelian case, use kron ?
        throw(MethodError(Fsymbol,(a,b,c,d,e,f)))
    end
end
function Rsymbol(a::P, b::P, c::P) where {P<:ProductSector}
    if fusiontype(P) isa Abelian || fusiontype(P) isa SimpleNonAbelian
        return prod(map(Rsymbol, a.sectors, b.sectors, c.sectors))
    else
        # TODO: DegenerateNonAbelian case, use kron ?
        throw(MethodError(Rsymbol,(a,b,c)))
    end
end
function Asymbol(a::P, b::P, c::P) where {P<:ProductSector}
    if fusiontype(P) isa Abelian || fusiontype(P) isa SimpleNonAbelian
        return prod(map(Asymbol, a.sectors, b.sectors, c.sectors))
    else
        # TODO: DegenerateNonAbelian case, use kron ?
        throw(MethodError(Asymbol,(a,b,c)))
    end
end
function Bsymbol(a::P, b::P, c::P) where {P<:ProductSector}
    if fusiontype(P) isa Abelian || fusiontype(P) isa SimpleNonAbelian
        return prod(map(Bsymbol, a.sectors, b.sectors, c.sectors))
    else
        # TODO: DegenerateNonAbelian case, use kron ?
        throw(MethodError(Bsymbol,(a,b,c)))
    end
end
frobeniusschur(p::ProductSector) = prod(map(frobeniusschur, p.sectors))

fusiontensor(a::ProductSector{T}, b::ProductSector{T}, c::ProductSector{T}, v::Nothing = nothing) where {T<:SectorTuple} =
    _kron(fusiontensor(a.sectors[1], b.sectors[1], c.sectors[1]), fusiontensor(ProductSector(tail(a.sectors)), ProductSector(tail(b.sectors)), ProductSector(tail(c.sectors))))

fusiontensor(a::ProductSector{T}, b::ProductSector{T}, c::ProductSector{T}, v::Nothing = nothing) where {T<:Tuple{<:Sector}} =
    fusiontensor(a.sectors[1], b.sectors[1], c.sectors[1])

fusiontype(::Type{<:ProductSector{T}}) where {T<:SectorTuple} = _fusiontype(T)
_fusiontype(::Type{Tuple{}}) = Abelian()
_fusiontype(::Type{T}) where {T<:SectorTuple} = fusiontype(Base.tuple_type_head(T)) & _fusiontype(Base.tuple_type_tail(T))

braidingtype(::Type{<:ProductSector{T}}) where {T<:SectorTuple} = _braidingtype(T)
_braidingtype(::Type{Tuple{}}) = Bosonic()
_braidingtype(::Type{T}) where {T<:SectorTuple} = braidingtype(Base.tuple_type_head(T)) & _braidingtype(Base.tuple_type_tail(T))

fermionparity(P::ProductSector) = _fermionparity(P.sectors)
_fermionparity(::Tuple{}) = false
_fermionparity(t::Tuple) = xor(fermionparity(t[1]), _fermionparity(tail(t)))

dim(p::ProductSector) = prod(dim, p.sectors)

Base.isequal(p1::ProductSector, p2::ProductSector) = isequal(p1.sectors, p2.sectors)
Base.hash(p::ProductSector, h::UInt64) = hash(p.sectors, h)

# Default construction from tensor product of sectors
#-----------------------------------------------------
×(S1, S2, S3, S4...) = ×(×(S1, S2), S3, S4...)

×(S1::Sector, S2::Sector) = ProductSector((S1, S2))
×(P1::ProductSector, S2::Sector) = ProductSector(tuple(P1.sectors..., S2))
×(S1::Sector, P2::ProductSector) = ProductSector(tuple(S1, P2.sectors...))
×(P1::ProductSector, P2::ProductSector) = ProductSector(tuple(P1.sectors..., P2.sectors...))

×(G1::Type{ProductSector{Tuple{}}}, G2::Type{ProductSector{T}}) where {T<:SectorTuple} = G2
×(G1::Type{ProductSector{T1}}, G2::Type{ProductSector{T2}}) where {T1<:SectorTuple,T2<:SectorTuple} =
    tuple_type_head(T1) × (ProductSector{tuple_type_tail(T1)} × G2)
×(G1::Type{ProductSector{Tuple{}}}, G2::Type{<:Sector}) = ProductSector{Tuple{G2}}
×(G1::Type{ProductSector{T}}, G2::Type{<:Sector}) where {T<:SectorTuple} = Base.tuple_type_head(T) × (ProductSector{Base.tuple_type_tail(T)} × G2)
×(G1::Type{<:Sector}, G2::Type{ProductSector{T}}) where {T<:SectorTuple} = ProductSector{Base.tuple_type_cons(G1,T)}
×(G1::Type{<:Sector}, G2::Type{<:Sector}) = ProductSector{Tuple{G1,G2}}

function Base.show(io::IO, P::ProductSector)
    sectors = P.sectors
    sep = get(io, :compact, false) ? ", " : " × "
    print(io,"(")
    for i = 1:length(sectors)
        i == 1 || print(io, sep)
        print(io, sectors[i])
    end
    print(io,")")
end

function Base.show(io::IO, ::Type{ProductSector{T}}) where {T<:SectorTuple}
    sectors = T.parameters
    print(io,"(")
    for i = 1:length(sectors)
        i == 1 || print(io, " × ")
        print(io, sectors[i])
    end
    print(io,")")
end
