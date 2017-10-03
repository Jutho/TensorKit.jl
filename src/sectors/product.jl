# Direct product of different sectors
#==============================================================================#
const SectorTuple = Tuple{Vararg{Sector}}

struct ProductSector{T<:SectorTuple} <: Sector
    sectors::T
end
ProductSector{T}(args...) where {T<:SectorTuple} = ProductSector{T}(args)
Base.convert(::Type{ProductSector{T}}, t::Tuple) where {T<:SectorTuple} = ProductSector{T}(convert(T, t))

Base.one(::Type{ProductSector{T}}) where {G<:Sector, T<:Tuple{G}} = ProductSector((one(G),))
Base.one(::Type{ProductSector{T}}) where {G<:Sector, T<:Tuple{G,Vararg{Sector}}} = one(G) × one(ProductSector{tuple_type_tail(T)})

Base.conj(p::ProductSector) = ProductSector(map(conj, p.sectors))
⊗(p1::P, p2::P) where {P<:ProductSector} = SectorSet{P}(product(map(⊗,p1.sectors,p2.sectors)...))
Nsymbol(a::P, b::P, c::P) where {P<:ProductSector} = prod(map(Nsymbol, a.sectors, b.sectors, c.sectors))

fusiontype(::Type{<:ProductSector{T}}) where {T<:SectorTuple} = _fusiontype(T)
_fusiontype(::Type{Tuple{}}) = Abelian
_fusiontype(::Type{T}) where {T<:SectorTuple} = fusiontype(tuple_type_head(T)) & _fusiontype(tuple_type_tail(T))

braidingtype(::Type{<:ProductSector{T}}) where {T<:SectorTuple} = _braidingtype(T)
_braidingtype(::Type{Tuple{}}) = Abelian
_braidingtype(::Type{<:SectorTuple}) = fusiontype(tuple_type_head(T)) & _braidingtype(tuple_type_tail(T))

fermionparity(P::ProductSector) = _fermionparity(P.sectors)
_fermionparity(::Tuple{}) = false
_fermionparity(t::Tuple) = xor(fermionparity(t[1]), _fermionparity(tail(t)))

dim(p::ProductSector) = prod(dim, p.sectors)

# Default construction from tensor product of sectors
#-----------------------------------------------------
Base.:×(S1::Sector, S2::Sector) = ProductSector((S1, S2))
Base.:×(P1::ProductSector, S2::Sector) = ProductSector(tuple(P1.sectors..., S2))
Base.:×(S1::Sector, P2::ProductSector) = ProductSector(tuple(S1, P2.sectors...))
Base.:×(P1::ProductSector, P2::ProductSector) = ProductSector(tuple(P1.sectors..., P2.sectors...))

Base.:×(G1::Type{ProductSector{Tuple{}}}, G2::Type{ProductSector{T}}) where {T<:SectorTuple} = G2
Base.:×(G1::Type{ProductSector{T1}}, G2::Type{ProductSector{T2}}) where {T1<:SectorTuple,T2<:SectorTuple} =
    tuple_type_head(T1) ⊗ (ProductSector{tuple_type_tail(T1)} ⊗ G2)
Base.:×(G1::Type{ProductSector{Tuple{}}}, G2::Type{<:Sector}) = ProductSector{Tuple{G2}}
Base.:×(G1::Type{ProductSector{T}}, G2::Type{<:Sector}) where {T<:SectorTuple} = tuple_type_head(T) ⊗ (ProductSector{tuple_type_tail(T)} ⊗ G2)
Base.:×(G1::Type{<:Sector}, G2::Type{ProductSector{T}}) where {T<:SectorTuple} = ProductSector{tuple_type_cons(G1,T)}
Base.:×(G1::Type{<:Sector}, G2::Type{<:Sector}) = ProductSector{Tuple{G1,G2}}

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
