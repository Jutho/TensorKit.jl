# General definition: might need to be generalized for nonabelian sectors
immutable InvariantSpace{G<:Sector,S<:UnitaryRepresentationSpace,N} <: TensorSpace{S,N}
    spaces::NTuple{N,S}
    dims::Dict{NTuple{N,G},Int}
end

# Additional constructors
InvariantSpace(P::InvariantSpace) = P
invariant(P::InvariantSpace) = P

# Specific constructors for Abelian Sectors
function InvariantSpace{G<:Abelian,N}(spaces::NTuple{N,AbelianSpace{G}})
    sectorlist=_invariantsectors(spaces)
    dims=Dict{eltype(sectorlist),Int}()
    sizehint(dims,length(sectorlist))
    for s in sectorlist
        dims[s]=_dim(spaces,s)
    end
    return InvariantSpace{G,AbelianSpace{G},N}(spaces,dims)
end

InvariantSpace{G<:Abelian}(V::AbelianSpace{G},Vlist::AbelianSpace{G}...) = InvariantSpace(tuple(V,Vlist...))
InvariantSpace{G<:Abelian}(P::ProductSpace{AbelianSpace{G},0}) = InvariantSpace{G,AbelianSpace{G},0}((),[()=>1])
InvariantSpace{G<:Abelian}(P::ProductSpace{AbelianSpace{G}}) = InvariantSpace(P.spaces)

invariant{G<:Abelian}(V::AbelianSpace{G}) = InvariantSpace(tuple(V))
invariant{G<:Abelian}(P::ProductSpace{AbelianSpace{G},0}) = InvariantSpace{G,AbelianSpace{G},0}((),[()=>1])
invariant{G<:Abelian}(P::ProductSpace{AbelianSpace{G}}) = InvariantSpace(P.spaces)

# Interaction with product spaces
⊗{G,S}(P1::InvariantSpace{G,S}, P2::InvariantSpace{G,S}) = InvariantSpace(tuple(P1.spaces..., P2.spaces...))
⊗{G,S}(P1::ProductSpace{S}, P2::InvariantSpace{G,S}) = ProductSpace(tuple(P1.spaces..., P2.spaces...))
⊗{G,S}(P1::InvariantSpace{G,S}, P2::ProductSpace{S}) = ProductSpace(tuple(P1.spaces..., P2.spaces...))
⊗{G,S}(V1::S, P2::InvariantSpace{G,S}) = ProductSpace(tuple(V1, P2.spaces...))
⊗{G,S}(P1::InvariantSpace{G,S}, V2::S) = ProductSpace(tuple(P1.spaces..., V2))

# Functionality for extracting and iterating over spaces
Base.length{G,S,N}(P::InvariantSpace{G,S,N}) = N
Base.endof(P::InvariantSpace) = length(P)
Base.getindex(P::InvariantSpace, n::Integer) = P.spaces[n]
Base.getindex{G,S,N}(P::InvariantSpace{G,S,N}, r)=ProductSpace{S,length(r)}(P.spaces[r])

Base.reverse{G,S,N}(P::InvariantSpace{G,S,N})=InvariantSpace{G,S,N}(reverse(P.spaces),[reverse(c)=>dim(P,c) for c in sectors(P)])
Base.map(f::Base.Callable,P::InvariantSpace) = map(f,P.spaces) # required to make map(dim,P) efficient

Base.start(P::InvariantSpace) = start(P.spaces)
Base.next(P::InvariantSpace, state) = next(P.spaces, state)
Base.done(P::InvariantSpace, state) = done(P.spaces, state)

# Corresponding methods
sectors(P::InvariantSpace) = keys(P.dims)
dim(P::InvariantSpace) = sum(values(P.dims))
dim{G,S,N}(P::InvariantSpace{G,S,N},sector::NTuple{N,G})=get(P.dims,sector,0)
iscnumber(P::InvariantSpace) = length(P)==0 || all(iscnumber,P)

# Convention on dual, conj, transpose and ctranspose of tensor product spaces
dual{G,S,N}(P::InvariantSpace{G,S,N}) = InvariantSpace{G,S,N}(ntuple(N,n->dual(P[n])),[map(conj,s)=>dim(P,s) for s in sectors(P)])
Base.conj(P::InvariantSpace) = dual(P) # since all AbelianSpaces are EuclideanSpaces

Base.transpose(P::InvariantSpace) = reverse(P)
Base.ctranspose(P::InvariantSpace) = reverse(conj(P))

# Promotion and conversion
==(P1::InvariantSpace,P2::InvariantSpace) = (P1.spaces == P2.spaces)
issubspace(P1::InvariantSpace,P2::ProductSpace) = P1.spaces==P2.spaces

# Show method
function Base.show(io::IO, P::InvariantSpace)
    print(io,"InvariantSpace(")
    showcompact(io,P.dims)
    print(io,")")
end

