# General definition: might need to be generalized for nonabelian sectors
immutable InvariantSpace{G<:Sector,S<:UnitaryRepresentationSpace,N} <: TensorSpace{S,N}
    spaces::NTuple{N,S}
    dims::Dict{NTuple{N,G},Int}
end

# Additional constructors
InvariantSpace(P::InvariantSpace) = P
invariant(P::InvariantSpace) = P

# Specific constructors for Abelian Sectors
function InvariantSpace{S<:AbelianSpace,N}(spaces::NTuple{N,S})
    sectorlist=_invariantsectors(spaces)
    dims=Dict{eltype(sectorlist),Int}()
    sizehint(dims,length(sectorlist))
    for s in sectorlist
        dims[s]=_dim(spaces,s)
    end
    return InvariantSpace(spaces,dims)
end

InvariantSpace{S<:AbelianSpace}(V::S,Vlist::S...) = InvariantSpace(tuple(V,Vlist...))
InvariantSpace{S<:AbelianSpace}(P::ProductSpace{S}) = InvariantSpace(P.spaces)

invariant{S<:AbelianSpace}(V::S) = InvariantSpace(tuple(V))
invariant{S<:AbelianSpace}(P::ProductSpace{S}) = InvariantSpace(P.spaces)

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
==(P::InvariantSpace,V::ElementarySpace) = length(P)==1 && P[1]==V
==(V::ElementarySpace,P::InvariantSpace) = length(P)==1 && P[1]==V

issubspace(P1::InvariantSpace,P2::ProductSpace) = P1.spaces==P2.spaces

# Show method
function Base.show(io::IO, P::InvariantSpace)
    print("invariant(")
    for i in 1:length(P)
        i==1 || print(io," ⊗ ")
        show(io, P[i])
    end
    print(")")
end

# Auxiliary functions
_dim{G<:Sector}(spaces::NTuple{1,UnitaryRepresentationSpace{G}},sectors::NTuple{1,G})=dim(spaces[1],sectors[1])
_dim{G<:Sector,N}(spaces::NTuple{N,UnitaryRepresentationSpace{G}},sectors::NTuple{N,G})=_dim(spaces[1:div(N,2)],sectors[1:div(N,2)])*_dim(spaces[div(N,2)+1:end],sectors[div(N,2)+1:end])

using Cartesian
@ngenerate N Vector{NTuple{N,G}} function _invariantsectors{G<:Abelian,N}(spaces::NTuple{N,AbelianSpace{G}})
    @nexprs N i->(s_i=collect(sectors(spaces[i])))
    sectorlist=Array(NTuple{N,G},0)
    @nloops N i d->1:length(s_d) begin
        sector=@ntuple N k->s_k[i_k]
        if prod(sector)==one(G)
            push!(sectorlist,sector)
        end
    end
    return sectorlist
end
