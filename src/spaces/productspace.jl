"""
    struct ProductSpace{S<:ElementarySpace, N} <: CompositeSpace{S}

A `ProductSpace` is a tensor product space of `N` vector spaces of type
`S<:ElementarySpace`. Only tensor products between [`ElementarySpace`](@ref) objects of the
same type are allowed.
"""
struct ProductSpace{S<:ElementarySpace, N} <: CompositeSpace{S}
    spaces::NTuple{N, S}
end
ProductSpace(spaces::Vararg{S,N}) where {S<:ElementarySpace, N} =
    ProductSpace{S,N}(spaces)
ProductSpace{S,N}(spaces::Vararg{S,N}) where {S<:ElementarySpace, N} =
    ProductSpace{S,N}(spaces)

# Corresponding methods
#-----------------------
"""
    dims(::ProductSpace{S,N}) -> Dims{N} = NTuple{N,Int}

Return the dimensions of the spaces in the tensor product space as a tuple of integers.
"""
dims(P::ProductSpace) = map(dim, P.spaces)
dim(P::ProductSpace, n::Int) = dim(P.spaces[n])
dim(P::ProductSpace) = prod(dims(P))

Base.axes(P::ProductSpace) = map(axes, P.spaces)
Base.axes(P::ProductSpace, n::Int) = axes(P.spaces[n])

dual(P::ProductSpace{<:ElementarySpace,0}) = P
dual(P::ProductSpace) = ProductSpace(map(dual, reverse(P.spaces)))

# Base.conj(P::ProductSpace) = ProductSpace(map(conj, P.spaces))

function Base.show(io::IO, P::ProductSpace{S}) where {S<:ElementarySpace}
    spaces = P.spaces
    if length(spaces) == 0
        print(io,"ProductSpace{", S, ",0}")
    end
    if length(spaces) == 1
        print(io,"ProductSpace")
    end
    print(io,"(")
    for i in 1:length(spaces)
        i==1 || print(io," ⊗ ")
        show(io, spaces[i])
    end
    print(io,")")
end

# more specific methods
sectors(P::ProductSpace) = _sectors(P, sectortype(P))
_sectors(P::ProductSpace{<:ElementarySpace, N}, ::Type{Trivial}) where {N} =
    (ntuple(n->Trivial(), StaticLength{N}()),) # speed up sectors for ungraded spaces
_sectors(P::ProductSpace{<:ElementarySpace, N}, ::Type{<:Sector}) where {N} =
    product(map(sectors, P.spaces)...)

hassector(V::ProductSpace{<:ElementarySpace,N}, s::NTuple{N}) where {N} =
    reduce(&, map(hassector, V.spaces, s); init = true)

dims(P::ProductSpace{<:ElementarySpace, N}, sector::NTuple{N,<:Sector}) where {N} =
    map(dim, P.spaces, sector)
dim(P::ProductSpace{<:ElementarySpace, N}, sector::NTuple{N,<:Sector}) where {N} =
    prod(dims(P, sector))

Base.axes(P::ProductSpace{<:ElementarySpace,N}, sectors::NTuple{N,<:Sector}) where {N} =
    map(axes, P.spaces, sectors)

"""
    dims(::ProductSpace{S,N}) -> Dims{N} = NTuple{N,Int}

Return the dimensions of the spaces in the tensor product space as a tuple of integers.
"""
function blocksectors(P::ProductSpace{S,N}) where {S,N}
    G = sectortype(S)
    if G == Trivial
        return (Trivial(),)
    end
    bs = Vector{G}()
    if N == 0
        push!(bs, one(G))
    elseif N == 1
        for s in sectors(P)
            push!(bs, first(s))
        end
    else
        for s in sectors(P)
            for c in ⊗(s...)
                if !(c in bs)
                    push!(bs, c)
                end
            end
        end
        # return foldl(union!, Set{G}(), (⊗(s...) for s in sectors(P)))
    end
    return bs
end
function blockdim(P::ProductSpace, c::Sector)
    sectortype(P) == typeof(c) || throw(SectorMismatch())
    d = 0
    for s in sectors(P)
        ds = dim(P, s)
        for f in fusiontrees(s, c)
            d += ds
        end
    end
    return d
end

Base.:(==)(P1::ProductSpace, P2::ProductSpace) = (P1.spaces == P2.spaces)

# Default construction from product of spaces
#---------------------------------------------
⊗(V1::S, V2::S) where {S<:ElementarySpace}= ProductSpace((V1, V2))
⊗(P1::ProductSpace{S}, V2::S) where {S<:ElementarySpace} =
    ProductSpace(tuple(P1.spaces..., V2))
⊗(V1::S, P2::ProductSpace{S}) where {S<:ElementarySpace} =
    ProductSpace(tuple(V1, P2.spaces...))
⊗(P1::ProductSpace{S}, P2::ProductSpace{S}) where {S<:ElementarySpace} =
    ProductSpace(tuple(P1.spaces..., P2.spaces...))
⊗(P::ProductSpace{S,0}, ::ProductSpace{S,0}) where {S<:ElementarySpace} = P
⊗(P::ProductSpace{S}, ::ProductSpace{S,0}) where {S<:ElementarySpace} = P
⊗(::ProductSpace{S,0}, P::ProductSpace{S}) where {S<:ElementarySpace} = P
⊗(V::ElementarySpace) = ProductSpace((V,))
⊗(P::ProductSpace) = P

# unit element with respect to the monoidal structure of taking tensor products
"""
    one(::S) where {S<:ElementarySpace} -> ProductSpace{S,0}
    one(::ProductSpace{S}) where {S<:ElementarySpace} -> ProductSpace{S,0}

Return a tensor product of zero spaces of type `S`, i.e. this is the unit object under the
tensor product operation, such that `V ⊗ one(V) == V`.
"""
Base.one(V::VectorSpace) = one(typeof(V))
Base.one(::Type{<:ProductSpace{S}}) where {S<:ElementarySpace} = ProductSpace{S,0}(())
Base.one(::Type{S}) where {S<:ElementarySpace} = ProductSpace{S,0}(())

Base.convert(::Type{<:ProductSpace}, V::ElementarySpace) = ProductSpace((V,))
Base.literal_pow(::typeof(^), V::ElementarySpace, p::Val) = ProductSpace(ntuple(n->V, p))
Base.convert(::Type{S}, P::ProductSpace{S,0}) where {S<:ElementarySpace} = oneunit(S)
Base.convert(::Type{S}, P::ProductSpace{S}) where {S<:ElementarySpace} = fuse(P.spaces...)
fuse(P::ProductSpace{S,0}) where {S<:ElementarySpace} = oneunit(S)
fuse(P::ProductSpace{S}) where {S<:ElementarySpace} = fuse(P.spaces...)

# Functionality for extracting and iterating over spaces
#--------------------------------------------------------
Base.length(P::ProductSpace) = length(P.spaces)
Base.getindex(P::ProductSpace, n::Integer) = P.spaces[n]
Base.getindex(P::ProductSpace{S}, I::NTuple{N,Integer}) where {S<:ElementarySpace,N} =
    ProductSpace{S,N}(TupleTools.getindices(P.spaces, I))

Base.iterate(P::ProductSpace) = Base.iterate(P.spaces)
Base.iterate(P::ProductSpace, s) = Base.iterate(P.spaces, s)

Base.eltype(P::ProductSpace{S}) where {S<:ElementarySpace} = S

Base.IteratorEltype(P::ProductSpace) = Base.IteratorEltype(P.spaces)
Base.IteratorSize(P::ProductSpace) = Base.IteratorSize(P.spaces)
