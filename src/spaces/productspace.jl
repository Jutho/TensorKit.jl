"""
    struct ProductSpace{S<:ElementarySpace, N} <: CompositeSpace{S}

A `ProductSpace` is a tensor product space of `N` vector spaces of type
`S<:ElementarySpace`. Only tensor products between [`ElementarySpace`](@ref) objects of the
same type are allowed.
"""
struct ProductSpace{S<:ElementarySpace,N} <: CompositeSpace{S}
    spaces::NTuple{N,S}
    ProductSpace{S,N}(spaces::NTuple{N,S}) where {S<:ElementarySpace,N} = new{S,N}(spaces)
end

function ProductSpace{S,N}(spaces::Vararg{S,N}) where {S<:ElementarySpace,N}
    return ProductSpace{S,N}(spaces)
end

function ProductSpace{S}(spaces::Tuple{Vararg{S}}) where {S<:ElementarySpace}
    return ProductSpace{S,length(spaces)}(spaces)
end
ProductSpace{S}(spaces::Vararg{S}) where {S<:ElementarySpace} = ProductSpace{S}(spaces)

function ProductSpace(spaces::Tuple{S,Vararg{S}}) where {S<:ElementarySpace}
    return ProductSpace{S,length(spaces)}(spaces)
end
function ProductSpace(space1::ElementarySpace, rspaces::Vararg{ElementarySpace})
    return ProductSpace((space1, rspaces...))
end

ProductSpace(P::ProductSpace) = P

# constructors with conversion behaviour
function ProductSpace{S,N}(V::Vararg{ElementarySpace,N}) where {S<:ElementarySpace,N}
    return ProductSpace{S,N}(V)
end
function ProductSpace{S}(V::Vararg{ElementarySpace}) where {S<:ElementarySpace}
    return ProductSpace{S}(V)
end

function ProductSpace{S,N}(V::Tuple{Vararg{ElementarySpace,N}}) where {S<:ElementarySpace,N}
    return ProductSpace{S}(convert.(S, V))
end
function ProductSpace{S}(V::Tuple{Vararg{ElementarySpace}}) where {S<:ElementarySpace}
    return ProductSpace{S}(convert.(S, V))
end
function ProductSpace(V::Tuple{ElementarySpace,Vararg{ElementarySpace}})
    return ProductSpace(promote(V...))
end

# Corresponding methods
#-----------------------
"""
    dims(::ProductSpace{S, N}) -> Dims{N} = NTuple{N, Int}

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
        print(io, "ProductSpace{", S, ", 0}")
    end
    if length(spaces) == 1
        print(io, "ProductSpace")
    end
    print(io, "(")
    for i in 1:length(spaces)
        i == 1 || print(io, " ⊗ ")
        show(io, spaces[i])
    end
    return print(io, ")")
end

# more specific methods
"""
    sectors(P::ProductSpace{S, N}) where {S<:ElementarySpace}

Return an iterator over all possible combinations of sectors (represented as an
`NTuple{N, sectortype(S)}`) that can appear within the tensor product space `P`.
"""
sectors(P::ProductSpace) = _sectors(P, sectortype(P))
function _sectors(P::ProductSpace{<:ElementarySpace,N}, ::Type{Trivial}) where {N}
    return OneOrNoneIterator(dim(P) != 0, ntuple(n -> Trivial(), N))
end
function _sectors(P::ProductSpace{<:ElementarySpace,N}, ::Type{<:Sector}) where {N}
    return product(map(sectors, P.spaces)...)
end

"""
    hassector(P::ProductSpace{S, N}, s::NTuple{N, sectortype(S)}) where {S<:ElementarySpace}
    -> Bool

Query whether `P` has a non-zero degeneracy of sector `s`, representing a combination of
sectors on the individual tensor indices.
"""
function hassector(V::ProductSpace{<:ElementarySpace,N}, s::NTuple{N}) where {N}
    return reduce(&, map(hassector, V.spaces, s); init=true)
end

"""
    dims(P::ProductSpace{S, N}, s::NTuple{N, sectortype(S)}) where {S<:ElementarySpace}
    -> Dims{N} = NTuple{N, Int}

Return the degeneracy dimensions corresponding to a tuple of sectors `s` for each of the
spaces in the tensor product `P`.
"""
function dims(P::ProductSpace{<:ElementarySpace,N}, sector::NTuple{N,<:Sector}) where {N}
    return map(dim, P.spaces, sector)
end

"""
    dim(P::ProductSpace{S, N}, s::NTuple{N, sectortype(S)}) where {S<:ElementarySpace}
    -> Int

Return the total degeneracy dimension corresponding to a tuple of sectors for each of the
spaces in the tensor product, obtained as `prod(dims(P, s))``.
"""
function dim(P::ProductSpace{<:ElementarySpace,N}, sector::NTuple{N,<:Sector}) where {N}
    return reduce(*, dims(P, sector); init=1)
end

function Base.axes(P::ProductSpace{<:ElementarySpace,N},
                   sectors::NTuple{N,<:Sector}) where {N}
    return map(axes, P.spaces, sectors)
end

"""
    blocksectors(P::ProductSpace)

Return an iterator over the different unique coupled sector labels, i.e. the different
fusion outputs that can be obtained by fusing the sectors present in the different spaces
that make up the `ProductSpace` instance.
"""
function blocksectors(P::ProductSpace{S,N}) where {S,N}
    I = sectortype(S)
    if I == Trivial
        return OneOrNoneIterator(dim(P) != 0, Trivial())
    end
    bs = Vector{I}()
    if N == 0
        push!(bs, one(I))
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
    end
    return bs
end

"""
    hasblock(P::ProductSpace, c::Sector)

Query whether a coupled sector `c` appears with nonzero dimension in `P`, i.e. whether
`blockdim(P, c) > 0`.

See also [`blockdim`](@ref) and [`blocksectors`](@ref).
"""
function hasblock(P::ProductSpace, c::Sector)
    sectortype(P) == typeof(c) || throw(SectorMismatch())
    for s in sectors(P)
        if !isempty(fusiontrees(s, c))
            return true
        end
    end
    return false
end

"""
    blockdim(P::ProductSpace, c::Sector)

Return the total dimension of a coupled sector `c` in the product space, by summing over
all `dim(P, s)` for all tuples of sectors `s::NTuple{N, <:Sector}` that can fuse to  `c`,
counted with the correct multiplicity (i.e. number of ways in which `s` can fuse to `c`).

See also [`hasblock`](@ref) and [`blocksectors`](@ref).
"""
function blockdim(P::ProductSpace, c::Sector)
    sectortype(P) == typeof(c) || throw(SectorMismatch())
    d = 0
    for s in sectors(P)
        ds = dim(P, s)
        d += length(fusiontrees(s, c)) * ds
    end
    return d
end

Base.:(==)(P1::ProductSpace, P2::ProductSpace) = (P1.spaces == P2.spaces)

Base.hash(P::ProductSpace, h::UInt) = hash(P.spaces, h)

# Default construction from product of spaces
#---------------------------------------------
⊗(V::ElementarySpace...) = ProductSpace(V...)
⊗(P::ProductSpace) = P
function ⊗(P1::ProductSpace{S}, P2::ProductSpace{S}) where {S<:ElementarySpace}
    return ProductSpace{S}(tuple(P1.spaces..., P2.spaces...))
end

# unit element with respect to the monoidal structure of taking tensor products
"""
    one(::S) where {S<:ElementarySpace} -> ProductSpace{S, 0}
    one(::ProductSpace{S}) where {S<:ElementarySpace} -> ProductSpace{S, 0}

Return a tensor product of zero spaces of type `S`, i.e. this is the unit object under the
tensor product operation, such that `V ⊗ one(V) == V`.
"""
Base.one(V::VectorSpace) = one(typeof(V))
Base.one(::Type{<:ProductSpace{S}}) where {S<:ElementarySpace} = ProductSpace{S,0}(())
Base.one(::Type{S}) where {S<:ElementarySpace} = ProductSpace{S,0}(())

Base.:^(V::ElementarySpace, N::Int) = ProductSpace{typeof(V),N}(ntuple(n -> V, N))
Base.:^(V::ProductSpace, N::Int) = ⊗(ntuple(n -> V, N)...)
function Base.literal_pow(::typeof(^), V::ElementarySpace, p::Val{N}) where {N}
    return ProductSpace{typeof(V),N}(ntuple(n -> V, p))
end

fuse(P::ProductSpace{S,0}) where {S<:ElementarySpace} = oneunit(S)
fuse(P::ProductSpace{S}) where {S<:ElementarySpace} = fuse(P.spaces...)

"""
    insertunit(P::ProductSpace, i::Int = length(P)+1; dual = false, conj = false)

For `P::ProductSpace{S,N}`, this adds an extra tensor product factor at position
`1 <= i <= N+1` (last position by default) which is just the `S`-equivalent of the
underlying field of scalars, i.e. `oneunit(S)`. With the keyword arguments, one can choose
to insert the conjugated or dual space instead, which are all isomorphic to the field of
scalars.
"""
function insertunit(P::ProductSpace, i::Int=length(P) + 1; dual=false, conj=false)
    u = oneunit(spacetype(P))
    if dual
        u = TensorKit.dual(u)
    end
    if conj
        u = TensorKit.conj(u)
    end
    return ProductSpace(TupleTools.insertafter(P.spaces, i - 1, (u,)))
end

# Functionality for extracting and iterating over spaces
#--------------------------------------------------------
Base.length(P::ProductSpace) = length(P.spaces)
Base.getindex(P::ProductSpace, n::Integer) = P.spaces[n]

@inline function Base.iterate(P::ProductSpace, ::Val{i}=Val(1)) where {i}
    if i > length(P)
        return nothing
    else
        return P.spaces[i], Val(i + 1)
    end
end
Base.indexed_iterate(P::ProductSpace, args...) = Base.indexed_iterate(P.spaces, args...)

Base.eltype(::Type{<:ProductSpace{S}}) where {S<:ElementarySpace} = S
Base.eltype(P::ProductSpace) = eltype(typeof(P))

Base.IteratorEltype(::Type{<:ProductSpace}) = Base.HasEltype()
Base.IteratorSize(::Type{<:ProductSpace}) = Base.HasLength()

Base.reverse(P::ProductSpace) = ProductSpace(reverse(P.spaces))

# Promotion and conversion
# ------------------------
function Base.promote_rule(::Type{S}, ::Type{<:ProductSpace{S}}) where {S<:ElementarySpace}
    return ProductSpace{S}
end

# ProductSpace to ElementarySpace
Base.convert(::Type{S}, P::ProductSpace{S,0}) where {S<:ElementarySpace} = oneunit(S)
Base.convert(::Type{S}, P::ProductSpace{S}) where {S<:ElementarySpace} = fuse(P.spaces...)

# ElementarySpace to ProductSpace
Base.convert(::Type{<:ProductSpace}, V::S) where {S<:ElementarySpace} = ⊗(V)
