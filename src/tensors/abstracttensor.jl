# abstracttensor.jl
#
# Abstract Tensor type
#----------------------
"""
    abstract type AbstractTensorMap{E<:Number, S<:IndexSpace, N₁, N₂} end

Abstract supertype of all tensor maps, i.e. linear maps between tensor products of vector
spaces of type `S<:IndexSpace`, with element type `E`. An `AbstractTensorMap` maps from an
input space of type `ProductSpace{S, N₂}` to an output space of type `ProductSpace{S, N₁}`.
"""
abstract type AbstractTensorMap{E<:Number,S<:IndexSpace,N₁,N₂} end

"""
    AbstractTensor{E,S,N} = AbstractTensorMap{E,S,N,0}

Abstract supertype of all tensors, i.e. elements in the tensor product space of type
`ProductSpace{S, N}`, with element type `E`.

An `AbstractTensor{E, S, N}` is actually a special case `AbstractTensorMap{E, S, N, 0}`,
i.e. a tensor map with only non-trivial output spaces.
"""
const AbstractTensor{E,S,N} = AbstractTensorMap{E,S,N,0}

# tensor characteristics
#------------------------
Base.eltype(::Type{<:AbstractTensorMap{E}}) where {E} = E

"""
    spacetype(::Union{T,Type{T}}) where {T<:AbstractTensorMap} -> Type{S<:IndexSpace}

Return the type of the elementary space `S` of a tensor.
"""
spacetype(::Type{<:AbstractTensorMap{E,S}}) where {E,S<:IndexSpace} = S

"""
    sectortype(::Union{T,Type{T}}) where {T<:AbstractTensorMap} -> Type{I<:Sector}

Return the type of sector `I` of a tensor.
"""
sectortype(::Type{T}) where {T<:AbstractTensorMap} = sectortype(spacetype(T))

function InnerProductStyle(::Type{T}) where {T<:AbstractTensorMap}
    return InnerProductStyle(spacetype(T))
end

"""
    field(::Union{T,Type{T}}) where {T<:AbstractTensorMap} -> Type{𝕂<:Field}

Return the type of field `𝕂` of a tensor.
"""
field(::Type{T}) where {T<:AbstractTensorMap} = field(spacetype(T))

"""
    numout(::Union{T,Type{T}}) where {T<:AbstractTensorMap} -> Int

Return the number of output spaces of a tensor. This is equivalent to the number of spaces in the codomain of that tensor.

See also [`numin`](@ref) and [`numind`](@ref).
"""
numout(::Type{<:AbstractTensorMap{E,S,N₁}}) where {E,S,N₁} = N₁

"""
    numin(::Union{T,Type{T}}) where {T<:AbstractTensorMap} -> Int

Return the number of input spaces of a tensor. This is equivalent to the number of spaces in the domain of that tensor.

See also [`numout`](@ref) and [`numind`](@ref).
"""
numin(::Type{<:AbstractTensorMap{E,S,N₁,N₂}}) where {E,S,N₁,N₂} = N₂

"""
    numind(::Union{T,Type{T}}) where {T<:AbstractTensorMap} -> Int

Return the total number of input and output spaces of a tensor. This is equivalent to the
total number of spaces in the domain and codomain of that tensor.

See also [`numout`](@ref) and [`numin`](@ref).
"""
numind(::Type{T}) where {T<:AbstractTensorMap} = numin(T) + numout(T)

function similarstoragetype(TT::Type{<:AbstractTensorMap}, ::Type{T}) where {T}
    return Core.Compiler.return_type(similar, Tuple{storagetype(TT),Type{T}})
end

spacetype(t::AbstractTensorMap) = spacetype(typeof(t))
sectortype(t::AbstractTensorMap) = sectortype(typeof(t))
InnerProductStyle(t::AbstractTensorMap) = InnerProductStyle(typeof(t))
field(t::AbstractTensorMap) = field(typeof(t))
numout(t::AbstractTensorMap) = numout(typeof(t))
numin(t::AbstractTensorMap) = numin(typeof(t))
numind(t::AbstractTensorMap) = numind(typeof(t))

storagetype(t::AbstractTensorMap) = storagetype(typeof(t))
similarstoragetype(t::AbstractTensorMap, T) = similarstoragetype(typeof(t), T)

const order = numind

@doc """
    codomain(t::AbstractTensorMap{E,S,N₁,N₂}, [i::Int]) -> ProductSpace{S,N₁}

Return the codomain of a tensor, i.e. the product space of the output spaces. If `i` is
specified, return the `i`-th output space. Implementations should provide `codomain(t)`.

See also [`domain`](@ref) and [`space`](@ref).
""" codomain

codomain(t::AbstractTensorMap, i) = codomain(t)[i]
target(t::AbstractTensorMap) = codomain(t) # categorical terminology

@doc """
    domain(t::AbstractTensorMap{E,S,N₁,N₂}, [i::Int]) -> ProductSpace{S,N₂}

Return the domain of a tensor, i.e. the product space of the input spaces. If `i` is
specified, return the `i`-th input space. Implementations should provide `domain(t)`.

See also [`codomain`](@ref) and [`space`](@ref).
""" domain

domain(t::AbstractTensorMap, i) = domain(t)[i]
source(t::AbstractTensorMap) = domain(t) # categorical terminology

"""
    space(t::AbstractTensorMap{E,S,N₁,N₂}, [i::Int]) -> HomSpace{S,N₁,N₂}

The index information of a tensor, i.e. the `HomSpace` of its domain and codomain. If `i` is specified, return the `i`-th index space.
"""
space(t::AbstractTensorMap) = HomSpace(codomain(t), domain(t))
space(t::AbstractTensorMap, i::Int) = space(t)[i]

"""
    dim(t::AbstractTensorMap) -> Int

The total number of free parameters of a tensor, discounting the entries that are fixed by
symmetry. This is also the dimension of the `HomSpace` on which the `TensorMap` is defined.
"""
dim(t::AbstractTensorMap) = dim(space(t))

"""
    codomainind(::Union{T,Type{T}}) where {T<:AbstractTensorMap} -> Tuple{Int}

Return all indices of the codomain of a tensor.

See also [`domainind`](@ref) and [`allind`](@ref).
"""
function codomainind(::Type{T}) where {T<:AbstractTensorMap}
    return ntuple(identity, numout(T))
end
codomainind(t::AbstractTensorMap) = codomainind(typeof(t))

"""
    domainind(::Union{T,Type{T}}) where {T<:AbstractTensorMap} -> Tuple{Int}

Return all indices of the domain of a tensor.

See also [`codomainind`](@ref) and [`allind`](@ref).
"""
function domainind(::Type{T}) where {T<:AbstractTensorMap}
    return ntuple(n -> numout(T) + n, numin(T))
end
domainind(t::AbstractTensorMap) = domainind(typeof(t))

"""
    allind(::Union{T,Type{T}}) where {T<:AbstractTensorMap} -> Tuple{Int}

Return all indices of a tensor, i.e. the indices of its domain and codomain.

See also [`codomainind`](@ref) and [`domainind`](@ref).
"""
function allind(::Type{T}) where {T<:AbstractTensorMap}
    return ntuple(n -> n, numind(T))
end
allind(t::AbstractTensorMap) = allind(typeof(t))

function adjointtensorindex(t::AbstractTensorMap, i)
    return ifelse(i <= numout(t), numin(t) + i, i - numout(t))
end

function adjointtensorindices(t::AbstractTensorMap, indices::IndexTuple)
    return map(i -> adjointtensorindex(t, i), indices)
end

function adjointtensorindices(t::AbstractTensorMap, p::Index2Tuple)
    return adjointtensorindices(t, p[1]), adjointtensorindices(t, p[2])
end

@doc """
    blocks(t::AbstractTensorMap) -> SectorDict{<:Sector,<:DenseMatrix}

Return an iterator over all blocks of a tensor, i.e. all coupled sectors and their
corresponding blocks.

See also [`block`](@ref), [`blocksectors`](@ref), [`blockdim`](@ref) and [`hasblock`](@ref).
""" blocks

@doc """
    block(t::AbstractTensorMap, c::Sector) -> DenseMatrix

Return the block of a tensor corresponding to a coupled sector `c`.

See also [`blocks`](@ref), [`blocksectors`](@ref), [`blockdim`](@ref) and [`hasblock`](@ref).
""" block

"""
    hasblock(t::AbstractTensorMap, c::Sector) -> Bool

Verify whether a tensor has a block corresponding to a coupled sector `c`.
"""
hasblock

@doc """
    blocksectors(t::AbstractTensorMap)

Return an iterator over all coupled sectors of a tensor.
""" blocksectors(::AbstractTensorMap)

@doc """
    blockdim(t::AbstractTensorMap, c::Sector) -> Base.Dims

Return the dimensions of the block of a tensor corresponding to a coupled sector `c`.
""" blockdim(::AbstractTensorMap, ::Sector)

@doc """
    fusiontrees(t::AbstractTensorMap)

Return an iterator over all splitting - fusion tree pairs of a tensor.
""" fusiontrees(::AbstractTensorMap)

# Similar
#---------
# The implementation is written for similar(t, TorA, V::TensorMapSpace) -> TensorMap
# and all other methods are just filling in default arguments
# 4 arguments
@doc """
    similar(t::AbstractTensorMap, [AorT=storagetype(t)], [V=space(t)])
    similar(t::AbstractTensorMap, [AorT=storagetype(t)], codomain, domain)

Creates an uninitialized mutable tensor with the given scalar or storagetype `AorT` and
structure `V` or `codomain ← domain`, based on the source tensormap. The second and third
arguments are both optional, defaulting to the given tensor's `storagetype` and `space`.
The structure may be specified either as a single `HomSpace` argument or as `codomain` and
`domain`.

By default, this will result in `TensorMap{T}(undef, V)` when custom objects do not
specialize this method.
""" Base.similar(::AbstractTensorMap, args...)

function Base.similar(t::AbstractTensorMap, ::Type{T}, codomain::TensorSpace{S},
                      domain::TensorSpace{S}) where {T,S}
    return similar(t, T, codomain ← domain)
end
# 3 arguments
function Base.similar(t::AbstractTensorMap, codomain::TensorSpace{S},
                      domain::TensorSpace{S}) where {S}
    return similar(t, storagetype(t), codomain ← domain)
end
function Base.similar(t::AbstractTensorMap, ::Type{T}, codomain::TensorSpace) where {T}
    return similar(t, T, codomain ← one(codomain))
end
# 2 arguments
function Base.similar(t::AbstractTensorMap, codomain::TensorSpace)
    return similar(t, storagetype(t), codomain ← one(codomain))
end
Base.similar(t::AbstractTensorMap, P::TensorMapSpace) = similar(t, storagetype(t), P)
Base.similar(t::AbstractTensorMap, ::Type{T}) where {T} = similar(t, T, space(t))
# 1 argument
Base.similar(t::AbstractTensorMap) = similar(t, storagetype(t), space(t))

# generic implementation for AbstractTensorMap -> returns `TensorMap`
function Base.similar(::AbstractTensorMap, ::Type{TorA},
                      P::TensorMapSpace{S}) where {TorA<:MatOrNumber,S}
    N₁ = length(codomain(P))
    N₂ = length(domain(P))
    TT = tensormaptype(S, N₁, N₂, TorA)
    return TT(undef, codomain(P), domain(P))
end

# Equality and approximality
#----------------------------
function Base.:(==)(t1::AbstractTensorMap, t2::AbstractTensorMap)
    (codomain(t1) == codomain(t2) && domain(t1) == domain(t2)) || return false
    for c in blocksectors(t1)
        block(t1, c) == block(t2, c) || return false
    end
    return true
end
function Base.hash(t::AbstractTensorMap, h::UInt)
    h = hash(codomain(t), h)
    h = hash(domain(t), h)
    for (c, b) in blocks(t)
        h = hash(c, hash(b, h))
    end
    return h
end

function Base.isapprox(t1::AbstractTensorMap, t2::AbstractTensorMap;
                       atol::Real=0,
                       rtol::Real=Base.rtoldefault(scalartype(t1), scalartype(t2), atol))
    d = norm(t1 - t2)
    if isfinite(d)
        return d <= max(atol, rtol * max(norm(t1), norm(t2)))
    else
        return false
    end
end

# Conversion to Array:
#----------------------
# probably not optimized for speed, only for checking purposes
function Base.convert(::Type{Array}, t::AbstractTensorMap)
    I = sectortype(t)
    if I === Trivial
        convert(Array, t[])
    else
        cod = codomain(t)
        dom = domain(t)
        local A
        for (f₁, f₂) in fusiontrees(t)
            F₁ = convert(Array, f₁)
            F₂ = convert(Array, f₂)
            sz1 = size(F₁)
            sz2 = size(F₂)
            d1 = TupleTools.front(sz1)
            d2 = TupleTools.front(sz2)
            F = reshape(reshape(F₁, TupleTools.prod(d1), sz1[end]) *
                        reshape(F₂, TupleTools.prod(d2), sz2[end])', (d1..., d2...))
            if !(@isdefined A)
                if eltype(F) <: Complex
                    T = complex(float(scalartype(t)))
                elseif eltype(F) <: Integer
                    T = scalartype(t)
                else
                    T = float(scalartype(t))
                end
                A = fill(zero(T), (dims(cod)..., dims(dom)...))
            end
            Aslice = StridedView(A)[axes(cod, f₁.uncoupled)..., axes(dom, f₂.uncoupled)...]
            axpy!(1, StridedView(_kron(convert(Array, t[f₁, f₂]), F)), Aslice)
        end
        return A
    end
end
