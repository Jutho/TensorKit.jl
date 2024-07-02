# abstracttensor.jl
#
# Abstract Tensor type
#----------------------
"""
    abstract type AbstractTensorMap{S<:IndexSpace, N₁, N₂} end

Abstract supertype of all tensor maps, i.e. linear maps between tensor products
of vector spaces of type `S<:IndexSpace`. An `AbstractTensorMap` maps from
an input space of type `ProductSpace{S, N₂}` to an output space of type
`ProductSpace{S, N₁}`.
"""
abstract type AbstractTensorMap{S<:IndexSpace,N₁,N₂} end
"""
    AbstractTensor{S<:IndexSpace, N} = AbstractTensorMap{S, N, 0}

Abstract supertype of all tensors, i.e. elements in the tensor product space
of type `ProductSpace{S, N}`, built from elementary spaces of type `S<:IndexSpace`.

An `AbstractTensor{S, N}` is actually a special case `AbstractTensorMap{S, N, 0}`,
i.e. a tensor map with only a non-trivial output space.
"""
const AbstractTensor{S<:IndexSpace,N} = AbstractTensorMap{S,N,0}

# tensor characteristics
#------------------------
Base.eltype(::Union{T,Type{T}}) where {T<:AbstractTensorMap} = scalartype(T)

"""
    spacetype(::Union{T,Type{T}}) where {T<:AbstractTensorMap} -> Type{S<:IndexSpace}

Return the type of the elementary space `S` of a tensor.
"""
spacetype(::Type{<:AbstractTensorMap{S}}) where {S<:IndexSpace} = S

"""
    sectortype(::Union{T,Type{T}}) where {T<:AbstractTensorMap} -> Type{I<:Sector}

Return the type of sector `I` of a tensor.
"""
sectortype(::Type{<:AbstractTensorMap{S}}) where {S<:IndexSpace} = sectortype(S)

function InnerProductStyle(::Type{<:AbstractTensorMap{S}}) where {S<:IndexSpace}
    return InnerProductStyle(S)
end

"""
    field(::Union{T,Type{T}}) where {T<:AbstractTensorMap} -> Type{𝕂<:Field}

Return the type of field `𝕂` of a tensor.
"""
field(::Type{<:AbstractTensorMap{S}}) where {S<:IndexSpace} = field(S)

"""
    numout(::Union{T,Type{T}}) where {T<:AbstractTensorMap} -> Int

Return the number of output spaces of a tensor. This is equivalent to the number of spaces in the codomain of that tensor.

See also [`numin`](@ref) and [`numind`](@ref).
"""
numout(::Type{<:AbstractTensorMap{<:IndexSpace,N₁,N₂}}) where {N₁,N₂} = N₁

"""
    numin(::Union{T,Type{T}}) where {T<:AbstractTensorMap} -> Int

Return the number of input spaces of a tensor. This is equivalent to the number of spaces in the domain of that tensor.

See also [`numout`](@ref) and [`numind`](@ref).
"""
numin(::Type{<:AbstractTensorMap{<:IndexSpace,N₁,N₂}}) where {N₁,N₂} = N₂

"""
    numind(::Union{T,Type{T}}) where {T<:AbstractTensorMap} -> Int

Return the total number of input and output spaces of a tensor. This is equivalent to the
total number of spaces in the domain and codomain of that tensor.

See also [`numout`](@ref) and [`numin`](@ref).
"""
numind(::Type{<:AbstractTensorMap{<:IndexSpace,N₁,N₂}}) where {N₁,N₂} = N₁ + N₂

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
    codomain(t::AbstractTensorMap{S,N₁,N₂}, [i::Int]) -> ProductSpace{S,N₁}

Return the codomain of a tensor, i.e. the product space of the output spaces. If `i` is
specified, return the `i`-th output space. Implementations should provide `codomain(t)`.

See also [`domain`](@ref) and [`space`](@ref).
""" codomain

codomain(t::AbstractTensorMap, i) = codomain(t)[i]
target(t::AbstractTensorMap) = codomain(t) # categorical terminology

@doc """
    domain(t::AbstractTensorMap{S,N₁,N₂}, [i::Int]) -> ProductSpace{S,N₂}

Return the domain of a tensor, i.e. the product space of the input spaces. If `i` is
specified, return the `i`-th input space. Implementations should provide `domain(t)`.

See also [`codomain`](@ref) and [`space`](@ref).
""" domain

domain(t::AbstractTensorMap, i) = domain(t)[i]
source(t::AbstractTensorMap) = domain(t) # categorical terminology

"""
    space(t::AbstractTensorMap{S,N₁,N₂}, [i::Int]) -> HomSpace{S,N₁,N₂}

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
function codomainind(::Type{<:AbstractTensorMap{<:IndexSpace,N₁,N₂}}) where {N₁,N₂}
    return ntuple(n -> n, N₁)
end
codomainind(t::AbstractTensorMap) = codomainind(typeof(t))

"""
    domainind(::Union{T,Type{T}}) where {T<:AbstractTensorMap} -> Tuple{Int}

Return all indices of the domain of a tensor.

See also [`codomainind`](@ref) and [`allind`](@ref).
"""
function domainind(::Type{<:AbstractTensorMap{<:IndexSpace,N₁,N₂}}) where {N₁,N₂}
    return ntuple(n -> N₁ + n, N₂)
end
domainind(t::AbstractTensorMap) = domainind(typeof(t))

"""
    allind(::Union{T,Type{T}}) where {T<:AbstractTensorMap} -> Tuple{Int}

Return all indices of a tensor, i.e. the indices of its domain and codomain.

See also [`codomainind`](@ref) and [`domainind`](@ref).
"""
function allind(::Type{<:AbstractTensorMap{<:IndexSpace,N₁,N₂}}) where {N₁,N₂}
    return ntuple(n -> n, N₁ + N₂)
end
allind(t::AbstractTensorMap) = allind(typeof(t))

function adjointtensorindex(::AbstractTensorMap{<:IndexSpace,N₁,N₂}, i) where {N₁,N₂}
    return ifelse(i <= N₁, N₂ + i, i - N₁)
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
function Base.convert(::Type{Array}, t::AbstractTensorMap{S,N₁,N₂}) where {S,N₁,N₂}
    I = sectortype(t)
    if I === Trivial
        convert(Array, t[])
    else
        cod = codomain(t)
        dom = domain(t)
        local A
        for (f₁, f₂) in fusiontrees(t)
            F = convert(Array, (f₁, f₂))
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
