# TensorMap & Tensor:
# general tensor implementation with arbitrary symmetries
#==========================================================#
#! format: off
"""
    struct TensorMap{S<:IndexSpace, N₁, N₂, ...} <: AbstractTensorMap{S, N₁, N₂}

Specific subtype of [`AbstractTensorMap`](@ref) for representing tensor maps (morphisms in
a tensor category) whose data is stored in blocks of some subtype of `DenseMatrix`.
"""
struct TensorMap{S<:IndexSpace, N₁, N₂, I<:Sector, A<:Union{<:DenseMatrix,SectorDict{I,<:DenseMatrix}}, F₁, F₂} <: AbstractTensorMap{S, N₁, N₂}
    data::A
    codom::ProductSpace{S,N₁}
    dom::ProductSpace{S,N₂}
    rowr::SectorDict{I,FusionTreeDict{F₁,UnitRange{Int}}}
    colr::SectorDict{I,FusionTreeDict{F₂,UnitRange{Int}}}
    function TensorMap{S, N₁, N₂, I, A, F₁, F₂}(data::A,
                codom::ProductSpace{S,N₁}, dom::ProductSpace{S,N₂},
                rowr::SectorDict{I,FusionTreeDict{F₁,UnitRange{Int}}},
                colr::SectorDict{I,FusionTreeDict{F₂,UnitRange{Int}}}) where
                    {S<:IndexSpace, N₁, N₂, I<:Sector, A<:SectorDict{I,<:DenseMatrix},
                     F₁<:FusionTree{I,N₁}, F₂<:FusionTree{I,N₂}}
        T = scalartype(valtype(data))
        T ⊆ field(S) || @warn("scalartype(data) = $T ⊈ $(field(S)))", maxlog = 1)
        return new{S,N₁,N₂,I,A,F₁,F₂}(data, codom, dom, rowr, colr)
    end
    function TensorMap{S,N₁,N₂,Trivial,A,Nothing,Nothing}(data::A,
                                                          codom::ProductSpace{S,N₁},
                                                          dom::ProductSpace{S,N₂}) where
             {S<:IndexSpace,N₁,N₂,A<:DenseMatrix}
        T = scalartype(data)
        T ⊆ field(S) ||
            @warn("scalartype(data) = $T ⊈ $(field(S)))", maxlog = 1)
        return new{S,N₁,N₂,Trivial,A,Nothing,Nothing}(data, codom, dom)
    end
end
#! format: on

"""
    Tensor{S, N, I, A, F₁, F₂} = TensorMap{S, N, 0, I, A, F₁, F₂}

Specific subtype of [`AbstractTensor`](@ref) for representing tensors whose data is stored
in blocks of some subtype of `DenseMatrix`.

A `Tensor{S, N, I, A, F₁, F₂}` is actually a special case `TensorMap{S, N, 0, I, A, F₁, F₂}`,
i.e. a tensor map with only a non-trivial output space.
"""
const Tensor{S,N,I,A,F₁,F₂} = TensorMap{S,N,0,I,A,F₁,F₂}
"""
    TrivialTensorMap{S<:IndexSpace, N₁, N₂, A<:DenseMatrix} = TensorMap{S, N₁, N₂, Trivial, 
                                                                        A, Nothing, Nothing}

A special case of [`TensorMap`](@ref) for representing tensor maps with trivial symmetry,
i.e., whose `sectortype` is `Trivial`.
"""
const TrivialTensorMap{S,N₁,N₂,A<:DenseMatrix} = TensorMap{S,N₁,N₂,Trivial,A,
                                                           Nothing,Nothing}
"""
    TrivialTensor{S, N, A} = TrivialTensorMap{S, N, 0, A}

A special case of [`Tensor`](@ref) for representing tensors with trivial symmetry, i.e.,
whose `sectortype` is `Trivial`.
"""
const TrivialTensor{S,N,A} = TrivialTensorMap{S,N,0,A}
"""
    tensormaptype(::Type{S}, N₁::Int, N₂::Int, [::Type{T}]) where {S<:IndexSpace,T} -> ::Type{<:TensorMap}

Return the fully specified type of a tensor map with elementary space `S`, `N₁` output
spaces and `N₂` input spaces, either with scalar type `T` or with storage type `T`.
"""
function tensormaptype(::Type{S}, N₁::Int, N₂::Int, ::Type{T}) where {S,T}
    I = sectortype(S)
    if T <: DenseMatrix
        M = T
    elseif T <: Number
        M = Matrix{T}
    else
        throw(ArgumentError("the final argument of `tensormaptype` should either be the scalar or the storage type, i.e. a subtype of `Number` or of `DenseMatrix`"))
    end
    if I === Trivial
        return TensorMap{S,N₁,N₂,I,M,Nothing,Nothing}
    else
        F₁ = fusiontreetype(I, N₁)
        F₂ = fusiontreetype(I, N₂)
        return TensorMap{S,N₁,N₂,I,SectorDict{I,M},F₁,F₂}
    end
end
tensormaptype(S, N₁, N₂=0) = tensormaptype(S, N₁, N₂, Float64)

# Basic methods for characterising a tensor:
#--------------------------------------------
codomain(t::TensorMap) = t.codom
domain(t::TensorMap) = t.dom

blocksectors(t::TrivialTensorMap) = OneOrNoneIterator(dim(t) != 0, Trivial())
blocksectors(t::TensorMap) = keys(t.data)

"""
    storagetype(::Union{T,Type{T}}) where {T<:TensorMap} -> Type{A<:DenseMatrix}

Return the type of the storage `A` of the tensor map.
"""
function storagetype(::Type{<:TensorMap{<:IndexSpace,N₁,N₂,Trivial,A}}) where
         {N₁,N₂,A<:DenseMatrix}
    return A
end
function storagetype(::Type{<:TensorMap{<:IndexSpace,N₁,N₂,I,<:SectorDict{I,A}}}) where
         {N₁,N₂,I<:Sector,A<:DenseMatrix}
    return A
end

dim(t::TensorMap) = mapreduce(x -> length(x[2]), +, blocks(t); init=0)

# General TensorMap constructors
#--------------------------------
# constructor starting from block data
"""
    TensorMap(data::AbstractDict{<:Sector,<:DenseMatrix}, codomain::ProductSpace{S,N₁},
                domain::ProductSpace{S,N₂}) where {S<:ElementarySpace,N₁,N₂}
    TensorMap(data, codomain ← domain)
    TensorMap(data, domain → codomain)

Construct a `TensorMap` by explicitly specifying its block data.

## Arguments
- `data::AbstractDict{<:Sector,<:DenseMatrix}`: dictionary containing the block data for
  each coupled sector `c` as a `DenseMatrix` of size
  `(blockdim(codomain, c), blockdim(domain, c))`.
- `codomain::ProductSpace{S,N₁}`: the codomain as a `ProductSpace` of `N₁` spaces of type
  `S<:ElementarySpace`.
- `domain::ProductSpace{S,N₂}`: the domain as a `ProductSpace` of `N₂` spaces of type
  `S<:ElementarySpace`.

Alternatively, the domain and codomain can be specified by passing a [`HomSpace`](@ref)
using the syntax `codomain ← domain` or `domain → codomain`.
"""
function TensorMap(data::AbstractDict{<:Sector,<:DenseMatrix}, codom::ProductSpace{S,N₁},
                   dom::ProductSpace{S,N₂}) where {S<:IndexSpace,N₁,N₂}
    I = sectortype(S)
    I == keytype(data) || throw(SectorMismatch())
    if I == Trivial
        if dim(dom) != 0 && dim(codom) != 0
            return TensorMap(data[Trivial()], codom, dom)
        else
            return TensorMap(valtype(data)(undef, dim(codom), dim(dom)), codom, dom)
        end
    end
    blocksectoriterator = blocksectors(codom ← dom)
    for c in blocksectoriterator
        haskey(data, c) || throw(SectorMismatch("no data for block sector $c"))
    end
    rowr, rowdims = _buildblockstructure(codom, blocksectoriterator)
    colr, coldims = _buildblockstructure(dom, blocksectoriterator)
    for (c, b) in data
        c in blocksectoriterator || isempty(b) ||
            throw(SectorMismatch("data for block sector $c not expected"))
        isempty(b) || size(b) == (rowdims[c], coldims[c]) ||
            throw(DimensionMismatch("wrong size of block for sector $c"))
    end
    F₁ = fusiontreetype(I, N₁)
    F₂ = fusiontreetype(I, N₂)
    if !isreal(I)
        data2 = SectorDict(c => complex(data[c]) for c in blocksectoriterator)
        A = typeof(data2)
        return TensorMap{S,N₁,N₂,I,A,F₁,F₂}(data2, codom, dom, rowr, colr)
    else
        data2 = SectorDict(c => data[c] for c in blocksectoriterator)
        A = typeof(data2)
        return TensorMap{S,N₁,N₂,I,A,F₁,F₂}(data2, codom, dom, rowr, colr)
    end
end

# constructor from general callable that produces block data
function TensorMap(f, codom::ProductSpace{S,N₁},
                   dom::ProductSpace{S,N₂}) where {S<:IndexSpace,N₁,N₂}
    I = sectortype(S)
    if I == Trivial
        d1 = dim(codom)
        d2 = dim(dom)
        data = f((d1, d2))
        A = typeof(data)
        return TensorMap{S,N₁,N₂,Trivial,A,Nothing,Nothing}(data, codom, dom)
    end
    blocksectoriterator = blocksectors(codom ← dom)
    rowr, rowdims = _buildblockstructure(codom, blocksectoriterator)
    colr, coldims = _buildblockstructure(dom, blocksectoriterator)
    if !isreal(I)
        data = SectorDict(c => complex(f((rowdims[c], coldims[c])))
                          for c in blocksectoriterator)
    else
        data = SectorDict(c => f((rowdims[c], coldims[c])) for c in blocksectoriterator)
    end
    F₁ = fusiontreetype(I, N₁)
    F₂ = fusiontreetype(I, N₂)
    A = typeof(data)
    return TensorMap{S,N₁,N₂,I,A,F₁,F₂}(data, codom, dom, rowr, colr)
end

# auxiliary function
function _buildblockstructure(P::ProductSpace{S,N}, blocksectors) where {S<:IndexSpace,N}
    I = sectortype(S)
    F = fusiontreetype(I, N)
    treeranges = SectorDict{I,FusionTreeDict{F,UnitRange{Int}}}()
    blockdims = SectorDict{I,Int}()
    for s in sectors(P)
        for c in blocksectors
            offset = get!(blockdims, c, 0)
            treerangesc = get!(treeranges, c) do
                return FusionTreeDict{F,UnitRange{Int}}()
            end
            for f in fusiontrees(s, c, map(isdual, P.spaces))
                r = (offset + 1):(offset + dim(P, s))
                push!(treerangesc, f => r)
                offset = last(r)
            end
            blockdims[c] = offset
        end
    end
    return treeranges, blockdims
end

"""
    TensorMap([f, eltype,] codomain::ProductSpace{S,N₁}, domain::ProductSpace{S,N₂})
                where {S<:ElementarySpace,N₁,N₂}
    TensorMap([f, eltype,], codomain ← domain)
    TensorMap([f, eltype,], domain → codomain)

Construct a `TensorMap` from a general callable that produces block data for each coupled
sector.

## Arguments
- `f`: callable object that returns a `DenseMatrix`, or `UndefInitializer`.
- `eltype::Type{<:Number}`: element type of the data.
- `codomain::ProductSpace{S,N₁}`: the codomain as a `ProductSpace` of `N₁` spaces of type
  `S<:ElementarySpace`.
- `domain::ProductSpace{S,N₂}`: the domain as a `ProductSpace` of `N₂` spaces of type
  `S<:ElementarySpace`.

If `eltype` is left unspecified, `f` should support the calling syntax `f(::Tuple{Int,Int})`
such that `f((m, n))` returns a `DenseMatrix` with `size(f((m, n))) == (m, n)`. If `eltype` is
specified, `f` is instead called as `f(eltype, (m, n))`. In the case where `f` is left
unspecified or `undef` is passed explicitly, a `TensorMap` with uninitialized data is
generated.

Alternatively, the domain and codomain can be specified by passing a [`HomSpace`](@ref)
using the syntax `codomain ← domain` or `domain → codomain`.
"""
function TensorMap(f, ::Type{T}, codom::ProductSpace{S},
                   dom::ProductSpace{S}) where {S<:IndexSpace,T<:Number}
    return TensorMap(d -> f(T, d), codom, dom)
end

function TensorMap(::Type{T}, codom::ProductSpace{S},
                   dom::ProductSpace{S}) where {S<:IndexSpace,T<:Number}
    return TensorMap(d -> Array{T}(undef, d), codom, dom)
end

function TensorMap(::UndefInitializer, ::Type{T}, codom::ProductSpace{S},
                   dom::ProductSpace{S}) where {S<:IndexSpace,T<:Number}
    return TensorMap(d -> Array{T}(undef, d), codom, dom)
end

function TensorMap(::UndefInitializer, codom::ProductSpace{S},
                   dom::ProductSpace{S}) where {S<:IndexSpace}
    return TensorMap(undef, Float64, codom, dom)
end

function TensorMap(::Type{T}, codom::TensorSpace{S},
                   dom::TensorSpace{S}) where {T<:Number,S<:IndexSpace}
    return TensorMap(T, convert(ProductSpace, codom), convert(ProductSpace, dom))
end

function TensorMap(dataorf, codom::TensorSpace{S},
                   dom::TensorSpace{S}) where {S<:IndexSpace}
    return TensorMap(dataorf, convert(ProductSpace, codom), convert(ProductSpace, dom))
end

function TensorMap(dataorf, ::Type{T}, codom::TensorSpace{S},
                   dom::TensorSpace{S}) where {T<:Number,S<:IndexSpace}
    return TensorMap(dataorf, T, convert(ProductSpace, codom), convert(ProductSpace, dom))
end

function TensorMap(codom::TensorSpace{S}, dom::TensorSpace{S}) where {S<:IndexSpace}
    return TensorMap(Float64, convert(ProductSpace, codom), convert(ProductSpace, dom))
end

function TensorMap(dataorf, T::Type{<:Number}, P::TensorMapSpace{S}) where {S<:IndexSpace}
    return TensorMap(dataorf, T, codomain(P), domain(P))
end

function TensorMap(dataorf, P::TensorMapSpace{S}) where {S<:IndexSpace}
    return TensorMap(dataorf, codomain(P), domain(P))
end

function TensorMap(T::Type{<:Number}, P::TensorMapSpace{S}) where {S<:IndexSpace}
    return TensorMap(T, codomain(P), domain(P))
end

TensorMap(P::TensorMapSpace{S}) where {S<:IndexSpace} = TensorMap(codomain(P), domain(P))

function Tensor(dataorf, T::Type{<:Number}, P::TensorSpace{S}) where {S<:IndexSpace}
    return TensorMap(dataorf, T, P, one(P))
end

Tensor(dataorf, P::TensorSpace{S}) where {S<:IndexSpace} = TensorMap(dataorf, P, one(P))

Tensor(T::Type{<:Number}, P::TensorSpace{S}) where {S<:IndexSpace} = TensorMap(T, P, one(P))

Tensor(P::TensorSpace{S}) where {S<:IndexSpace} = TensorMap(P, one(P))

# constructor starting from a dense array
"""
    TensorMap(data::DenseArray, codomain::ProductSpace{S,N₁}, domain::ProductSpace{S,N₂};
                    tol=sqrt(eps(real(float(eltype(data)))))) where {S<:ElementarySpace,N₁,N₂}
    TensorMap(data, codomain ← domain)
    TensorMap(data, domain → codomain)

Construct a `TensorMap` from a plain multidimensional array.

## Arguments
- `data::DenseArray`: tensor data as a plain array.
- `codomain::ProductSpace{S,N₁}`: the codomain as a `ProductSpace` of `N₁` spaces of type
  `S<:ElementarySpace`.
- `domain::ProductSpace{S,N₂}`: the domain as a `ProductSpace` of `N₂` spaces of type
  `S<:ElementarySpace`.
- `tol=sqrt(eps(real(float(eltype(data)))))::Float64`: 
    
Here, `data` can be specified in two ways. It can either be a `DenseArray` of rank `N₁ + N₂`
whose size matches that of the domain and codomain spaces,
`size(data) == (dims(codomain)..., dims(domain)...)`, or a `DenseMatrix` where
`size(data) == (dim(codomain), dim(domain))`. The `TensorMap` constructor will then
reconstruct the tensor data such that the resulting tensor `t` satisfies
`data == convert(Array, t)`. For the case where `sectortype(S) == Trivial`, the `data` array
is simply reshaped into matrix form and referred to as such in the resulting `TensorMap`
instance. When `S<:GradedSpace`, the `data` array has to be compatible with how how each
sector in every space `V` is assigned to an index range within `1:dim(V)`.

Alternatively, the domain and codomain can be specified by passing a [`HomSpace`](@ref)
using the syntax `codomain ← domain` or `domain → codomain`.

!!! note
    This constructor only works for `sectortype` values for which conversion to a plain
    array is possible, and only in the case where the `data` actually respects the specified
    symmetry structure.
"""
function TensorMap(data::DenseArray, codom::ProductSpace{S,N₁}, dom::ProductSpace{S,N₂};
                   tol=sqrt(eps(real(float(eltype(data)))))) where {S<:IndexSpace,N₁,N₂}
    (d1, d2) = (dim(codom), dim(dom))
    if !(length(data) == d1 * d2 || size(data) == (d1, d2) ||
         size(data) == (dims(codom)..., dims(dom)...))
        throw(DimensionMismatch())
    end

    if sectortype(S) === Trivial
        data2 = reshape(data, (d1, d2))
        A = typeof(data2)
        return TensorMap{S,N₁,N₂,Trivial,A,Nothing,Nothing}(data2, codom, dom)
    end

    t = TensorMap(undef, eltype(data), codom, dom)
    project_symmetric!(t, data)

    if !isapprox(data, convert(Array, t); atol=tol)
        throw(ArgumentError("Data has non-zero elements at incompatible positions"))
    end

    return t
end

"""
    project_symmetric!(t::TensorMap, data::DenseArray) -> TensorMap

Project the data from a dense array `data` into the tensor map `t`. This function discards 
any data that does not fit the symmetry structure of `t`.
"""
function project_symmetric!(t::TensorMap, data::DenseArray)
    if sectortype(t) === Trivial
        copy!(t.data, reshape(data, size(t.data)))
        return t
    end

    for (f₁, f₂) in fusiontrees(t)
        F = convert(Array, (f₁, f₂))
        b = zeros(eltype(data), dims(codomain(t), f₁.uncoupled)...,
                  dims(domain(t), f₂.uncoupled)...)
        szbF = _interleave(size(b), size(F))
        dataslice = sreshape(StridedView(data)[axes(codomain(t), f₁.uncoupled)...,
                                               axes(domain(t), f₂.uncoupled)...], szbF)
        # project (can this be done in one go?)
        d = inv(dim(f₁.coupled))
        for k in eachindex(b)
            b[k] = 1
            projector = _kron(b, F) # probably possible to re-use memory
            t[f₁, f₂][k] = dot(projector, dataslice) * d
            b[k] = 0
        end
    end

    return t
end

# Efficient copy constructors
#-----------------------------
Base.copy(t::TrivialTensorMap) = typeof(t)(copy(t.data), t.codom, t.dom)
Base.copy(t::TensorMap) = typeof(t)(deepcopy(t.data), t.codom, t.dom, t.rowr, t.colr)

# Similar
#---------
# 4 arguments
function Base.similar(t::AbstractTensorMap, T::Type, codomain::VectorSpace,
                      domain::VectorSpace)
    return similar(t, T, codomain ← domain)
end
# 3 arguments
function Base.similar(t::AbstractTensorMap, codomain::VectorSpace, domain::VectorSpace)
    return similar(t, scalartype(t), codomain ← domain)
end
function Base.similar(t::AbstractTensorMap, T::Type, codomain::VectorSpace)
    return similar(t, T, codomain ← one(codomain))
end
# 2 arguments
function Base.similar(t::AbstractTensorMap, codomain::VectorSpace)
    return similar(t, scalartype(t), codomain ← one(codomain))
end
Base.similar(t::AbstractTensorMap, P::TensorMapSpace) = similar(t, scalartype(t), P)
Base.similar(t::AbstractTensorMap, T::Type) = similar(t, T, space(t))
# 1 argument
Base.similar(t::AbstractTensorMap) = similar(t, scalartype(t), space(t))

# actual implementation
function Base.similar(t::TensorMap{S}, ::Type{T}, P::TensorMapSpace{S}) where {T,S}
    N₁ = length(codomain(P))
    N₂ = length(domain(P))
    I = sectortype(S)
    # speed up specialized cases
    if I === Trivial
        data = similar(t.data, T, (dim(codomain(P)), dim(domain(P))))
        A = typeof(data)
        return TrivialTensorMap{S,N₁,N₂,A}(data, codomain(P), domain(P))
    end
    F₁ = fusiontreetype(I, N₁)
    F₂ = fusiontreetype(I, N₂)
    if space(t) == P
        data = SectorDict(c => similar(b, T) for (c, b) in blocks(t))
        A = typeof(data)
        return TensorMap{S,N₁,N₂,I,A,F₁,F₂}(data, codomain(P), domain(P), t.rowr, t.colr)
    end

    blocksectoriterator = blocksectors(P)
    # try to recycle rowr
    if codomain(P) == codomain(t) && all(c -> haskey(t.rowr, c), blocksectoriterator)
        if length(t.rowr) == length(blocksectoriterator)
            rowr = t.rowr
        else
            rowr = SectorDict(c => t.rowr[c] for c in blocksectoriterator)
        end
        rowdims = SectorDict(c => size(block(t, c), 1) for c in blocksectoriterator)
    elseif codomain(P) == domain(t) && all(c -> haskey(t.colr, c), blocksectoriterator)
        if length(t.colr) == length(blocksectoriterator)
            rowr = t.colr
        else
            rowr = SectorDict(c => t.colr[c] for c in blocksectoriterator)
        end
        rowdims = SectorDict(c => size(block(t, c), 2) for c in blocksectoriterator)
    else
        rowr, rowdims = _buildblockstructure(codomain(P), blocksectoriterator)
    end
    # try to recylce colr
    if domain(P) == codomain(t) && all(c -> haskey(t.rowr, c), blocksectoriterator)
        if length(t.rowr) == length(blocksectoriterator)
            colr = t.rowr
        else
            colr = SectorDict(c => t.rowr[c] for c in blocksectoriterator)
        end
        coldims = SectorDict(c => size(block(t, c), 1) for c in blocksectoriterator)
    elseif domain(P) == domain(t) && all(c -> haskey(t.colr, c), blocksectoriterator)
        if length(t.colr) == length(blocksectoriterator)
            colr = t.colr
        else
            colr = SectorDict(c => t.colr[c] for c in blocksectoriterator)
        end
        coldims = SectorDict(c => size(block(t, c), 2) for c in blocksectoriterator)
    else
        colr, coldims = _buildblockstructure(domain(P), blocksectoriterator)
    end
    M = similarstoragetype(t, T)
    data = SectorDict{I,M}(c => M(undef, (rowdims[c], coldims[c]))
                           for c in blocksectoriterator)
    A = typeof(data)
    return TensorMap{S,N₁,N₂,I,A,F₁,F₂}(data, codomain(P), domain(P), rowr, colr)
end

function Base.complex(t::AbstractTensorMap)
    if scalartype(t) <: Complex
        return t
    else
        return copy!(similar(t, complex(scalartype(t))), t)
    end
end

# Conversion between TensorMap and Dict, for read and write purpose
#------------------------------------------------------------------
function Base.convert(::Type{Dict}, t::AbstractTensorMap)
    d = Dict{Symbol,Any}()
    d[:codomain] = repr(codomain(t))
    d[:domain] = repr(domain(t))
    data = Dict{String,Any}()
    for (c, b) in blocks(t)
        data[repr(c)] = Array(b)
    end
    d[:data] = data
    return d
end
function Base.convert(::Type{TensorMap}, d::Dict{Symbol,Any})
    try
        codomain = eval(Meta.parse(d[:codomain]))
        domain = eval(Meta.parse(d[:domain]))
        data = SectorDict(eval(Meta.parse(c)) => b for (c, b) in d[:data])
        return TensorMap(data, codomain, domain)
    catch e # sector unknown in TensorKit.jl; user-defined, hopefully accessible in Main
        codomain = Base.eval(Main, Meta.parse(d[:codomain]))
        domain = Base.eval(Main, Meta.parse(d[:domain]))
        data = SectorDict(Base.eval(Main, Meta.parse(c)) => b for (c, b) in d[:data])
        return TensorMap(data, codomain, domain)
    end
end

# Getting and setting the data
#------------------------------
hasblock(t::TrivialTensorMap, ::Trivial) = !isempty(t.data)
hasblock(t::TensorMap, s::Sector) = haskey(t.data, s)

block(t::TrivialTensorMap, ::Trivial) = t.data
function block(t::TensorMap, s::Sector)
    sectortype(t) == typeof(s) || throw(SectorMismatch())
    A = valtype(t.data)
    if haskey(t.data, s)
        return t.data[s]
    else # at least one of the two matrix dimensions will be zero
        return storagetype(t)(undef, (blockdim(codomain(t), s), blockdim(domain(t), s)))
    end
end

function blocks(t::TensorMap{<:IndexSpace,N₁,N₂,Trivial}) where {N₁,N₂}
    return SingletonDict(Trivial() => t.data)
end
blocks(t::TensorMap) = t.data

fusiontrees(t::TrivialTensorMap) = ((nothing, nothing),)
fusiontrees(t::TensorMap) = TensorKeyIterator(t.rowr, t.colr)

"""
    Base.getindex(t::TensorMap{<:IndexSpace,N₁,N₂,I},
                  sectors::NTuple{N₁+N₂,I}) where {N₁,N₂,I<:Sector} 
        -> StridedViews.StridedView
    t[sectors]

Return a view into the data slice of `t` corresponding to the splitting - fusion tree pair
with combined uncoupled charges `sectors`. In particular, if `sectors == (s1..., s2...)`
where `s1` and `s2` correspond to the coupled charges in the codomain and domain
respectively, then a `StridedViews.StridedView` of size
`(dims(codomain(t), s1)..., dims(domain(t), s2))` is returned.

This method is only available for the case where `FusionStyle(I) isa UniqueFusion`,
since it assumes a  uniquely defined coupled charge.
"""
@inline function Base.getindex(t::TensorMap{<:IndexSpace,N₁,N₂,I},
                               sectors::Tuple{Vararg{I}}) where {N₁,N₂,I<:Sector}
    FusionStyle(I) isa UniqueFusion ||
        throw(SectorMismatch("Indexing with sectors only possible if unique fusion"))
    s1 = TupleTools.getindices(sectors, codomainind(t))
    s2 = map(dual, TupleTools.getindices(sectors, domainind(t)))
    c1 = length(s1) == 0 ? one(I) : (length(s1) == 1 ? s1[1] : first(⊗(s1...)))
    @boundscheck begin
        c2 = length(s2) == 0 ? one(I) : (length(s2) == 1 ? s2[1] : first(⊗(s2...)))
        c2 == c1 || throw(SectorMismatch("Not a valid sector for this tensor"))
        hassector(codomain(t), s1) && hassector(domain(t), s2)
    end
    f₁ = FusionTree(s1, c1, map(isdual, tuple(codomain(t)...)))
    f₂ = FusionTree(s2, c1, map(isdual, tuple(domain(t)...)))
    @inbounds begin
        return t[f₁, f₂]
    end
end
@propagate_inbounds function Base.getindex(t::TensorMap, sectors::Tuple)
    return t[map(sectortype(t), sectors)]
end

"""
    Base.getindex(t::TensorMap{<:IndexSpace,N₁,N₂,I},
                  f₁::FusionTree{I,N₁},
                  f₂::FusionTree{I,N₂}) where {N₁,N₂,I<:Sector}
        -> StridedViews.StridedView
    t[f₁, f₂]

Return a view into the data slice of `t` corresponding to the splitting - fusion tree pair
`(f₁, f₂)`. In particular, if `f₁.coupled == f₂.coupled == c`, then a
`StridedViews.StridedView` of size
`(dims(codomain(t), f₁.uncoupled)..., dims(domain(t), f₂.uncoupled))` is returned which
represents the slice of `block(t, c)` whose row indices correspond to `f₁.uncoupled` and
column indices correspond to `f₂.uncoupled`.
"""
@inline function Base.getindex(t::TensorMap{<:IndexSpace,N₁,N₂,I},
                               f₁::FusionTree{I,N₁},
                               f₂::FusionTree{I,N₂}) where {N₁,N₂,I<:Sector}
    c = f₁.coupled
    @boundscheck begin
        c == f₂.coupled || throw(SectorMismatch())
        haskey(t.rowr[c], f₁) || throw(SectorMismatch())
        haskey(t.colr[c], f₂) || throw(SectorMismatch())
    end
    @inbounds begin
        d = (dims(codomain(t), f₁.uncoupled)..., dims(domain(t), f₂.uncoupled)...)
        return sreshape(StridedView(t.data[c])[t.rowr[c][f₁], t.colr[c][f₂]], d)
    end
end

"""
    Base.setindex!(t::TensorMap{<:IndexSpace,N₁,N₂,I},
                   v,
                   f₁::FusionTree{I,N₁},
                   f₂::FusionTree{I,N₂}) where {N₁,N₂,I<:Sector}
    t[f₁, f₂] = v

Copies `v` into the  data slice of `t` corresponding to the splitting - fusion tree pair
`(f₁, f₂)`. Here, `v` can be any object that can be copied into a `StridedViews.StridedView`
of size `(dims(codomain(t), f₁.uncoupled)..., dims(domain(t), f₂.uncoupled))` using
`Base.copy!`.

See also [`Base.getindex(::TensorMap{<:IndexSpace,N₁,N₂,I<:Sector}, ::FusionTree{I<:Sector,N₁}, ::FusionTree{I<:Sector,N₂})`](@ref)
"""
@propagate_inbounds function Base.setindex!(t::TensorMap{<:IndexSpace,N₁,N₂,I},
                                            v,
                                            f₁::FusionTree{I,N₁},
                                            f₂::FusionTree{I,N₂}) where {N₁,N₂,I<:Sector}
    return copy!(getindex(t, f₁, f₂), v)
end

# For a tensor with trivial symmetry, allow no argument indexing
"""
    Base.getindex(t::TrivialTensorMap)
    t[]

Return a view into the data of `t` as a `StridedViews.StridedView` of size
`(dims(codomain(t))..., dims(domain(t))...)`.
"""
@inline function Base.getindex(t::TrivialTensorMap)
    return sreshape(StridedView(t.data), (dims(codomain(t))..., dims(domain(t))...))
end
@inline Base.setindex!(t::TrivialTensorMap, v) = copy!(getindex(t), v)

# For a tensor with trivial symmetry, fusiontrees returns (nothing,nothing)
@inline Base.getindex(t::TrivialTensorMap, ::Nothing, ::Nothing) = getindex(t)
@inline Base.setindex!(t::TrivialTensorMap, v, ::Nothing, ::Nothing) = setindex!(t, v)

# For a tensor with trivial symmetry, allow direct indexing
"""
    Base.getindex(t::TrivialTensorMap, indices::Vararg{Int})
    t[indices]

Return a view into the data slice of `t` corresponding to `indices`, by slicing the
`StridedViews.StridedView` into the full data array.
"""
@inline function Base.getindex(t::TrivialTensorMap, indices::Vararg{Int})
    data = t[]
    @boundscheck checkbounds(data, indices...)
    @inbounds v = data[indices...]
    return v
end
"""
    Base.setindex!(t::TrivialTensorMap, v, indices::Vararg{Int})
    t[indices] = v

Assigns `v` to the data slice of `t` corresponding to `indices`.
"""
@inline function Base.setindex!(t::TrivialTensorMap, v, indices::Vararg{Int})
    data = t[]
    @boundscheck checkbounds(data, indices...)
    @inbounds data[indices...] = v
    return v
end

# Show
#------
function Base.summary(io::IO, t::TensorMap)
    return print(io, "TensorMap(", space(t), ")")
end
function Base.show(io::IO, t::TensorMap{S}) where {S<:IndexSpace}
    if get(io, :compact, false)
        print(io, "TensorMap(", space(t), ")")
        return
    end
    println(io, "TensorMap(", space(t), "):")
    if sectortype(S) == Trivial
        Base.print_array(io, t[])
        println(io)
    elseif FusionStyle(sectortype(S)) isa UniqueFusion
        for (f₁, f₂) in fusiontrees(t)
            println(io, "* Data for sector ", f₁.uncoupled, " ← ", f₂.uncoupled, ":")
            Base.print_array(io, t[f₁, f₂])
            println(io)
        end
    else
        for (f₁, f₂) in fusiontrees(t)
            println(io, "* Data for fusiontree ", f₁, " ← ", f₂, ":")
            Base.print_array(io, t[f₁, f₂])
            println(io)
        end
    end
end

# Real and imaginary parts
#---------------------------
function Base.real(t::AbstractTensorMap{S}) where {S}
    # `isreal` for a `Sector` returns true iff the F and R symbols are real. This guarantees
    # that the real/imaginary part of a tensor `t` can be obtained by just taking
    # real/imaginary part of the degeneracy data.
    if isreal(sectortype(S))
        realdata = Dict(k => real(v) for (k, v) in blocks(t))
        return TensorMap(realdata, codomain(t), domain(t))
    else
        msg = "`real` has not been implemented for `AbstractTensorMap{$(S)}`."
        throw(ArgumentError(msg))
    end
end

function Base.imag(t::AbstractTensorMap{S}) where {S}
    # `isreal` for a `Sector` returns true iff the F and R symbols are real. This guarantees
    # that the real/imaginary part of a tensor `t` can be obtained by just taking
    # real/imaginary part of the degeneracy data.
    if isreal(sectortype(S))
        imagdata = Dict(k => imag(v) for (k, v) in blocks(t))
        return TensorMap(imagdata, codomain(t), domain(t))
    else
        msg = "`imag` has not been implemented for `AbstractTensorMap{$(S)}`."
        throw(ArgumentError(msg))
    end
end

# Conversion and promotion:
#---------------------------
Base.convert(::Type{TensorMap}, t::TensorMap) = t
function Base.convert(::Type{TensorMap}, t::AbstractTensorMap)
    return copy!(TensorMap(undef, scalartype(t), codomain(t), domain(t)), t)
end

function Base.convert(T::Type{TensorMap{S,N₁,N₂,I,A,F₁,F₂}},
                      t::AbstractTensorMap{S,N₁,N₂}) where {S,N₁,N₂,I,A,F₁,F₂}
    if typeof(t) == T
        return t
    else
        data = Dict{I,storagetype(T)}(c => convert(storagetype(T), b)
                                      for (c, b) in blocks(t))
        return TensorMap(data, codomain(t), domain(t))
    end
end

function Base.promote_rule(::Type{<:T1},
                           t2::Type{<:T2}) where {S,N₁,N₂,T1<:TensorMap{S,N₁,N₂},
                                                  T2<:TensorMap{S,N₁,N₂}}
    return tensormaptype(S, N₁, N₂, promote_type(storagetype(T1), storagetype(T2)))
end
