# TensorMap & Tensor:
# general tensor implementation with arbitrary symmetries
#==========================================================#
#! format: off
"""
    struct TensorMap{E, S<:IndexSpace, N₁, N₂, ...} <: AbstractTensorMap{E, S, N₁, N₂}

Specific subtype of [`AbstractTensorMap`](@ref) for representing tensor maps (morphisms in
a tensor category) whose data is stored in blocks of some subtype of `DenseMatrix`.
"""
struct TensorMap{E, S<:IndexSpace, N₁, N₂, I<:Sector, A<:Union{<:DenseMatrix{E},SectorDict{I,<:DenseMatrix{E}}},
                 F₁, F₂} <: AbstractTensorMap{E, S, N₁, N₂}
    data::A
    codom::ProductSpace{S,N₁}
    dom::ProductSpace{S,N₂}
    rowr::SectorDict{I,FusionTreeDict{F₁,UnitRange{Int}}}
    colr::SectorDict{I,FusionTreeDict{F₂,UnitRange{Int}}}
    
    # uninitialized constructors
    function TensorMap{E,S,N₁,N₂,Trivial,A,Nothing,Nothing}(::UndefInitializer, codom::ProductSpace{S,N₁}, dom::ProductSpace{S,N₂}) where {E,S<:IndexSpace,N₁,N₂,A<:DenseMatrix{E}}
        data = A(undef, dim(codom), dim(dom))
        return TensorMap{E,S,N₁,N₂,Trivial,A,Nothing,Nothing}(data, codom, dom)
    end
    function TensorMap{E,S,N₁,N₂,I,A,F₁,F₂}(::UndefInitializer, codom::TensorSpace{S},
                                            dom::TensorSpace{S}) where {E,S<:IndexSpace,N₁,N₂,
                                            I<:Sector,A<:SectorDict{I,<:DenseMatrix{E}},F₁,F₂}
        I === sectortype(S) || throw(SectorMismatch())
        blocksectoriterator = blocksectors(codom ← dom)
        rowr, rowdims = _buildblockstructure(codom, blocksectoriterator)
        colr, coldims = _buildblockstructure(dom, blocksectoriterator)
        data = SectorDict(c => valtype(A)(undef, rowdims[c], coldims[c]) for c in blocksectoriterator)
        return TensorMap{E,S,N₁,N₂,I,A,F₁,F₂}(data, codom, dom, rowr, colr)
    end
    
    # constructors from data
    function TensorMap{E,S,N₁,N₂,I,A,F₁,F₂}(data::A,
                codom::ProductSpace{S,N₁}, dom::ProductSpace{S,N₂},
                rowr::SectorDict{I,FusionTreeDict{F₁,UnitRange{Int}}},
                colr::SectorDict{I,FusionTreeDict{F₂,UnitRange{Int}}}) where
                    {E,S<:IndexSpace, N₁, N₂, I<:Sector, A<:SectorDict{I,<:DenseMatrix{E}},
                     F₁<:FusionTree{I,N₁}, F₂<:FusionTree{I,N₂}}
        E ⊆ field(S) || @warn("scalartype(data) = $T ⊈ $(field(S)))", maxlog = 1)
        return new{E,S,N₁,N₂,I,A,F₁,F₂}(data, codom, dom, rowr, colr)
    end
    function TensorMap{E,S,N₁,N₂,Trivial,A,Nothing,Nothing}(data::A,
                                                          codom::ProductSpace{S,N₁},
                                                          dom::ProductSpace{S,N₂}) where
             {E,S<:IndexSpace,N₁,N₂,A<:DenseMatrix{E}}
        E ⊆ field(S) || @warn("scalartype(data) = $T ⊈ $(field(S)))", maxlog = 1)
        return new{E,S,N₁,N₂,Trivial,A,Nothing,Nothing}(data, codom, dom)
    end
end
#! format: on

"""
    Tensor{E, S, N, I, A, F₁, F₂} = TensorMap{E, S, N, 0, I, A, F₁, F₂}

Specific subtype of [`AbstractTensor`](@ref) for representing tensors whose data is stored
in blocks of some subtype of `DenseMatrix`.

A `Tensor{E, S, N, I, A, F₁, F₂}` is actually a special case `TensorMap{E, S, N, 0, I, A, F₁, F₂}`,
i.e. a tensor map with only a non-trivial output space.
"""
const Tensor{E,S,N,I,A,F₁,F₂} = TensorMap{E,S,N,0,I,A,F₁,F₂}

"""
    TrivialTensorMap{E, S, N₁, N₂, A<:DenseMatrix} = TensorMap{E, S, N₁, N₂, Trivial, 
                                                                        A, Nothing, Nothing}

A special case of [`TensorMap`](@ref) for representing tensor maps with trivial symmetry,
i.e., whose `sectortype` is `Trivial`.
"""
const TrivialTensorMap{E,S,N₁,N₂,A<:DenseMatrix} = TensorMap{E,S,N₁,N₂,Trivial,A,Nothing,
                                                             Nothing}

"""
    TrivialTensor{E, S, N, A} = TrivialTensorMap{E, S, N, 0, A}

A special case of [`Tensor`](@ref) for representing tensors with trivial symmetry, i.e.,
whose `sectortype` is `Trivial`.
"""
const TrivialTensor{E,S,N,A} = TrivialTensorMap{E,S,N,0,A}

# TODO: check if argument order should change
"""
    tensormaptype(::Type{S}, N₁::Int, N₂::Int, [::Type{T}]) where {S<:IndexSpace,T} -> ::Type{<:TensorMap}

Return the fully specified type of a tensor map with elementary space `S`, `N₁` output
spaces and `N₂` input spaces, either with scalar type `T` or with storage type `T`.
"""
function tensormaptype(::Type{S}, N₁::Int, N₂::Int, ::Type{T}) where {S,T<:MatOrNumber}
    I = sectortype(S)
    if T <: DenseMatrix
        M = T
        E = scalartype(T)
    elseif T <: Number
        M = Matrix{T}
        E = T
    else
        throw(ArgumentError("the final argument of `tensormaptype` should either be the scalar or the storage type, i.e. a subtype of `Number` or of `DenseMatrix`"))
    end
    if I === Trivial
        return TensorMap{E,S,N₁,N₂,I,M,Nothing,Nothing}
    else
        F₁ = fusiontreetype(I, N₁)
        F₂ = fusiontreetype(I, N₂)
        return TensorMap{E,S,N₁,N₂,I,SectorDict{I,M},F₁,F₂}
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
function storagetype(::Type{<:TrivialTensorMap{E,S,N₁,N₂,A}}) where
         {E,S,N₁,N₂,A}
    return A
end
function storagetype(::Type{<:TensorMap{E,S,N₁,N₂,I,<:SectorDict{I,A}}}) where
         {E,S,N₁,N₂,I<:Sector,A<:DenseMatrix}
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
function TensorMap(data::AbstractDict{<:Sector,<:DenseMatrix},
                   V::TensorMapSpace{S,N₁,N₂}) where {S,N₁,N₂}
    dom = domain(V)
    codom = codomain(V)
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
    data2 = if isreal(I)
        SectorDict(c => data[c] for c in blocksectoriterator)
    else
        SectorDict(c => complex(data[c]) for c in blocksectoriterator)
    end
    TT = tensormaptype(S, N₁, N₂, valtype(data))
    return TT(data2, codom, dom, rowr, colr)
end
function TensorMap(data::AbstractDict{<:Sector,<:DenseMatrix}, codom::TensorSpace{S},
                   dom::TensorSpace{S}) where {S}
    return TensorMap(data, codom ← dom)
end

"""
    TensorMap{E}(undef, codomain::ProductSpace{S,N₁}, domain::ProductSpace{S,N₂})
                where {E,S,N₁,N₂}
    TensorMap{E}(undef, codomain ← domain)
    TensorMap{E}(undef, domain → codomain)
    # expert mode: select storage type `A`
    TensorMap{E,S,N₁,N₂,I,A}(undef, codomain::ProductSpace{S,N₁}, domain::ProductSpace{S,N₂})

Construct a `TensorMap` with uninitialized data.
"""
function TensorMap{E}(::UndefInitializer, V::TensorMapSpace{S,N₁,N₂}) where {E,S,N₁,N₂}
    TT = tensormaptype(S, N₁, N₂, E) # construct full type
    return TT(undef, codomain(V), domain(V))
end
function TensorMap{E}(::UndefInitializer, codomain::TensorSpace{S},
                      domain::TensorSpace{S}) where {E,S}
    return TensorMap{E}(undef, codomain ← domain)
end
function Tensor{E}(::UndefInitializer, V::TensorSpace{S}) where {E,S}
    return TensorMap{E}(undef, V ← one(V))
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

@doc """
    zeros([T=Float64,], codomain::ProductSpace{S,N₁}, domain::ProductSpace{S,N₂}) where {S,N₁,N₂,T}
    zeros([T=Float64,], codomain ← domain)

Create a `TensorMap` with element type `T`, of all zeros with spaces specified by `codomain` and `domain`.
"""
Base.zeros

@doc """
    ones([T=Float64,], codomain::ProductSpace{S,N₁}, domain::ProductSpace{S,N₂}) where {S,N₁,N₂,T}
    ones([T=Float64,], codomain ← domain)
    
Create a `TensorMap` with element type `T`, of all ones with spaces specified by `codomain` and `domain`.
"""
Base.ones

for (fname, felt) in ((:zeros, :zero), (:ones, :one))
    @eval begin
        function Base.$fname(codomain::TensorSpace{S},
                             domain::TensorSpace{S}=one(codomain)) where {S<:IndexSpace}
            return Base.$fname(codomain ← domain)
        end
        function Base.$fname(::Type{T}, codomain::TensorSpace{S},
                             domain::TensorSpace{S}=one(codomain)) where {T,S<:IndexSpace}
            return Base.$fname(T, codomain ← domain)
        end
        Base.$fname(V::TensorMapSpace) = Base.$fname(Float64, V)
        function Base.$fname(::Type{T}, V::TensorMapSpace) where {T}
            t = TensorMap{T}(undef, V)
            fill!(t, $felt(T))
            return t
        end
    end
end

for randfun in (:rand, :randn, :randexp)
    randfun! = Symbol(randfun, :!)
    _docstr = """
        $randfun([rng=default_rng()], [T=Float64], codomain::ProductSpace{S,N₁},
                     domain::ProductSpace{S,N₂}) where {S,N₁,N₂,T} -> t
        $randfun([rng=default_rng()], [T=Float64], codomain ← domain) -> t
        
        Generate a tensor `t` with entries generated by `$randfun)`.
    """
    _docstr! = """
        $randfun!([rng=default_rng()], t::AbstractTensorMap) -> t
        
    Fill the tensor `t` with entries generated by `$randfun!`.
    """

    @eval begin
        @doc $_docstr Random.$randfun
        @doc $_docstr! Random.$randfun!

        # converting `codomain` and `domain` into `HomSpace`
        function Random.$randfun(codomain::TensorSpace{S},
                                 domain::TensorSpace{S}) where {S<:IndexSpace}
            return Random.$randfun(codomain ← domain)
        end
        function Random.$randfun(::Type{T}, codomain::TensorSpace{S},
                                 domain::TensorSpace{S}) where {T,S<:IndexSpace}
            return Random.$randfun(T, codomain ← domain)
        end
        function Random.$randfun(rng::Random.AbstractRNG, ::Type{T},
                                 codomain::TensorSpace{S},
                                 domain::TensorSpace{S}) where {T,S<:IndexSpace}
            return Random.$randfun(rng, T, codomain ← domain)
        end

        # accepting single `TensorSpace`
        Random.$randfun(codomain::TensorSpace) = Random.$randfun(codomain ← one(codomain))
        function Random.$randfun(::Type{T}, codomain::TensorSpace) where {T}
            return Random.$randfun(T, codomain ← one(codomain))
        end
        function Random.$randfun(rng::Random.AbstractRNG, ::Type{T},
                                 codomain::TensorSpace) where {T}
            return Random.$randfun(rng, T, codomain ← one(domain))
        end

        # filling in default eltype
        Random.$randfun(V::TensorMapSpace) = Random.$randfun(Float64, V)
        function Random.$randfun(rng::Random.AbstractRNG, V::TensorMapSpace)
            return Random.$randfun(rng, Float64, V)
        end

        # filling in default rng
        function Random.$randfun(::Type{T}, V::TensorMapSpace) where {T}
            return Random.$randfun(Random.default_rng(), T, V)
        end
        Random.$randfun!(t::AbstractTensorMap) = Random.$randfun!(Random.default_rng(), t)

        # implementation
        function Random.$randfun(rng::Random.AbstractRNG, ::Type{T},
                                 V::TensorMapSpace) where {T}
            t = TensorMap{T}(undef, V)
            Random.$randfun!(rng, t)
            return t
        end

        function Random.$randfun!(rng::Random.AbstractRNG, t::AbstractTensorMap)
            for (_, b) in blocks(t)
                Random.$randfun!(rng, b)
            end
            return t
        end
    end
end

# constructor starting from a dense array
"""
    TensorMap(data::DenseArray, codomain::ProductSpace{S,N₁}, domain::ProductSpace{S,N₂};
                    tol=sqrt(eps(real(float(eltype(data)))))) where {S<:ElementarySpace,N₁,N₂}
    TensorMap(data, codomain ← domain; tol=sqrt(eps(real(float(eltype(data))))))
    TensorMap(data, domain → codomain; tol=sqrt(eps(real(float(eltype(data))))))

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
function TensorMap(data::DenseArray, V::TensorMapSpace{S,N₁,N₂};
                   tol=sqrt(eps(real(float(eltype(data)))))) where {S<:IndexSpace,N₁,N₂}
    codom = codomain(V)
    dom = domain(V)
    (d1, d2) = (dim(codom), dim(dom))
    if !(length(data) == d1 * d2 || size(data) == (d1, d2) ||
         size(data) == (dims(codom)..., dims(dom)...))
        throw(DimensionMismatch())
    end

    if sectortype(S) === Trivial
        data2 = reshape(data, (d1, d2))
        A = typeof(data2)
        return tensormaptype(S, N₁, N₂, A)(data2, codom, dom)
    end

    t = TensorMap{eltype(data)}(undef, codom, dom)
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
function TensorMap(data::DenseArray, codom::TensorSpace{S},
                   dom::TensorSpace{S}; kwargs...) where {S}
    return TensorMap(data, codom ← dom; kwargs...)
end

# Efficient copy constructors
#-----------------------------
Base.copy(t::TrivialTensorMap) = typeof(t)(copy(t.data), t.codom, t.dom)
Base.copy(t::TensorMap) = typeof(t)(deepcopy(t.data), t.codom, t.dom, t.rowr, t.colr)

# specializations when data can be re-used
function Base.similar(t::TensorMap, ::Type{TorA},
                      P::TensorMapSpace{S}) where {TorA<:MatOrNumber,S}
    N₁ = length(codomain(P))
    N₂ = length(domain(P))
    I = sectortype(S)

    # speed up specialized cases
    TT = tensormaptype(S, N₁, N₂, TorA)
    I === Trivial && return TT(undef, codomain(P), domain(P))

    if space(t) == P
        data = if TorA <: Number
            SectorDict(c => similar(b, TorA) for (c, b) in blocks(t))
        else
            SectorDict(c => similar(TorA, size(b)) for (c, b) in blocks(t))
        end
        return TT(data, codomain(P), domain(P), t.rowr, t.colr)
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
    # try to recycle colr
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
    M = storagetype(TT)
    data = SectorDict{I,M}(c => M(undef, (rowdims[c], coldims[c]))
                           for c in blocksectoriterator)
    return TT(data, codomain(P), domain(P), rowr, colr)
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

function blocks(t::TrivialTensorMap)
    return SingletonDict(Trivial() => t.data)
end
blocks(t::TensorMap) = t.data

fusiontrees(t::TrivialTensorMap) = ((nothing, nothing),)
fusiontrees(t::TensorMap) = TensorKeyIterator(t.rowr, t.colr)

"""
    Base.getindex(t::TensorMap
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
@inline function Base.getindex(t::TensorMap, sectors::Tuple{I,Vararg{I}}) where {I<:Sector}
    I === sectortype(t) || throw(SectorMismatch("Not a valid sectortype for this tensor."))
    length(sectors) == numind(t) ||
        throw(ArgumentError("Number of sectors does not match."))
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
    Base.getindex(t::TensorMap{E,S,N₁,N₂,I},
                  f₁::FusionTree{I,N₁},
                  f₂::FusionTree{I,N₂}) where {E,SN₁,N₂,I<:Sector}
        -> StridedViews.StridedView
    t[f₁, f₂]

Return a view into the data slice of `t` corresponding to the splitting - fusion tree pair
`(f₁, f₂)`. In particular, if `f₁.coupled == f₂.coupled == c`, then a
`StridedViews.StridedView` of size
`(dims(codomain(t), f₁.uncoupled)..., dims(domain(t), f₂.uncoupled))` is returned which
represents the slice of `block(t, c)` whose row indices correspond to `f₁.uncoupled` and
column indices correspond to `f₂.uncoupled`.
"""
@inline function Base.getindex(t::TensorMap{E,S,N₁,N₂,I},
                               f₁::FusionTree{I,N₁},
                               f₂::FusionTree{I,N₂}) where {E,S,N₁,N₂,I<:Sector}
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
    Base.setindex!(t::TensorMap{E,S,N₁,N₂,I},
                   v,
                   f₁::FusionTree{I,N₁},
                   f₂::FusionTree{I,N₂}) where {E,S,N₁,N₂,I<:Sector}
    t[f₁, f₂] = v

Copies `v` into the  data slice of `t` corresponding to the splitting - fusion tree pair
`(f₁, f₂)`. Here, `v` can be any object that can be copied into a `StridedViews.StridedView`
of size `(dims(codomain(t), f₁.uncoupled)..., dims(domain(t), f₂.uncoupled))` using
`Base.copy!`.

See also [`Base.getindex(::TensorMap{E,S,N₁,N₂,I<:Sector}, ::FusionTree{I<:Sector,N₁}, ::FusionTree{I<:Sector,N₂})`](@ref)
"""
@propagate_inbounds function Base.setindex!(t::TensorMap{E,S,N₁,N₂,I},
                                            v,
                                            f₁::FusionTree{I,N₁},
                                            f₂::FusionTree{I,N₂}) where {E,S,N₁,N₂,
                                                                         I<:Sector}
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
function Base.show(io::IO, t::TensorMap)
    if get(io, :compact, false)
        print(io, "TensorMap(", space(t), ")")
        return
    end
    println(io, "TensorMap(", space(t), "):")
    if sectortype(t) == Trivial
        Base.print_array(io, t[])
        println(io)
    elseif FusionStyle(sectortype(t)) isa UniqueFusion
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
function Base.real(t::AbstractTensorMap)
    # `isreal` for a `Sector` returns true iff the F and R symbols are real. This guarantees
    # that the real/imaginary part of a tensor `t` can be obtained by just taking
    # real/imaginary part of the degeneracy data.
    if isreal(sectortype(t))
        realdata = Dict(k => real(v) for (k, v) in blocks(t))
        return TensorMap(realdata, codomain(t), domain(t))
    else
        msg = "`real` has not been implemented for `$(typeof(t))`."
        throw(ArgumentError(msg))
    end
end

function Base.imag(t::AbstractTensorMap)
    # `isreal` for a `Sector` returns true iff the F and R symbols are real. This guarantees
    # that the real/imaginary part of a tensor `t` can be obtained by just taking
    # real/imaginary part of the degeneracy data.
    if isreal(sectortype(t))
        imagdata = Dict(k => imag(v) for (k, v) in blocks(t))
        return TensorMap(imagdata, codomain(t), domain(t))
    else
        msg = "`imag` has not been implemented for `$(typeof(t))`."
        throw(ArgumentError(msg))
    end
end

# Conversion and promotion:
#---------------------------
Base.convert(::Type{TensorMap}, t::TensorMap) = t
function Base.convert(::Type{TensorMap}, t::AbstractTensorMap)
    return copy!(TensorMap{scalartype(t)}(undef, space(t)), t)
end

function Base.convert(T::Type{<:TensorMap{E,S,N₁,N₂}},
                      t::AbstractTensorMap{<:Any,S,N₁,N₂}) where {E,S,N₁,N₂}
    if typeof(t) === T
        return t
    else
        data = Dict{sectortype(T),storagetype(T)}(c => convert(storagetype(T), b)
                                                  for (c, b) in blocks(t))
        return TensorMap(data, codomain(t), domain(t))
    end
end

function Base.promote_rule(::Type{<:T₁},
                           ::Type{<:T₂}) where {S,N₁,N₂,
                                                T₁<:TensorMap{<:Any,S,N₁,N₂},
                                                T₂<:TensorMap{<:Any,S,N₁,N₂}}
    return tensormaptype(S, N₁, N₂, promote_type(storagetype(T₁), storagetype(T₂)))
end
