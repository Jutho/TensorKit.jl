# TensorMap & Tensor:
# general tensor implementation with arbitrary symmetries
#==========================================================#
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
        eltype(valtype(data)) ⊆ field(S) ||
            @warn("eltype(data) = $(eltype(data)) ⊈ $(field(S)))", maxlog=1)
        new{S, N₁, N₂, I, A, F₁, F₂}(data, codom, dom, rowr, colr)
    end
    function TensorMap{S, N₁, N₂, Trivial, A, Nothing, Nothing}(data::A,
                codom::ProductSpace{S,N₁}, dom::ProductSpace{S,N₂}) where
                    {S<:IndexSpace, N₁, N₂, A<:DenseMatrix}
        eltype(data) ⊆ field(S) ||
            @warn("eltype(data) = $(eltype(data)) ⊈ $(field(S)))", maxlog=1)
        new{S, N₁, N₂, Trivial, A, Nothing, Nothing}(data, codom, dom)
    end
end

const Tensor{S<:IndexSpace, N, I<:Sector, A, F₁, F₂} = TensorMap{S, N, 0, I, A, F₁, F₂}
const TrivialTensorMap{S<:IndexSpace, N₁, N₂, A<:DenseMatrix} = TensorMap{S, N₁, N₂, Trivial, A, Nothing, Nothing}

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
tensormaptype(S, N₁, N₂ = 0) = tensormaptype(S, N₁, N₂, Float64)

# Basic methods for characterising a tensor:
#--------------------------------------------
codomain(t::TensorMap) = t.codom
domain(t::TensorMap) = t.dom

blocksectors(t::TrivialTensorMap) = TrivialOrEmptyIterator(dim(t) == 0)
blocksectors(t::TensorMap) = keys(t.data)

storagetype(::Type{<:TensorMap{<:IndexSpace,N₁,N₂,Trivial,A}}) where
    {N₁,N₂,A<:DenseMatrix} = A
storagetype(::Type{<:TensorMap{<:IndexSpace,N₁,N₂,I,<:SectorDict{I,A}}}) where
    {N₁,N₂,I<:Sector,A<:DenseMatrix} = A

dim(t::TensorMap) = mapreduce(x->length(x[2]), +, blocks(t); init = 0)

# General TensorMap constructors
#--------------------------------
# with data
function TensorMap(data::DenseArray, codom::ProductSpace{S,N₁}, dom::ProductSpace{S,N₂};
            tol = sqrt(eps(real(float(eltype(data)))))) where {S<:IndexSpace, N₁, N₂}
    (d1, d2) = (dim(codom), dim(dom))
    if !(length(data) == d1*d2 || size(data) == (d1, d2) ||
        size(data) == (dims(codom)..., dims(dom)...))
        throw(DimensionMismatch())
    end
    if sectortype(S) === Trivial
        data2 = reshape(data, (d1, d2))
        A = typeof(data2)
        return TensorMap{S, N₁, N₂, Trivial, A, Nothing, Nothing}(data2, codom, dom)
    else
        t = TensorMap(zeros, eltype(data), codom, dom)
        ta = convert(Array, t)
        l = length(ta)
        basis = zeros(eltype(ta), (l, dim(t)))
        qdims = zeros(real(eltype(ta)), (dim(t),))
        i = 1
        for (c,b) in blocks(t)
            for k = 1:length(b)
                b[k] = 1
                copyto!(view(basis, :, i), reshape(convert(Array, t), (l,)))
                # TODO: change this to `copy!` once we drop support for Julia 1.4
                qdims[i] = dim(c)
                b[k] = 0
                i += 1
            end
        end
        rhs = reshape(data, (l,))
        if FusionStyle(sectortype(t)) isa UniqueFusion
            lhs = basis'*rhs
        else
            lhs = Diagonal(qdims) \ (basis'*rhs)
        end
        if norm(basis*lhs - rhs) > tol
            throw(ArgumentError("Data has non-zero elements at incompatible positions"))
        end
        if eltype(lhs) != eltype(t)
            t2 = TensorMap(zeros, promote_type(eltype(lhs), eltype(t)), codom, dom)
        else
            t2 = t
        end
        i = 1
        for (c,b) in blocks(t2)
            for k = 1:length(b)
                b[k] = lhs[i]
                i += 1
            end
        end
        return t2
    end
end

function TensorMap(data::AbstractDict{<:Sector,<:DenseMatrix}, codom::ProductSpace{S,N₁}, dom::ProductSpace{S,N₂}) where {S<:IndexSpace, N₁, N₂}
    I = sectortype(S)
    I == keytype(data) || throw(SectorMismatch())
    if I == Trivial
        if dim(dom) != 0 && dim(codom) != 0
            return TensorMap(data[Trivial()], codom, dom)
        else
            return TensorMap(valtype(data)(undef, dim(codom), dim(dom)), codom, dom)
        end
    end
    F₁ = fusiontreetype(I, N₁)
    F₂ = fusiontreetype(I, N₂)
    rowr = SectorDict{I, FusionTreeDict{F₁, UnitRange{Int}}}()
    colr = SectorDict{I, FusionTreeDict{F₂, UnitRange{Int}}}()
    rowdims = SectorDict{I, Int}()
    coldims = SectorDict{I, Int}()
    if N₁ == 0 || N₂ == 0
        blocksectoriterator = (one(I),)
    elseif N₂ <= N₁
        blocksectoriterator = blocksectors(dom)
    else
        blocksectoriterator = blocksectors(codom)
    end
    for s1 in sectors(codom)
        for c in blocksectoriterator
            offset1 = get!(rowdims, c, 0)
            rowrc = get!(rowr, c) do
                FusionTreeDict{F₁, UnitRange{Int}}()
            end
            for f1 in fusiontrees(s1, c, map(isdual, codom.spaces))
                r = (offset1 + 1):(offset1 + dim(codom, s1))
                push!(rowrc, f1 => r)
                offset1 = last(r)
            end
            rowdims[c] = offset1
        end
    end
    for s2 in sectors(dom)
        for c in blocksectoriterator
            offset2 = get!(coldims, c, 0)
            colrc = get!(colr, c) do
                FusionTreeDict{F₂, UnitRange{Int}}()
            end
            for f2 in fusiontrees(s2, c, map(isdual, dom.spaces))
                r = (offset2 + 1):(offset2 + dim(dom, s2))
                push!(colrc, f2 => r)
                offset2 = last(r)
            end
            coldims[c] = offset2
        end
    end
    for c in blocksectoriterator
        dim1 = get!(rowdims, c, 0)
        dim2 = get!(coldims, c, 0)
        if dim1 == 0 || dim2 == 0
            delete!(rowr, c)
            delete!(colr, c)
        else
            (haskey(data, c) && size(data[c]) == (dim1, dim2)) ||
            throw(DimensionMismatch())
        end
    end
    if !isreal(I) && eltype(valtype(data)) <: Real
        b = valtype(data)(undef, (0,0))
        V = typeof(complex(b))
        K = keytype(data)
        data2 = SectorDict{K,V}((c=>complex(data[c])) for c in blocksectoriterator)
        A = typeof(data2)
        return TensorMap{S, N₁, N₂, I, A, F₁, F₂}(data2, codom, dom, rowr, colr)
    else
        V = valtype(data)
        K = keytype(data)
        data2 = SectorDict{K,V}((c=>data[c]) for c in blocksectoriterator)
        A = typeof(data2)
        return TensorMap{S, N₁, N₂, I, A, F₁, F₂}(data2, codom, dom, rowr, colr)
    end
end

# without data: generic constructor from callable:
function TensorMap(f, codom::ProductSpace{S,N₁}, dom::ProductSpace{S,N₂}) where {S<:IndexSpace, N₁, N₂}
    I = sectortype(S)
    if I == Trivial
        d1 = dim(codom)
        d2 = dim(dom)
        data = f((d1,d2))
        A = typeof(data)
        return TensorMap{S, N₁, N₂, Trivial, A, Nothing, Nothing}(data, codom, dom)
    else
        F₁ = fusiontreetype(I, N₁)
        F₂ = fusiontreetype(I, N₂)
        # TODO: the current approach is not very efficient and somewhat wasteful
        sampledata = f((1,1))
        if !isreal(I) && eltype(sampledata) <: Real
            A = typeof(complex(sampledata))
        else
            A = typeof(sampledata)
        end
        data = SectorDict{I,A}()
        rowr = SectorDict{I, FusionTreeDict{F₁, UnitRange{Int}}}()
        colr = SectorDict{I, FusionTreeDict{F₂, UnitRange{Int}}}()
        rowdims = SectorDict{I, Int}()
        coldims = SectorDict{I, Int}()
        if N₁ == 0 || N₂ == 0
            blocksectoriterator = (one(I),)
        elseif N₂ <= N₁
            blocksectoriterator = blocksectors(dom)
        else
            blocksectoriterator = blocksectors(codom)
        end
        for s1 in sectors(codom)
            for c in blocksectoriterator
                offset1 = get!(rowdims, c, 0)
                rowrc = get!(rowr, c) do
                    FusionTreeDict{F₁, UnitRange{Int}}()
                end
                for f1 in fusiontrees(s1, c, map(isdual, codom.spaces))
                    r = (offset1 + 1):(offset1 + dim(codom, s1))
                    push!(rowrc, f1 => r)
                    offset1 = last(r)
                end
                rowdims[c] = offset1
            end
        end
        for s2 in sectors(dom)
            for c in blocksectoriterator
                offset2 = get!(coldims, c, 0)
                colrc = get!(colr, c) do
                    FusionTreeDict{F₂, UnitRange{Int}}()
                end
                for f2 in fusiontrees(s2, c, map(isdual, dom.spaces))
                    r = (offset2 + 1):(offset2 + dim(dom, s2))
                    push!(colrc, f2 => r)
                    offset2 = last(r)
                end
                coldims[c] = offset2
            end
        end
        for c in blocksectoriterator
            dim1 = get!(rowdims, c, 0)
            dim2 = get!(coldims, c, 0)
            if dim1 == 0 || dim2 == 0
                delete!(rowr, c)
                delete!(colr, c)
            else
                data[c] = f((dim1, dim2))
            end
        end
        return TensorMap{S, N₁, N₂, I, SectorDict{I,A}, F₁, F₂}(data, codom, dom, rowr, colr)
    end
end

TensorMap(f,
            ::Type{T},
            codom::ProductSpace{S},
            dom::ProductSpace{S}) where {S<:IndexSpace, T<:Number} =
    TensorMap(d->f(T, d), codom, dom)

TensorMap(::Type{T},
            codom::ProductSpace{S},
            dom::ProductSpace{S}) where {S<:IndexSpace, T<:Number} =
    TensorMap(d->Array{T}(undef, d), codom, dom)

TensorMap(::UndefInitializer,
            ::Type{T},
            codom::ProductSpace{S},
            dom::ProductSpace{S}) where {S<:IndexSpace, T<:Number} =
    TensorMap(d->Array{T}(undef, d), codom, dom)

TensorMap(::UndefInitializer,
            codom::ProductSpace{S},
            dom::ProductSpace{S}) where {S<:IndexSpace} =
    TensorMap(undef, Float64, codom, dom)

TensorMap(::Type{T},
            codom::TensorSpace{S},
            dom::TensorSpace{S}) where {T<:Number, S<:IndexSpace} =
    TensorMap(T, convert(ProductSpace, codom), convert(ProductSpace, dom))

TensorMap(dataorf, codom::TensorSpace{S}, dom::TensorSpace{S}) where {S<:IndexSpace} =
    TensorMap(dataorf, convert(ProductSpace, codom), convert(ProductSpace, dom))

TensorMap(dataorf, ::Type{T},
            codom::TensorSpace{S},
            dom::TensorSpace{S}) where {T<:Number, S<:IndexSpace} =
    TensorMap(dataorf, T, convert(ProductSpace, codom), convert(ProductSpace, dom))

TensorMap(codom::TensorSpace{S}, dom::TensorSpace{S}) where {S<:IndexSpace} =
    TensorMap(Float64, convert(ProductSpace, codom), convert(ProductSpace, dom))

TensorMap(dataorf, T::Type{<:Number}, P::TensorMapSpace{S}) where {S<:IndexSpace} =
    TensorMap(dataorf, T, codomain(P), domain(P))

TensorMap(dataorf, P::TensorMapSpace{S}) where {S<:IndexSpace} =
    TensorMap(dataorf, codomain(P), domain(P))

TensorMap(T::Type{<:Number}, P::TensorMapSpace{S}) where {S<:IndexSpace} =
    TensorMap(T, codomain(P), domain(P))

TensorMap(P::TensorMapSpace{S}) where {S<:IndexSpace} = TensorMap(codomain(P), domain(P))

Tensor(dataorf, T::Type{<:Number}, P::TensorSpace{S}) where {S<:IndexSpace} =
    TensorMap(dataorf, T, P, one(P))

Tensor(dataorf, P::TensorSpace{S}) where {S<:IndexSpace} = TensorMap(dataorf, P, one(P))

Tensor(T::Type{<:Number}, P::TensorSpace{S}) where {S<:IndexSpace} = TensorMap(T, P, one(P))

Tensor(P::TensorSpace{S}) where {S<:IndexSpace} = TensorMap(P, one(P))

# Efficient copy constructors
#-----------------------------
function Base.copy(t::TrivialTensorMap{S, N₁, N₂, A}) where {S, N₁, N₂, A}
    return TrivialTensorMap{S, N₁, N₂, A}(copy(t.data), t.codom, t.dom)
end
function Base.copy(t::TensorMap{S, N₁, N₂, I, A, F₁, F₂}) where {S, N₁, N₂, I, A, F₁, F₂}
    return TensorMap{S, N₁, N₂, I, A, F₁, F₂}(deepcopy(t.data), t.codom, t.dom, t.rowr, t.colr)
end

# Similar
#---------
Base.similar(t::AbstractTensorMap, T::Type, codomain::VectorSpace, domain::VectorSpace) =
    similar(t, T, codomain←domain)
Base.similar(t::AbstractTensorMap, codomain::VectorSpace, domain::VectorSpace) =
    similar(t, codomain←domain)

Base.similar(t::AbstractTensorMap{S}, ::Type{T},
                P::TensorMapSpace{S} = (domain(t) → codomain(t))) where {T,S} =
    TensorMap(d->similarstoragetype(t, T)(undef, d), P)
Base.similar(t::AbstractTensorMap{S}, ::Type{T}, P::TensorSpace{S}) where {T,S} =
    Tensor(d->similarstoragetype(t, T)(undef, d), P)
Base.similar(t::AbstractTensorMap{S},
                P::TensorMapSpace{S} = (domain(t) → codomain(t))) where {S} =
    TensorMap(d->storagetype(t)(undef, d), P)
Base.similar(t::AbstractTensorMap{S}, P::TensorSpace{S}) where {S} =
    Tensor(d->storagetype(t)(undef, d), P)

function Base.complex(t::AbstractTensorMap)
    if eltype(t) <: Complex
        return t
    elseif t.data isa AbstractArray
        return TensorMap(complex(t.data), codomain(t), domain(t))
    else
        data = SectorDict(c=>complex(d) for (c,d) in t.data)
        return TensorMap(data, codomain(t), domain(t))
    end
end

# Conversion between TensorMap and Dict, for read and write purpose
#------------------------------------------------------------------
function Base.convert(::Type{Dict}, t::AbstractTensorMap)
    d = Dict{Symbol,Any}()
    d[:codomain] = repr(codomain(t))
    d[:domain] = repr(domain(t))
    data = Dict{String,Any}()
    for (c,b) in blocks(t)
        data[repr(c)] = Array(b)
    end
    d[:data] = data
    return d
end
function Base.convert(::Type{TensorMap}, d::Dict{Symbol,Any})
    try
        codomain = eval(Meta.parse(d[:codomain]))
        domain = eval(Meta.parse(d[:domain]))
        data = SectorDict(eval(Meta.parse(c))=>b for (c,b) in d[:data])
        return TensorMap(data, codomain, domain)
    catch e # sector unknown in TensorKit.jl; user-defined, hopefully accessible in Main
        codomain = Base.eval(Main, Meta.parse(d[:codomain]))
        domain = Base.eval(Main, Meta.parse(d[:domain]))
        data = SectorDict(Base.eval(Main, Meta.parse(c))=>b for (c,b) in d[:data])
        return TensorMap(data, codomain, domain)
    end
end

# Getting and setting the data
#------------------------------
hasblock(t::TrivialTensorMap, ::Trivial) = true
hasblock(t::TensorMap, s::Sector) = haskey(t.data, s)

block(t::TrivialTensorMap, ::Trivial) = t.data
function block(t::TensorMap, s::Sector)
    sectortype(t) == typeof(s) || throw(SectorMismatch())
    A = valtype(t.data)
    if haskey(t.data, s)
        return t.data[s]
    else # at least one of the two matrix dimensions will be zero
        return storagetype(t)(undef, (blockdim(codomain(t),s), blockdim(domain(t), s)))
    end
end

blocks(t::TensorMap{<:IndexSpace,N₁,N₂,Trivial}) where {N₁,N₂} =
    SingletonDict(Trivial()=>t.data)
blocks(t::TensorMap) = t.data

fusiontrees(t::TrivialTensorMap) = ((nothing, nothing),)
fusiontrees(t::TensorMap) = TensorKeyIterator(t.rowr, t.colr)

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
    f1 = FusionTree(s1, c1, map(isdual, tuple(codomain(t)...)))
    f2 = FusionTree(s2, c1, map(isdual, tuple(domain(t)...)))
    @inbounds begin
        return t[f1,f2]
    end
end
@propagate_inbounds Base.getindex(t::TensorMap, sectors::Tuple) =
    t[map(sectortype(t), sectors)]

@inline function Base.getindex(t::TensorMap{<:IndexSpace,N₁,N₂,I},
        f1::FusionTree{I,N₁}, f2::FusionTree{I,N₂}) where {N₁,N₂,I<:Sector}

    c = f1.coupled
    @boundscheck begin
        c == f2.coupled || throw(SectorMismatch())
        haskey(t.rowr[c], f1) || throw(SectorMismatch())
        haskey(t.colr[c], f2) || throw(SectorMismatch())
    end
    @inbounds begin
        d = (dims(codomain(t), f1.uncoupled)..., dims(domain(t), f2.uncoupled)...)
        return sreshape(StridedView(t.data[c])[t.rowr[c][f1], t.colr[c][f2]], d)
    end
end
@propagate_inbounds Base.setindex!(t::TensorMap{<:IndexSpace,N₁,N₂,I},
                                    v,
                                    f1::FusionTree{I,N₁},
                                    f2::FusionTree{I,N₂}) where {N₁,N₂,I<:Sector} =
                                    copy!(getindex(t, f1, f2), v)

# For a tensor with trivial symmetry, allow no argument indexing
@inline Base.getindex(t::TrivialTensorMap) =
    sreshape(StridedView(t.data), (dims(codomain(t))..., dims(domain(t))...))
@inline Base.setindex!(t::TrivialTensorMap, v) = copy!(getindex(t), v)

# For a tensor with trivial symmetry, fusiontrees returns (nothing,nothing)
@inline Base.getindex(t::TrivialTensorMap, ::Nothing, ::Nothing) = getindex(t)
@inline Base.setindex!(t::TrivialTensorMap, v, ::Nothing, ::Nothing) = setindex!(t, v)

# For a tensor with trivial symmetry, allow direct indexing
@inline function Base.getindex(t::TrivialTensorMap, indices::Vararg{Int})
    data = t[]
    @boundscheck checkbounds(data, indices...)
    @inbounds v = data[indices...]
    return v
end
@inline function Base.setindex!(t::TrivialTensorMap, v, indices::Vararg{Int})
    data = t[]
    @boundscheck checkbounds(data, indices...)
    @inbounds data[indices...] = v
    return v
end

# Show
#------
function Base.summary(t::TensorMap)
    print("TensorMap(", codomain(t), " ← ", domain(t), ")")
end
function Base.show(io::IO, t::TensorMap{S}) where {S<:IndexSpace}
    if get(io, :compact, false)
        print(io, "TensorMap(", codomain(t), " ← ", domain(t), ")")
        return
    end
    println(io, "TensorMap(", codomain(t), " ← ", domain(t), "):")
    if sectortype(S) == Trivial
        Base.print_array(io, t[])
        println(io)
    elseif FusionStyle(sectortype(S)) isa UniqueFusion
        for (f1,f2) in fusiontrees(t)
            println(io, "* Data for sector ", f1.uncoupled, " ← ", f2.uncoupled, ":")
            Base.print_array(io, t[f1,f2])
            println(io)
        end
    else
        for (f1,f2) in fusiontrees(t)
            println(io, "* Data for fusiontree ", f1, " ← ", f2, ":")
            Base.print_array(io, t[f1,f2])
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
Base.convert(::Type{TensorMap}, t::AbstractTensorMap) =
    copy!(TensorMap(undef, eltype(t), codomain(t), domain(t)), t)

function Base.convert(T::Type{TensorMap{S,N₁,N₂,I,A,F1,F2}},
                        t::AbstractTensorMap{S,N₁,N₂}) where {S,N₁,N₂,I,A,F1,F2}
    if typeof(t) == T
        return t
    else
        data = Dict(c=>convert(storagetype(T), b) for (c,b) in blocks(t))
        return TensorMap(data, codomain(t), domain(t))
    end
end
