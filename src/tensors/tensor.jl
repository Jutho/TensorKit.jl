# TensorMap & Tensor:
# general tensor implementation with arbitrary symmetries
#==========================================================#
struct TensorMap{S<:IndexSpace, N₁, N₂, A, F₁, F₂} <: AbstractTensorMap{S, N₁, N₂}
    data::A
    codom::ProductSpace{S,N₁}
    dom::ProductSpace{S,N₂}
    rowr::Dict{F₁,UnitRange{Int}}
    colr::Dict{F₂,UnitRange{Int}}
    function TensorMap{S,N₁,N₂}(data, spaces::TensorMapSpace{S,N₁,N₂}) where {S<:IndexSpace, N₁, N₂}
        codom = spaces[2]
        dom = spaces[1]
        G = sectortype(S)
        if G == Trivial
            data2 = validatedata(data, codom, dom, fieldtype(S), sectortype(S))
            new{S,N₁,N₂,typeof(data2), Void, Void}(data2, codom, dom)
        else
            F₁ = fusiontreetype(G,Val(N₁))
            F₂ = fusiontreetype(G,Val(N₂))
            data2, rowr, colr = validatedata(data, codom, dom, fieldtype(S), sectortype(S))
            new{S,N₁,N₂,typeof(data2),F₁,F₂}(data2, codom, dom, rowr, colr)
        end
    end
end

const Tensor{S<:IndexSpace, N, A, F₁, F₂} = TensorMap{S, N, 0, A, F₁, F₂}

blocksectors(t::TensorMap{<:IndexSpace, N₁, N₂, <:AbstractArray}) where {N₁,N₂} = (Trivial(),)
blocksectors(t::TensorMap{<:IndexSpace, N₁, N₂, <:Associative}) where {N₁,N₂} = keys(t.data)

function blocksectors(codom::ProductSpace{S,N₁}, dom::ProductSpace{S,N₂}) where {S,N₁,N₂}
    G = sectortype(S)
    if G == Trivial
        return (Trivial(),)
    end
    if N₁ == 0
        c1 = Set{G}((one(G),))
    elseif N₁ == 1
        c1 = Set{G}(first(s) for s in sectors(codom))
    else
        c1 = foldl(union!, Set{G}(), (⊗(s...) for s in sectors(codom)))
    end
    if N₂ == 0
        c2 = Set{G}((one(G),))
    elseif N₂ == 1
        c2 = Set{G}(first(s) for s in sectors(dom))
    else
        c2 = foldl(union!, Set{G}(), (⊗(s...) for s in sectors(dom)))
    end

    return intersect(c1,c2)
end
function validatedata(data::AbstractArray, codom, dom, k::Field, ::Type{Trivial})
    if ndims(data) == 2
        size(data) == (dim(codom), dim(dom)) || size(data) == (dims(codom)..., dims(dom)...) || throw(DimensionMismatch())
    elseif ndims(data) == 1
        length(data) == dim(codom) * dim(dom) || throw(DimensionMismatch())
    else
        size(data) == (dims(codom)..., dims(dom)...) || throw(DimensionMismatch())
    end
    eltype(data) ⊆ k || warn("eltype(data) = $(eltype(data)) ⊈ $k)")
    return reshape(data, (dim(codom), dim(dom)))
end
function validatedata(data::Associative{G, <:AbstractMatrix}, codom::ProductSpace{S,N₁}, dom::ProductSpace{S,N₂}, k::Field, ::Type{G}) where {S<:IndexSpace, G<:Sector, N₁,N₂}
    F₁ = fusiontreetype(G,Val(N₁))
    F₂ = fusiontreetype(G,Val(N₂))
    rowr = Dict{F₁, UnitRange{Int}}()
    colr = Dict{F₂, UnitRange{Int}}()
    for c in blocksectors(codom, dom)
        offset1 = 0
        for s1 in sectors(codom)
            for f in fusiontrees(s1, c)
                r = offset1 .+ (1:dim(codom, s1))
                rowr[f] = r
                offset1 = last(r)
            end
        end
        offset2 = 0
        for s2 in sectors(dom)
            for f in fusiontrees(s2, c)
                r = offset2 .+ (1:dim(dom, s2))
                colr[f] = r
                offset2 = last(r)
            end
        end
        (haskey(data, c) && size(data[c]) == (offset1, offset2)) || throw(DimensionMismatch())
        eltype(data[c]) ⊆ k || warn("eltype(data) = $(eltype(data)) ⊈ $k)")
    end
    return data, rowr, colr
end
# TODO: allow to start from full data (a single AbstractArray) and create the dictionary, in the first place for Abelian sectors, or for e.g. SU₂ using Wigner 3j symbols

# Basic methods for characterising a tensor:
#--------------------------------------------
codomain(t::TensorMap) = t.codom
domain(t::TensorMap) = t.dom

Base.eltype(::Type{<:TensorMap{<:IndexSpace,N₁,N₂,<:AbstractArray{T}}}) where {T,N₁,N₂} = T
Base.eltype(::Type{<:TensorMap{<:IndexSpace,N₁,N₂,<:Associative{<:Any,<:AbstractMatrix{T}}}}) where {T,N₁,N₂} = T

Base.length(t::TensorMap) = sum(length, blocks(t)) # total number of free parameters, in order to use e.g. KrylovKit

# General TensorMap constructors
#--------------------------------
# with data
TensorMap(data::Union{AbstractArray,Associative}, P::TensorMapSpace{S,N₁,N₂}) where {S<:IndexSpace, N₁, N₂} = TensorMap{S,N₁,N₂}(data, P)

# without data: generic constructor from callable:
TensorMap(f, T::Type{<:Number}, P::TensorMapSpace) = TensorMap(generatedata(f, T, P.second, P.first), P)
TensorMap(f, P::TensorMapSpace) = TensorMap(generatedata(f, P.second, P.first), P)

# uninitialized tensor
TensorMap(T::Type{<:Number}, P::TensorMapSpace) = TensorMap(generatedata(Array{T}, P.second, P.first), P)
TensorMap(P::TensorMapSpace) = TensorMap(Float64, P)

Tensor(dataorf, T::Type{<:Number}, P::TensorSpace{S}) where {S<:IndexSpace} = TensorMap(dataorf, T, one(P)→P)
Tensor(dataorf, P::TensorSpace{S}) where {S<:IndexSpace} = TensorMap(dataorf, one(P)→P)
Tensor(T::Type{<:Number}, P::TensorSpace{S}) where {S<:IndexSpace} = TensorMap(T, one(P)→P)
Tensor(P::TensorSpace{S}) where {S<:IndexSpace} = TensorMap(one(P)→P)

# generate data:
generatedata(A::Type{<:AbstractArray}, T::Type{<:Number}, codom::ProductSpace, dom::ProductSpace) = _generatedata(A{T}, codom, dom, sectortype(codom))
generatedata(f, T::Type{<:Number}, codom::ProductSpace, dom::ProductSpace) = _generatedata(f, T, codom, dom, sectortype(codom))
generatedata(f, codom::ProductSpace, dom::ProductSpace) = _generatedata(f, codom, dom, sectortype(codom))

_generatedata(f, T::Type{<:Number}, codom::ProductSpace, dom::ProductSpace, ::Type{Trivial}) = f(T, (dim(codom), dim(dom)))
_generatedata(f, codom::ProductSpace, dom::ProductSpace, ::Type{Trivial}) = f((dim(codom), dim(dom)))

function _generatedata(f, T::Type{<:Number}, codom::ProductSpace{S,N₁}, dom::ProductSpace{S,N₂}, G::Type{<:Sector}) where {S,N₁,N₂}
    F₁ = fusiontreetype(G, Val(N₁))
    F₂ = fusiontreetype(G, Val(N₂))
    rowr = Dict{F₁, UnitRange{Int}}()
    colr = Dict{F₂, UnitRange{Int}}()
    A = typeof(f(T,(1,1)))
    data = Dict{G,A}()
    for c in blocksectors(codom, dom)
        dim1 = 0
        for s1 in sectors(codom)
            for f1 in fusiontrees(s1, c)
                r = dim1 .+ (1:dim(codom, s1))
                dim1 = last(r)
                rowr[f1] = r
            end
        end
        dim2 = 0
        for s2 in sectors(dom)
            for f2 in fusiontrees(s2, c)
                r = dim2 .+ (1:dim(dom, s2))
                dim2 = last(r)
                colr[f2] = r
            end
        end
        data[c] = f(T, (dim1, dim2))
    end
    return data
end
function _generatedata(f, codom::ProductSpace{S,N₁}, dom::ProductSpace{S,N₂}, G::Type{<:Sector}) where {S,N₁,N₂}
    F₁ = fusiontreetype(G, Val(N₁))
    F₂ = fusiontreetype(G, Val(N₂))
    rowr = Dict{F₁, UnitRange{Int}}()
    colr = Dict{F₂, UnitRange{Int}}()
    A = typeof(f((1,1)))
    data = Dict{G,A}()
    for c in blocksectors(codom, dom)
        dim1 = 0
        for s1 in sectors(codom)
            for f1 in fusiontrees(s1, c)
                r = dim1 .+ (1:dim(codom, s1))
                dim1 = last(r)
                rowr[f1] = r
            end
        end
        dim2 = 0
        for s2 in sectors(dom)
            for f2 in fusiontrees(s2, c)
                r = dim2 .+ (1:dim(dom, s2))
                dim2 = last(r)
                colr[f2] = r
            end
        end
        data[c] = f((dim1, dim2))
    end
    return data
end

# Getting and setting the data
#------------------------------
hasblock(t::TensorMap{<:ElementarySpace,N₁,N₂,<:Associative}, s::Sector) where {N₁,N₂} = haskey(t.data, s)
hasblock(t::TensorMap{<:ElementarySpace,N₁,N₂,<:AbstractArray}, ::Trivial) where {N₁,N₂} = true

block(t::TensorMap{S,N₁,N₂,<:Associative}, s::Sector) where {S,N₁,N₂} = sectortype(S) == typeof(s) ? t.data[s] : throw(SectorMismatch())
block(t::TensorMap{S,N₁,N₂,<:AbstractArray}, ::Trivial) where {S,N₁,N₂} = t.data

blocks(t::TensorMap{S,N₁,N₂,<:Associative}) where {S,N₁,N₂} = values(t.data)
blocks(t::TensorMap{S,N₁,N₂,<:AbstractArray}) where {S,N₁,N₂} = (t.data,)

function Base.getindex(t::TensorMap{S,N₁,N₂}, f1::FusionTree{G,N₁}, f2::FusionTree{G,N₂}) where {S,N₁,N₂,G}
    c = f1.incoming
    c == f2.incoming || throw(SectorMismatch())
    checksectors(codomain(t), f1.outgoing) && checksectors(domain(t), f2.outgoing)
    return splitdims(view(t.data[c], t.rowr[f1], t.colr[f2]), dims(codomain(t), f1.outgoing), dims(domain(t), f2.outgoing))
end
Base.setindex!(t::TensorMap{S,N₁,N₂}, v, f1::FusionTree{G,N₁}, f2::FusionTree{G,N₂}) where {S,N₁,N₂,G} = copy!(getindex(t, f1, f2), v)

function Base.getindex(t::Tensor{S,N}, f::FusionTree{G,N}) where {S,N,G}
    f.incoming == one(G) || throw(SectorMismatch())
    checksectors(codomain(t), f.outgoing)
    return splitdims(view(t.data[one(G)], t.rowr[f], :), dims(codomain(t), f.outgoing), ())
end
Base.setindex!(t::TensorMap{S,N}, v, f::FusionTree{G,N}) where {S,N,G} = copy!(getindex(t, f), v)

# For a tensor with trivial symmetry, allow direct indexing
@inline Base.getindex(t::TensorMap{<:Any,N₁,N₂,<:AbstractArray}) where {N₁,N₂} = splitdims(t.data, dims(codomain(t)), dims(domain(t)))
@inline function Base.getindex(t::TensorMap{<:Any,N₁,N₂,<:AbstractArray}, I::Vararg{Int}) where {N₁,N₂}
    data = splitdims(t.data, dims(codom), dims(dom))
    @boundscheck checkbounds(data, I)
    @inbounds v = data[I...]
    return v
end
@inline function Base.setindex!(t::TensorMap{<:Any,N₁,N₂,<:AbstractArray}, v, I::Vararg{Int}) where {N₁,N₂}
    data = splitdims(t.data, dims(codom), dims(dom))
    @boundscheck checkbounds(data, I)
    @inbounds data[I...] = v
    return v
end

# Similar
#---------
Base.similar(t::TensorMap{S}, ::Type{T}, P::TensorMapSpace{S} = (domain(t)=>codomain(t))) where {T,S} = TensorMap(d->similar(first(blocks(t)), T, d), P)
Base.similar(t::TensorMap{S}, ::Type{T}, P::TensorSpace{S}) where {T,S} = Tensor(d->similar(first(blocks(t)), T, d), P)
Base.similar(t::TensorMap{S}, P::TensorMapSpace{S} = (domain(t)=>codomain(t))) where {S} = TensorMap(d->similar(first(blocks(t)), d), P)
Base.similar(t::TensorMap{S}, P::TensorSpace{S}) where {S} = Tensor(d->similar(first(blocks(t)), d), P)

# Copy and fill tensors:
# ------------------------
function Base.copy!(tdest::TensorMap, tsource::TensorMap)
    codomain(tdest) == codomain(tsource) && domain(tdest) == domain(tsource) || throw(SpaceError())
    for c in blocksectors(tdest)
        copy!(block(tdest, c), block(tsource, c))
    end
    return tdest
end
function Base.fill!(t::TensorMap, value::Number)
    for b in blocks(t)
        fill!(b, value)
    end
    return t
end

# Equality and approximality
#----------------------------
function Base.:(==)(t1::TensorMap, t2::TensorMap)
    (codomain(t1)==codomain(t2) && domain(t1) == domain(t2)) || return false
    for c in blocksectors(t1)
        block(t1, c) == block(t2, c) || return false
    end
    return true
end

function Base.isapprox(t1::TensorMap, t2::TensorMap; atol::Real=0, rtol::Real=Base.rtoldefault(eltype(t1), eltype(t2), atol))
    d = vecnorm(t1 - t2)
    if isfinite(d)
        return d <= max(atol, rtol*max(vecnorm(t1), vecnorm(t2)))
    else
        return false
    end
end


# Conversion and promotion:
#---------------------------
# TODO

# Basic vector space methods:
# ---------------------------
function Base.scale!(t1::TensorMap, t2::TensorMap, α::Number)
    (codomain(t1)==codomain(t2) && domain(t1) == domain(t2)) || throw(SpaceError())
    for c in blocksectors(t1)
        scale!(block(t1, c), block(t2, c), α)
    end
    return t1
end

function Base.LinAlg.axpy!(α::Number, t1::TensorMap, t2::TensorMap)
    (codomain(t1)==codomain(t2) && domain(t1) == domain(t2)) || throw(SpaceError())
    for c in blocksectors(t1)
        Base.LinAlg.axpy!(α, block(t1, c), block(t2, c))
    end
    return t2
end

# inner product and norm only valid for spaces with Euclidean inner product
function Base.vecdot(t1::TensorMap{S}, t2::TensorMap{S}) where {S<:EuclideanSpace}
    (codomain(t1) == codomain(t2) && domain(t1) == domain(t2)) || throw(SpaceMismatch())
    return sum(dim(c)*vecdot(block(t1,c), block(t2,c)) for c in blocksectors(t1))
end

Base.vecnorm(t::TensorMap{<:EuclideanSpace}, p::Real) = vecnorm((convert(real(eltype(t)),dim(c)^(1/p)*vecnorm(block(t,c), p)) for c in blocksectors(t)), p)

# TensorMap multiplication:
#--------------------------
function Base.A_mul_B!(tC::TensorMap, tA::TensorMap,  tB::TensorMap)
    (codomain(tC) == codomain(tA) && domain(tC) == domain(tB) && domain(tA) == codomain(tB)) || throw(SpaceMismatch())
    for c in blocksectors(tC)
        if hasblock(tA, c) # then also tB should have such a block
            A_mul_B!(block(tC, c), block(tA, c), block(tB, c))
        else
            fill!(block(tC, c), 0)
        end
    end
    return tC
end

# Orthogonal factorizations: only correct if Euclidean inner product
#--------------------------------------------------------------------
function leftorth!(t::TensorMap{S}) where {S<:EuclideanSpace}
    if isa(t.data, AbstractArray)
        Q, R = leftorth!(t.data)
        V = S(size(Q,2))
        return TensorMap(Q, codomain(t)←V), TensorMap(R, V←domain(t))
    else
        it = blocksectors(t)
        c, s = next(it, start(it))
        Q, R = leftorth!(t.data[c])
        Qdata = Dict(c => Q)
        Rdata = Dict(c => R)
        dims = Dict(c => size(Q, 2))
        while !done(it, s)
            c, s = next(it, s)
            Q, R = leftorth!(t.data[c])
            Qdata[c] = Q
            Rdata[c] = R
            dims[c] = size(Q, 2)
        end
        V = S(dims)
        return TensorMap(Qdata, codomain(t)←V), TensorMap(Rdata, V←domain(t))
    end
end
function leftnull!(t::TensorMap{S}) where {S<:EuclideanSpace}
    if isa(t.data, AbstractArray)
        N = leftnull!(t.data)
        V = S(size(N, 2))
        return TensorMap(N, codomain(t)←V)
    else
        it = blocksectors(t)
        c, s = next(it, start(it))
        N = leftnull!(t.data[c])
        Ndata = Dict(c => N)
        dims = Dict(c => size(N, 2))
        while !done(it, s)
            c, s = next(it, s)
            N = leftnull!(t.data[c])
            Ndata[c] = N
        end
        V = S(dims)
        return TensorMap(Ndata, codomain(t)←V)
    end
end
function rightorth!(t::TensorMap{S}) where {S<:EuclideanSpace}
    if isa(t.data, AbstractArray)
        L, Q = rightorth!(t.data)
        V = S(size(Q, 1))
        return TensorMap(L, codomain(t)←V), TensorMap(Q, V←domain(t))
    else
        it = blocksectors(t)
        c, s = next(it, start(it))
        L, Q = rightorth!(t.data[c])
        Ldata = Dict(c => L)
        Qdata = Dict(c => Q)
        dims = Dict(c => size(Q, 1))
        while !done(it, s)
            c, s = next(it, s)
            L, Q = rightorth!(t.data[c])
            Ldata[c] = L
            Qdata[c] = Q
            dims[c] = size(Q, 1)
        end
        V = S(dims)
        return TensorMap(Ldata, codomain(t)←V), TensorMap(Qdata, V←domain(t))
    end
end
function rightnull!(t::TensorMap{S}) where {S<:EuclideanSpace}
    if isa(t.data, AbstractArray)
        N = rightnull!(t.data)
        V = S(size(N, 1))
        return TensorMap(N, V←domain(t))
    else
        it = blocksectors(t)
        c, s = next(it, start(it))
        N = rightnull!(t.data[c])
        Ndata = Dict(c => N)
        dims = Dict(c => size(N, 1))
        while !done(it, s)
            c, s = next(it, s)
            N = rightnull!(t.data[c])
            Ndata[c] = N
            dims[c] = size(N, 1)
        end
        V = S(dims)
        return TensorMap(Ndata, V←domain(t))
    end
end
function svd!(t::TensorMap{S}, trunc::TruncationScheme = NoTruncation(), p::Real = 2) where {S<:EuclideanSpace}
    if isa(t.data, AbstractArray)
        U,Σ,V = svd!(t.data)
        dmax = length(Σ)
        Σ, truncerr = _truncate!(Σ, trunc, p)
        d = length(Σ)
        W = S(d)
        if d < dmax
            U = U[:,1:d]
            V = V[1:d,:]
            Σ = Σ[1:d]
        end
        return TensorMap(U, codomain(t)←W), TensorMap(diagm(Σ), W←W), TensorMap(V, W←domain(t)), truncerr
        #TODO: make this work with Diagonal(Σ) in such a way that it is type stable and
        # robust for all further operations on that tensor
    else
        it = blocksectors(t)
        c, s = next(it, start(it))
        U,Σ,V = svd!(t.data[c])
        Udata = Dict(c => U)
        Σdata = Dict(c => Σ)
        Vdata = Dict(c => V)
        dims = Dict{typeof(c),Int}(c=> length(Σ))
        while !done(it, s)
            c, s = next(it, s)
            U,Σ,V = svd!(t.data[c])
            Udata[c] = U
            Σdata[c] = Σ
            Vdata[c] = V
            dims[c] = length(Σ)
        end
        Σdata, truncerr = _truncate!(Σdata, trunc, p)

        for c in it
            truncdim = length(Σdata[c])
            if truncdim != dims[c]
                dims[c] = truncdim
                Udata[c] = Udata[c][:, 1:truncdim]
                Vdata[c] = Vdata[c][1:truncdim, :]
            end
        end
        W = S(dims)
        return TensorMap(Udata, codomain(t)←W), TensorMap(Dict(c=>diagm(Σ) for (c,Σ) in Σdata), W←W), TensorMap(Vdata, W←domain(t)), truncerr
    end
end

function _truncate!(v::AbstractVector, trunc::TruncationScheme, p::Real = 2)
    fullnorm = vecnorm(v, p)
    truncerr = zero(fullnorm)
    if isa(trunc, NoTruncation)
        # don't do anything
    elseif isa(trunc, TruncationError)
        dmax = length(v)
        dtrunc = dmax
        while true
            dtrunc -= 1
            truncerr = vecnorm(view(v, dtrunc+1:dmax), p)
            if truncerr / fullnorm > trunc.ϵ
                dtrunc += 1
                break
            end
        end
        truncerr = vecnorm(view(v, dtrunc+1:dmax), p)
        resize!(v, dtrunc)
    elseif isa(trunc, TruncationDimension)
        dmax = length(v)
        dtrunc = min(dmax, trunc.dim)
        truncerr = vecnorm(view(v, dtrunc+1:dmax), p)
        resize!(v, dtrunc)
    else
        error("unknown truncation scheme")
    end
    return v, truncerr
end
function _truncate!(V::Associative{G,<:AbstractVector}, trunc::TruncationScheme, p = 2) where {G<:Sector}
    T = real(eltype(valtype(V)))
    fullnorm::T = vecnorm((convert(T, dim(c))^(1/p)*vecnorm(v, p) for (c,v) in V), p)
    truncerr::T = zero(fullnorm)
    it = keys(V)
    if isa(trunc, NoTruncation)
        # don't do anything
    elseif isa(trunc, TruncationError)
        truncdim = Dict{G,Int}(c=>length(v) for (c,v) in V)
        while true
            cmin = mininum(c->sqrt(dim(c))*V[c][truncdim[c]], keys(V))
            truncdim[cmin] -= 1
            truncerr = vecnorm((convert(T, dim(c))^(1/p)*vecnorm(view(Σdata[c],truncdim[c]+1:maxdim[c]), p) for c in it), p)
            if truncerr / fullnorm > trunc.ϵ
                truncdim[cmin] += 1
                break
            end
        end
        truncerr = vecnorm((convert(T, dim(c))^(1/p)*vecnorm(view(Σdata[c],truncdim[c]+1:maxdim[c]), p) for c in it), p)
        for c in it
            resize!(V[c], truncdim[c])
        end
    elseif isa(trunc, TruncationDimension)
        truncdim = Dict{G,Int}(c=>length(v) for (c,v) in V)
        while sum(c->dim(c)*truncdim[c], it) > trunc.dim
            cmin = mininum(c->dim(c)^(1/p)*V[c][truncdim[c]], it)
            truncdim[cmin] -= 1
        end
        truncerr = vecnorm((convert(T, dim(c))^(1/p)*vecnorm(view(Σdata[c],truncdim[c]+1:maxdim[c]), p) for c in it), p)
        for c in it
            resize!(V[c], truncdim[c])
        end
    elseif isa(trunc, TruncationSpace)
        for c in it
            if length(V[c]) > dim(trunc.space, c)
                resize!(V[c], dim(trunc.space, c))
            end
        end
    else
        error("unknown truncation scheme")
    end
    return V, truncerr
end

# Index manipulations
#---------------------
using Base.Iterators.filter
fusiontrees(t::TensorMap) = filter(fs->(fs[1].incoming == fs[2].incoming), product(keys(t.rowr), keys(t.colr)))

# TODO: reconsider whether we need repartitionind!, or if we just want permuteind!
function repartitionind!(tdst::TensorMap{S,N₁,N₂}, tsrc::TensorMap{S,N₁′,N₂′}) where {S,N₁,N₂,N₁′,N₂′}
    space1 = codomain(tdst) ⊗ dual(domain(tdst))
    space2 = codomain(tsrc) ⊗ dual(domain(tsrc))
    space1 == space2 || throw(SpaceMismatch())
    p = (ntuple(n->n, Val{N₁′})..., ntuple(n->N₁′+N₂′+1-n, Val{N₂′}))
    p1 = tselect(p, ntuple(n->n, Val{N₁}))
    p2 = reverse(tselect(p, ntuple(n->N₁+n, Val{N₂})))
    pdata = (p1..., p2...)

    if sectortype(S) == Trivial
        TensorOperations.add!(1, tsrc[], Val{:N}, 0, tdst[], pdata)
    else
        fill!(tdst, 0)
        for (f1,f2) in fusiontrees(t)
            for ((f1′,f2′), coeff) in repartition(f1, f2, Val{N₁})
                TensorOperations.add!(coeff, tsrc[f1,f2], Val{:N}, 1, tdst[f1′,f2′], pdata)
            end
        end
    end
    return tdst
end

function permuteind!(tdst::TensorMap{S,N₁,N₂}, tsrc::TensorMap{S}, p1::NTuple{N₁,Int}, p2::NTuple{N₂,Int} = ()) where {S,N₁,N₂}
    # TODO: Frobenius-Schur indicators!, and fermions!
    space1 = codomain(tdst) ⊗ dual(domain(tdst))
    space2 = codomain(tsrc) ⊗ dual(domain(tsrc))

    N₁′, N₂′ = length(codomain(tsrc)), length(domain(tsrc))
    p = linearizepermutation(p1, p2, N₁′, N₂′)

    isperm(p) && length(p) == N₁′+N₂′ || throw(ArgumentError("not a valid permutation: $p1 & $p2"))
    space1 == space2[p] || throw(SpaceMismatch())

    pdata = (p1..., p2...)
    if sectortype(S) == Trivial
        TensorOperations.add!(1, tsrc[], Val{:N}, 0, tdst[], pdata)
    else
        fill!(tdst, 0)
        for (f1,f2) in fusiontrees(tsrc)
            for ((f1′,f2′), coeff) in permute(f1, f2, p1, p2)
                TensorOperations.add!(coeff, tsrc[f1,f2], Val{:N}, 1, tdst[f1′,f2′], pdata)
            end
        end
    end
    return tdst
end

# do we need those?
function splitind! end#

function fuseind! end

# Adjoint (complex/Hermitian conjugation)
#-----------------------------------------
function adjoint!(tdst::TensorMap{<:EuclideanSpace}, tsrc::TensorMap{<:EuclideanSpace})
    (codomain(tdst) == domain(tsrc) && domain(tdst) == codomain(tsrc)) || throw(SpaceMismatch())

    for c in blocksectors(t)
        adjoint!(block(tdst, c), block(tsrc, c))
    end
    return tdst
end
