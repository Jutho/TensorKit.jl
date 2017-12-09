# TensorMap & Tensor:
# general tensor implementation with arbitrary symmetries
#==========================================================#
struct TensorMap{S<:IndexSpace, N₁, N₂, A, F₁, F₂} <: AbstractTensorMap{S, N₁, N₂}
    data::A
    codom::ProductSpace{S,N₁}
    dom::ProductSpace{S,N₂}
    rowr::Dict{F₁,UnitRange{Int}}
    colr::Dict{F₂,UnitRange{Int}}
    function TensorMap{S, N₁, N₂, A, F₁, F₂}(data::A, codom::ProductSpace{S,N₁}, dom::ProductSpace{S,N₂}, rowr::Dict{F₁,UnitRange{Int}}, colr::Dict{F₂,UnitRange{Int}}) where {S<:IndexSpace, N₁, N₂, A<:Associative, F₁, F₂}
        new{S, N₁, N₂, A, F₁, F₂}(data, codom, dom, rowr, colr)
    end
    function TensorMap{S, N₁, N₂, A, Void, Void}(data::A, codom::ProductSpace{S,N₁}, dom::ProductSpace{S,N₂}) where {S<:IndexSpace, N₁, N₂, A<:AbstractMatrix}
        new{S, N₁, N₂, A, Void, Void}(data, codom, dom)
    end
end

const Tensor{S<:IndexSpace, N, A, F₁, F₂} = TensorMap{S, N, 0, A, F₁, F₂}

# Basic methods for characterising a tensor:
#--------------------------------------------
codomain(t::TensorMap) = t.codom
domain(t::TensorMap) = t.dom

blocksectors(t::TensorMap{<:IndexSpace, N₁, N₂, <:AbstractArray}) where {N₁,N₂} = (Trivial(),)
blocksectors(t::TensorMap{<:IndexSpace, N₁, N₂, <:Associative}) where {N₁,N₂} = keys(t.data)

Base.eltype(::Type{<:TensorMap{<:IndexSpace,N₁,N₂,<:AbstractArray{T}}}) where {T,N₁,N₂} = T
Base.eltype(::Type{<:TensorMap{<:IndexSpace,N₁,N₂,<:Associative{<:Any,<:AbstractMatrix{T}}}}) where {T,N₁,N₂} = T

Base.length(t::TensorMap) = sum(length(b) for (c,b) in blocks(t)) # total number of free parameters, in order to use e.g. KrylovKit

# General TensorMap constructors
#--------------------------------
# with data
function TensorMap(data::AbstractArray, codom::ProductSpace{S,N₁}, dom::ProductSpace{S,N₂}) where {S<:IndexSpace, N₁, N₂}
    if sectortype(S) == Trivial # For now, we can only accept array data for Trivial sectortype
        if ndims(data) == 2
            size(data) == (dim(codom), dim(dom)) || size(data) == (dims(codom)..., dims(dom)...) || throw(DimensionMismatch())
        elseif ndims(data) == 1
            length(data) == dim(codom) * dim(dom) || throw(DimensionMismatch())
        else
            size(data) == (dims(codom)..., dims(dom)...) || throw(DimensionMismatch())
        end
        eltype(data) ⊆ fieldtype(S) || warn("eltype(data) = $(eltype(data)) ⊈ $(fieldtype(S)))")
        data2 = reshape(data, (dim(codom), dim(dom)))
        A = typeof(data2)
        return TensorMap{S,N₁,N₂,A,Void,Void}(data2, codom, dom)
    else
        # TODO: allow to start from full data (a single AbstractArray) and create the dictionary, in the first place for Abelian sectors, or for e.g. SU₂ using Wigner 3j symbols
        throw(SectorMismatch())
    end
end

function TensorMap(data::A, codom::ProductSpace{S,N₁}, dom::ProductSpace{S,N₂}) where {A<:Associative, S<:IndexSpace, N₁, N₂}
    G = sectortype(S)
    G == keytype(data) || throw(SectorMismatch())
    F₁ = fusiontreetype(G, StaticLength(N₁))
    F₂ = fusiontreetype(G, StaticLength(N₂))
    rowr = Dict{F₁, UnitRange{Int}}()
    colr = Dict{F₂, UnitRange{Int}}()
    for c in blocksectors(codom, dom)
        offset1 = 0
        for s1 in sectors(codom)
            for f in fusiontrees(s1, c)
                r = (offset1 + 1):(offset1 + dim(codom, s1))
                rowr[f] = r
                offset1 = last(r)
            end
        end
        offset2 = 0
        for s2 in sectors(dom)
            for f in fusiontrees(s2, c)
                r = (offset2 + 1):(offset2 + dim(dom, s2))
                colr[f] = r
                offset2 = last(r)
            end
        end
        (haskey(data, c) && size(data[c]) == (offset1, offset2)) || throw(DimensionMismatch())
        eltype(data[c]) ⊆ fieldtype(S) || warn("eltype(data) = $(eltype(data[c])) ⊈ $(fieldtype(S)))")
    end
    return TensorMap{S, N₁, N₂, A, F₁, F₂}(data, codom, dom, rowr, colr)
end

# without data: generic constructor from callable:
function TensorMap(f, codom::ProductSpace{S,N₁}, dom::ProductSpace{S,N₂}) where {S<:IndexSpace, N₁, N₂}
    G = sectortype(S)
    if G == Trivial
        data = f((dim(codom), dim(dom)))
        return TensorMap{S, N₁, N₂, typeof(data), Void, Void}(data, codom, dom)
    else
        F₁ = fusiontreetype(G, StaticLength(N₁))
        F₂ = fusiontreetype(G, StaticLength(N₂))
        rowr = Dict{F₁, UnitRange{Int}}()
        colr = Dict{F₂, UnitRange{Int}}()
        A = typeof(f((1,1)))
        data = Dict{G,A}()
        for c in blocksectors(codom, dom)
            offset1 = 0
            for s1 in sectors(codom)
                for f1 in fusiontrees(s1, c)
                    r = (offset1 + 1):(offset1 + dim(codom, s1))
                    rowr[f1] = r
                    offset1 = last(r)
                end
            end
            dim1 = offset1
            offset2 = 0
            for s2 in sectors(dom)
                for f2 in fusiontrees(s2, c)
                    r = (offset2 + 1):(offset2 + dim(dom, s2))
                    colr[f2] = r
                    offset2 = last(r)
                end
            end
            dim2 = offset2
            data[c] = f((dim1, dim2))
        end
        return TensorMap{S, N₁, N₂, typeof(data), F₁, F₂}(data, codom, dom, rowr, colr)
    end
end
TensorMap(f, ::Type{T}, codom::ProductSpace{S}, dom::ProductSpace{S}) where {S<:IndexSpace, T<:Number} =
    TensorMap(d->f(T, d), codom, dom)
TensorMap(::Type{T}, codom::ProductSpace{S}, dom::ProductSpace{S}) where {S<:IndexSpace, T<:Number} =
    TensorMap(d->Array{T}(uninitalized, d), codom, dom)
TensorMap(I::UniformScaling, ::Type{T}, codom::ProductSpace{S}, dom::ProductSpace{S}) where {S<:IndexSpace, T<:Number} =
    TensorMap(d->Array{T}(I, d), codom, dom)
TensorMap(I::UniformScaling, codom::ProductSpace{S}, dom::ProductSpace{S}) where {S<:IndexSpace} = TensorMap(I, Float64, codom, dom)
TensorMap(::Uninitialized, ::Type{T}, codom::ProductSpace{S}, dom::ProductSpace{S}) where {S<:IndexSpace, T<:Number} =
    TensorMap(d->Array{T}(uninitialized, d), codom, dom)
TensorMap(::Uninitialized, codom::ProductSpace{S}, dom::ProductSpace{S}) where {S<:IndexSpace} = TensorMap(uninitialized, Float64, codom, dom)

TensorMap(::Type{T}, codom::TensorSpace{S}, dom::TensorSpace{S}) where {T<:Number, S<:IndexSpace} = TensorMap(T, convert(ProductSpace, codom), convert(ProductSpace, dom))
TensorMap(dataorf, codom::TensorSpace{S}, dom::TensorSpace{S}) where {S<:IndexSpace} = TensorMap(dataorf, convert(ProductSpace, codom), convert(ProductSpace, dom))
TensorMap(dataorf, ::Type{T}, codom::TensorSpace{S}, dom::TensorSpace{S}) where {T<:Number, S<:IndexSpace} = TensorMap(dataorf, T, convert(ProductSpace, codom), convert(ProductSpace, dom))
TensorMap(codom::TensorSpace{S}, dom::TensorSpace{S}) where {S<:IndexSpace} = TensorMap(Float64, convert(ProductSpace, codom), convert(ProductSpace, dom))

TensorMap(dataorf, T::Type{<:Number}, P::TensorMapSpace{S}) where {S<:IndexSpace} = TensorMap(dataorf, T, P[2], P[1])
TensorMap(dataorf, P::TensorMapSpace{S}) where {S<:IndexSpace} = TensorMap(dataorf, P[2], P[1])
TensorMap(T::Type{<:Number}, P::TensorMapSpace{S}) where {S<:IndexSpace} = TensorMap(T, P[2], P[1])
TensorMap(P::TensorMapSpace{S}) where {S<:IndexSpace} = TensorMap(P[2], P[1])

Tensor(dataorf, T::Type{<:Number}, P::TensorSpace{S}) where {S<:IndexSpace} = TensorMap(dataorf, T, P, one(P))
Tensor(dataorf, P::TensorSpace{S}) where {S<:IndexSpace} = TensorMap(dataorf, P, one(P))
Tensor(T::Type{<:Number}, P::TensorSpace{S}) where {S<:IndexSpace} = TensorMap(T, P, one(P))
Tensor(P::TensorSpace{S}) where {S<:IndexSpace} = TensorMap(P, one(P))

# Special purpose constructors
#------------------------------
Base.zero(t::AbstractTensorMap) = fill!(similar(t), 0)
function Base.one(t::AbstractTensorMap)
    domain(t) == codomain(t) || throw(SectorMismatch("no identity if domain and codomain are different"))
    eye(eltype(t), domain(t))
end
function one!(t::AbstractTensorMap)
    domain(t) == codomain(t) || throw(SectorMismatch("no identity if domain and codomain are different"))
    for (c,b) in blocks(t)
        copy!(b, I)
    end
    return t
end

# Getting and setting the data
#------------------------------
hasblock(t::TensorMap{<:IndexSpace,N₁,N₂,<:Associative}, s::Sector) where {N₁,N₂} = haskey(t.data, s)
hasblock(t::TensorMap{<:IndexSpace,N₁,N₂,<:AbstractArray}, ::Trivial) where {N₁,N₂} = true

function block(t::TensorMap{S,N₁,N₂,<:Associative}, s::Sector) where {S,N₁,N₂}
    sectortype(S) == typeof(s) || throw(SectorMismatch())
    A = valtype(t.data)
    if haskey(t.data, s)
        return t.data[s]
    else # at least one of the two matrix dimensions will be zero
        return A(uninitialized, (blockdim(codomain(t),s), blockdim(domain(t), s)))
    end
end
block(t::TensorMap{S,N₁,N₂,<:AbstractArray}, ::Trivial) where {S,N₁,N₂} = t.data

blocks(t::TensorMap{S,N₁,N₂,<:Associative}) where {S<:IndexSpace,N₁,N₂} = (c=>t.data[c] for c in blocksectors(t))
blocks(t::TensorMap{S,N₁,N₂,<:AbstractArray}) where {S<:IndexSpace,N₁,N₂} = (Trivial()=>t.data,)

fusiontrees(t::TensorMap) = filter(fs->(fs[1].incoming == fs[2].incoming), product(keys(t.rowr), keys(t.colr)))

function Base.getindex(t::TensorMap{S,N₁,N₂}, sectors::Tuple{Vararg{G}}) where {S<:IndexSpace,N₁,N₂,G<:Sector}
    (N₁+N₂ == length(sectors) && sectortype(S) == G) || throw(SectorMismatch("Sectors $sectors not valid for tensor in $(codomain(t))←$(domain(t))"))
    fusiontype(G) == Abelian || throw(SectorMismatch("Indexing with sectors only possible if abelian"))
    s1 = ntuple(n->sectors[n], StaticLength(N₁))
    s2 = ntuple(n->sectors[N₁+n], StaticLength(N₂))
    c1 = length(s1) == 0 ? one(G) : (length(s1) == 1 ? s1[1] : first(⊗(s1...)))
    @boundscheck begin
        c2 = length(s2) == 0 ? one(G) : (length(s2) == 1 ? s2[1] : first(⊗(s1...)))
        c2 == c1 || throw(SectorMismatch())
        checksectors(codomain(t), s1) && checksectors(domain(t), s2)
    end
    f1 = FusionTree(s1,c1)
    f2 = FusionTree(s2,c1)
    return t[f1,f2]
end
Base.getindex(t::TensorMap, sectors::Tuple) = t[map(sectortype(t), sectors)]

function Base.getindex(t::TensorMap{S,N₁,N₂}, f1::FusionTree{G,N₁}, f2::FusionTree{G,N₂}) where {S,N₁,N₂,G}
    c = f1.incoming
    @boundscheck begin
        c == f2.incoming || throw(SectorMismatch())
        checksectors(codomain(t), f1.outgoing) && checksectors(domain(t), f2.outgoing)
    end
    return splitdims(sview(t.data[c], t.rowr[f1], t.colr[f2]), dims(codomain(t), f1.outgoing), dims(domain(t), f2.outgoing))
end
@propagate_inbounds Base.setindex!(t::TensorMap{S,N₁,N₂}, v, f1::FusionTree{G,N₁}, f2::FusionTree{G,N₂}) where {S,N₁,N₂,G} = copy!(getindex(t, f1, f2), v)

function Base.getindex(t::Tensor{S,N}, f::FusionTree{G,N}) where {S,N,G}
    @boundscheck begin
        f.incoming == one(G) || throw(SectorMismatch())
        checksectors(codomain(t), f.outgoing)
    end
    return splitdims(sview(t.data[one(G)], t.rowr[f], :), dims(codomain(t), f.outgoing), ())
end
@propagate_inbounds Base.setindex!(t::TensorMap{S,N}, v, f::FusionTree{G,N}) where {S,N,G} = copy!(getindex(t, f), v)

# For a tensor with trivial symmetry, allow direct indexing
Base.getindex(t::TensorMap{<:Any,N₁,N₂,<:AbstractArray}) where {N₁,N₂} = splitdims(t.data, dims(codomain(t)), dims(domain(t)))
Base.setindex!(t::TensorMap{<:Any,N₁,N₂,<:AbstractArray}, v) where {N₁,N₂} = copy!(splitdims(t.data, dims(codomain(t)), dims(domain(t))), v)

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

# Show
#------
function Base.showcompact(io::IO, t::TensorMap)
    print(io, "TensorMap(", codomain(t), " ← ", domain(t), ")")
end
function Base.show(io::IO, t::TensorMap{S}) where {S<:IndexSpace}
    println(io, "TensorMap(", codomain(t), " ← ", domain(t), "):")
    if sectortype(S) == Trivial && isa(t.data, AbstractArray)
        Base.showarray(io, t[], false; header = false)
        println(io)
    elseif fusiontype(sectortype(S)) == Abelian
        for (f1,f2) in fusiontrees(t)
            println(io, "* Data for sector ", f1.outgoing, " ← ", f2.outgoing, ":")
            Base.showarray(io, t[f1,f2], false; header = false)
            println(io)
        end
    else
        for (f1,f2) in fusiontrees(t)
            println(io, "* Data for fusiontree ", f1, " ← ", f2, ":")
            Base.showarray(io, t[f1,f2], false; header = false)
            println(io)
        end
    end
end

# Similar
#---------
Base.similar(t::TensorMap{S}, ::Type{T}, P::TensorMapSpace{S} = (domain(t)=>codomain(t))) where {T,S} = TensorMap(d->similar(first(blocks(t))[2], T, d), P)
Base.similar(t::TensorMap{S}, ::Type{T}, P::TensorSpace{S}) where {T,S} = Tensor(d->similar(first(blocks(t))[2], T, d), P)
Base.similar(t::TensorMap{S}, P::TensorMapSpace{S} = (domain(t)=>codomain(t))) where {S} = TensorMap(d->similar(first(blocks(t))[2], d), P)
Base.similar(t::TensorMap{S}, P::TensorSpace{S}) where {S} = Tensor(d->similar(first(blocks(t))[2], d), P)

# Copy and fill tensors:
# ------------------------
function Base.copy!(tdest::TensorMap, tsource::TensorMap)
    codomain(tdest) == codomain(tsource) && domain(tdest) == domain(tsource) || throw(SpaceMismatch())
    for c in blocksectors(tdest)
        copy!(block(tdest, c), block(tsource, c))
    end
    return tdest
end
function Base.fill!(t::TensorMap, value::Number)
    for (c,b) in blocks(t)
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
    (codomain(t1)==codomain(t2) && domain(t1) == domain(t2)) || throw(SpaceMismatch())
    for c in blocksectors(t1)
        scale!(block(t1, c), block(t2, c), α)
    end
    return t1
end

function Base.LinAlg.axpy!(α::Number, t1::TensorMap, t2::TensorMap)
    (codomain(t1)==codomain(t2) && domain(t1) == domain(t2)) || throw(SpaceMisMatch())
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

Base.vecnorm(t::TensorMap{<:EuclideanSpace}, p::Real) = _vecnorm(blocks(t), p)
function _vecnorm(blockiterator, p::Real)
    if p == Inf
        maximum(vecnorm(b, p) for (c,b) in blockiterator)
    elseif p == -Inf
        minimum(vecnorm(b, p) for (c,b) in blockiterator)
    elseif p == 1
        sum(dim(c)*vecnorm(b, p) for (c,b) in blockiterator)
    else
        s = sum(dim(c)*vecnorm(b, p)^p for (c,b) in blockiterator)
        return exp(log(s)/p) # (s^(1/p) is promoting Float32 to Float64)
    end
end

# TensorMap multiplication:
#--------------------------
function Base.A_mul_B!(tC::TensorMap, tA::TensorMap,  tB::TensorMap)
    if !(codomain(tC) == codomain(tA) && domain(tC) == domain(tB) && domain(tA) == codomain(tB))
        @show codomain(tA), domain(tA)
        @show codomain(tB), domain(tB)
        @show codomain(tC), domain(tC)
        throw(SpaceMismatch())
    end
    for c in blocksectors(tC)
        A_mul_B!(block(tC, c), block(tA, c), block(tB, c))
    end
    return tC
end

# Orthogonal factorizations: only correct if Euclidean inner product
#--------------------------------------------------------------------
function leftorth!(t::TensorMap{S}, alg::OrthogonalFactorizationAlgorithm = QRpos()) where {S<:EuclideanSpace}
    if isa(t.data, AbstractArray)
        Q, R = leftorth!(t.data, alg)
        V = S(size(Q,2))
        return TensorMap(Q, codomain(t)←V), TensorMap(R, V←domain(t))
    else
        Qdata = empty(t.data)
        Rdata = empty(t.data)
        dims = Dict{sectortype(t), Int}()
        for c in blocksectors(t)
            Q, R = leftorth!(block(t,c), alg)
            Qdata[c] = Q
            Rdata[c] = R
            dims[c] = size(Q, 2)
        end
        V = S(dims)
        return TensorMap(Qdata, codomain(t)←V), TensorMap(Rdata, V←domain(t))
    end
end
function leftnull!(t::TensorMap{S}, alg::OrthogonalFactorizationAlgorithm = QRpos()) where {S<:EuclideanSpace}
    if isa(t.data, AbstractArray)
        N = leftnull!(t.data, alg)
        W = S(size(N, 2))
        return TensorMap(N, codomain(t)←W)
    else
        V = codomain(t)
        Ndata = empty(t.data)
        dims = Dict{sectortype(t), Int}()
        for c in blocksectors(V)
            N = leftnull!(block(t,c), alg)
            Ndata[c] = N
            dims[c] = size(N,2)
        end
        W = S(dims)
        return TensorMap(Ndata, V←W)
    end
end
function rightorth!(t::TensorMap{S}, alg::OrthogonalFactorizationAlgorithm = LQpos()) where {S<:EuclideanSpace}
    if isa(t.data, AbstractArray)
        L, Q = rightorth!(t.data, alg)
        V = S(size(Q, 1))
        return TensorMap(L, codomain(t)←V), TensorMap(Q, V←domain(t))
    else
        Ldata = empty(t.data)
        Qdata = empty(t.data)
        dims = Dict{sectortype(t), Int}()
        for c in blocksectors(t)
            L, Q = rightorth!(block(t,c), alg)
            Ldata[c] = L
            Qdata[c] = Q
            dims[c] = size(Q, 1)
        end
        V = S(dims)
        return TensorMap(Ldata, codomain(t)←V), TensorMap(Qdata, V←domain(t))
    end
end
function rightnull!(t::TensorMap{S}, alg::OrthogonalFactorizationAlgorithm = LQpos()) where {S<:EuclideanSpace}
    if isa(t.data, AbstractArray)
        N = rightnull!(t.data, alg)
        W = S(size(N, 1))
        return TensorMap(N, W←domain(t))
    else
        V = domain(t)
        Ndata = empty(t.data)
        A = valtype(Ndata)
        dims = Dict{sectortype(t), Int}()
        for c in blocksectors(V)
            N = rightnull!(block(t,c), alg)
            Ndata[c] = N
            dims[c] = size(N,1)
        end
        W = S(dims)
        return TensorMap(Ndata, W←V)
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
        return TensorMap(U, codomain(t)←W), TensorMap(Matrix(Diagonal(Σ)), W←W), TensorMap(V, W←domain(t)), truncerr
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
        return TensorMap(Udata, codomain(t)←W), TensorMap(Dict(c=>Matrix(Diagonal(Σ)) for (c,Σ) in Σdata), W←W), TensorMap(Vdata, W←domain(t)), truncerr
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
    it = keys(V)
    fullnorm = _vecnorm((c=>V[c] for c in it), p)
    truncerr = zero(fullnorm)
    if isa(trunc, NoTruncation)
        # don't do anything
    elseif isa(trunc, TruncationError)
        truncdim = Dict{G,Int}(c=>length(v) for (c,v) in V)
        while true
            cmin = mininum(c->sqrt(dim(c))*V[c][truncdim[c]], keys(V))
            truncdim[cmin] -= 1
            truncerr = _vecnorm((c=>view(Σdata[c],truncdim[c]+1:maxdim[c]) for c in it), p)
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

# Adjoint (complex/Hermitian conjugation)
#-----------------------------------------
function adjoint!(tdst::TensorMap{<:EuclideanSpace}, tsrc::TensorMap{<:EuclideanSpace})
    (codomain(tdst) == domain(tsrc) && domain(tdst) == codomain(tsrc)) || throw(SpaceMismatch())

    for c in blocksectors(t)
        adjoint!(block(tdst, c), block(tsrc, c))
    end
    return tdst
end
