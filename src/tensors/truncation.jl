# truncation.jl
#
# Implements truncation schemes for truncating a tensor with svd, leftorth or rightorth

abstract type TruncationScheme end

struct NoTruncation <: TruncationScheme
end
notrunc() = NoTruncation()

struct TruncationError{T<:Real} <: TruncationScheme
    ϵ::T
end
truncerr(epsilon::Real) = TruncationError(epsilon)

struct TruncationDimension <: TruncationScheme
    dim::Int
end
truncdim(d::Int) = TruncationDimension(d)

struct TruncationSpace{S<:ElementarySpace} <: TruncationScheme
    space::S
end
truncspace(space::ElementarySpace) = TruncationSpace(space)

struct TruncateBelow{T<:Real} <: TruncationScheme
    ϵ::T
end
truncbelow(epsilon::Real) = TruncateBelow(epsilon)

# For a single vector
function _truncate!(v::AbstractVector, ::NoTruncation, p::Real = 2)
    return v, zero(norm(v, p))
end

function _truncate!(v::AbstractVector, trunc::TruncationError, p::Real = 2)
    truncerr = abs(zero(eltype(v)))
    dmax = length(v)
    dtrunc = dmax
    while true
        dtrunc -= 1
        truncerr = norm(view(v, dtrunc+1:dmax), p)
        if truncerr > trunc.ϵ
            dtrunc += 1
            break
        end
    end
    truncerr = norm(view(v, dtrunc+1:dmax), p)
    resize!(v, dtrunc)
    return v, truncerr
end

function _truncate!(v::AbstractVector, trunc::TruncationDimension, p::Real = 2)
    dtrunc = min(length(v), trunc.dim)
    truncerr = norm(view(v, dtrunc+1:length(v)), p)
    resize!(v, dtrunc)
    return v, truncerr
end

_truncate!(v::AbstractVector, trunc::TruncationSpace, p::Real = 2) = _truncate!(v, dim(trunc.space), p)

function _truncate!(v::AbstractVector, trunc::TruncateBelow, p::Real = 2)
    dtrunc   = findlast(x->(x>trunc.ϵ), v)
    truncerr = norm(view(v, dtrunc+1:length(v)), p)
    resize!(v, dtrunc)
    return v, truncerr
end

# For SectorDict
function _truncate!(V::SectorDict{G,<:AbstractVector}, ::NoTruncation, p = 2) where {G<:Sector}
    return V, zero(_norm(V, p))
end
function _truncate!(V::SectorDict{G,<:AbstractVector}, trunc::TruncationError, p = 2) where {G<:Sector}
    truncdim = SectorDict{G,Int}(c=>length(v) for (c,v) in V)
    maxdim = copy(truncdim)
    it = keys(V)
    while true
        s = start(it)
        c, s = next(it, s)
        while truncdim[c] == 0
            c, s = next(it, s)
        end
        cmin = c
        vmin = dim(c)^(1/p)*V[c][truncdim[c]]
        while !done(it, s)
            c, s = next(it, s)
            if truncdim[c] > 0
                v = dim(c)^(1/p)*V[c][truncdim[c]]
                if v < vmin
                    cmin, vmin = c, v
                end
            end
        end
        truncdim[cmin] -= 1
        truncerr = _norm((c=>view(v,truncdim[c]+1:length(v)) for (c,v) in V), p)
        if truncerr > trunc.ϵ
            truncdim[cmin] += 1
            break
        end
    end
    truncerr = _norm((c=>view(v,truncdim[c]+1:length(v)) for (c,v) in V), p)
    for c in it
        resize!(V[c], truncdim[c])
    end
    return V, truncerr
end
function _truncate!(V::SectorDict{G,<:AbstractVector}, trunc::TruncationDimension, p = 2) where {G<:Sector}
    truncdim = SectorDict{G,Int}(c=>length(v) for (c,v) in V)
    it = keys(V)
    while sum(c->dim(c)*truncdim[c], it) > trunc.dim
        s = start(it)
        c, s = next(it, s)
        while truncdim[c] == 0
            c, s = next(it, s)
        end
        cmin = c
        vmin = dim(c)^(1/p)*V[c][truncdim[c]]
        while !done(it, s)
            c, s = next(it, s)
            if truncdim[c] > 0
                v = dim(c)^(1/p)*V[c][truncdim[c]]
                if v < vmin
                    cmin, vmin = c, v
                end
            end
        end
        truncdim[cmin] -= 1
    end
    truncerr = _norm((c=>view(v,truncdim[c]+1:length(v)) for (c,v) in V), p)
    for c in it
        resize!(V[c], truncdim[c])
    end
    return V, truncerr
end
function _truncate!(V::SectorDict{G,<:AbstractVector}, trunc::TruncationSpace, p = 2) where {G<:Sector}
    truncdim = SectorDict{G,Int}(c=>min(lenth(v), dim(trunc.space,c)) for (c,v) in V)
    truncerr = _norm((c=>view(v,truncdim[c]+1:length(v)) for (c,v) in V), p)
    for c in keys(V)
        resize!(V[c], truncdim[c])
    end
    return V, truncerr
end
function _truncate!(V::SectorDict{G,<:AbstractVector}, trunc::TruncateBelow, p = 2) where {G<:Sector}
    truncdim = SectorDict{G,Int}(c=>length(v) for (c,v) in V)
    it = keys(V)
    for c in it
        newdim = findlast(x->(x>trunc.ϵ), V[c] )
        if newdim == nothing
            truncdim[c] = 0
        else
            truncdim[c] = newdim
        end
    end
    truncerr = _norm((c=>view(v,truncdim[c]+1:length(v)) for (c,v) in V), p)
    for c in it
        resize!(V[c], truncdim[c])
    end
    return V, truncerr
end

# Combine truncations
struct MultipleTruncation{T<:Tuple{Vararg{<:TruncationScheme}}} <: TruncationScheme
    truncations::T
end
Base.:&(a::MultipleTruncation, b::MultipleTruncation) = MultipleTruncation((a.truncations...,b.truncations...))
Base.:&(a::MultipleTruncation, b::TruncationScheme) = MultipleTruncation((a.truncations...,b))
Base.:&(a::TruncationScheme, b::MultipleTruncation) = MultipleTruncation((a,b.truncations...))
Base.:&(a::TruncationScheme, b::TruncationScheme) = MultipleTruncation((a,b))

# TODO: this is not really correct because of relative error in TruncationError;
# maybe TruncationError should use absolute error as measure
function _truncate!(v, trunc::MultipleTruncation, p::Real = 2)
    v, truncerrs = __truncate!(v, trunc.truncations, p)
    return v, norm(truncerrs, p)
end
function __truncate!(v, trunc::Tuple{Vararg{<:TruncationScheme}}, p::Real = 2)
    v, truncerr1 = _truncate!(v, first(trunc), p)
    v, truncerrtail = __truncate!(v, tail(trunc), p)
    return v, (truncerr1, truncerrtail...)
end
function __truncate!(v, trunc::Tuple{<:TruncationScheme}, p::Real = 2)
    v, truncerr1 = _truncate!(v, first(trunc), p)
    return v, (truncerr1,)
end
