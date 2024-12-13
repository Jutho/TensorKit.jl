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

struct TruncationCutoff{T<:Real} <: TruncationScheme
    ϵ::T
    add_back::Int
end
truncbelow(epsilon::Real, add_back::Int=0) = TruncationCutoff(epsilon, add_back)

# Compute the total truncation error given truncation dimensions
function _compute_truncerr(Σdata, truncdim, p=2)
    I = keytype(Σdata)
    S = scalartype(valtype(Σdata))
    return _norm((c => view(v, (truncdim[c] + 1):length(v)) for (c, v) in Σdata), p,
                 zero(S))
end

# Compute truncation dimensions
function _compute_truncdim(Σdata, ::NoTruncation, p=2)
    I = keytype(Σdata)
    truncdim = SectorDict{I,Int}(c => length(v) for (c, v) in Σdata)
    return truncdim
end
function _compute_truncdim(Σdata, trunc::TruncationError, p=2)
    I = keytype(Σdata)
    S = real(eltype(valtype(Σdata)))
    truncdim = SectorDict{I,Int}(c => length(Σc) for (c, Σc) in Σdata)
    truncerr = zero(S)
    while true
        cmin = _findnexttruncvalue(Σdata, truncdim, p)
        isnothing(cmin) && break
        truncdim[cmin] -= 1
        truncerr = _compute_truncerr(Σdata, truncdim, p)
        if truncerr > trunc.ϵ
            truncdim[cmin] += 1
            break
        end
    end
    return truncdim
end
function _compute_truncdim(Σdata, trunc::TruncationDimension, p=2)
    I = keytype(Σdata)
    truncdim = SectorDict{I,Int}(c => length(v) for (c, v) in Σdata)
    while sum(dim(c) * d for (c, d) in truncdim) > trunc.dim
        cmin = _findnexttruncvalue(Σdata, truncdim, p)
        isnothing(cmin) && break
        truncdim[cmin] -= 1
    end
    return truncdim
end
function _compute_truncdim(Σdata, trunc::TruncationSpace, p=2)
    I = keytype(Σdata)
    truncdim = SectorDict{I,Int}(c => min(length(v), dim(trunc.space, c))
                                 for (c, v) in Σdata)
    return truncdim
end

function _compute_truncdim(Σdata, trunc::TruncationCutoff, p=2)
    I = keytype(Σdata)
    truncdim = SectorDict{I,Int}(c => length(v) for (c, v) in Σdata)
    for (c, v) in Σdata
        newdim = findlast(Base.Fix2(>, trunc.ϵ), v)
        if newdim === nothing
            truncdim[c] = 0
        else
            truncdim[c] = newdim
        end
    end
    for i in 1:(trunc.add_back)
        cmax = _findnextgrowvalue(Σdata, truncdim, p)
        isnothing(cmax) && break
        truncdim[cmax] += 1
    end
    return truncdim
end

# Combine truncations
struct MultipleTruncation{T<:Tuple{Vararg{TruncationScheme}}} <: TruncationScheme
    truncations::T
end
function Base.:&(a::MultipleTruncation, b::MultipleTruncation)
    return MultipleTruncation((a.truncations..., b.truncations...))
end
function Base.:&(a::MultipleTruncation, b::TruncationScheme)
    return MultipleTruncation((a.truncations..., b))
end
function Base.:&(a::TruncationScheme, b::MultipleTruncation)
    return MultipleTruncation((a, b.truncations...))
end
Base.:&(a::TruncationScheme, b::TruncationScheme) = MultipleTruncation((a, b))

function _compute_truncdim(Σdata, trunc::MultipleTruncation, p::Real=2)
    truncdim = _compute_truncdim(Σdata, trunc.truncations[1], p)
    for k in 2:length(trunc.truncations)
        truncdimₖ = _compute_truncdim(Σdata, trunc.truncations[k], p)
        for (c, d) in truncdim
            truncdim[c] = min(d, truncdimₖ[c])
        end
    end
    return truncdim
end

# auxiliary function
function _findnexttruncvalue(Σdata, truncdim::SectorDict{I,Int}, p::Real) where {I<:Sector}
    # early return
    (isempty(Σdata) || all(iszero, values(truncdim))) && return nothing

    # find some suitable starting candidate
    cmin = findfirst(>(0), truncdim)
    σmin = dim(cmin)^inv(p) * Σdata[cmin][truncdim[cmin]]

    # find the actual minimum singular value
    for (c, σs) in Σdata
        if truncdim[c] > 0
            σ = dim(c)^inv(p) * σs[truncdim[c]]
            if σ < σmin
                cmin, σmin = c, σ
            end
        end
    end
    return cmin
end
function _findnextgrowvalue(Σdata, truncdim::SectorDict{I,Int}, p::Real) where {I<:Sector}
    istruncated = SectorDict{I,Bool}(c => (d < length(Σdata[c])) for (c, d) in truncdim)
    # early return
    (isempty(Σdata) || !any(values(istruncated))) && return nothing

    # find some suitable starting candidate
    cmax = findfirst(istruncated)
    σmax = Σdata[cmax][truncdim[cmax] + 1]

    # find the actual maximal singular value
    for (c, σs) in Σdata
        if istruncated[c]
            σ = σs[truncdim[c] + 1]
            if σ > σmax
                cmax, σmax = c, σ
            end
        end
    end
    return cmax
end
