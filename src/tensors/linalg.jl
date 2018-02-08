# Basic algebra
#---------------
Base.copy(t::AbstractTensorMap) = Base.copy!(similar(t), t)

Base.:-(t::AbstractTensorMap) = mul!(similar(t), t, -one(eltype(t)))
function Base.:+(t1::AbstractTensorMap, t2::AbstractTensorMap)
    T = promote_type(eltype(t1), eltype(t2))
    return axpy!(one(T), t2, copy!(similar(t1, T), t1))
end
function Base.:-(t1::AbstractTensorMap, t2::AbstractTensorMap)
    T = promote_type(eltype(t1), eltype(t2))
    return axpy!(-one(T), t2, copy!(similar(t1, T), t1))
end

Base.:*(t::AbstractTensorMap, α::Number) = mul!(similar(t, promote_type(eltype(t), typeof(α))), t, α)
Base.:*(α::Number, t::AbstractTensorMap) = mul!(similar(t, promote_type(eltype(t), typeof(α))), α, t)
Base.:/(t::AbstractTensorMap, α::Number) = *(t, one(α)/α)
Base.:\(α::Number, t::AbstractTensorMap) = *(t, one(α)/α)

@static if VERSION < v"v0.7-"
    scale!(t::AbstractTensorMap, α::Number) = mul!(t, t, α)
    scale!(α::Number, t::AbstractTensorMap) = mul!(t, α, t)
    scale!(tdest::AbstractTensorMap, α::Number, tsrc::AbstractTensorMap) = mul!(tdest, α, tsrc)
    scale!(tdest::AbstractTensorMap, tsrc::AbstractTensorMap, α::Number) = mul!(tdest, tsrc, α)
end

normalize!(t::AbstractTensorMap, p::Real = 2) = mul!(t, t, inv(vecnorm(t, p)))
normalize(t::AbstractTensorMap, p::Real = 2) = mul!(similar(t), t, inv(vecnorm(t, p)))

Base.:*(t1::AbstractTensorMap, t2::AbstractTensorMap) = mul!(similar(t1, promote_type(eltype(t1),eltype(t2)), codomain(t1)←domain(t2)), t1, t2)
Base.exp(t::AbstractTensorMap) = exp!(copy(t))

# Special purpose constructors
#------------------------------
Base.zero(t::AbstractTensorMap) = fill!(similar(t), 0)
function Base.one(t::AbstractTensorMap)
    domain(t) == codomain(t) || throw(SectorMismatch("no identity if domain and codomain are different"))
    TensorMap(I, eltype(t), domain(t), domain(t))
end
function one!(t::AbstractTensorMap)
    domain(t) == codomain(t) || throw(SectorMismatch("no identity if domain and codomain are different"))
    for (c,b) in blocks(t)
        copy!(b, I)
    end
    return t
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

function Base.isapprox(t1::AbstractTensorMap, t2::AbstractTensorMap; atol::Real=0, rtol::Real=Base.rtoldefault(eltype(t1), eltype(t2), atol))
    d = vecnorm(t1 - t2)
    if isfinite(d)
        return d <= max(atol, rtol*max(vecnorm(t1), vecnorm(t2)))
    else
        return false
    end
end

# In-place methods
#------------------
# Copy, adjoint! and fill
function Base.copy!(tdest::AbstractTensorMap, tsource::AbstractTensorMap)
    codomain(tdest) == codomain(tsource) && domain(tdest) == domain(tsource) || throw(SpaceMismatch())
    for c in blocksectors(tdest)
        copy!(block(tdest, c), block(tsource, c))
    end
    return tdest
end
function adjoint!(tdest::AbstractTensorMap, tsource::AbstractTensorMap)
    codomain(tdest) == domain(tsource) && domain(tdest) == codomain(tsource) || throw(SpaceMismatch())
    for c in blocksectors(tdest)
        adjoint!(block(tdest, c), block(tsource, c))
    end
    return tdest
end
function Base.fill!(t::AbstractTensorMap, value::Number)
    for (c,b) in blocks(t)
        fill!(b, value)
    end
    return t
end

# basic vector space methods: addition and scalar multiplication
function mul!(t1::AbstractTensorMap, t2::AbstractTensorMap, α::Number)
    (codomain(t1)==codomain(t2) && domain(t1) == domain(t2)) || throw(SpaceMismatch())
    for c in blocksectors(t1)
        mul!(block(t1, c), block(t2, c), α)
    end
    return t1
end
function mul!(t1::AbstractTensorMap, α::Number, t2::AbstractTensorMap)
    (codomain(t1)==codomain(t2) && domain(t1) == domain(t2)) || throw(SpaceMismatch())
    for c in blocksectors(t1)
        mul!(block(t1, c), α, block(t2, c))
    end
    return t1
end
function axpy!(α::Number, t1::AbstractTensorMap, t2::AbstractTensorMap)
    (codomain(t1)==codomain(t2) && domain(t1) == domain(t2)) || throw(SpaceMisMatch())
    for c in blocksectors(t1)
        axpy!(α, block(t1, c), block(t2, c))
    end
    return t2
end

# inner product and norm only valid for spaces with Euclidean inner product
function vecdot(t1::AbstractTensorMap{S}, t2::AbstractTensorMap{S}) where {S<:EuclideanSpace}
    (codomain(t1) == codomain(t2) && domain(t1) == domain(t2)) || throw(SpaceMismatch())
    return sum(dim(c)*vecdot(block(t1,c), block(t2,c)) for c in blocksectors(t1))
end

vecnorm(t::AbstractTensorMap{<:EuclideanSpace}, p::Real) = _vecnorm(blocks(t), p)
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
function mul!(tC::AbstractTensorMap, tA::AbstractTensorMap,  tB::AbstractTensorMap)
    if !(codomain(tC) == codomain(tA) && domain(tC) == domain(tB) && domain(tA) == codomain(tB))
        throw(SpaceMismatch())
    end
    for c in blocksectors(tC)
        if hasblock(tA, c) # then also tB should have such a block
            mul!(block(tC, c), block(tA, c), block(tB, c))
        else
            fill!(block(tC, c), 0)
        end
    end
    return tC
end

# TensorMap exponentation:
function exp!(t::TensorMap)
    domain(t) == codomain(t) || error("Exponentional of a tensor only exist when domain == codomain.")
    for (c,b) in blocks(t)
        copyto!(b,exp!(b))
    end
    return t
end
