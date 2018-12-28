# Basic algebra
#---------------
Base.copy(t::AbstractTensorMap) = Base.copyto!(similar(t), t)

Base.:-(t::AbstractTensorMap) = mul!(similar(t), t, -one(eltype(t)))
function Base.:+(t1::AbstractTensorMap, t2::AbstractTensorMap)
    T = promote_type(eltype(t1), eltype(t2))
    return axpy!(one(T), t2, copyto!(similar(t1, T), t1))
end
function Base.:-(t1::AbstractTensorMap, t2::AbstractTensorMap)
    T = promote_type(eltype(t1), eltype(t2))
    return axpy!(-one(T), t2, copyto!(similar(t1, T), t1))
end

Base.:*(t::AbstractTensorMap, α::Number) = mul!(similar(t, promote_type(eltype(t), typeof(α))), t, α)
Base.:*(α::Number, t::AbstractTensorMap) = mul!(similar(t, promote_type(eltype(t), typeof(α))), α, t)
Base.:/(t::AbstractTensorMap, α::Number) = *(t, one(α)/α)
Base.:\(α::Number, t::AbstractTensorMap) = *(t, one(α)/α)

LinearAlgebra.normalize!(t::AbstractTensorMap, p::Real = 2) = rmul!(t, inv(norm(t, p)))
LinearAlgebra.normalize(t::AbstractTensorMap, p::Real = 2) = mul!(similar(t), t, inv(norm(t, p)))

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
        copyto!(b, I)
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
    d = norm(t1 - t2)
    if isfinite(d)
        return d <= max(atol, rtol*max(norm(t1), norm(t2)))
    else
        return false
    end
end

# In-place methods
#------------------
# Wrapping the blocks in a StridedView enables multithreading if JULIA_NUM_THREADS > 1
# Copy, adjoint! and fill:
function Base.copyto!(tdest::AbstractTensorMap, tsource::AbstractTensorMap)
    codomain(tdest) == codomain(tsource) && domain(tdest) == domain(tsource) || throw(SpaceMismatch())
    for c in blocksectors(tdest)
        copyto!(StridedView(block(tdest, c)), StridedView(block(tsource, c)))
    end
    return tdest
end
function Base.fill!(t::AbstractTensorMap, value::Number)
    for (c,b) in blocks(t)
        fill!(b, value)
    end
    return t
end
function LinearAlgebra.adjoint!(tdest::AbstractTensorMap, tsource::AbstractTensorMap)
    codomain(tdest) == domain(tsource) && domain(tdest) == codomain(tsource) || throw(SpaceMismatch())
    for c in blocksectors(tdest)
        adjoint!(StridedView(block(tdest, c)), StridedView(block(tsource, c)))
    end
    return tdest
end

# Basic vector space methods: addition and scalar multiplication
LinearAlgebra.rmul!(t::AbstractTensorMap, α::Number) = mul!(t, t, α)
LinearAlgebra.lmul!(α::Number, t::AbstractTensorMap) = mul!(t, α, t)

function LinearAlgebra.mul!(t1::AbstractTensorMap, t2::AbstractTensorMap, α::Number)
    (codomain(t1)==codomain(t2) && domain(t1) == domain(t2)) || throw(SpaceMismatch())
    for c in blocksectors(t1)
        mul!(StridedView(block(t1, c)), StridedView(block(t2, c)), α)
    end
    return t1
end
function LinearAlgebra.mul!(t1::AbstractTensorMap, α::Number, t2::AbstractTensorMap)
    (codomain(t1)==codomain(t2) && domain(t1) == domain(t2)) || throw(SpaceMismatch())
    for c in blocksectors(t1)
        mul!(StridedView(block(t1, c)), α, StridedView(block(t2, c)))
    end
    return t1
end
function LinearAlgebra.axpy!(α::Number, t1::AbstractTensorMap, t2::AbstractTensorMap)
    (codomain(t1)==codomain(t2) && domain(t1) == domain(t2)) || throw(SpaceMismatch())
    for c in blocksectors(t1)
        axpy!(α, StridedView(block(t1, c)), StridedView(block(t2, c)))
    end
    return t2
end
function LinearAlgebra.axpby!(α::Number, t1::AbstractTensorMap, β::Number, t2::AbstractTensorMap)
    (codomain(t1)==codomain(t2) && domain(t1) == domain(t2)) || throw(SpaceMismatch())
    for c in blocksectors(t1)
        axpby!(α, StridedView(block(t1, c)), β, StridedView(block(t2, c)))
    end
    return t2
end

# inner product and norm only valid for spaces with Euclidean inner product
function LinearAlgebra.dot(t1::AbstractTensorMap{S}, t2::AbstractTensorMap{S}) where {S<:EuclideanSpace}
    (codomain(t1) == codomain(t2) && domain(t1) == domain(t2)) || throw(SpaceMismatch())
    return sum(dim(c)*dot(block(t1,c), block(t2,c)) for c in blocksectors(t1))
end

LinearAlgebra.norm(t::AbstractTensorMap{<:EuclideanSpace}, p::Real) = _norm(blocks(t), p)
function _norm(blockiterator, p::Real)
    if p == Inf
        maximum(norm(b, p) for (c,b) in blockiterator)
    elseif p == -Inf
        minimum(norm(b, p) for (c,b) in blockiterator)
    elseif p == 1
        sum(dim(c)*norm(b, p) for (c,b) in blockiterator)
    else
        s = sum(dim(c)*norm(b, p)^p for (c,b) in blockiterator)
        return exp(log(s)/p) # (s^(1/p) is promoting Float32 to Float64)
    end
end

# TensorMap multiplication:
function LinearAlgebra.mul!(tC::AbstractTensorMap, tA::AbstractTensorMap,  tB::AbstractTensorMap, α = true, β = false)
    if !(codomain(tC) == codomain(tA) && domain(tC) == domain(tB) && domain(tA) == codomain(tB))
        throw(SpaceMismatch())
    end
    for c in blocksectors(tC)
        if hasblock(tA, c) # then also tB should have such a block
            A = block(tA, c)
            B = block(tB, c)
            C = block(tC, c)
            if isbitstype(eltype(A)) && isbitstype(eltype(B)) && isbitstype(eltype(C))
                @unsafe_strided A B C mul!(C, A, B, α, β)
            else
                mul!(StridedView(C), StridedView(A), StridedView(B), α, β)
            end
        elseif β != one(β)
            rmul!(block(tC, c), β)
        end
    end
    return tC
end

# TensorMap exponentation:
function exp!(t::TensorMap)
    domain(t) == codomain(t) || error("Exponentional of a tensor only exist when domain == codomain.")
    for (c,b) in blocks(t)
        copyto!(b, exp!(b))
    end
    return t
end

# hcat and vcat of tensors
function Base.hcat(t1::AbstractTensorMap{S,N₁,1}, t2::AbstractTensorMap{S,N₁,1}) where {S,N₁}
    codomain(t1) == codomain(t2) || throw(SpaceMismatch())

    V1, = domain(t1)
    V2, = domain(t2)
    isdual(V1) == isdual(V2) || throw(SpaceMismatch("cannot hcat tensors whose domain has non-matching duality"))

    V = V1 ⊕ V2
    t = TensorMap(undef, promote_type(eltype(t1),eltype(t2)), codomain(t1), V)
    for c in sectors(V)
        block(t,c)[:,1:dim(V1, c)] .= block(t1,c)
        block(t,c)[:,dim(V1,c) .+ (1:dim(V2, c))] .= block(t2,c)
    end
    return t
end
function Base.vcat(t1::AbstractTensorMap{S,1,N₂}, t2::AbstractTensorMap{S,1,N₂}) where {S,N₂}
    domain(t1) == domain(t2) || throw(SpaceMismatch())

    V1, = codomain(t1)
    V2, = codomain(t2)
    isdual(V1) == isdual(V2) || throw(SpaceMismatch("cannot hcat tensors whose domain has non-matching duality"))

    V = V1 ⊕ V2
    t = TensorMap(undef, promote_type(eltype(t1),eltype(t2)), V, domain(t1))
    for c in sectors(V)
        block(t,c)[1:dim(V1, c),:] .= block(t1,c)
        block(t,c)[dim(V1,c) .+ (1:dim(V2, c)),:] .= block(t2,c)
    end
    return t
end
