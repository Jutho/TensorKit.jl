# DiagonalTensorMap
#==========================================================#
struct DiagonalTensorMap{T,S<:IndexSpace,A<:DenseVector{T}} <: AbstractTensorMap{T,S,1,1}
    data::A
    domain::S # equals codomain

    # uninitialized constructors
    function DiagonalTensorMap{T,S,A}(::UndefInitializer,
                                      dom::S) where {T,S<:IndexSpace,A<:DenseVector{T}}
        data = A(undef, reduceddim(dom))
        if !isbitstype(T)
            zerovector!(data)
        end
        return DiagonalTensorMap{T,S,A}(data, dom)
    end
    # constructors from data
    function DiagonalTensorMap{T,S,A}(data::A,
                                      dom::S) where {T,S<:IndexSpace,A<:DenseVector{T}}
        T ⊆ field(S) || @warn("scalartype(data) = $T ⊈ $(field(S)))", maxlog = 1)
        return new{T,S,A}(data, dom)
    end
end

# Basic methods for characterising a tensor:
#--------------------------------------------
space(d::DiagonalTensorMap) = d.domain ← d.domain

storagetype(::Type{<:DiagonalTensorMap{T,S,A}}) where {T,S,A<:DenseVector{T}} = A

# DiagonalTensorMap constructors
#--------------------------------
# undef constructors
"""
    DiagonalTensorMap{T}(undef, domain::S) where {T,S<:IndexSpace}
    # expert mode: select storage type `A`
    DiagonalTensorMap{T,S,A}(undef, domain::S) where {T,S<:IndexSpace,A<:DenseVector{T}}

Construct a `DiagonalTensorMap` with uninitialized data.
"""
function DiagonalTensorMap{T}(::UndefInitializer, V::S) where {T,S<:IndexSpace}
    return DiagonalTensorMap{T,S,Vector{T}}(undef, V)
end
DiagonalTensorMap(::UndefInitializer, V::IndexSpace) = DiagonalTensorMap{Float64}(undef, V)

function DiagonalTensorMap{T}(data::A, V::S) where {T,S<:IndexSpace,A<:DenseVector{T}}
    length(data) == reduceddim(V) ||
        throw(DimensionMismatch("length(data) = $(length(data)) is not compatible with the space $V"))
    return DiagonalTensorMap{T,S,A}(data, V)
end

function DiagonalTensorMap(data::DenseVector{T}, V::IndexSpace) where {T}
    return DiagonalTensorMap{T}(data, V)
end

# TODO: more constructors needed?

# Special case adjoint:
#-----------------------
Base.adjoint(d::DiagonalTensorMap{<:Real}) = d
Base.adjoint(d::DiagonalTensorMap{<:Complex}) = DiagonalTensorMap(conj(d.data), d.domain)

# Efficient copy constructors and TensorMap converters
#-----------------------------------------------------
Base.copy(d::DiagonalTensorMap) = typeof(d)(copy(d.data), d.domain)

function Base.copy!(t::AbstractTensorMap, d::DiagonalTensorMap)
    space(t) == space(d) || throw(SpaceMismatch())
    for (c, b) in blocks(d)
        copy!(block(t, c), b)
    end
    return t
end
TensorMap(d::DiagonalTensorMap) = copy!(similar(d), d)
Base.convert(::Type{TensorMap}, d::DiagonalTensorMap) = TensorMap(d)

function Base.convert(::Type{DiagonalTensorMap{T,S,A}},
                      d::DiagonalTensorMap{T,S,A}) where {T,S,A}
    return d
end
function Base.convert(D::Type{<:DiagonalTensorMap}, d::DiagonalTensorMap)
    return DiagonalTensorMap(convert(storagetype(D), d.data), d.domain)
end

# Complex, real and imaginary parts
#-----------------------------------
for f in (:real, :imag, :complex)
    @eval begin
        function Base.$f(d::DiagonalTensorMap)
            return DiagonalTensorMap($f(d.data), d.domain)
        end
    end
end

# Getting and setting the data at the block level
#-------------------------------------------------
function block(d::DiagonalTensorMap, s::Sector)
    sectortype(d) == typeof(s) || throw(SectorMismatch())
    offset = 0
    dom = domain(d)[1]
    for c in blocksectors(d)
        if c < s
            offset += dim(dom, c)
        elseif c == s
            r = offset .+ (1:dim(dom, c))
            return Diagonal(view(d.data, r))
        else # s not in sectors(t)
            break
        end
    end
    return Diagonal(view(d.data, 1:0))
end

blocks(t::DiagonalTensorMap) = BlockIterator(t, diagonalblockstructure(space(t)))
blocktype(::Type{TT}) where {TT<:DiagonalTensorMap} = Diagonal{eltype(TT),storagetype(TT)}

function Base.iterate(iter::BlockIterator{<:DiagonalTensorMap}, state...)
    next = iterate(iter.structure, state...)
    isnothing(next) && return next
    (c, r), newstate = next
    return c => Diagonal(view(iter.t.data, r)), newstate
end

function Base.getindex(iter::BlockIterator{<:DiagonalTensorMap}, c::Sector)
    sectortype(iter.t) === typeof(c) || throw(SectorMismatch())
    r = get(iter.structure, c, 1:0)
    return Diagonal(view(iter.t.data, r))
end

# Indexing and getting and setting the data at the subblock level
#-----------------------------------------------------------------
@inline function Base.getindex(d::DiagonalTensorMap,
                               f₁::FusionTree{I,1},
                               f₂::FusionTree{I,1}) where {I<:Sector}
    s = f₁.uncoupled[1]
    s == f₁.coupled == f₂.uncoupled[1] == f₂.coupled || throw(SectorMismatch())
    return block(d, s)
    # TODO: do we want a StridedView here? Then we need to allocate a new matrix.
end

function Base.setindex!(d::DiagonalTensorMap,
                        v,
                        f₁::FusionTree{I,1},
                        f₂::FusionTree{I,1}) where {I<:Sector}
    return copy!(getindex(d, f₁, f₂), v)
end

function Base.getindex(d::DiagonalTensorMap)
    sectortype(d) === Trivial || throw(SectorMismatch())
    return Diagonal(d.data)
end

# Index manipulations
# -------------------
function has_shared_permute(d::DiagonalTensorMap, (p₁, p₂)::Index2Tuple)
    if p₁ === (1,) && p₂ === (2,)
        return true
    elseif p₁ === (2,) && p₂ === (1,) # transpose
        return sectortype(d) === Trivial
    else
        return false
    end
end

function permute(d::DiagonalTensorMap, (p₁, p₂)::Index2Tuple{1,1};
                 copy::Bool=false)
    if p₁ === (1,) && p₂ === (2,)
        return copy ? Base.copy(d) : d
    elseif p₁ === (2,) && p₂ === (1,) # transpose
        if has_shared_permute(d, (p₁, p₂)) # tranpose for bosonic sectors
            return DiagonalTensorMap(copy ? Base.copy(d.data) : d.data, dual(d.domain))
        end
        d′ = typeof(d)(undef, dual(d.domain))
        for (c, b) in blocks(d)
            f = only(fusiontrees(codomain(d), c))
            ((f′, _), coeff) = only(permute(f, f, p₁, p₂))
            c′ = f′.coupled
            scale!(block(d′, c′), b, coeff)
        end
        return d′
    else
        throw(ArgumentError("invalid permutation $((p₁, p₂)) for tensor in space $(space(d))"))
    end
end

# VectorInterface
# ---------------
function VectorInterface.zerovector(d::DiagonalTensorMap, ::Type{S}) where {S<:Number}
    return DiagonalTensorMap(zerovector(d.data, S), d.domain)
end
function VectorInterface.add(ty::DiagonalTensorMap, tx::DiagonalTensorMap,
                             α::Number, β::Number)
    domain(ty) == domain(tx) || throw(SpaceMismatch("$(space(ty)) ≠ $(space(tx))"))
    T = VectorInterface.promote_add(ty, tx, α, β)
    return add!(scale!(zerovector(ty, T), ty, β), tx, α) # zerovector instead of similar preserves diagonal structure
end

# TensorOperations
# ----------------
function TO.tensoradd_type(TC, A::DiagonalTensorMap, ::Index2Tuple{1,1}, ::Bool)
    M = similarstoragetype(A, TC)
    return DiagonalTensorMap{TC,spacetype(A),M}
end

function TO.tensorcontract_type(TC, A::DiagonalTensorMap, ::Index2Tuple{1,1}, ::Bool,
                                B::DiagonalTensorMap, ::Index2Tuple{1,1}, ::Bool,
                                ::Index2Tuple{1,1})
    M = similarstoragetype(A, TC)
    M == similarstoragetype(B, TC) ||
        throw(ArgumentError("incompatible storage types:\n$(M) ≠ $(similarstoragetype(B, TC))"))
    spacetype(A) == spacetype(B) || throw(SpaceMismatch("incompatible space types"))
    return DiagonalTensorMap{TC,spacetype(A),M}
end

function TO.tensoralloc(::Type{DiagonalTensorMap{T,S,M}},
                        structure::TensorMapSpace{S,1,1},
                        istemp::Val,
                        allocator=TO.DefaultAllocator()) where {T,S,M}
    domain(structure) == codomain(structure) || throw(ArgumentError("domain ≠ codomain"))
    V = only(domain(structure))
    dim = reduceddim(V)
    data = TO.tensoralloc(M, dim, istemp, allocator)
    return DiagonalTensorMap{T,S,M}(data, V)
end

# Linear Algebra and factorizations
# ---------------------------------
function one!(d::DiagonalTensorMap)
    fill!(d.data, one(eltype(d.data)))
    return d
end
function Base.one(d::DiagonalTensorMap)
    return DiagonalTensorMap(one.(d.data), d.domain)
end
function Base.zero(d::DiagonalTensorMap)
    return DiagonalTensorMap(zero.(d.data), d.domain)
end

function LinearAlgebra.mul!(dC::DiagonalTensorMap,
                            dA::DiagonalTensorMap,
                            dB::DiagonalTensorMap,
                            α::Number, β::Number)
    dC.domain == dA.domain == dB.domain || throw(SpaceMismatch())
    mul!(Diagonal(dC.data), Diagonal(dA.data), Diagonal(dB.data), α, β)
    return dC
end

Base.inv(d::DiagonalTensorMap) = DiagonalTensorMap(inv.(d.data), d.domain)
function Base.:\(d1::DiagonalTensorMap, d2::DiagonalTensorMap)
    d1.domain == d2.domain || throw(SpaceMismatch())
    return DiagonalTensorMap(d1.data .\ d2.data, d1.domain)
end
function Base.:/(d1::DiagonalTensorMap, d2::DiagonalTensorMap)
    d1.domain == d2.domain || throw(SpaceMismatch())
    return DiagonalTensorMap(d1.data ./ d2.data, d1.domain)
end
function LinearAlgebra.pinv(d::DiagonalTensorMap; kwargs...)
    T = eltype(d.data)
    atol = get(kwargs, :atol, zero(real(T)))
    if iszero(atol)
        rtol = get(kwargs, :rtol, zero(real(T)))
    else
        rtol = sqrt(eps(real(float(oneunit(T))))) * length(d.data)
    end
    pdata = let tol = max(atol, rtol * maximum(abs, d.data))
        map(x -> abs(x) < tol ? zero(x) : pinv(x), d.data)
    end
    return DiagonalTensorMap(pdata, d.domain)
end
function LinearAlgebra.isposdef(d::DiagonalTensorMap)
    return all(isposdef, d.data)
end

function eig!(d::DiagonalTensorMap)
    return d, one(d)
end
function eigh!(d::DiagonalTensorMap{<:Real})
    return d, one(d)
end
function eigh!(d::DiagonalTensorMap{<:Complex})
    # TODO: should this test for hermiticity? `eigh!(::TensorMap)` also does not do this.
    return DiagonalTensorMap(real(d.data), d.domain), one(d)
end

function leftorth!(d::DiagonalTensorMap; alg=QR(), kwargs...)
    @assert alg isa Union{QR,QL}
    return one(d), d # TODO: this is only correct for `alg = QR()` or `alg = QL()`
end
function rightorth!(d::DiagonalTensorMap; alg=LQ(), kwargs...)
    @assert alg isa Union{LQ,RQ}
    return d, one(d) # TODO: this is only correct for `alg = LQ()` or `alg = RQ()`
end
# not much to do here:
leftnull!(d::DiagonalTensorMap; kwargs...) = leftnull!(TensorMap(d); kwargs...)
rightnull!(d::DiagonalTensorMap; kwargs...) = rightnull!(TensorMap(d); kwargs...)

function tsvd!(d::DiagonalTensorMap; trunc=NoTruncation(), p::Real=2, alg=SDD())
    return _tsvd!(d, alg, trunc, p)
end
# helper function
function _compute_svddata!(d::DiagonalTensorMap, alg::Union{SVD,SDD})
    InnerProductStyle(d) === EuclideanInnerProduct() || throw_invalid_innerproduct(:tsvd!)
    I = sectortype(d)
    dims = SectorDict{I,Int}()
    generator = Base.Iterators.map(blocks(d)) do (c, b)
        lb = length(b.diag)
        U = zerovector!(similar(b.diag, lb, lb))
        V = zerovector!(similar(b.diag, lb, lb))
        p = sortperm(b.diag; by=abs, rev=true)
        for (i, pi) in enumerate(p)
            U[pi, i] = MatrixAlgebra.safesign(b.diag[pi])
            V[i, pi] = 1
        end
        Σ = abs.(view(b.diag, p))
        dims[c] = lb
        return c => (U, Σ, V)
    end
    SVDdata = SectorDict(generator)
    return SVDdata, dims
end

# matrix functions
for f in
    (:exp, :cos, :sin, :tan, :cot, :cosh, :sinh, :tanh, :coth, :atan, :acot, :asinh, :sqrt,
     :log, :asin, :acos, :acosh, :atanh, :acoth)
    @eval Base.$f(d::DiagonalTensorMap) = DiagonalTensorMap($f.(d.data), d.domain)
end

# Show
#------
function Base.summary(io::IO, t::DiagonalTensorMap)
    return print(io, "DiagonalTensorMap(", space(t), ")")
end
function Base.show(io::IO, t::DiagonalTensorMap)
    summary(io, t)
    get(io, :compact, false) && return nothing
    println(io, ":")

    if sectortype(t) == Trivial
        Base.print_array(io, Diagonal(t.data))
        println(io)
    else
        for (c, b) in blocks(t)
            println(io, "* Data for sector ", c, ":")
            Base.print_array(io, b)
            println(io)
        end
    end
    return nothing
end
