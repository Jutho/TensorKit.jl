# Basic algebra
#---------------
Base.copy(t::AbstractTensorMap) = Base.copy!(similar(t), t)

Base.:-(t::AbstractTensorMap) = VectorInterface.scale(t, -one(scalartype(t)))

Base.:+(t1::AbstractTensorMap, t2::AbstractTensorMap) = VectorInterface.add(t1, t2)
function Base.:-(t1::AbstractTensorMap, t2::AbstractTensorMap)
    return VectorInterface.add(t1, t2, -one(scalartype(t1)))
end

Base.:*(t::AbstractTensorMap, α::Number) = VectorInterface.scale(t, α)
Base.:*(α::Number, t::AbstractTensorMap) = VectorInterface.scale(t, α)

Base.:/(t::AbstractTensorMap, α::Number) = *(t, one(scalartype(t)) / α)
Base.:\(α::Number, t::AbstractTensorMap) = *(t, one(scalartype(t)) / α)

LinearAlgebra.normalize!(t::AbstractTensorMap, p::Real=2) = scale!(t, inv(norm(t, p)))
LinearAlgebra.normalize(t::AbstractTensorMap, p::Real=2) = scale(t, inv(norm(t, p)))

"""
    compose(t1::AbstractTensorMap, t2::AbstractTensorMap) -> AbstractTensorMap

Return the `AbstractTensorMap` that implements the composition of the two tensor maps `t1`
and `t2`.
"""
function compose(t1::AbstractTensorMap, t2::AbstractTensorMap)
    return mul!(similar(t1, promote_type(scalartype(t1), scalartype(t2)),
                        compose(space(t1), space(t2))), t1, t2)
end
Base.:*(t1::AbstractTensorMap, t2::AbstractTensorMap) = compose(t1, t2)

Base.exp(t::AbstractTensorMap) = exp!(copy(t))
function Base.:^(t::AbstractTensorMap, p::Integer)
    return p < 0 ? Base.power_by_squaring(inv(t), -p) : Base.power_by_squaring(t, p)
end

# Special purpose constructors
#------------------------------
Base.zero(t::AbstractTensorMap) = VectorInterface.zerovector(t)
function Base.one(t::AbstractTensorMap)
    domain(t) == codomain(t) ||
        throw(SectorMismatch("no identity if domain and codomain are different"))
    return one!(similar(t))
end
function one!(t::AbstractTensorMap)
    domain(t) == codomain(t) ||
        throw(SectorMismatch("no identity if domain and codomain are different"))
    for (c, b) in blocks(t)
        MatrixAlgebra.one!(b)
    end
    return t
end

"""
    id([T::Type=Float64,] V::TensorSpace) -> TensorMap

Construct the identity endomorphism on space `V`, i.e. return a `t::TensorMap` with
`domain(t) == codomain(t) == V`, where either `scalartype(t) = T` if `T` is a `Number` type
or `storagetype(t) = T` if `T` is a `DenseMatrix` type.
"""
id(V::TensorSpace) = id(Float64, V)
function id(::Type{A}, V::TensorSpace{S}) where {A<:MatOrNumber,S}
    W = V ← V
    N = length(domain(W))
    t = tensormaptype(S, N, N, A)(undef, codomain(W), domain(W))
    return one!(t)
end

"""
    isomorphism([T::Type=Float64,] codomain::TensorSpace, domain::TensorSpace) -> TensorMap
    isomorphism([T::Type=Float64,] codomain ← domain) -> TensorMap
    isomorphism([T::Type=Float64,] domain → codomain) -> TensorMap

Construct a specific isomorphism between the codomain and the domain, i.e. return a
`t::TensorMap` where either `scalartype(t) = T` if `T` is a `Number` type or
`storagetype(t) = T` if `T` is a `DenseMatrix` type. If the spaces are not isomorphic, an
error will be thrown.

!!! note
    There is no canonical choice for a specific isomorphism, but the current choice is such
    that `isomorphism(cod, dom) == inv(isomorphism(dom, cod))`.

See also [`unitary`](@ref) when `InnerProductStyle(cod) === EuclideanProduct()`.
"""
function isomorphism(::Type{A}, V::TensorMapSpace{S,N₁,N₂}) where {A<:MatOrNumber,S,N₁,N₂}
    codomain(V) ≅ domain(V) ||
        throw(SpaceMismatch("codomain and domain are not isomorphic: $V"))
    t = tensormaptype(S, N₁, N₂, A)(undef, codomain(V), domain(V))
    for (_, b) in blocks(t)
        MatrixAlgebra.one!(b)
    end
    return t
end

"""
    unitary([T::Type=Float64,] codomain::TensorSpace, domain::TensorSpace) -> TensorMap
    unitary([T::Type=Float64,] codomain ← domain) -> TensorMap
    unitary([T::Type=Float64,] domain → codomain) -> TensorMap

Construct a specific unitary morphism between the codomain and the domain, i.e. return a
`t::TensorMap` where either `scalartype(t) = T` if `T` is a `Number` type or
`storagetype(t) = T` if `T` is a `DenseMatrix` type. If the spaces are not isomorphic, or
the spacetype does not have a Euclidean inner product, an error will be thrown.

!!! note
    There is no canonical choice for a specific unitary, but the current choice is such that
    `unitary(cod, dom) == inv(unitary(dom, cod)) = adjoint(unitary(dom, cod))`.

See also [`isomorphism`](@ref) and [`isometry`](@ref).
"""
function unitary(::Type{A}, V::TensorMapSpace{S,N₁,N₂}) where {A<:MatOrNumber,S,N₁,N₂}
    InnerProductStyle(S) === EuclideanProduct() || throw_invalid_innerproduct(:unitary)
    return isomorphism(A, V)
end

"""
    isometry([T::Type=Float64,] codomain::TensorSpace, domain::TensorSpace) -> TensorMap
    isometry([T::Type=Float64,] codomain ← domain) -> TensorMap
    isometry([T::Type=Float64,] domain → codomain) -> TensorMap

Construct a specific isometry between the codomain and the domain, i.e. return a
`t::TensorMap` where either `scalartype(t) = T` if `T` is a `Number` type or
`storagetype(t) = T` if `T` is a `DenseMatrix` type. The isometry `t` then satisfies
`t' * t = id(domain)` and `(t * t')^2 = t * t'`. If the spaces do not allow for such an 
isometric inclusion, an error will be thrown.

See also [`isomorphism`](@ref) and [`unitary`](@ref).
"""
function isometry(::Type{A}, V::TensorMapSpace{S,N₁,N₂}) where {A<:MatOrNumber,S,N₁,N₂}
    InnerProductStyle(S) === EuclideanProduct() || throw_invalid_innerproduct(:isometry)
    domain(V) ≾ codomain(V) ||
        throw(SpaceMismatch("$V does not allow for an isometric inclusion"))
    t = tensormaptype(S, N₁, N₂, A)(undef, codomain(V), domain(V))
    for (_, b) in blocks(t)
        MatrixAlgebra.one!(b)
    end
    return t
end

# expand methods with default arguments
for morphism in (:isomorphism, :unitary, :isometry)
    @eval begin
        $morphism(V::TensorMapSpace) = $morphism(Float64, V)
        $morphism(codomain::TensorSpace, domain::TensorSpace) = $morphism(codomain ← domain)
        function $morphism(::Type{T}, codomain::TensorSpace,
                           domain::TensorSpace) where {T<:MatOrNumber}
            return $morphism(T, codomain ← domain)
        end
        $morphism(t::AbstractTensorMap) = $morphism(storagetype(t), space(t))
    end
end

# In-place methods
#------------------
# Wrapping the blocks in a StridedView enables multithreading if JULIA_NUM_THREADS > 1
# TODO: reconsider this strategy, consider spawning different threads for different blocks

# Copy, adjoint! and fill:
function Base.copy!(tdst::AbstractTensorMap, tsrc::AbstractTensorMap)
    space(tdst) == space(tsrc) || throw(SpaceMismatch("$(space(tdst)) ≠ $(space(tsrc))"))
    for c in blocksectors(tdst)
        copy!(StridedView(block(tdst, c)), StridedView(block(tsrc, c)))
    end
    return tdst
end
function Base.fill!(t::AbstractTensorMap, value::Number)
    for (c, b) in blocks(t)
        fill!(b, value)
    end
    return t
end
function LinearAlgebra.adjoint!(tdst::AbstractTensorMap,
                                tsrc::AbstractTensorMap)
    InnerProductStyle(tdst) === EuclideanProduct() || throw_invalid_innerproduct(:adjoint!)
    space(tdst) == adjoint(space(tsrc)) ||
        throw(SpaceMismatch("$(space(tdst)) ≠ adjoint($(space(tsrc)))"))
    for c in blocksectors(tdst)
        adjoint!(StridedView(block(tdst, c)), StridedView(block(tsrc, c)))
    end
    return tdst
end

# Basic vector space methods: recycle VectorInterface implementation
function LinearAlgebra.rmul!(t::AbstractTensorMap, α::Number)
    return iszero(α) ? zerovector!(t) : scale!(t, α)
end
function LinearAlgebra.lmul!(α::Number, t::AbstractTensorMap)
    return iszero(α) ? zerovector!(t) : scale!(t, α)
end

function LinearAlgebra.mul!(t1::AbstractTensorMap, t2::AbstractTensorMap, α::Number)
    return scale!(t1, t2, α)
end
function LinearAlgebra.mul!(t1::AbstractTensorMap, α::Number, t2::AbstractTensorMap)
    return scale!(t1, t2, α)
end

# TODO: remove VectorInterface namespace when we renamed TensorKit.add!
function LinearAlgebra.axpy!(α::Number, t1::AbstractTensorMap, t2::AbstractTensorMap)
    return VectorInterface.add!(t2, t1, α)
end
function LinearAlgebra.axpby!(α::Number, t1::AbstractTensorMap, β::Number,
                              t2::AbstractTensorMap)
    return VectorInterface.add!(t2, t1, α, β)
end

# inner product and norm only valid for spaces with Euclidean inner product
LinearAlgebra.dot(t1::AbstractTensorMap, t2::AbstractTensorMap) = inner(t1, t2)

function LinearAlgebra.norm(t::AbstractTensorMap, p::Real=2)
    InnerProductStyle(t) === EuclideanProduct() || throw_invalid_innerproduct(:norm)
    return _norm(blocks(t), p, float(zero(real(scalartype(t)))))
end
function _norm(blockiter, p::Real, init::Real)
    if p == Inf
        return mapreduce(max, blockiter; init=init) do (c, b)
            return isempty(b) ? init : oftype(init, LinearAlgebra.normInf(b))
        end
    elseif p == 2
        return sqrt(mapreduce(+, blockiter; init=init) do (c, b)
                        return isempty(b) ? init :
                               oftype(init, dim(c) * LinearAlgebra.norm2(b)^2)
                    end)
    elseif p == 1
        return mapreduce(+, blockiter; init=init) do (c, b)
            return isempty(b) ? init : oftype(init, dim(c) * sum(abs, b))
        end
    elseif p > 0
        s = mapreduce(+, blockiter; init=init) do (c, b)
            return isempty(b) ? init : oftype(init, dim(c) * LinearAlgebra.normp(b, p)^p)
        end
        return s^inv(oftype(s, p))
    else
        msg = "Norm with non-positive p is not defined for `AbstractTensorMap`"
        throw(ArgumentError(msg))
    end
end

# TensorMap trace
function LinearAlgebra.tr(t::AbstractTensorMap)
    domain(t) == codomain(t) ||
        throw(SpaceMismatch("Trace of a tensor only exist when domain == codomain"))
    return sum(dim(c) * tr(b) for (c, b) in blocks(t))
end

# TensorMap multiplication
function LinearAlgebra.mul!(tC::AbstractTensorMap,
                            tA::AbstractTensorMap,
                            tB::AbstractTensorMap, α=true, β=false)
    compose(space(tA), space(tB)) == space(tC) ||
        throw(SpaceMismatch("$(space(tC)) ≠ $(space(tA)) * $(space(tB))"))

    for c in blocksectors(tC)
        if hasblock(tA, c) # then also tB should have such a block
            A = block(tA, c)
            B = block(tB, c)
            C = block(tC, c)
            mul!(StridedView(C), StridedView(A), StridedView(B), α, β)
        elseif β != one(β)
            rmul!(block(tC, c), β)
        end
    end
    return tC
end
# TODO: reconsider wrapping the blocks in a StridedView, consider spawning threads for different blocks

# TensorMap inverse
function Base.inv(t::AbstractTensorMap)
    cod = codomain(t)
    dom = domain(t)
    for c in union(blocksectors(cod), blocksectors(dom))
        blockdim(cod, c) == blockdim(dom, c) ||
            throw(SpaceMismatch("codomain $cod and domain $dom are not isomorphic: no inverse"))
    end
    if sectortype(t) === Trivial
        return TensorMap(inv(block(t, Trivial())), domain(t) ← codomain(t))
    else
        data = empty(t.data)
        for (c, b) in blocks(t)
            data[c] = inv(b)
        end
        return TensorMap(data, domain(t) ← codomain(t))
    end
end
function LinearAlgebra.pinv(t::AbstractTensorMap; kwargs...)
    if sectortype(t) === Trivial
        return TensorMap(pinv(block(t, Trivial()); kwargs...), domain(t) ← codomain(t))
    else
        data = empty(t.data)
        for (c, b) in blocks(t)
            data[c] = pinv(b; kwargs...)
        end
        return TensorMap(data, domain(t) ← codomain(t))
    end
end
function Base.:(\)(t1::AbstractTensorMap, t2::AbstractTensorMap)
    codomain(t1) == codomain(t2) ||
        throw(SpaceMismatch("non-matching codomains in t1 \\ t2"))
    if sectortype(t1) === Trivial
        data = block(t1, Trivial()) \ block(t2, Trivial())
        return TensorMap(data, domain(t1) ← domain(t2))
    else
        cod = codomain(t1)
        data = SectorDict(c => block(t1, c) \ block(t2, c)
                          for c in blocksectors(codomain(t1)))
        return TensorMap(data, domain(t1) ← domain(t2))
    end
end
function Base.:(/)(t1::AbstractTensorMap, t2::AbstractTensorMap)
    domain(t1) == domain(t2) ||
        throw(SpaceMismatch("non-matching domains in t1 / t2"))
    if sectortype(t1) === Trivial
        data = block(t1, Trivial()) / block(t2, Trivial())
        return TensorMap(data, codomain(t1) ← codomain(t2))
    else
        data = SectorDict(c => block(t1, c) / block(t2, c)
                          for c in blocksectors(domain(t1)))
        return TensorMap(data, codomain(t1) ← codomain(t2))
    end
end

# TensorMap exponentation:
function exp!(t::TensorMap)
    domain(t) == codomain(t) ||
        error("Exponentional of a tensor only exist when domain == codomain.")
    for (c, b) in blocks(t)
        copy!(b, LinearAlgebra.exp!(b))
    end
    return t
end

# Sylvester equation with TensorMap objects:
function LinearAlgebra.sylvester(A::AbstractTensorMap,
                                 B::AbstractTensorMap,
                                 C::AbstractTensorMap)
    (codomain(A) == domain(A) == codomain(C) && codomain(B) == domain(B) == domain(C)) ||
        throw(SpaceMismatch())
    cod = domain(A)
    dom = codomain(B)
    sylABC(c) = sylvester(block(A, c), block(B, c), block(C, c))
    data = SectorDict(c => sylABC(c) for c in blocksectors(cod ← dom))
    return TensorMap(data, cod ← dom)
end

# functions that map ℝ to (a subset of) ℝ
for f in (:cos, :sin, :tan, :cot, :cosh, :sinh, :tanh, :coth, :atan, :acot, :asinh)
    sf = string(f)
    @eval function Base.$f(t::AbstractTensorMap)
        domain(t) == codomain(t) ||
            error("$sf of a tensor only exist when domain == codomain.")
        I = sectortype(t)
        T = similarstoragetype(t, float(scalartype(t)))
        if sectortype(t) === Trivial
            local data::T
            if scalartype(t) <: Real
                data = real($f(block(t, Trivial())))
            else
                data = $f(block(t, Trivial()))
            end
            return TensorMap(data, codomain(t), domain(t))
        else
            if scalartype(t) <: Real
                datadict = SectorDict{I,T}(c => real($f(b)) for (c, b) in blocks(t))
            else
                datadict = SectorDict{I,T}(c => $f(b) for (c, b) in blocks(t))
            end
            return TensorMap(datadict, codomain(t), domain(t))
        end
    end
end
# functions that don't map ℝ to (a subset of) ℝ
for f in (:sqrt, :log, :asin, :acos, :acosh, :atanh, :acoth)
    sf = string(f)
    @eval function Base.$f(t::AbstractTensorMap)
        domain(t) == codomain(t) ||
            error("$sf of a tensor only exist when domain == codomain.")
        I = sectortype(t)
        T = similarstoragetype(t, complex(float(scalartype(t))))
        if sectortype(t) === Trivial
            data::T = $f(block(t, Trivial()))
            return TensorMap(data, codomain(t), domain(t))
        else
            datadict = SectorDict{I,T}(c => $f(b) for (c, b) in blocks(t))
            return TensorMap(datadict, codomain(t), domain(t))
        end
    end
end

# concatenate tensors
function catdomain(t1::T, t2::T) where {S,N₁,T<:AbstractTensorMap{<:Any,S,N₁,1}}
    codomain(t1) == codomain(t2) ||
        throw(SpaceMismatch("codomains of tensors to concatenate must match:\n" *
                            "$(codomain(t1)) ≠ $(codomain(t2))"))
    V1, = domain(t1)
    V2, = domain(t2)
    isdual(V1) == isdual(V2) ||
        throw(SpaceMismatch("cannot horizontally concatenate tensors whose domain has non-matching duality"))

    V = V1 ⊕ V2
    t = TensorMap(undef, promote_type(scalartype(t1), scalartype(t2)), codomain(t1), V)
    for c in sectors(V)
        block(t, c)[:, 1:dim(V1, c)] .= block(t1, c)
        block(t, c)[:, dim(V1, c) .+ (1:dim(V2, c))] .= block(t2, c)
    end
    return t
end
function catcodomain(t1::T, t2::T) where {S,N₂,T<:AbstractTensorMap{<:Any,S,1,N₂}}
    domain(t1) == domain(t2) ||
        throw(SpaceMismatch("domains of tensors to concatenate must match:\n" *
                            "$(domain(t1)) ≠ $(domain(t2))"))
    V1, = codomain(t1)
    V2, = codomain(t2)
    isdual(V1) == isdual(V2) ||
        throw(SpaceMismatch("cannot vertically concatenate tensors whose codomain has non-matching duality"))

    V = V1 ⊕ V2
    t = TensorMap(undef, promote_type(scalartype(t1), scalartype(t2)), V, domain(t1))
    for c in sectors(V)
        block(t, c)[1:dim(V1, c), :] .= block(t1, c)
        block(t, c)[dim(V1, c) .+ (1:dim(V2, c)), :] .= block(t2, c)
    end
    return t
end

# tensor product of tensors
"""
    ⊗(t1::AbstractTensorMap, t2::AbstractTensorMap, ...) -> TensorMap
    otimes(t1::AbstractTensorMap, t2::AbstractTensorMap, ...) -> TensorMap

Compute the tensor product between two `AbstractTensorMap` instances, which results in a
new `TensorMap` instance whose codomain is `codomain(t1) ⊗ codomain(t2)` and whose domain
is `domain(t1) ⊗ domain(t2)`.
"""
function ⊗(t1::AbstractTensorMap, t2::AbstractTensorMap)
    (S = spacetype(t1)) === spacetype(t2) ||
        throw(SpaceMismatch("spacetype(t1) ≠ spacetype(t2)"))
    cod1, cod2 = codomain(t1), codomain(t2)
    dom1, dom2 = domain(t1), domain(t2)
    cod = cod1 ⊗ cod2
    dom = dom1 ⊗ dom2
    t = zeros(promote_type(scalartype(t1), scalartype(t2)), cod ← dom)
    if sectortype(S) === Trivial
        d1 = dim(cod1)
        d2 = dim(cod2)
        d3 = dim(dom1)
        d4 = dim(dom2)
        m1 = reshape(t1[], (d1, 1, d3, 1))
        m2 = reshape(t2[], (1, d2, 1, d4))
        m = reshape(t[], (d1, d2, d3, d4))
        m .= m1 .* m2
    else
        for (f1l, f1r) in fusiontrees(t1)
            for (f2l, f2r) in fusiontrees(t2)
                c1 = f1l.coupled # = f1r.coupled
                c2 = f2l.coupled # = f2r.coupled
                for c in c1 ⊗ c2
                    degeneracyiter = FusionStyle(c) isa GenericFusion ?
                                     (1:Nsymbol(c1, c2, c)) : (nothing,)
                    for μ in degeneracyiter
                        for (fl, coeff1) in merge(f1l, f2l, c, μ)
                            for (fr, coeff2) in merge(f1r, f2r, c, μ)
                                d1 = dim(cod1, f1l.uncoupled)
                                d2 = dim(cod2, f2l.uncoupled)
                                d3 = dim(dom1, f1r.uncoupled)
                                d4 = dim(dom2, f2r.uncoupled)
                                m1 = reshape(t1[f1l, f1r], (d1, 1, d3, 1))
                                m2 = reshape(t2[f2l, f2r], (1, d2, 1, d4))
                                m = reshape(t[fl, fr], (d1, d2, d3, d4))
                                m .+= coeff1 .* conj(coeff2) .* m1 .* m2
                            end
                        end
                    end
                end
            end
        end
    end
    return t
end

# deligne product of tensors
function ⊠(t1::AbstractTensorMap, t2::AbstractTensorMap)
    S1 = spacetype(t1)
    I1 = sectortype(S1)
    S2 = spacetype(t2)
    I2 = sectortype(S2)
    codom1 = codomain(t1) ⊠ one(S2)
    dom1 = domain(t1) ⊠ one(S2)
    data1 = SectorDict{I1 ⊠ I2,storagetype(t1)}(c ⊠ one(I2) => b for (c, b) in blocks(t1))
    t1′ = TensorMap(data1, codom1, dom1)
    codom2 = one(S1) ⊠ codomain(t2)
    dom2 = one(S1) ⊠ domain(t2)
    data2 = SectorDict{I1 ⊠ I2,storagetype(t2)}(one(I1) ⊠ c => b for (c, b) in blocks(t2))
    t2′ = TensorMap(data2, codom2, dom2)
    return t1′ ⊗ t2′
end
