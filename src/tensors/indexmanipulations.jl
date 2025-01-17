# Index manipulations
#---------------------
"""
    flip(t::AbstractTensorMap, I) -> t′::AbstractTensorMap

Return a new tensor that is isomorphic to `t` but where the arrows on the indices `i` that satisfy
`i ∈ I` are flipped, i.e. `space(t′, i) = flip(space(t, i))`.
"""
function flip(t::AbstractTensorMap, I)
    P = flip(space(t), I)
    t′ = similar(t, P)
    for (f₁, f₂) in fusiontrees(t)
        f₁′, f₂′ = f₁, f₂
        factor = one(scalartype(t))
        for i in I
            (f₁′, f₂′), s = only(flip(f₁′, f₂′, i))
            factor *= s
        end
        scale!(t′[f₁′, f₂′], t[f₁, f₂], factor)
    end
    return t′
end

"""
    permute!(tdst::AbstractTensorMap, tsrc::AbstractTensorMap, (p₁, p₂)::Index2Tuple)
        -> tdst

Write into `tdst` the result of permuting the indices of `tsrc`.
The codomain and domain of `tdst` correspond to the indices in `p₁` and `p₂` of `tsrc` respectively.
                
See [`permute`](@ref) for creating a new tensor and [`add_permute!`](@ref) for a more general version.
"""
@propagate_inbounds function Base.permute!(tdst::AbstractTensorMap,
                                           tsrc::AbstractTensorMap,
                                           p::Index2Tuple)
    return add_permute!(tdst, tsrc, p, One(), Zero())
end

"""
    permute(tsrc::AbstractTensorMap, (p₁, p₂)::Index2Tuple;
            copy::Bool=false)
        -> tdst::TensorMap

Return tensor `tdst` obtained by permuting the indices of `tsrc`.
The codomain and domain of `tdst` correspond to the indices in `p₁` and `p₂` of `tsrc` respectively.

If `copy=false`, `tdst` might share data with `tsrc` whenever possible. Otherwise, a copy is always made.

To permute into an existing destination, see [permute!](@ref) and [`add_permute!`](@ref)
"""
function permute(t::AbstractTensorMap, (p₁, p₂)::Index2Tuple{N₁,N₂};
                 copy::Bool=false) where {N₁,N₂}
    space′ = permute(space(t), (p₁, p₂))
    # share data if possible
    if !copy && p₁ === codomainind(t) && p₂ === domainind(t)
        return t
    end
    # general case
    @inbounds begin
        return permute!(similar(t, space′), t, (p₁, p₂))
    end
end
function permute(t::TensorMap, (p₁, p₂)::Index2Tuple{N₁,N₂}; copy::Bool=false) where {N₁,N₂}
    space′ = permute(space(t), (p₁, p₂))
    # share data if possible
    if !copy
        if p₁ === codomainind(t) && p₂ === domainind(t)
            return t
        elseif has_shared_permute(t, (p₁, p₂))
            return TensorMap(t.data, space′)
        end
    end
    # general case
    @inbounds begin
        return permute!(similar(t, space′), t, (p₁, p₂))
    end
end
function permute(t::AdjointTensorMap, (p₁, p₂)::Index2Tuple; copy::Bool=false)
    p₁′ = adjointtensorindices(t, p₂)
    p₂′ = adjointtensorindices(t, p₁)
    return adjoint(permute(adjoint(t), (p₁′, p₂′); copy))
end
function permute(t::AbstractTensorMap, p::IndexTuple; copy::Bool=false)
    return permute(t, (p, ()); copy)
end

function has_shared_permute(t::AbstractTensorMap, (p₁, p₂)::Index2Tuple)
    return (p₁ === codomainind(t) && p₂ === domainind(t))
end
function has_shared_permute(t::TensorMap, (p₁, p₂)::Index2Tuple)
    if p₁ === codomainind(t) && p₂ === domainind(t)
        return true
    elseif sectortype(t) === Trivial
        stridet = i -> stride(t[], i)
        sizet = i -> size(t[], i)
        canfuse1, d1, s1 = TO._canfuse(sizet.(p₁), stridet.(p₁))
        canfuse2, d2, s2 = TO._canfuse(sizet.(p₂), stridet.(p₂))
        return canfuse1 && canfuse2 && s1 == 1 && (d2 == 1 || s2 == d1)
    else
        return false
    end
end
function has_shared_permute(t::AdjointTensorMap, (p₁, p₂)::Index2Tuple)
    p₁′ = adjointtensorindices(t, p₂)
    p₂′ = adjointtensorindices(t, p₁)
    return has_shared_permute(t', (p₁′, p₂′))
end

# Braid
"""
    braid!(tdst::AbstractTensorMap, tsrc::AbstractTensorMap,
           (p₁, p₂)::Index2Tuple, levels::Tuple)
        -> tdst

Write into `tdst` the result of braiding the indices of `tsrc`.
The codomain and domain of `tdst` correspond to the indices in `p₁` and `p₂` of `tsrc` respectively.
Here, `levels` is a tuple of length `numind(tsrc)` that assigns a level or height to the indices of `tsrc`,
which determines whether they will braid over or under any other index with which they have to change places.

See [`braid`](@ref) for creating a new tensor and [`add_braid!`](@ref) for a more general version.
"""
@propagate_inbounds function braid!(tdst::AbstractTensorMap,
                                    tsrc::AbstractTensorMap,
                                    p::Index2Tuple,
                                    levels::IndexTuple)
    return add_braid!(tdst, tsrc, p, levels, One(), Zero())
end

"""
    braid(tsrc::AbstractTensorMap, (p₁, p₂)::Index2Tuple, levels::IndexTuple;
          copy::Bool = false)
        -> tdst::TensorMap

Return tensor `tdst` obtained by braiding the indices of `tsrc`.
The codomain and domain of `tdst` correspond to the indices in `p₁` and `p₂` of `tsrc` respectively.
Here, `levels` is a tuple of length `numind(tsrc)` that assigns a level or height to the indices of `tsrc`,
which determines whether they will braid over or under any other index with which they have to change places.

If `copy=false`, `tdst` might share data with `tsrc` whenever possible. Otherwise, a copy is always made.

To braid into an existing destination, see [braid!](@ref) and [`add_braid!`](@ref)
"""
function braid(t::AbstractTensorMap, (p₁, p₂)::Index2Tuple, levels::IndexTuple;
               copy::Bool=false)
    @assert length(levels) == numind(t)
    if BraidingStyle(sectortype(t)) isa SymmetricBraiding
        return permute(t, (p₁, p₂); copy=copy)
    end
    if !copy && p₁ == codomainind(t) && p₂ == domainind(t)
        return t
    end
    # general case
    space′ = permute(space(t), (p₁, p₂))
    @inbounds begin
        return braid!(similar(t, space′), t, (p₁, p₂), levels)
    end
end
# TODO: braid for `AdjointTensorMap`; think about how to map the `levels` argument.

# Transpose
_transpose_indices(t::AbstractTensorMap) = (reverse(domainind(t)), reverse(codomainind(t)))

"""
    transpose!(tdst::AbstractTensorMap, tsrc::AbstractTensorMap,
               (p₁, p₂)::Index2Tuple)
        -> tdst

Write into `tdst` the result of transposing the indices of `tsrc`.
The codomain and domain of `tdst` correspond to the indices in `p₁` and `p₂` of `tsrc` respectively.
The new index positions should be attainable without any indices crossing each other, i.e.,
the permutation `(p₁..., reverse(p₂)...)` should constitute a cyclic permutation of `(codomainind(tsrc)..., reverse(domainind(tsrc))...)`.

See [`transpose`](@ref) for creating a new tensor and [`add_transpose!`](@ref) for a more general version.
"""
function LinearAlgebra.transpose!(tdst::AbstractTensorMap,
                                  tsrc::AbstractTensorMap,
                                  (p₁, p₂)::Index2Tuple=_transpose_indices(t))
    return add_transpose!(tdst, tsrc, (p₁, p₂), One(), Zero())
end

"""
    transpose(tsrc::AbstractTensorMap, (p₁, p₂)::Index2Tuple;
              copy::Bool=false)
        -> tdst::TensorMap

Return tensor `tdst` obtained by transposing the indices of `tsrc`.
The codomain and domain of `tdst` correspond to the indices in `p₁` and `p₂` of `tsrc` respectively.
The new index positions should be attainable without any indices crossing each other, i.e.,
the permutation `(p₁..., reverse(p₂)...)` should constitute a cyclic permutation of `(codomainind(tsrc)..., reverse(domainind(tsrc))...)`.

If `copy=false`, `tdst` might share data with `tsrc` whenever possible. Otherwise, a copy is always made.

To permute into an existing destination, see [permute!](@ref) and [`add_permute!`](@ref)
"""
function LinearAlgebra.transpose(t::AbstractTensorMap,
                                 (p₁, p₂)::Index2Tuple=_transpose_indices(t);
                                 copy::Bool=false)
    if sectortype(t) === Trivial
        return permute(t, (p₁, p₂); copy=copy)
    end
    if !copy && p₁ == codomainind(t) && p₂ == domainind(t)
        return t
    end
    # general case
    space′ = permute(space(t), (p₁, p₂))
    @inbounds begin
        return transpose!(similar(t, space′), t, (p₁, p₂))
    end
end

function LinearAlgebra.transpose(t::AdjointTensorMap,
                                 (p₁, p₂)::Index2Tuple=_transpose_indices(t);
                                 copy::Bool=false)
    p₁′ = map(n -> adjointtensorindex(t, n), p₂)
    p₂′ = map(n -> adjointtensorindex(t, n), p₁)
    return adjoint(transpose(adjoint(t), (p₁′, p₂′); copy=copy))
end

"""
    repartition!(tdst::AbstractTensorMap{S}, tsrc::AbstractTensorMap{S}) where {S} -> tdst

Write into `tdst` the result of repartitioning the indices of `tsrc`. This is just a special
case of a transposition that only changes the number of in- and outgoing indices.

See [`repartition`](@ref) for creating a new tensor.
"""
function repartition!(tdst::AbstractTensorMap{S}, tsrc::AbstractTensorMap{S}) where {S}
    numind(tsrc) == numind(tdst) ||
        throw(ArgumentError("tsrc and tdst should have an equal amount of indices"))
    all_inds = (codomainind(tsrc)..., reverse(domainind(tsrc))...)
    p₁ = ntuple(i -> all_inds[i], numout(tdst))
    p₂ = reverse(ntuple(i -> all_inds[i + numout(tdst)], numin(tdst)))
    return transpose!(tdst, tsrc, (p₁, p₂))
end

"""
    repartition(tsrc::AbstractTensorMap{S}, N₁::Int, N₂::Int; copy::Bool=false) where {S}
        -> tdst::AbstractTensorMap{S,N₁,N₂}

Return tensor `tdst` obtained by repartitioning the indices of `t`.
The codomain and domain of `tdst` correspond to the first `N₁` and last `N₂` spaces of `t`,
respectively.

If `copy=false`, `tdst` might share data with `tsrc` whenever possible. Otherwise, a copy is always made.

To repartition into an existing destination, see [repartition!](@ref).
"""
@constprop :aggressive function repartition(t::AbstractTensorMap, N₁::Int,
                                            N₂::Int=numind(t) - N₁;
                                            copy::Bool=false)
    N₁ + N₂ == numind(t) ||
        throw(ArgumentError("Invalid repartition: $(numind(t)) to ($N₁, $N₂)"))
    all_inds = (codomainind(t)..., reverse(domainind(t))...)
    p₁ = ntuple(i -> all_inds[i], N₁)
    p₂ = reverse(ntuple(i -> all_inds[i + N₁], N₂))
    return transpose(t, (p₁, p₂); copy)
end

# Twist
"""
    twist!(t::AbstractTensorMap, i::Int; inv::Bool=false) -> t
    twist!(t::AbstractTensorMap, is; inv::Bool=false) -> t

Apply a twist to the `i`th index of `t`, or all indices in `is`, storing the result in `t`.
If `inv=true`, use the inverse twist.

See [`twist`](@ref) for creating a new tensor.
"""
function twist!(t::AbstractTensorMap, is; inv::Bool=false)
    if !all(in(allind(t)), is)
        msg = "Can't twist indices $is of a tensor with only $(numind(t)) indices."
        throw(ArgumentError(msg))
    end
    (BraidingStyle(sectortype(t)) == Bosonic() || isempty(is)) && return t
    N₁ = numout(t)
    for (f₁, f₂) in fusiontrees(t)
        θ = prod(i -> i <= N₁ ? twist(f₁.uncoupled[i]) : twist(f₂.uncoupled[i - N₁]), is)
        inv && (θ = θ')
        rmul!(t[f₁, f₂], θ)
    end
    return t
end

"""
    twist(tsrc::AbstractTensorMap, i::Int; inv::Bool=false) -> tdst
    twist(tsrc::AbstractTensorMap, is; inv::Bool=false) -> tdst

Apply a twist to the `i`th index of `tsrc` and return the result as a new tensor.
If `inv=true`, use the inverse twist.

See [`twist!`](@ref) for storing the result in place.
"""
twist(t::AbstractTensorMap, i; inv::Bool=false) = twist!(copy(t), i; inv)

"""
    insertleftunit(tsrc::AbstractTensorMap, i::Int=numind(t) + 1;
                   conj=false, dual=false, copy=false) -> tdst

Insert a trivial vector space, isomorphic to the underlying field, at position `i`.
More specifically, adds a left monoidal unit or its dual.

If `copy=false`, `tdst` might share data with `tsrc` whenever possible. Otherwise, a copy is always made.

See also [`insertrightunit`](@ref) and [`removeunit`](@ref).
"""
@constprop :aggressive function insertleftunit(t::AbstractTensorMap, i::Int=numind(t) + 1;
                                               copy::Bool=true, conj::Bool=false,
                                               dual::Bool=false)
    W = insertleftunit(space(t), i; conj, dual)
    tdst = similar(t, W)
    for (c, b) in blocks(t)
        copy!(block(tdst, c), b)
    end
    return tdst
end
@constprop :aggressive function insertleftunit(t::TensorMap, i::Int=numind(t) + 1;
                                               copy::Bool=false, conj::Bool=false,
                                               dual::Bool=false)
    W = insertleftunit(space(t), i; conj, dual)
    return TensorMap{scalartype(t)}(copy ? Base.copy(t.data) : t.data, W)
end

"""
    insertrightunit(tsrc::AbstractTensorMap, i::Int=numind(t);
                    conj=false, dual=false, copy=false) -> tdst

Insert a trivial vector space, isomorphic to the underlying field, after position `i`.
More specifically, adds a right monoidal unit or its dual.

If `copy=false`, `tdst` might share data with `tsrc` whenever possible. Otherwise, a copy is always made.

See also [`insertleftunit`](@ref) and [`removeunit`](@ref).
"""
@constprop :aggressive function insertrightunit(t::AbstractTensorMap, i::Int=numind(t);
                                                copy::Bool=true, kwargs...)
    W = insertrightunit(space(t), i; kwargs...)
    tdst = similar(t, W)
    for (c, b) in blocks(t)
        copy!(block(tdst, c), b)
    end
    return tdst
end
@constprop :aggressive function insertrightunit(t::TensorMap, i::Int=numind(t);
                                                copy::Bool=false, kwargs...)
    W = insertrightunit(space(t), i; kwargs...)
    return TensorMap{scalartype(t)}(copy ? Base.copy(t.data) : t.data, W)
end

"""
    removeunit(tsrc::AbstractTensorMap, i::Int; copy=false) -> tdst

This removes a trivial tensor product factor at position `1 ≤ i ≤ N`.
For this to work, that factor has to be isomorphic to the field of scalars.

If `copy=false`, `tdst` might share data with `tsrc` whenever possible. Otherwise, a copy is always made.

This operation undoes the work of [`insertunit`](@ref).
"""
@constprop :aggressive function removeunit(t::TensorMap, i::Int; copy::Bool=false)
    W = removeunit(space(t), i)
    return TensorMap{scalartype(t)}(copy ? Base.copy(t.data) : t.data, W)
end
@constprop :aggressive function removeunit(t::AbstractTensorMap, i::Int; copy::Bool=true)
    W = removeunit(space(t), i)
    tdst = similar(t, W)
    for (c, b) in blocks(t)
        copy!(block(tdst, c), b)
    end
    return tdst
end

# Fusing and splitting
# TODO: add functionality for easy fusing and splitting of tensor indices

#-------------------------------------
# Full implementations based on `add`
#-------------------------------------
"""
    add_permute!(tdst::AbstractTensorMap, tsrc::AbstractTensorMap, (p₁, p₂)::Index2Tuple,
                 α::Number, β::Number, backend...)

Return the updated `tdst`, which is the result of adding `α * tsrc` to `tdst` after permuting 
the indices of `tsrc` according to `(p₁, p₂)`.

See also [`permute`](@ref), [`permute!`](@ref), [`add_braid!`](@ref), [`add_transpose!`](@ref).
"""
@propagate_inbounds function add_permute!(tdst::AbstractTensorMap,
                                          tsrc::AbstractTensorMap,
                                          p::Index2Tuple,
                                          α::Number,
                                          β::Number,
                                          backend...)
    transformer = treepermuter(tdst, tsrc, p)
    return add_transform!(tdst, tsrc, p, transformer, α, β, backend...)
end

"""
    add_braid!(tdst::AbstractTensorMap, tsrc::AbstractTensorMap, (p₁, p₂)::Index2Tuple,
               levels::IndexTuple, α::Number, β::Number, backend...)

Return the updated `tdst`, which is the result of adding `α * tsrc` to `tdst` after braiding
the indices of `tsrc` according to `(p₁, p₂)` and `levels`.

See also [`braid`](@ref), [`braid!`](@ref), [`add_permute!`](@ref), [`add_transpose!`](@ref).
"""
@propagate_inbounds function add_braid!(tdst::AbstractTensorMap,
                                        tsrc::AbstractTensorMap,
                                        p::Index2Tuple,
                                        levels::IndexTuple,
                                        α::Number,
                                        β::Number,
                                        backend...)
    length(levels) == numind(tsrc) ||
        throw(ArgumentError("incorrect levels $levels for tensor map $(codomain(tsrc)) ← $(domain(tsrc))"))

    levels1 = TupleTools.getindices(levels, codomainind(tsrc))
    levels2 = TupleTools.getindices(levels, domainind(tsrc))
    # TODO: arg order for tensormaps is different than for fusiontrees
    transformer = treebraider(tdst, tsrc, p, (levels1, levels2))
    return add_transform!(tdst, tsrc, p, transformer, α, β, backend...)
end

"""
    add_transpose!(tdst::AbstractTensorMap, tsrc::AbstractTensorMap, (p₁, p₂)::Index2Tuple,
                   α::Number, β::Number, backend...)

Return the updated `tdst`, which is the result of adding `α * tsrc` to `tdst` after transposing
the indices of `tsrc` according to `(p₁, p₂)`.

See also [`transpose`](@ref), [`transpose!`](@ref), [`add_permute!`](@ref), [`add_braid!`](@ref).
"""
@propagate_inbounds function add_transpose!(tdst::AbstractTensorMap,
                                            tsrc::AbstractTensorMap,
                                            p::Index2Tuple,
                                            α::Number,
                                            β::Number,
                                            backend...)
    transformer = treetransposer(tdst, tsrc, p)
    return add_transform!(tdst, tsrc, p, transformer, α, β, backend...)
end

# Implementation
# --------------
"""
    add_transform!(C, A, pA, transformer, α, β, [backend], [allocator])

Return the updated `C`, which is the result of adding `α * A` to `β * B`,
permuting the data with `pA` while transforming the fusiontrees with `transformer`.
"""
function add_transform!(C::AbstractTensorMap, A::AbstractTensorMap, pA::Index2Tuple,
                        transformer, α::Number, β::Number)
    return add_transform!(C, A, pA, transformer, α, β, TO.DefaultBackend())
end
function add_transform!(C::AbstractTensorMap, A::AbstractTensorMap, pA::Index2Tuple,
                        transformer, α::Number, β::Number, backend)
    return add_transform!(C, A, pA, transformer, α, β, backend, TO.DefaultAllocator())
end
function add_transform!(C::AbstractTensorMap, A::AbstractTensorMap, pA::Index2Tuple,
                        transformer, α::Number, β::Number, backend, allocator)
    if backend isa TO.DefaultBackend
        newbackend = TO.select_backend(add_transform!, C, A)
        return add_transform!(C, A, pA, transformer, α, β, newbackend, allocator)
    elseif backend isa TO.NoBackend # error for missing backend
        TC = typeof(C)
        TA = typeof(A)
        throw(ArgumentError("No suitable backend found for `add_transform!` and tensor types $TC and $TA"))
    else # error for unknown backend
        TC = typeof(C)
        TA = typeof(A)
        throw(ArgumentError("Unknown backend $backend for `add_transform!` and tensor types $TC and $TA"))
    end
end
function TO.select_backend(::typeof(add_transform!), C::AbstractTensorMap,
                           A::AbstractTensorMap)
    return TensorKitBackend()
end

function add_transform!(tdst::AbstractTensorMap,
                        tsrc::AbstractTensorMap,
                        (p₁, p₂)::Index2Tuple,
                        transformer,
                        α::Number,
                        β::Number,
                        backend::TensorKitBackend, allocator)
    @boundscheck begin
        permute(space(tsrc), (p₁, p₂)) == space(tdst) ||
            throw(SpaceMismatch("source = $(codomain(tsrc))←$(domain(tsrc)),
            dest = $(codomain(tdst))←$(domain(tdst)), p₁ = $(p₁), p₂ = $(p₂)"))
    end

    if p₁ === codomainind(tsrc) && p₂ === domainind(tsrc)
        add!(tdst, tsrc, α, β)
    else
        add_transform_kernel!(tdst, tsrc, (p₁, p₂), transformer, α, β, backend, allocator)
    end

    return tdst
end

function add_transform_kernel!(tdst::TensorMap,
                               tsrc::TensorMap,
                               (p₁, p₂)::Index2Tuple,
                               ::TrivialTreeTransformer,
                               α::Number,
                               β::Number,
                               backend::TensorKitBackend, allocator)
    return TO.tensoradd!(tdst[], tsrc[], (p₁, p₂), false, α, β, backend.arraybackend,
                         allocator)
end

function add_transform_kernel!(tdst::TensorMap,
                               tsrc::TensorMap,
                               (p₁, p₂)::Index2Tuple,
                               transformer::AbelianTreeTransformer,
                               α::Number,
                               β::Number,
                               backend::TensorKitBackend, allocator)
    structure_dst = transformer.structure_dst.fusiontreestructure
    structure_src = transformer.structure_src.fusiontreestructure

    tforeach(transformer.rows, transformer.cols, transformer.vals;
             scheduler=backend.subblockscheduler) do row, col, val
        sz_dst, str_dst, offset_dst = structure_dst[col]
        subblock_dst = StridedView(tdst.data, sz_dst, str_dst, offset_dst)

        sz_src, str_src, offset_src = structure_src[row]
        subblock_src = StridedView(tsrc.data, sz_src, str_src, offset_src)

        return TO.tensoradd!(subblock_dst, subblock_src, (p₁, p₂), false, α * val, β,
                             backend.arraybackend, allocator)
    end

    return nothing
end

function add_transform_kernel!(tdst::TensorMap,
                               tsrc::TensorMap,
                               (p₁, p₂)::Index2Tuple,
                               transformer::GenericTreeTransformer,
                               α::Number,
                               β::Number,
                               backend::TensorKitBackend, allocator)
    structure_dst = transformer.structure_dst.fusiontreestructure
    structure_src = transformer.structure_src.fusiontreestructure

    rows = rowvals(transformer.matrix)
    vals = nonzeros(transformer.matrix)

    tforeach(axes(transformer.matrix, 2); scheduler=backend.subblockscheduler) do j
        sz_dst, str_dst, offset_dst = structure_dst[j]
        subblock_dst = StridedView(tdst.data, sz_dst, str_dst, offset_dst)
        nzrows = nzrange(transformer.matrix, j)

        # treat first entry
        sz_src, str_src, offset_src = structure_src[rows[first(nzrows)]]
        subblock_src = StridedView(tsrc.data, sz_src, str_src, offset_src)
        TO.tensoradd!(subblock_dst, subblock_src, (p₁, p₂), false, α * vals[first(nzrows)],
                      β,
                      backend.arraybackend, allocator)

        # treat remaining entries
        for i in @view(nzrows[2:end])
            sz_src, str_src, offset_src = structure_src[rows[i]]
            subblock_src = StridedView(tsrc.data, sz_src, str_src, offset_src)
            TO.tensoradd!(subblock_dst, subblock_src, (p₁, p₂), false, α * vals[i], One(),
                          backend.arraybackend, allocator)
        end
    end

    return tdst
end

function add_transform_kernel!(tdst::AbstractTensorMap,
                               tsrc::AbstractTensorMap,
                               (p₁, p₂)::Index2Tuple,
                               fusiontreetransform::Function,
                               α::Number,
                               β::Number,
                               backend::TensorKitBackend, allocator)
    I = sectortype(spacetype(tdst))

    if I === Trivial
        _add_trivial_kernel!(tdst, tsrc, (p₁, p₂), fusiontreetransform, α, β,
                             backend, allocator)
    elseif FusionStyle(I) isa UniqueFusion
        _add_abelian_kernel!(tdst, tsrc, (p₁, p₂), fusiontreetransform, α, β,
                             backend, allocator)
    else
        _add_general_kernel!(tdst, tsrc, (p₁, p₂), fusiontreetransform, α, β,
                             backend, allocator)
    end

    return nothing
end

# internal methods: no argument types
function _add_trivial_kernel!(tdst, tsrc, p, fusiontreetransform, α, β,
                              backend::TensorKitBackend, allocator)
    TO.tensoradd!(tdst[], tsrc[], p, false, α, β, backend.arraybackend, allocator)
    return nothing
end

function _add_abelian_kernel!(tdst, tsrc, p, fusiontreetransform, α, β,
                              backend::TensorKitBackend, allocator)
    tforeach(fusiontrees(tsrc); scheduler=backend.subblockscheduler) do (f₁, f₂)
        return _add_abelian_block!(tdst, tsrc, p, fusiontreetransform,
                                   f₁, f₂, α, β, backend.arraybackend, allocator)
    end
    return nothing
end

function _add_abelian_block!(tdst, tsrc, p, fusiontreetransform, f₁, f₂, α, β,
                             backend, allocator)
    (f₁′, f₂′), coeff = first(fusiontreetransform(f₁, f₂))
    @inbounds TO.tensoradd!(tdst[f₁′, f₂′], tsrc[f₁, f₂], p, false, α * coeff, β,
                            backend, allocator)
    return nothing
end

function _add_general_kernel!(tdst, tsrc, p, fusiontreetransform, α, β, backend, allocator)
    if iszero(β)
        tdst = zerovector!(tdst)
    elseif β != 1
        tdst = scale!(tdst, β)
    end
    β′ = One()
    if backend.subblockscheduler isa SerialScheduler
        for (f₁, f₂) in fusiontrees(tsrc)
            for ((f₁′, f₂′), coeff) in fusiontreetransform(f₁, f₂)
                @inbounds TO.tensoradd!(tdst[f₁′, f₂′], tsrc[f₁, f₂], p, false, α * coeff,
                                        β′, backend.arraybackend, allocator)
            end
        end
    else
        tforeach(Iterators.product(sectors(codomain(tsrc)), sectors(domain(tsrc)));
                 scheduler=backend.subblockscheduler) do (s₁,
                                                          s₂)
            return _add_nonabelian_sector!(tdts, tsrc, p, fusiontreetransform, s₁, s₂, α,
                                           β′, backend.arraybackend, allocator)
        end
    end
    return nothing
end

# TODO: β argument is weird here because it has to be 1
function _add_nonabelian_sector!(tdst, tsrc, p, fusiontreetransform, s₁, s₂, α, β,
                                 backend, allocator)
    for (f₁, f₂) in fusiontrees(tsrc)
        (f₁.uncoupled == s₁ && f₂.uncoupled == s₂) || continue
        for ((f₁′, f₂′), coeff) in fusiontreetransform(f₁, f₂)
            @inbounds TO.tensoradd!(tdst[f₁′, f₂′], tsrc[f₁, f₂], p, false, α * coeff, β,
                                    backend, allocator)
        end
    end
    return nothing
end
