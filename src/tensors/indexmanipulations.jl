# Index manipulations
#---------------------
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
    return add_permute!(tdst, tsrc, p, true, false)
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
    cod = ProductSpace{spacetype(t),N₁}(map(n -> space(t, n), p₁))
    dom = ProductSpace{spacetype(t),N₂}(map(n -> dual(space(t, n)), p₂))
    # share data if possible
    if !copy
        if p₁ === codomainind(t) && p₂ === domainind(t)
            return t
        elseif has_shared_permute(t, (p₁, p₂))
            return TensorMap(reshape(t.data, dim(cod), dim(dom)), cod, dom)
        end
    end
    # general case
    @inbounds begin
        return permute!(similar(t, cod ← dom), t, (p₁, p₂))
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
    return add_braid!(tdst, tsrc, p, levels, true, false)
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
    cod = ProductSpace{spacetype(t)}(map(n -> space(t, n), p₁))
    dom = ProductSpace{spacetype(t)}(map(n -> dual(space(t, n)), p₂))
    @inbounds begin
        return braid!(similar(t, cod ← dom), t, (p₁, p₂), levels)
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
    return add_transpose!(tdst, tsrc, (p₁, p₂), true, false)
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
    cod = ProductSpace{spacetype(t)}(map(n -> space(t, n), p₁))
    dom = ProductSpace{spacetype(t)}(map(n -> dual(space(t, n)), p₂))
    @inbounds begin
        return transpose!(similar(t, cod ← dom), t, (p₁, p₂))
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
    twist!(t::AbstractTensorMap, i::Int; inv::Bool=false)
        -> t

Apply a twist to the `i`th index of `t`, storing the result in `t`.
If `inv=true`, use the inverse twist.

See [`twist`](@ref) for creating a new tensor.
"""
function twist!(t::AbstractTensorMap, i::Int; inv::Bool=false)
    if i > numind(t)
        msg = "Can't twist index $i of a tensor with only $(numind(t)) indices."
        throw(ArgumentError(msg))
    end
    BraidingStyle(sectortype(t)) == Bosonic() && return t
    N₁ = numout(t)
    for (f₁, f₂) in fusiontrees(t)
        θ = i <= N₁ ? twist(f₁.uncoupled[i]) : twist(f₂.uncoupled[i - N₁])
        inv && (θ = θ')
        rmul!(t[f₁, f₂], θ)
    end
    return t
end

"""
    twist(t::AbstractTensorMap, i::Int; inv::Bool=false)
        -> t

Apply a twist to the `i`th index of `t` and return the result as a new tensor.
If `inv=true`, use the inverse twist.

See [`twist!`](@ref) for storing the result in place.
"""
twist(t::AbstractTensorMap, i::Int; inv::Bool=false) = twist!(copy(t), i; inv=inv)

# Fusing and splitting
# TODO: add functionality for easy fusing and splitting of tensor indices

#-------------------------------------
# Full implementations based on `add`
#-------------------------------------
@propagate_inbounds function add_permute!(tdst::AbstractTensorMap{E,S,N₁,N₂},
                                          tsrc::AbstractTensorMap,
                                          p::Index2Tuple{N₁,N₂},
                                          α::Number,
                                          β::Number,
                                          backend::Backend...) where {E,S,N₁,N₂}
    treepermuter(f₁, f₂) = permute(f₁, f₂, p[1], p[2])
    return add_transform!(tdst, tsrc, p, treepermuter, α, β, backend...)
end

@propagate_inbounds function add_braid!(tdst::AbstractTensorMap{E,S,N₁,N₂},
                                        tsrc::AbstractTensorMap,
                                        p::Index2Tuple{N₁,N₂},
                                        levels::IndexTuple,
                                        α::Number,
                                        β::Number,
                                        backend::Backend...) where {E,S,N₁,N₂}
    length(levels) == numind(tsrc) ||
        throw(ArgumentError("incorrect levels $levels for tensor map $(codomain(tsrc)) ← $(domain(tsrc))"))

    levels1 = TupleTools.getindices(levels, codomainind(tsrc))
    levels2 = TupleTools.getindices(levels, domainind(tsrc))
    # TODO: arg order for tensormaps is different than for fusiontrees
    treebraider(f₁, f₂) = braid(f₁, f₂, levels1, levels2, p...)
    return add_transform!(tdst, tsrc, p, treebraider, α, β, backend...)
end

@propagate_inbounds function add_transpose!(tdst::AbstractTensorMap{E,S,N₁,N₂},
                                            tsrc::AbstractTensorMap,
                                            p::Index2Tuple{N₁,N₂},
                                            α::Number,
                                            β::Number,
                                            backend::Backend...) where {E,S,N₁,N₂}
    treetransposer(f₁, f₂) = transpose(f₁, f₂, p[1], p[2])
    return add_transform!(tdst, tsrc, p, treetransposer, α, β, backend...)
end

function add_transform!(tdst::AbstractTensorMap{E,S,N₁,N₂},
                        tsrc::AbstractTensorMap,
                        (p₁, p₂)::Index2Tuple{N₁,N₂},
                        fusiontreetransform,
                        α::Number,
                        β::Number,
                        backend::Backend...) where {E,S,N₁,N₂}
    @boundscheck begin
        all(i -> space(tsrc, p₁[i]) == space(tdst, i), 1:N₁) ||
            throw(SpaceMismatch("source = $(codomain(tsrc))←$(domain(tsrc)),
            dest = $(codomain(tdst))←$(domain(tdst)), p₁ = $(p₁), p₂ = $(p₂)"))
        all(i -> space(tsrc, p₂[i]) == space(tdst, N₁ + i), 1:N₂) ||
            throw(SpaceMismatch("source = $(codomain(tsrc))←$(domain(tsrc)),
            dest = $(codomain(tdst))←$(domain(tdst)), p₁ = $(p₁), p₂ = $(p₂)"))
    end

    I = sectortype(S)
    if p₁ == codomainind(tsrc) && p₂ == domainind(tsrc)
        add!(tdst, tsrc, α, β)
    elseif I === Trivial
        _add_trivial_kernel!(tdst, tsrc, (p₁, p₂), fusiontreetransform, α, β, backend...)
    elseif FusionStyle(I) isa UniqueFusion
        _add_abelian_kernel!(tdst, tsrc, (p₁, p₂), fusiontreetransform, α, β, backend...)
    else
        _add_general_kernel!(tdst, tsrc, (p₁, p₂), fusiontreetransform, α, β, backend...)
    end
    return tdst
end

# internal methods: no argument types
function _add_trivial_kernel!(tdst, tsrc, p, fusiontreetransform, α, β, backend...)
    TO.tensoradd!(tdst[], p, tsrc[], :N, α, β, backend...)
    return nothing
end

function _add_abelian_kernel!(tdst, tsrc, p, fusiontreetransform, α, β, backend...)
    if Threads.nthreads() > 1
        Threads.@sync for (f₁, f₂) in fusiontrees(tsrc)
            Threads.@spawn _add_abelian_block!(tdst, tsrc, p, fusiontreetransform,
                                               f₁, f₂, α, β, backend...)
        end
    else
        for (f₁, f₂) in fusiontrees(tsrc)
            _add_abelian_block!(tdst, tsrc, p, fusiontreetransform,
                                f₁, f₂, α, β, backend...)
        end
    end
    return tdst
end

function _add_abelian_block!(tdst, tsrc, p, fusiontreetransform, f₁, f₂, α, β, backend...)
    (f₁′, f₂′), coeff = first(fusiontreetransform(f₁, f₂))
    TO.tensoradd!(tdst[f₁′, f₂′], p, tsrc[f₁, f₂], :N, α * coeff, β, backend...)
    return nothing
end

function _add_general_kernel!(tdst, tsrc, p, fusiontreetransform, α, β, backend...)
    if iszero(β)
        tdst = zerovector!(tdst)
    elseif β != 1
        tdst = scale!(tdst, β)
    end
    if Threads.nthreads() > 1
        Threads.@sync for s₁ in sectors(codomain(tsrc)), s₂ in sectors(domain(tsrc))
            Threads.@spawn _add_nonabelian_sector!(tdst, tsrc, p, fusiontreetransform, s₁,
                                                   s₂, α, β, backend...)
        end
    else
        for (f₁, f₂) in fusiontrees(tsrc)
            for ((f₁′, f₂′), coeff) in fusiontreetransform(f₁, f₂)
                TO.tensoradd!(tdst[f₁′, f₂′], p, tsrc[f₁, f₂], :N, α * coeff, true,
                              backend...)
            end
        end
    end
    return nothing
end

function _add_nonabelian_sector!(tdst, tsrc, p, fusiontreetransform, s₁, s₂, α, β,
                                 backend...)
    for (f₁, f₂) in fusiontrees(tsrc)
        (f₁.uncoupled == s₁ && f₂.uncoupled == s₂) || continue
        for ((f₁′, f₂′), coeff) in fusiontreetransform(f₁, f₂)
            TO.tensoradd!(tdst[f₁′, f₂′], p, tsrc[f₁, f₂], :N, α * coeff, true, backend...)
        end
    end
    return nothing
end
