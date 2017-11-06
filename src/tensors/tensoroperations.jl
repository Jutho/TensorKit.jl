# Add support for TensorOperations.jl
function similar_from_indices(T::Type, p1::IndexTuple, p2::IndexTuple, t::AbstractTensorMap)
    s = codomain(t) ⊗ dual(domain(t))
    cod = s[map(n->tensor2spaceindex(t,n), p1)]
    dom = dual(s[map(n->tensor2spaceindex(t,n), reverse(p2))])
    return similar(t, T, cod←dom)
end
function similar_from_indices(T::Type, oindA::IndexTuple, oindB::IndexTuple, p1::IndexTuple, p2::IndexTuple, tA::AbstractTensorMap, tB::AbstractTensorMap)
    sA = codomain(tA) ⊗ dual(domain(tA))
    sB = codomain(tB) ⊗ dual(domain(tB))
    s = sA[map(n->tensor2spaceindex(tA,n), oindA)] ⊗ sB[map(n->tensor2spaceindex(tB,n), oindB)]
    cod = s[p1]
    dom = dual(s[reverse(p2)])
    return similar(tA, T, cod←dom)
end

function add!(α, tsrc::AbstractTensorMap{S}, β, tdst::AbstractTensorMap{S,N₁,N₂}, p1::IndexTuple{N₁}, p2::IndexTuple{N₂}) where {S,N₁,N₂}
    # TODO: Frobenius-Schur indicators!, and fermions!
    all(i->space(tsrc, p1[i]) == space(tdst,i), 1:N₁) || throw(SpaceMismatch())
    all(i->space(tsrc, p2[i]) == space(tdst,N₁+i), 1:N₂) || throw(SpaceMismatch())

    pdata = (p1...,p2...)
    if sectortype(S) == Trivial
        if iszero(β)
            fill!(tdst, β)
        end
        if β != 1
            @inbounds axpby!(α, permutedims(tsrc[], pdata), β, tdst[])
        else
            @inbounds axpy!(α, permutedims(tsrc[], pdata), tdst[])
        end
    else
        if iszero(β)
            fill!(tdst, β)
        elseif β != 1
            scale!(tdst, β)
        end
        for (f1,f2) in fusiontrees(tsrc)
            for ((f1′,f2′), coeff) in permute(f1, f2, p1, p2)
                @inbounds axpy!(α*coeff, permutedims(tsrc[f1,f2], pdata), tdst[f1′,f2′])
            end
        end
    end
    return tdst
end

function contract!(α, A::AbstractTensorMap{S}, B::AbstractTensorMap{S}, β, C::AbstractTensorMap{S}, oindA::IndexTuple, cindA::IndexTuple, oindB::IndexTuple, cindB::IndexTuple, p1::IndexTuple, p2::IndexTuple) where {S}
    A′ = permuteind(A, oindA, cindA)
    B′ = permuteind(B, cindB, oindB)
    return add!(α, A′ * B′, β, C, p1, p2)
end

# Compatibility layer for working with the `@tensor` macro from TensorOperations
TensorOperations.numind(t::AbstractTensorMap) = numind(t)
TensorOperations.numind(T::Type{<:AbstractTensorMap}) = numind(T)

function TensorOperations.similar_from_indices(T::Type, p1::IndexTuple, p2::IndexTuple, t::AbstractTensorMap, V::Type{<:Val})
    if V == Val{:N}
        similar_from_indices(T, p1, p2, t)
    else
        p1 = map(n->adjointtensorindex(t,n), p1)
        p2 = map(n->adjointtensorindex(t,n), p2)
        similar_from_indices(T, p1, p2, adjoint(t))
    end
end
function TensorOperations.similar_from_indices(T::Type, oindA::IndexTuple, oindB::IndexTuple, p1::IndexTuple, p2::IndexTuple, tA::AbstractTensorMap{S}, tB::AbstractTensorMap{S}, VA::Type{<:Val}, VB::Type{<:Val}) where {S}
    if VA == Val{:N} && VB == Val{:N}
        similar_from_indices(T, oindA, oindB, p1, p2, tA, tB)
    elseif VA == Val{:N} && VB == Val{:C}
        oindB = map(n->adjointtensorindex(tB,n), oindB)
        similar_from_indices(T, oindA, oindB, p1, p2, tA, adjoint(tB))
    elseif VA == Val{:C} && VB == Val{:N}
        oindA = map(n->adjointtensorindex(tA,n), oindA)
        similar_from_indices(T, oindA, oindB, p1, p2, adjoint(tA), tB)
    else
        oindA = map(n->adjointtensorindex(tA,n), oindA)
        oindB = map(n->adjointtensorindex(tB,n), oindB)
        similar_from_indices(T, oindA, oindB, p1, p2, adjoint(tA), adjoint(tB))
    end
end

function TensorOperations.add!(α, tsrc::AbstractTensorMap{S}, V::Type{<:Val}, β, tdst::AbstractTensorMap{S,N₁,N₂}, p1::IndexTuple{N₁}, p2::IndexTuple{N₂}) where {S,N₁,N₂}
    if V == Val{:N}
        add!(α, tsrc, β, tdst, p1, p2)
    else
        p1 = map(n->adjointtensorindex(t,n), p1)
        p2 = map(n->adjointtensorindex(t,n), p2)
        add!(α, adjoint(tsrc), β, tdst, p1, p2)
    end
    return tdst
end

function TensorOperations.contract!(α, tA::AbstractTensorMap{S}, VA::Type{<:Val}, tB::AbstractTensorMap{S}, VB::Type{<:Val}, β, tC::AbstractTensorMap{S}, oindA, cindA, oindB, cindB, p1, p2) where {S}
    if VA == Val{:N} && VB == Val{:N}
        contract!(α, tA, tB, β, tC, oindA, cindA, oindB, cindB, p1, p2)
    elseif VA == Val{:N} && VB == Val{:C}
        oindB = map(n->adjointtensorindex(tB,n), oindB)
        cindB = map(n->adjointtensorindex(tB,n), cindB)
        contract!(α, tA, adjoint(tB), β, tC, oindA, cindA, oindB, cindB, p1, p2)
    elseif VA == Val{:C} && VB == Val{:N}
        oindA = map(n->adjointtensorindex(tA,n), oindA)
        cindA = map(n->adjointtensorindex(tA,n), cindA)
        contract!(α, adjoint(tA), tB, β, tC, oindA, cindA, oindB, cindB, p1, p2)
    else
        oindA = map(n->adjointtensorindex(tA,n), oindA)
        cindA = map(n->adjointtensorindex(tA,n), cindA)
        oindB = map(n->adjointtensorindex(tB,n), oindB)
        cindB = map(n->adjointtensorindex(tB,n), cindB)
        contract!(α, adjoint(tA), adjoint(tB), β, tC, oindA, cindA, oindB, cindB, p1, p2)
    end
    return tC
end
