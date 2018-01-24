# Add support for TensorOperations.jl
function similar_from_indices(::Type{T}, p1::IndexTuple, p2::IndexTuple, t::AbstractTensorMap) where {T}
    s = codomain(t) ⊗ dual(domain(t))
    cod = s[map(n->tensor2spaceindex(t,n), p1)]
    dom = dual(s[map(n->tensor2spaceindex(t,n), reverse(p2))])
    return similar(t, T, cod←dom)
end
function similar_from_indices(::Type{T}, oindA::IndexTuple, oindB::IndexTuple, p1::IndexTuple, p2::IndexTuple, tA::AbstractTensorMap{S}, tB::AbstractTensorMap{S}) where {T, S<:IndexSpace}
    sA = codomain(tA) ⊗ dual(domain(tA))
    sB = codomain(tB) ⊗ dual(domain(tB))
    s = sA[map(n->tensor2spaceindex(tA,n), oindA)] ⊗ sB[map(n->tensor2spaceindex(tB,n), oindB)]
    cod = s[p1]
    dom = dual(s[reverse(p2)])
    return similar(tA, T, cod←dom)
end

function unsafe_similar_from_indices(::Type{T}, p1::IndexTuple, p2::IndexTuple, t::AbstractTensorMap) where {T}
    s = codomain(t) ⊗ dual(domain(t))
    cod = s[map(n->tensor2spaceindex(t,n), p1)]
    dom = dual(s[map(n->tensor2spaceindex(t,n), reverse(p2))])
    return unsafe_similar(t, T, cod←dom)
end
function unsafe_similar_from_indices(::Type{T}, oindA::IndexTuple, oindB::IndexTuple, p1::IndexTuple, p2::IndexTuple, tA::AbstractTensorMap{S}, tB::AbstractTensorMap{S}) where {T, S<:IndexSpace}
    sA = codomain(tA) ⊗ dual(domain(tA))
    sB = codomain(tB) ⊗ dual(domain(tB))
    s = sA[map(n->tensor2spaceindex(tA,n), oindA)] ⊗ sB[map(n->tensor2spaceindex(tB,n), oindB)]
    cod = s[p1]
    dom = dual(s[reverse(p2)])
    return unsafe_similar(tA, T, cod←dom)
end
function unsafe_permuteind(t::AbstractTensorMap{S}, p1::IndexTuple{N₁},  p2::IndexTuple{N₂}=()) where {S,N₁,N₂}
    # share data if possible
    if (p1..., p2...) == ntuple(identity, StaticLength(N₁)+StaticLength(N₂))
        if isa(t, TensorMap{S,N₁,N₂})
            return t, false
        elseif isa(t, TensorMap) && S == Trivial
            spacet = codomain(t) ⊗ dual(domain(t))
            cod = spacet[map(n->tensor2spaceindex(t,n), p1)]
            dom = dual(spacet[map(n->tensor2spaceindex(t,n), reverse(p2))])
            return TensorMap(reshape(t.data, dim(cod), dim(dom)), cod, dom), false
        elseif isa(t, AdjointTensorMap)
            # TODO: can we share data for AdjointTensorMap?
        end
    end
    # general case
    @inbounds begin
        tnew = unsafe_similar_from_indices(eltype(t), p1, p2, t)
        return permuteind!(tnew, t, p1, p2), true
    end
end

scalar(t::AbstractTensorMap{S}) where {S<:IndexSpace} = dim(codomain(t)) == dim(domain(t)) == 1 ? first(blocks(t))[2][1,1] : throw(SpaceMismatch())

function add!(α, tsrc::AbstractTensorMap{S}, β, tdst::AbstractTensorMap{S,N₁,N₂}, p1::IndexTuple{N₁}, p2::IndexTuple{N₂}) where {S,N₁,N₂}
    # TODO: check Frobenius-Schur indicators!, and  add fermions!
    @boundscheck begin
        all(i->space(tsrc, p1[i]) == space(tdst,i), 1:N₁) || throw(SpaceMismatch("tsrc = $(codomain(tsrc))←$(domain(tsrc)), tdst = $(codomain(tdst))←$(domain(tdst)), p1 = $(p1), p2 = $(p2)"))
        all(i->space(tsrc, p2[i]) == space(tdst,N₁+i), 1:N₂) || throw(SpaceMismatch("tsrc = $(codomain(tsrc))←$(domain(tsrc)), tdst = $(codomain(tdst))←$(domain(tdst)), p1 = $(p1), p2 = $(p2)"))
    end

    cod = codomain(tsrc)
    dom = domain(tsrc)
    n = length(cod)

    pdata = (p1...,p2...)
    G = sectortype(S)
    if G == Trivial
        if iszero(β)
            scale!(tdst[], α, permutedims(tsrc[], pdata))
        elseif β == one(β)
            axpy!(α, permutedims(tsrc[], pdata), tdst[])
        else
            axpby!(α, permutedims(tsrc[], pdata), β, tdst[])
        end
    elseif fusiontype(G) == Abelian
        @inbounds for (f1,f2) in fusiontrees(tsrc)
            (f1′,f2′), coeff = first(permute(f1, f2, p1, p2))
            for i in p2
                if i <= n && !isdual(cod[i])
                    b = f1.outgoing[i]
                    coeff *= frobeniusschur(b) #*fermionparity(b)
                end
            end
            for i in p1
                if i > n && isdual(dom[i-n])
                    b = f2.outgoing[i-n]
                    coeff /= frobeniusschur(b) #*fermionparity(b)
                end
            end
            if iszero(β)
                scale!(tdst[f1′,f2′], α*coeff, permutedims(tsrc[f1,f2], pdata))
            elseif β == one(β)
                axpy!(α*coeff, permutedims(tsrc[f1,f2], pdata), tdst[f1′,f2′])
            else
                axpby!(α*coeff, permutedims(tsrc[f1,f2], pdata), β, tdst[f1′,f2′])
            end
        end
    else
        if iszero(β)
            fill!(tdst, β)
        elseif β != 1
            scale!(tdst, β)
        end
        @inbounds for (f1,f2) in fusiontrees(tsrc)
            for ((f1′,f2′), coeff) in permute(f1, f2, p1, p2)
                for i in p2
                    if i <= n && !isdual(cod[i])
                        b = f1.outgoing[i]
                        coeff *= frobeniusschur(b) #*fermionparity(b)
                    end
                end
                for i in p1
                    if i > n && isdual(dom[i-n])
                        b = f2.outgoing[i-n]
                        coeff /= frobeniusschur(b) #*fermionparity(b)
                    end
                end
                axpy!(α*coeff, permutedims(tsrc[f1,f2], pdata), tdst[f1′,f2′])
            end
        end
    end
    return tdst
end

# function contract!(α, A::AbstractTensorMap{S}, B::AbstractTensorMap{S}, β, C::AbstractTensorMap{S}, oindA::IndexTuple{N₁}, cindA::IndexTuple, oindB::IndexTuple{N₂}, cindB::IndexTuple, p1::IndexTuple, p2::IndexTuple) where {S<:IndexSpace,N₁,N₂}
#     if fusiontype(sectortype(S)) == Abelian
#         for (fA1,fA2) in fusiontrees(A)
#             ((fA1′,fA2′), coeff1), = permute(fA1,fA2,oindA,cindA)
#             for (fB1,fB2) in fusiontrees(B)
#                 ((fB1′,fB2′), coeff2), = permute(fB1,fB2,cindB,oindB)
#                 if fA2′ == fB1′
#                     ((fC1,fC2), coeff3), = permute(fA1′,fB2′,p1,p2)
#                     TensorOperations.contract!(α*coeff1*coeff2*coeff3, A[fA1,fA2], Val{:N}, B[fB1,fB2], Val{:N}, β, C[fC1,fC2], oindA, cindA, oindB, cindB, p1, p2)
#                 end
#             end
#         end
#     else
#         A′ = permuteind(A, oindA, cindA)
#         B′ = permuteind(B, cindB, oindB)
#         if α == 1 && β == 0 && p1 == ntuple(n->n, StaticLength(N₁)) && p2 == ntuple(n->(N₁+n), StaticLength(N₂))
#             A_mul_B!(C, A′, B′)
#         elseif α == 1 && β == 0 && isa(C, TensorMap) && sectortype(S) == Trivial && (p1...,p2...) == ntuple(n->n, StaticLength(N₁)+StaticLength(N₂))
#             p1′ = ntuple(n->n, StaticLength(N₁))
#             p2′ = ntuple(n->(N₁+n), StaticLength(N₂))
#             C′ = permuteind(C, p1′, p2′)
#             A_mul_B!(C′, A′, B′)
#         else
#             add!(α, A′ * B′, β, C, p1, p2)
#         end
#     end
#     return C
# end

function contract!(α, A::AbstractTensorMap{S}, B::AbstractTensorMap{S}, β, C::AbstractTensorMap{S}, oindA::IndexTuple{N₁}, cindA::IndexTuple, oindB::IndexTuple{N₂}, cindB::IndexTuple, p1::IndexTuple, p2::IndexTuple) where {S<:IndexSpace,N₁,N₂}
    A′ = permuteind(A, oindA, cindA)
    B′ = permuteind(B, cindB, oindB)
    if α == 1 && β == 0 && p1 == ntuple(n->n, StaticLength(N₁)) && p2 == ntuple(n->(N₁+n), StaticLength(N₂))
        mul!(C, A′, B′)
    elseif α == 1 && β == 0 && isa(C, TensorMap) && sectortype(S) == Trivial && (p1...,p2...) == ntuple(n->n, StaticLength(N₁)+StaticLength(N₂))
        p1′ = ntuple(n->n, StaticLength(N₁))
        p2′ = ntuple(n->(N₁+n), StaticLength(N₂))
        C′ = permuteind(C, p1′, p2′)
        mul!(C′, A′, B′)
    else
        C′ = A′ * B′
        add!(α, C′, β, C, p1, p2)
        # finalize(C′)
    end
    # finalize(A′)
    # finalize(B′)
    return C
end
#
# function contract!(α, A::AbstractTensorMap{S}, B::AbstractTensorMap{S}, β, C::AbstractTensorMap{S}, oindA::IndexTuple{N₁}, cindA::IndexTuple, oindB::IndexTuple{N₂}, cindB::IndexTuple, p1::IndexTuple, p2::IndexTuple) where {S<:IndexSpace,N₁,N₂}
#     A′, freeA = unsafe_permuteind(A, oindA, cindA)
#     B′, freeB = unsafe_permuteind(B, cindB, oindB)
#     if α == 1 && β == 0 && p1 == ntuple(n->n, StaticLength(N₁)) && p2 == ntuple(n->(N₁+n), StaticLength(N₂))
#         mul!(C, A′, B′)
#     elseif α == 1 && β == 0 && isa(C, TensorMap) && sectortype(S) == Trivial && (p1...,p2...) == ntuple(n->n, StaticLength(N₁)+StaticLength(N₂))
#         p1′ = ntuple(n->n, StaticLength(N₁))
#         p2′ = ntuple(n->(N₁+n), StaticLength(N₂))
#         C′ = permuteind(C, p1′, p2′)
#         mul!(C′, A′, B′)
#     else
#         C′ = unsafe_similar(A, promote_type(eltype(A), eltype(B)), codomain(A′)←domain(B′))
#         mul!(C′, A′, B′)
#         add!(α, C′, β, C, p1, p2)
#         unsafe_free(C′)
#     end
#     if freeA
#         unsafe_free(A′)
#     end
#     if freeB
#         unsafe_free(B′)
#     end
#     return C
# end

# Compatibility layer for working with the `@tensor` macro from TensorOperations
TensorOperations.numind(t::AbstractTensorMap) = numind(t)
TensorOperations.numind(T::Type{<:AbstractTensorMap}) = numind(T)

TensorOperations.similar_from_indices(T::Type, p::IndexTuple, t::AbstractTensorMap, V::Type{<:Val}) = TensorOperations.similar_from_indices(T, p, (), t, V)
TensorOperations.similar_from_indices(T::Type, oindA::IndexTuple, oindB::IndexTuple, p::IndexTuple, tA::AbstractTensorMap{S}, tB::AbstractTensorMap{S}, VA::Type{<:Val}, VB::Type{<:Val}) where {S} = TensorOperations.similar_from_indices(T, oindA, oindB, p, (), ta, tb, VA, VB)

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

TensorOperations.scalar(t::AbstractTensorMap) = scalar(t)

function TensorOperations.add!(α, tsrc::AbstractTensorMap{S}, V::Type{<:Val}, β, tdst::AbstractTensorMap{S,N₁,N₂}, p1::IndexTuple, p2::IndexTuple = ()) where {S,N₁,N₂}
    p = (p1..., p2...)
    if V == Val{:N}
        pl = ntuple(n->p[n], StaticLength(N₁))
        pr = ntuple(n->p[N₁+n], StaticLength(N₂))
        add!(α, tsrc, β, tdst, pl, pr)
    else
        pl = ntuple(n->adjointtensorindex(tsrc, p[n]), StaticLength(N₁))
        pr = ntuple(n->adjointtensorindex(tsrc, p[N₁+n]), StaticLength(N₂))
        add!(α, adjoint(tsrc), β, tdst, pl, pr)
    end
    return tdst
end

TensorOperations.contract!(α, tA::AbstractTensorMap{S}, VA::Type{<:Val}, tB::AbstractTensorMap{S}, VB::Type{<:Val}, β, tC::AbstractTensorMap{S}, oindA::IndexTuple, cindA::IndexTuple, oindB::IndexTuple, cindB::IndexTuple, p::IndexTuple, ::Type{Val{:BLAS}}) where {S} = TensorOperations.contract!(α, tA, VA, tB, VB, β, tC, oindA, cindA, oindB, cindB, p)

function TensorOperations.contract!(α, tA::AbstractTensorMap{S}, VA::Type{<:Val}, tB::AbstractTensorMap{S}, VB::Type{<:Val}, β, tC::AbstractTensorMap{S,N₁,N₂}, oindA::IndexTuple, cindA::IndexTuple, oindB::IndexTuple, cindB::IndexTuple, p1::IndexTuple, p2::IndexTuple = ()) where {S,N₁,N₂}
    p = (p1..., p2...)
    pl = ntuple(n->p[n], StaticLength(N₁))
    pr = ntuple(n->p[N₁+n], StaticLength(N₂))
    if VA == Val{:N} && VB == Val{:N}
        contract!(α, tA, tB, β, tC, oindA, cindA, oindB, cindB, pl, pr)
    elseif VA == Val{:N} && VB == Val{:C}
        oindB = map(n->adjointtensorindex(tB,n), oindB)
        cindB = map(n->adjointtensorindex(tB,n), cindB)
        contract!(α, tA, adjoint(tB), β, tC, oindA, cindA, oindB, cindB, pl, pr)
    elseif VA == Val{:C} && VB == Val{:N}
        oindA = map(n->adjointtensorindex(tA,n), oindA)
        cindA = map(n->adjointtensorindex(tA,n), cindA)
        contract!(α, adjoint(tA), tB, β, tC, oindA, cindA, oindB, cindB, pl, pr)
    else
        oindA = map(n->adjointtensorindex(tA,n), oindA)
        cindA = map(n->adjointtensorindex(tA,n), cindA)
        oindB = map(n->adjointtensorindex(tB,n), oindB)
        cindB = map(n->adjointtensorindex(tB,n), cindB)
        contract!(α, adjoint(tA), adjoint(tB), β, tC, oindA, cindA, oindB, cindB, pl, pr)
    end
    return tC
end
