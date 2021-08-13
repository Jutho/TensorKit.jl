# BraidingTensor:
# special (2,2) tensor that implements a standard braiding operation
#====================================================================#
"""
    struct BraidingTensor{S<:IndexSpace} <: AbstractTensorMap{S, 2, 2}

Specific subtype of [`AbstractTensorMap`](@ref) for representing the braiding tensor that
braids the first input over the second input; its inverse can be obtained as the adjoint.
"""
struct BraidingTensor{S<:IndexSpace, A} <: AbstractTensorMap{S, 2, 2}
    V1::S
    V2::S
    adjoint::Bool
    function BraidingTensor(V1::S, V2::S, adjoint::Bool = false, ::Type{A} = Matrix{ComplexF64}) where
                {S<:IndexSpace, A<:DenseMatrix}
        for a in sectors(V1)
            for b in sectors(V2)
                for c in (a ⊗ b)
                    Nsymbol(a, b, c) == Nsymbol(b, a, c) ||
                        throw(ArgumentError("Cannot define a braiding between $a and $b"))
                end
            end
        end
        return new{S,A}(V1, V2, adjoint)
        # partial construction: only construct rowr and colr when needed
    end
end

Base.adjoint(b::BraidingTensor{S,A}) where {S<:IndexSpace, A<:DenseMatrix} =
    BraidingTensor(b.V1, b.V2, !b.adjoint, A)

domain(b::BraidingTensor) = b.adjoint ? b.V2 ⊗ b.V1 : b.V1 ⊗ b.V2
codomain(b::BraidingTensor) = b.adjoint ? b.V1 ⊗ b.V2 : b.V2 ⊗ b.V1

storagetype(::Type{BraidingTensor{S,A}}) where {S<:IndexSpace, A<:DenseMatrix} = A

blocksectors(b::BraidingTensor) = blocksectors(b.V1 ⊗ b.V2)
hasblock(b::BraidingTensor, s::Sector) = s ∈ blocksectors(b)

function fusiontrees(b::BraidingTensor)
    codom = codomain(b)
    dom = domain(b)
    I = sectortype(b)
    F = fusiontreetype(I, 2)
    rowr = SectorDict{I, FusionTreeDict{F, UnitRange{Int}}}()
    colr = SectorDict{I, FusionTreeDict{F, UnitRange{Int}}}()
    for c in blocksectors(codom)
        rowrc = FusionTreeDict{F, UnitRange{Int}}()
        colrc = FusionTreeDict{F, UnitRange{Int}}()
        offset1 = 0
        for s1 in sectors(codom)
            for f1 in fusiontrees(s1, c, map(isdual, codom.spaces))
                r = (offset1 + 1):(offset1 + dim(codom, s1))
                push!(rowrc, f1 => r)
                offset1 = last(r)
            end
        end
        dim1 = offset1
        offset2 = 0
        for s2 in sectors(dom)
            for f2 in fusiontrees(s2, c, map(isdual, dom.spaces))
                r = (offset2 + 1):(offset2 + dim(dom, s2))
                push!(colrc, f2 => r)
                offset2 = last(r)
            end
        end
        dim2 = offset2
        push!(rowr, c=>rowrc)
        push!(colr, c=>colrc)
    end
    return TensorKeyIterator(rowr, colr)
end

function Base.getindex(b::BraidingTensor{S}) where S
    sectortype(S) == Trivial || throw(SectorMismatch())
    (V1,V2) = domain(b);
    return reshape(storagetype(b)(LinearAlgebra.I, dim(V1)*dim(V2), dim(V1)*dim(V2)),(dim(V2),dim(V1),dim(V1),dim(V2)));
end

@inline function Base.getindex(b::BraidingTensor, f1::FusionTree{I,2}, f2::FusionTree{I,2}) where {I<:Sector}
    I == sectortype(b) || throw(SectorMismatch())
    c = f1.coupled
    V1, V2 = domain(b)
    @boundscheck begin
        c == f2.coupled || throw(SectorMismatch())
        ((f1.uncoupled[1] ∈ sectors(V2)) && (f2.uncoupled[1] ∈ sectors(V1))) || throw(SectorMismatch())
        ((f1.uncoupled[2] ∈ sectors(V1)) && (f2.uncoupled[2] ∈ sectors(V2))) || throw(SectorMismatch())
    end
    @inbounds begin
        d = (dims(V2 ⊗ V1, f1.uncoupled)..., dims(V1 ⊗ V2, f2.uncoupled)...)
        n1 = d[1]*d[2]
        n2 = d[3]*d[4]
        a1, a2 = f2.uncoupled
        data = fill!(storagetype(b)(undef, (n1, n2)), zero(eltype(b)))
        if f1.uncoupled == (a2, a1)
            braiddict = artin_braid(f2, 1; inv = b.adjoint)
            r = get(braiddict, f1, zero(valtype(braiddict)))
            data[1:(n1+1):end] .= r # set diagonal
        end

        return sreshape(StridedView(data), d)
    end
end

Base.copy(b::BraidingTensor) = copy!(similar(b), b)
function Base.copy!(t::TensorMap, b::BraidingTensor)
    space(t) == space(b) || throw(SectorMismatch())
    fill!(t, zero(eltype(t)))
    for (f1, f2) in fusiontrees(t)
        if f1 == nothing || f2 == nothing
            copy!(t.data,one(t.data))
            return t
        end

        a1, a2 = f2.uncoupled
        c = f2.coupled
        f1.uncoupled == (a2, a1) || continue
        data = t[f1, f2]
        r = artin_braid(f2, 1; inv = b.adjoint)[f1]
        for i = 1:size(data, 1), j = 1:size(data, 2)
            data[i, j, j, i] = r
        end
    end
    return t
end
TensorMap(b::BraidingTensor) = copy(b)
Base.convert(::Type{TensorMap}, b::BraidingTensor) = copy(b)

function block(b::BraidingTensor, s::Sector)
    sectortype(b) == typeof(s) || throw(SectorMismatch())
    return block(copy(b), s) # stopgap awaiting actual implementation

    # V1, V2 = codomain(b)
    # n = blockdim(V1 ⊗ V2, s)
    # data = fill!(storagetype(b)(undef, (n, n)), eltype(b))
    # n == 0 && return data
    # for (f1, f2) in fusiontrees(b)
    #     f2.coupled == s || continue
    #     a1, a2 = f2.uncoupled
    #
end

function planar_contract!(α, A::BraidingTensor, B::AbstractTensorMap{S},
                            β, C::AbstractTensorMap{S},
                            oindA::IndexTuple{2}, cindA::IndexTuple{2},
                            oindB::IndexTuple{N₂}, cindB::IndexTuple,
                            p1::IndexTuple, p2::IndexTuple,
                            syms::Union{Nothing, NTuple{3, Symbol}}) where {S, N₂}
    codA = codomainind(A)
    domA = domainind(A)
    codB = codomainind(B)
    domB = domainind(B)
    oindA, cindA, oindB, cindB =  reorder_indices(codA, domA, codB, domB, oindA, cindA, oindB, cindB, p1, p2)

    braidingtensor_levels = A.adjoint ? (2,1,1,2) : (1,2,2,1);

    if iszero(β)
        fill!(C, β)
    elseif β != 1
        rmul!(C, β)
    end

    for (f1, f2) in fusiontrees(B)
        if f1 == nothing && f2 == nothing
            return TO.add!(α, B,:N,true,C, reverse(cindB),oindB);
        end

        fmap = Dict{Tuple{typeof(f1),typeof(f2)},eltype(f1)}();

        #transpose
        for ((f1′,f2′),coeff) in transpose(f1,f2,cindB,oindB)

            #artin braid
            braid_above = braidingtensor_levels[cindA[1]] > braidingtensor_levels[cindA[2]];
            for (f1′′,coeff′) in artin_braid(f1′,1,inv = braid_above)
                nk = (f1′′,f2′);
                nv = coeff′*coeff;

                fmap[nk] = get(fmap,nk,zero(nv)) + nv;
            end
        end

        for ((f1′,f2′),c) in fmap
            TO._add!(c*α, B[f1, f2], true,C[f1′,f2′], (reverse(cindB)...,oindB...));
        end
    end
    C
end
function planar_contract!(α, A::AbstractTensorMap{S}, B::BraidingTensor,
                            β, C::AbstractTensorMap{S},
                            oindA::IndexTuple{N₁}, cindA::IndexTuple{N₁},
                            oindB::IndexTuple{2}, cindB::IndexTuple{2},
                            p1::IndexTuple, p2::IndexTuple,
                            syms::Union{Nothing, NTuple{3, Symbol}}) where {S, N₁}
    codA = codomainind(A)
    domA = domainind(A)
    codB = codomainind(B)
    domB = domainind(B)
    oindA, cindA, oindB, cindB =  reorder_indices(codA, domA, codB, domB, oindA, cindA, oindB, cindB, p1, p2)

    braidingtensor_levels = B.adjoint ? (2,1,1,2) : (1,2,2,1);

    if iszero(β)
        fill!(C, β)
    elseif β != 1
        rmul!(C, β)
    end

    for (f1, f2) in fusiontrees(A)
        if f1 == nothing && f2 == nothing
            return TO.add!(α, A,:N,true,C, oindA,reverse(cindA));
        end

        fmap = Dict{Tuple{typeof(f1),typeof(f2)},eltype(f1)}();

        #transpose
        for ((f1′,f2′),coeff) in transpose(f1,f2,oindA,cindA)

            #artin braid
            braid_above = braidingtensor_levels[cindB[1]] > braidingtensor_levels[cindB[2]];
            for (f2′′,coeff′) in artin_braid(f2′,1,inv = braid_above)
                nk = (f1′,f2′′);
                nv = coeff′*coeff;

                fmap[nk] = get(fmap,nk,zero(nv)) + nv;
            end
        end

        for ((f1′,f2′),c) in fmap
            TO._add!(c*α, A[f1, f2], true,C[f1′,f2′], (oindA...,reverse(cindA)...));
        end
    end
    C
end

function TensorKit.planar_contract!(α, A::BraidingTensor, B::AbstractTensorMap{S},
                            β, C::AbstractTensorMap{S},
                            oindA::IndexTuple{0}, cindA::IndexTuple{4},
                            oindB::IndexTuple{N₂}, cindB::IndexTuple,
                            p1::IndexTuple, p2::IndexTuple,
                            syms::Union{Nothing, NTuple{3, Symbol}}) where {S, N₂}

    braidingtensor_levels = A.adjoint ? (2,1,1,2) : (1,2,2,1);
    codA = codomainind(A)
    domA = domainind(A)
    codB = codomainind(B)
    domB = domainind(B)

    oindA, cindA, oindB, cindB =    reorder_indices(codA, domA, codB, domB, oindA, cindA, oindB, cindB, p1, p2)

    if iszero(β)
        fill!(C, β)
    elseif β != 1
        rmul!(C, β)
    end

    for (f1, f2) in fusiontrees(B)
        if f1 == nothing && f2 == nothing
            TO.trace!(α, B, :N,true,C, oindB, (),(cindB[1],cindB[2]), (cindB[3],cindB[4]))
            break;
        end

        local fmap;

        braid_above = braidingtensor_levels[cindB[2]] > braidingtensor_levels[cindB[3]];

        for ((f1′,f2′),coeff) in transpose(f1,f2,cindB,oindB), #transpose
            (f1′′,coeff′) in artin_braid(f1′,2,inv = braid_above), #artin braid
            (f1_tr1,c_tr1) in elementary_trace(f1′′, 1), #trace
            (f1_tr2,c_tr2) in elementary_trace(f1_tr1,1) #trace
            nk = (f1_tr2,f2′);
            nv = coeff*coeff′*c_tr1*c_tr2*α;
            if @isdefined fmap
                fmap[nk] = get(fmap,nk,zero(nv)) + nv;
            else
                fmap = Dict(nk=>nv);
            end

        end

        for ((f1′,f2′),c) in fmap
            TO._trace!(c, B[f1, f2], true,C[f1′,f2′], oindB, (cindB[1],cindB[2]), (cindB[3],cindB[4]))
        end
    end

    C
end

function TensorKit.planar_contract!(α, A::AbstractTensorMap{S}, B::BraidingTensor,
                            β, C::AbstractTensorMap{S},
                            oindA::IndexTuple{N₁}, cindA::IndexTuple,
                            oindB::IndexTuple{0}, cindB::IndexTuple{4},
                            p1::IndexTuple, p2::IndexTuple,
                            syms::Union{Nothing, NTuple{3, Symbol}}) where {S, N₁}

    braidingtensor_levels = B.adjoint ? (2,1,1,2) : (1,2,2,1);

    codA = codomainind(A)
    domA = domainind(A)
    codB = codomainind(B)
    domB = domainind(B)

    oindA, cindA, oindB, cindB =    reorder_indices(codA, domA, codB, domB, oindA, cindA, oindB, cindB, p1, p2)


    if iszero(β)
        fill!(C, β)
    elseif β != 1
        rmul!(C, β)
    end

    braid_above = braidingtensor_levels[cindA[2]] > braidingtensor_levels[cindA[3]];

    for (f1, f2) in fusiontrees(A)
        if f1 == nothing && f2 == nothing
            TO.trace!(α, A, :N,true,C, oindA , (),(cindA[1],cindA[2]), (cindA[3],cindA[4]))
            break;
        end

        local fmap;
        for ((f1′,f2′),coeff) in transpose(f1,f2,oindA,cindA), #transpose
            (f2′′,coeff′) in artin_braid(f2′,2,inv = braid_above), #artin braid
            (f2_tr1,c_tr1) in elementary_trace(f2′′, 1), #trace
            (f2_tr2,c_tr2) in elementary_trace(f2_tr1,1) #trace

            nk = (f1′,f2_tr2);
            nv = coeff*coeff′*c_tr1*c_tr2*α;
            if @isdefined fmap
                fmap[nk] = get(fmap,nk,zero(nv)) + nv;
            else
                fmap = Dict(nk=>nv);
            end
        end

        for ((f1′,f2′),c) in fmap
            TO._trace!(c, A[f1, f2], true,C[f1′,f2′], oindA , (cindA[1],cindA[2]), (cindA[3],cindA[4]))
        end

    end
    C
end
