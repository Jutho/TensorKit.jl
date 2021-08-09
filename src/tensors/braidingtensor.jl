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
        push!(data, c=>f((dim1, dim2)))
        push!(rowr, c=>rowrc)
        push!(colr, c=>colrc)
    end
    return TensorKeyIterator(rowr, colr)
end

@inline function Base.getindex(b::BraidingTensor, f1::FusionTree{I,2}, f2::FusionTree{I,2}) where {I<:Sector}
    I == sectortype(b) || throw(SectorMismatch())
    c = f1.coupled
    V1, V2 = domain(b)
    @boundscheck begin
        c == f2.coupled || throw(SectorMismatch())
        ((f1.uncoupled[1] ∈ V2) && (f2.uncoupled[1] ∈ V1)) || throw(SectorMismatch())
        ((f1.uncoupled[2] ∈ V1) && (f2.uncoupled[2] ∈ V2)) || throw(SectorMismatch())
    end
    @inbounds begin
        d = (dims(V2 ⊗ V1, f1.uncoupled)..., dims(V1 ⊗ V2, f2.uncoupled)...)
        n1 = d[1]*d[2]
        n2 = d[3]*d[4]
        a1, a2 = f2.uncoupled
        data = fill!(storagetype(b)(undef, (n1, n2)), zero(eltype(b)))
        if f1.uncoupled == (a2, a1)
            braiddict = artin_braid(f2, 1; inv = b.adjoint)
            r = get(dict, f1, zero(valtype(braiddict)))
            data[1:(n1+1):end] .= r # set diagonal
        end
        return permutedims(sreshape(StridedView(data), d), (1,2,4,3))
    end
end

Base.copy(b::BraidingTensor) = copy!(similar(b), b)
function Base.copy!(t::TensorMap, b::BraidingTensor)
    space(t) == space(b) || throw(SectorMismatch())
    fill!(t, zero(eltype(t)))
    for (f1, f2) in fusiontrees(t)
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
