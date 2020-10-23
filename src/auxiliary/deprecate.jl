Base.@deprecate min(V1::ElementarySpace, V2::ElementarySpace) infimum(V1, V2)
Base.@deprecate max(V1::ElementarySpace, V2::ElementarySpace) supremum(V1, V2)
Base.@deprecate(infinum(V1, V2...), infimum(V1, V2...))

Base.@deprecate(fusiontreetype(::Type{I}, ::StaticLength{N}) where {I<:Sector, N},
    fusiontreetype(I, N))

abstract type RepresentationSpace{I<:Sector} end
Base.@deprecate(RepresentationSpace(args...), GradedSpace(args...))
Base.@deprecate(RepresentationSpace{I}(args...) where {I}, Vect[I](args...))

Base.@deprecate(×(a::Sector, b::Sector), ⊠(a,b))

Base.@deprecate(
    permuteind(t::TensorMap, p1::IndexTuple, p2::IndexTuple=(); copy::Bool = false),
    permute(t, p1, p2; copy = copy))

@noinline function Base.getindex(::ComplexNumbers, I::Type{<:Group})
    S = Rep[I]
    if repr(S) == "GradedSpace[Irrep[$I]]"
        Base.depwarn("`ℂ[$I]` is deprecated because $I represents a group rather than its representations, use `ℂ[Irrep[$I]]`, `Rep[$I]` or `GradedSpace[Irrep[$I]]` instead.",
        ((Base.Core).Typeof(Base.getindex)).name.mt.name)
    else
        Base.depwarn("`ℂ[$I]` is deprecated because $I represents a group rather than its representations, use `ℂ[Irrep[$I]]`, `Rep[$I]`, `$S` or `GradedSpace[Irrep[$I]]` instead.",
        ((Base.Core).Typeof(Base.getindex)).name.mt.name)
    end
    return Rep[I]
end

@noinline function ℤ₂(args...)
    Base.depwarn("`ℤ₂(args...)` is deprecated, use `Z2Irrep(args...)` or ``Irrep[ℤ₂](args...)` instead.", ((Base.Core).Typeof(ℤ₂)).name.mt.name)
    Irrep[ℤ₂](args...)
end
@noinline function ℤ₃(args...)
    Base.depwarn("`ℤ₃(args...)` is deprecated, use `Z3Irrep(args...)` or ``Irrep[ℤ₃](args...)` instead.", ((Base.Core).Typeof(ℤ₃)).name.mt.name)
    Irrep[ℤ₃](args...)
end
@noinline function ℤ₄(args...)
    Base.depwarn("`ℤ₄(args...)` is deprecated, use `Z4Irrep(args...)` or ``Irrep[ℤ₄](args...)` instead.", ((Base.Core).Typeof(ℤ₄)).name.mt.name)
    Irrep[ℤ₄](args...)
end
@noinline function U₁(args...)
    Base.depwarn("`U₁(args...)` is deprecated, use `U1Irrep(args...)` or ``Irrep[U₁](args...)` instead.", ((Base.Core).Typeof(U₁)).name.mt.name)
    Irrep[U₁](args...)
end
@noinline function CU₁(args...)
    Base.depwarn("`CU₁(args...)` is deprecated, use `CU1Irrep(args...)` or ``Irrep[CU₁](args...)` instead.", ((Base.Core).Typeof(CU₁)).name.mt.name)
    Irrep[CU₁](args...)
end
@noinline function SU₂(args...)
    Base.depwarn("`SU₂(args...)` is deprecated, use `SU2Irrep(args...)` or ``Irrep[SU₂](args...)` instead.", ((Base.Core).Typeof(SU₂)).name.mt.name)
    Irrep[SU₂](args...)
end

Base.@deprecate(permuteind!(tdst::AbstractTensorMap, tsrc::AbstractTensorMap, p1, p2),
                permute!(tdst, tsrc, p1, p2))

function Base.getindex(::Type{GradedSpace}, ::Type{I}) where {I<:Sector}
    @warn "`getindex(::Type{GradedSpace}, I::Type{<:Sector})` is deprecated, use `ℂ[I]`, `Vect[I]`, or, if `I == Irrep[G]`, `Rep[G]` instead." maxlog = 1
    return Vect[I]
end

function Base.getindex(::ComplexNumbers, d1::Pair{I, Int}, dims::Pair{I, Int}...) where {I<:Sector}
    @warn "`ℂ[s1=>n1, s2=>n2, ...]` is deprecated, use `ℂ[I](s1=>n1, s2=>n2, ...)`, `Vect[I](s1=>n1, s2=>n2, ...)` instead with `I = typeof(s1)`." maxlog = 1
    return Vect[I](d1, dims...)
end

function Base.getindex(::RealNumbers, d::Int)
    @warn "`ℝ[d]` is deprecated, use `ℝ^d` or `CartesianSpace(d)`." maxlog = 1
    return ℝ^d
end

function Base.getindex(::ComplexNumbers, d::Int)
    @warn "`ℂ[d]` is deprecated, use `ℂ^d` or `ComplexSpace(d)`." maxlog = 1
    return ℂ^d
end
