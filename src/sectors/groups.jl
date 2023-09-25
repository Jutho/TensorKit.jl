# Groups
#------------------------------------------------------------------------------#
abstract type Group end
abstract type AbelianGroup <: Group end

abstract type ℤ{N} <: AbelianGroup end
abstract type U₁ <: AbelianGroup end
abstract type SU{N} <: Group end
abstract type CU₁ <: Group end

const ℤ₂ = ℤ{2}
const ℤ₃ = ℤ{3}
const ℤ₄ = ℤ{4}
const SU₂ = SU{2}

type_repr(::Type{ℤ₂}) = "ℤ₂"
type_repr(::Type{ℤ₃}) = "ℤ₃"
type_repr(::Type{ℤ₄}) = "ℤ₄"
type_repr(::Type{SU₂}) = "SU₂"

const GroupTuple = Tuple{Vararg{Group}}

abstract type ProductGroup{T<:GroupTuple} <: Group end

×(a::Type{<:Group}, b::Type{<:Group}, c::Type{<:Group}...) = ×(×(a, b), c...)
×(G::Type{<:Group}) = ProductGroup{Tuple{G}}
×(G1::Type{ProductGroup{Tuple{}}},
G2::Type{ProductGroup{T}}) where {T<:GroupTuple} = G2
function ×(G1::Type{ProductGroup{T1}},
           G2::Type{ProductGroup{T2}}) where {T1<:GroupTuple,T2<:GroupTuple}
    return tuple_type_head(T1) × (ProductGroup{tuple_type_tail(T1)} × G2)
end
×(G1::Type{ProductGroup{Tuple{}}}, G2::Type{<:Group}) = ProductGroup{Tuple{G2}}
function ×(G1::Type{ProductGroup{T}}, G2::Type{<:Group}) where {T<:GroupTuple}
    return Base.tuple_type_head(T) × (ProductGroup{Base.tuple_type_tail(T)} × G2)
end
function ×(G1::Type{<:Group}, G2::Type{ProductGroup{T}}) where {T<:GroupTuple}
    return ProductGroup{Base.tuple_type_cons(G1, T)}
end
×(G1::Type{<:Group}, G2::Type{<:Group}) = ProductGroup{Tuple{G1,G2}}

function type_repr(G::Type{<:ProductGroup})
    T = G.parameters[1]
    groups = T.parameters
    if length(groups) == 1
        s = "ProductGroup{Tuple{" * type_repr(groups[1]) * "}}"
    else
        s = "("
        for i in 1:length(groups)
            if i != 1
                s *= " × "
            end
            s *= type_repr(groups[i])
        end
        s *= ")"
    end
    return s
end
