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

Base.show(io::IO, ::Type{ℤ₂}) = print(io, "ℤ₂")
Base.show(io::IO, ::Type{ℤ₃}) = print(io, "ℤ₃")
Base.show(io::IO, ::Type{ℤ₄}) = print(io, "ℤ₄")
Base.show(io::IO, ::Type{SU₂}) = print(io, "SU₂")

const GroupTuple = Tuple{Vararg{Group}}

abstract type ProductGroup{T<:GroupTuple} <: Group end

×(G1::Type{ProductGroup{Tuple{}}},
                    G2::Type{ProductGroup{T}}) where {T<:GroupTuple} = G2
×(G1::Type{ProductGroup{T1}},
                    G2::Type{ProductGroup{T2}}) where {T1<:GroupTuple, T2<:GroupTuple} =
    tuple_type_head(T1) × (ProductGroup{tuple_type_tail(T1)} × G2)
×(G1::Type{ProductGroup{Tuple{}}}, G2::Type{<:Group}) =
    ProductGroup{Tuple{G2}}
×(G1::Type{ProductGroup{T}}, G2::Type{<:Group}) where {T<:GroupTuple} =
    Base.tuple_type_head(T) × (ProductGroup{Base.tuple_type_tail(T)} × G2)
×(G1::Type{<:Group}, G2::Type{ProductGroup{T}}) where {T<:GroupTuple} =
    ProductGroup{Base.tuple_type_cons(G1, T)}
×(G1::Type{<:Group}, G2::Type{<:Group}) = ProductGroup{Tuple{G1, G2}}

function Base.show(io::IO, ::Type{ProductGroup{T}}) where {T<:GroupTuple}
    sectors = T.parameters
    print(io, "(")
    for i = 1:length(sectors)
        i == 1 || print(io, " × ")
        print(io, sectors[i])
    end
    print(io, ")")
end
