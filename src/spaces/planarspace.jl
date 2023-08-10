struct PlanarNumbers end # this is probably not a field?
const ℙ = PlanarNumbers()
Base.show(io::IO, ::PlanarNumbers) = print(io, "ℙ")

# convenience constructor
Base.:^(::PlanarNumbers, d::Int) = Vect[PlanarTrivial](PlanarTrivial() => d)

# convenience show
function Base.show(io::IO, V::GradedSpace{PlanarTrivial})
    return print(io, isdual(V) ? "(ℙ^$(dim(V)))'" : "ℙ^$(dim(V))")
end