using Test
using Random
using LinearAlgebra
using Combinatorics
using TensorKit
using TensorKit: ProductSector, fusiontensor
using TensorOperations
TensorOperations.disable_cache() # avoids memory overflow during CI?
using TupleTools
using TupleTools: StaticLength
using Base.Iterators: take, product

include("timedtest.jl")

using .TimedTests

const TK = TensorKit

Random.seed!(1234)

smallset(::Type{G}) where {G<:Sector} = take(values(G), 5)
function smallset(::Type{ProductSector{Tuple{G1,G2}}}) where {G1,G2}
    iter = product(smallset(G1),smallset(G2))
    s = collect(i × j for (i,j) in iter if dim(i)*dim(j) <= 6)
    return length(s) > 6 ? rand(s, 6) : s
end
function smallset(::Type{ProductSector{Tuple{G1,G2,G3}}}) where {G1,G2,G3}
    iter = product(smallset(G1),smallset(G2),smallset(G3))
    s = collect(i × j × k for (i,j,k) in iter if dim(i)*dim(j)*dim(k) <= 6)
    return length(s) > 6 ? rand(s, 6) : s
end
function randsector(::Type{G}) where {G<:Sector}
    s = collect(smallset(G))
    a = rand(s)
    while a == one(a) # don't use trivial label
        a = rand(s)
    end
    return a
end
function hasfusiontensor(G::Type{<:Sector})
    try
        fusiontensor(one(G), one(G), one(G))
        return true
    catch
        return false
    end
end

Ti = time()
include("sectors.jl")
include("fusiontrees.jl")
include("spaces.jl")
include("tensors.jl")
Tf = time()
printstyled("Finished all tests in ",
            string(round((Tf-Ti)/60; sigdigits=3)),
            " minutes."; bold = true, color = Base.info_color())
println()
