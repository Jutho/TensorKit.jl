# additional interface to deal with IsingBimod Sector
#------------------------------------------------------------------------------

# make this a separate module?

function dim(V::Vect[IsingBimod])
    T = Base.promote_op(*, Int, real(sectorscalartype(sectortype(V))))
    return reduce(+, dim(V, c) * dim(c) for c in sectors(V);
                  init=zero(T))
end

function scalar(t::AbstractTensorMap{T,Vect[IsingBimod],0,0}) where {T}
    Bs = collect(blocks(t))
    inds = findall(!iszero ∘ last, Bs)
    isempty(inds) && return zero(scalartype(t))
    return only(last(Bs[only(inds)]))
end

function Base.oneunit(S::Vect[IsingBimod])
    allequal(a.row for a in sectors(S)) && allequal(a.col for a in sectors(S)) ||
        throw(ArgumentError("sectors of $S are not all equal"))
    first(sectors(S)).row == first(sectors(S)).col ||
        throw(ArgumentError("sectors of $S are non-diagonal"))
    sector = one(first(sectors(S)))
    return spacetype(S)(sector => 1)
end

Base.zero(S::Type{Vect[IsingBimod]}) = Vect[IsingBimod]()

function blocksectors(W::TensorMapSpace{Vect[IsingBimod],N₁,N₂}) where {N₁,N₂}
    codom = codomain(W)
    dom = domain(W)
    if N₁ == 0 && N₂ == 0
        return (IsingBimod(1, 1, 0), IsingBimod(2, 2, 0))
    elseif N₁ == 0
        @assert N₂ != 0 "one of Type IsingBimod doesn't exist"
        return filter!(c -> c == leftone(c) == rightone(c), collect(blocksectors(dom))) # is this what we want? doesn't allow M/Mop to end at empty space
    elseif N₂ == 0
        @assert N₁ != 0 "one of Type IsingBimod doesn't exist"
        return filter!(c -> c == leftone(c) == rightone(c), collect(blocksectors(codom)))
    elseif N₂ <= N₁ # keep intersection
        return filter!(c -> hasblock(codom, c), collect(blocksectors(dom)))
    else
        return filter!(c -> hasblock(dom, c), collect(blocksectors(codom)))
    end
end

function rightoneunit(S::Vect[IsingBimod])
    allequal(a.col for a in sectors(S)) ||
        throw(ArgumentError("sectors of $S do not have the same rightone"))

    sector = rightone(first(sectors(S)))
    return spacetype(S)(sector => 1)
end

function leftoneunit(S::Vect[IsingBimod])
    allequal(a.row for a in sectors(S)) ||
        throw(ArgumentError("sectors of $S do not have the same leftone"))

    sector = leftone(first(sectors(S)))
    return spacetype(S)(sector => 1)
end

function insertrightunit(P::ProductSpace{Vect[IsingBimod],N}, ::Val{i};
                         conj::Bool=false,
                         dual::Bool=false) where {i,N}
    i > N && error("cannot insert a sensible right unit onto $P at index $(i+1)")
    # possible change to rightone of correct space for N = 0
    u = N > 0 ? rightoneunit(P[i]) : error("no unit object in $P")
    if dual
        u = TensorKit.dual(u)
    end
    if conj
        u = TensorKit.conj(u)
    end
    return ProductSpace(TupleTools.insertafter(P.spaces, i, (u,)))
end

# TODO?: overwrite defaults at level of HomSpace and TensorMap?
function insertleftunit(P::ProductSpace{Vect[IsingBimod],N}, ::Val{i}; # want no defaults?
                        conj::Bool=false,
                        dual::Bool=false) where {i,N}
    i > N && error("cannot insert a sensible left unit onto $P at index $i") # do we want this to error in the diagonal case?
    u = N > 0 ? leftoneunit(P[i]) : error("no unit object in $P")
    if dual
        u = TensorKit.dual(u)
    end
    if conj
        u = TensorKit.conj(u)
    end
    return ProductSpace(TupleTools.insertafter(P.spaces, i - 1, (u,)))
end

# is this even necessary? can let it error at fusiontrees.jl:93 from the one(IsingBimod) call
# but these errors are maybe more informative
function FusionTree(uncoupled::Tuple{IsingBimod,Vararg{IsingBimod}})
    coupled = collect(⊗(uncoupled...))
    if length(coupled) == 0 # illegal fusion somewhere
        throw(ArgumentError("Forbidden fusion with uncoupled sectors $uncoupled"))
    else # allowed fusions require inner lines
        error("fusion tree requires inner lines if `FusionStyle(I) <: MultipleFusion`")
    end
end

# this one might also be overkill, since `FusionTreeIterator`s don't check whether the fusion is allowed
function fusiontrees(uncoupled::Tuple{IsingBimod,Vararg{IsingBimod}})
    return throw(ArgumentError("coupled sector must be provided for IsingBimod fusion"))
end
