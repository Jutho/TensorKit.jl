immutable U1Charge <: Abelian
    charge::Int
end

*(c1::U1Charge,c2::U1Charge)=U1Charge(c1.charge+c2.charge)
Base.conj(c::U1Charge)=U1Charge(-c.charge)
Base.one(c::U1Charge)=U1Charge(0)
Base.one(::Type{U1Charge})=U1Charge(0)

immutable U1Space <: AbelianSpace{U1Charge}
    dims::Dict{U1Charge,Int}
    dual::Bool
    function U1Space(dims::Dict{U1Charge,Int},dual::Bool=false)
        charges=collect(keys(dims))
        d=[dims[c] for c in charges]
        ind=find(d.>0)
        isempty(ind) && throw(ArgumentError("Dimension of a vector space should be bigger than zero"))
        new([c=>dims[c] for c in charges[ind]],dual)
    end
end
U1Space(dims::Dict{Int,Int},dual::Bool=false)=U1Space([U1Charge(c)=>dims[c] for c in keys(dims)],dual)

# Corresponding methods:
sectors(V::U1Space) = keys(V.dims) # collect for consistency: all sectors output generate explicit vectors
sectortype(::Type{U1Space}) = U1Charge

dim(V::U1Space) = sum(values(V.dims))
dim(V::U1Space,c::U1Charge) = get(V.dims,c,0)
dual(V::U1Space) = U1Space([conj(c)=>dim(V,c) for c in sectors(V)], !V.dual)
cnumber(V::U1Space) = U1Space([U1Charge(0)=>1],V.dual)
cnumber(::Type{U1Space}) = U1Space([U1Charge(0)=>1])
iscnumber(V::U1Space) = V.dims == [U1Charge(0)=>1]

# Show methods
Base.show(io::IO, V::U1Space) = (print(io,"U1Space");showcompact(io,V.dims);print(io, V.dual ? "*" : ""))

# direct sum of U1Spaces
function directsum(V1::U1Space, V2::U1Space)
    V1.dual == V2.dual || throw(SpaceError("Direct sum of a vector space and its dual do not exist"))
    dims=[c1*c2=>0 for c1 in sectors(V1), c2 in sectors(V2)]
    for c1 in sectors(V1), c2 in sectors(V2)
        dims[c1*c2]+= dim(V1,c1)+dim(V2,c2)
    end
    return U1Space(dims,V1.dual)
end

# fusing and splitting U1Spaces
function fuse(V1::U1Space, V2::U1Space, V::U1Space)
    dims=[c1*c2=>0 for c1 in sectors(V1), c2 in sectors(V2)]
    for c1 in sectors(V1), c2 in sectors(V2)
        dims[c1*c2]+= dim(V1,c1)*dim(V2,c2)
    end
    return V.dims == dims
end

# # basis and basisvector
# typealias ComplexBasisVector BasisVector{U1Space,Int} # use integer from 1 to dim as identifier
# typealias ComplexBasis Basis{U1Space}

# Base.length(B::ComplexBasis) = dim(space(B))
# Base.start(B::ComplexBasis) = 1
# Base.next(B::ComplexBasis, state::Int) = (EuclideanBasisVector(space(B),state),state+1)
# Base.done(B::ComplexBasis, state::Int) = state>length(B)

# Base.to_index(b::ComplexBasisVector) = b.identifier
