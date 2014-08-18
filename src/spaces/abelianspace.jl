immutable AbelianSpace{G<:Abelian} <: UnitaryRepresentationSpace{G}
    dims::Dict{G,Int}
    dual::Bool
    function AbelianSpace(dims::Dict{G,Int},dual::Bool=false)
        charges=collect(keys(dims))
        d=[dims[c] for c in charges]
        ind=find(d.>0)
        isempty(ind) && throw(ArgumentError("Dimension of a vector space should be bigger than zero"))
        new([c=>dims[c] for c in charges[ind]],dual)
    end
end
AbelianSpace{G<:Abelian}(dims::Dict{G,Int},dual::Bool=false)=AbelianSpace{G}(dims,dual)

==(V1::AbelianSpace,V2::AbelianSpace)=(V1.dims==V2.dims && V1.dual==V2.dual)

# Corresponding methods:
sectors(V::AbelianSpace) = keys(V.dims)

dim(V::AbelianSpace) = sum(values(V.dims))
dim{G<:Abelian}(V::AbelianSpace{G},c::G) = get(V.dims,c,0)
dual(V::AbelianSpace) = AbelianSpace([conj(c)=>dim(V,c) for c in sectors(V)], !V.dual)
cnumber{G}(V::AbelianSpace{G}) = AbelianSpace([one(G)=>1],V.dual)
cnumber{G}(::Type{AbelianSpace{G}}) = AbelianSpace([one(G)=>1])
iscnumber{G}(V::AbelianSpace{G}) = V.dims == [one(G)=>1]

# Show methods
Base.show(io::IO, V::AbelianSpace) = (print(io,"AbelianSpace");showcompact(io,V.dims);print(io, V.dual ? "*" : ""))

# direct sum of AbelianSpaces
function directsum{G}(V1::AbelianSpace{G}, V2::AbelianSpace{G})
    V1.dual == V2.dual || throw(SpaceError("Direct sum of a vector space and its dual do not exist"))
    dims=[c1*c2=>0 for c1 in sectors(V1), c2 in sectors(V2)]
    for c1 in sectors(V1), c2 in sectors(V2)
        dims[c1*c2]+= dim(V1,c1)+dim(V2,c2)
    end
    return AbelianSpace(dims,V1.dual)
end

# fusing and splitting AbelianSpaces
function fuse{G}(V1::AbelianSpace{G}, V2::AbelianSpace{G}, V::AbelianSpace{G})
    dims=[c1*c2=>0 for c1 in sectors(V1), c2 in sectors(V2)]
    for c1 in sectors(V1), c2 in sectors(V2)
        dims[c1*c2]+= dim(V1,c1)*dim(V2,c2)
    end
    return V.dims == dims
end

# indexing using sectors
function Base.to_range{G}(s::G,V::AbelianSpace{G})
    offset=0
    for c in sort!(collect(sectors(V)),rev=V.dual,by=s->s.charge)
        if c!=s
            offset+=dim(V,c)
        else
            break
        end
    end
    return offset+(1:dim(V,s))
end