# tensor.jl
#
# Tensor provides a dense implementation of an AbstractTensor type without any
# symmetry assumptions, i.e. it describes tensors living in the full tensor
# product space of its index spaces.

#++++++++++++++
# Tensor type:
#++++++++++++++
# Type definition and constructors:
#-----------------------------------
struct TensorMap{T<:Number, S<:IndexSpace, N₁, N₂, A} <: AbstractTensorMap{T, S, N₁, N₂}
    data::A
    codom::ProductSpace{S,N₁}
    dom::ProductSpace{S,N₂}
    function TensorMap{T,S,N₁,N₂}(data, spaces::TensorMapSpace{S,N₁,N₂}) where {T<:Number, S<:IndexSpace, N₁, N₂}
        codom = spaces[2]
        dom = spaces[1]
        data2 = validatedata(data, codom, dom, fieldtype(S), sectortype(S))
        new{T,S,N₁,N₂,typeof(data2)}(data2, codom, dom)
    end
    function TensorMap{T,S,N,0}(data, space::TensorSpace{S,N}) where {T<:Number, S<:IndexSpace, N}
        codom = space
        dom = ProductSpace{S,0}(())
        data2 = validatedata(data, codom, dom, fieldtype(S), sectortype(S))
        new{T,S,N,0,typeof(data2)}(data2, codom, dom)
    end
end

const Tensor{T<:Number, S<:IndexSpace, N, A<:AbstractArray{T}} = TensorMap{T, S, N, 0, A}

function validatedata(data::AbstractArray, codom, dom, k::Field, ::Type{Trivial})
    size(data) == (dims(codom)..., dims(dom)...) || throw(DimensionMismatch())
    eltype(data) ⊆ k || warn("eltype(data) = $(eltype(data)) ⊈ $k)")
    return data
end
function validatedata(data::Dict{<:FusionTree{G,N₁,N₂},<:AbstractArray}, codom::ProductSpace{S,N₁}, dom::ProductSpace{S,N₁}, k::Field, ::Type{G}) where {S<:IndexSpace, G<:Sector, N₁,N₂}
    for tree in keys(data)
        sectorcodom = tree.outgoing
        sectordom = tree.incoming
        size(data[tree]) == (dims(codom, sectorcodom)..., dims(dom, sectordom)...) || throw(DimensionMismatch())
        eltype(data[tree]) ⊆ k || warn("eltype(data) = $(eltype(data)) ⊈ $k)")
    end
    return data
end
# TODO: allow to start from full data (a single AbstractArray) and create the dictionary, in the first place for Abelian sectors, or for e.g. SU₂ using Wigner 3j symbols

# Show method:
#-------------
function Base.show(io::IO, t::TensorMap{T,S,N₁,N₂}) where {T<:Number, S<:IndexSpace, N₁, N₂}
    print(io, "TensorMap{", T, "}( ", codom(t), " ← ", dom(t), " )")
    if !get(io, :compact, false)
        println(":")
        Base.showarray(io, t.data, false; header = false)
    end
end
function Base.show(io::IO, t::Tensor{T,S,N}) where {T<:Number, S<:IndexSpace, N}
    print(io," Tensor{", T, "}( ", codom(t), " )")
    if !get(io, :compact, false)
        println(":")
        Base.showarray(io, t.data, false; header = false)
    end
end

# Basic methods for characterising a tensor:
#--------------------------------------------
codom(t::TensorMap) = t.codom
dom(t::TensorMap) = t.dom
space(t::TensorMap) = t.codom ⊗ dual(t.dom)
space(t::TensorMap, n::Int) = space(t::TensorMap)[n]

# General Tensor constructors
#----------------------------
# with data
Tensor{T}(data::AbstractArray{T}, P::TensorSpace{S,N}) where {T<:Number, S<:IndexSpace, N} = Tensor{T,S,N}(data, P)

Tensor(data::AbstractArray{T}, P::TensorSpace) where {T<:Number} = Tensor{T}(data, P)

# without data: uninitialized
Tensor{T}(A::Type{<:AbstractArray{T}}, P::TensorSpace) where {T<:Number} = Tensor{T}(A(dims(P)), P)
Tensor(A::Type{<:AbstractArray{T}}, P::TensorSpace) where {T<:Number} = Tensor{T}(A, P)

Tensor{T}(P::TensorSpace) where {T<:Number} = Tensor(Array{T}, P)
Tensor(P::TensorSpace) = Tensor{Float64}(P)

# generic constructor from callable:
Tensor{T}(f, P::TensorSpace) where {T<:Number} = Tensor{T}(f(T, dims(P)), P)
Tensor(f, P::TensorSpace) = Tensor(f(dims(P)), P)

# for a single `IndexSpace`
Tensor{T}(arg, V::IndexSpace) where {T<:Number} = Tensor{T}(arg, TensorSpace(V))
Tensor(arg, V::IndexSpace) = Tensor(arg, TensorSpace(V))

Tensor(data::AbstractArray{T,N}) where {T<:Real, N} =
    Tensor{T}(data, TensorSpace{CartesianSpace,N}(map(CartesianSpace, size(data))))
Tensor(data::AbstractArray{T,N}) where {T<:Complex, N} =
    Tensor{T}(data, TensorSpace{ComplexSpace,N}(map(ComplexSpace, size(data))))


# General TensorMap constructors
#--------------------------------
# with data
TensorMap{T}(data::AbstractArray{T}, P::TensorMapSpace{S,N₁,N₂}) where {T<:Number, S<:IndexSpace, N₁, N₂} = TensorMap{T,S,N₁,N₂}(data, P)
TensorMap(data::AbstractArray{T}, P::TensorMapSpace) where {T<:Number} = Tensor{T}(data, P)

# without data: uninitialized
TensorMap{T}(A::Type{<:AbstractArray{T}}, P::TensorMapSpace) where {T<:Number} = Tensor{T}(A(dims(P)), P)
TensorMap(A::Type{<:AbstractArray{T}}, P::TensorMapSpace) where {T<:Number} = Tensor{T}(A, P)

TensorMap{T}(P::TensorMapSpace) where {T<:Number} = TensorMap{T}(Array{T}, P)
TensorMap(P::TensorMapSpace) = TensorMap{Float64}(P)

# generic constructor from callable:
TensorMap{T}(f, P::TensorMapSpace) where {T<:Number} = Tensor{T}(f(T, dims(P)), P)
TensorMap(f, P::TensorMapSpace) = Tensor(f(dims(P)), P)

# without spaces
TensorMap(data::AbstractMatrix{T}) where {T<:Real} = Tensor{T}(data, CartesianSpace(size(data,1)) ← CartesianSpace(size(data,2)))
TensorMap(data::AbstractMatrix{T}) where {T<:Complex} = Tensor{T}(data, ComplexSpace(size(data,1)) ← ComplexSpace(size(data,2)))

# Similar
#---------
Base.similar(t::TensorMap{T1, S}, ::Type{T2}, P::TensorMapSpace{S} = (dom(t)=>codom(t))) where {T1,T2,S} = Tensor(similar(t.data, T, dims(P)), P)
Base.similar(t::TensorMap{T1, S}, ::Type{T2}, P::TensorSpace{S}) where {T1,T2,S} = Tensor(similar(t.data, T, dims(P)), P)
Base.similar(t::TensorMap{T1, S}, ::Type{T2}, V::S) where {T1,T2,S} = similar(t, T2, TensorSpace(V))

Base.similar(t::TensorMap{T, S}, P::TensorMapSpace{S} = (dom(t)=>codom(t))) where {T,S} = similar(t, eltype(t), P)
Base.similar(t::TensorMap{T, S}, P::TensorSpace{S}) where {T,S} = simlar(t, eltype(t), P)
Base.similar(t::TensorMap{T, S}, V::S) where {T,S} = similar(t, eltype(t), V)

# Copy and fill tensors:
#------------------------
function Base.copy!(tdest::TensorMap, tsource::TensorMap)
    codom(tdest) == codom(tsource) && dom(tdest) == dom(tsource) || throw(SpaceError())
    copy!(tdest.data, tsource.data)
    return tdest
end
Base.fill!(tdest::TensorMap, value::Number)=fill!(tdest.data, value)

# Conversion and promotion:
#---------------------------
Base.promote_rule(::Type{<:TensorMap{T1,S}},::Type{<:TensorMap{T2,S}}) where {T1, T2, S} = TensorMap{promote_type(T1,T2),S}

Base.convert(::Type{TensorMap{T,S}}, t::Tensor{T,S}) where {T,S} = t
Base.convert(::Type{TensorMap{T1,S}}, t::Tensor{T2,S}) where {T1,T2,S} = copy!(similar(t, T1), t)

# TODO: Check whether we need anything of this old stuff
# Base.promote_rule{S,T1,T2,N1,N2}(::Type{Tensor{S,T1,N1}},::Type{Tensor{S,T2,N2}})=Tensor{S,promote_type(T1,T2)}
# Base.promote_rule{S,T1,T2}(::Type{Tensor{S,T1}},::Type{Tensor{S,T2}})=Tensor{S,promote_type(T1,T2)}
#
# Base.promote_rule{S,T1,T2,N}(::Type{AbstractTensor{S,ProductSpace,T1,N}},::Type{Tensor{S,T2,N}})=AbstractTensor{S,ProductSpace,promote_type(T1,T2),N}
# Base.promote_rule{S,T1,T2,N1,N2}(::Type{AbstractTensor{S,ProductSpace,T1,N1}},::Type{Tensor{S,T2,N2}})=AbstractTensor{S,ProductSpace,promote_type(T1,T2)}
# Base.promote_rule{S,T1,T2}(::Type{AbstractTensor{S,ProductSpace,T1}},::Type{Tensor{S,T2}})=AbstractTensor{S,ProductSpace,promote_type(T1,T2)}


# Base.convert{S,T,N}(::Type{Tensor{S,T,N}},t::Tensor{S,T,N})=t
# Base.convert{S,T1,T2,N}(::Type{Tensor{S,T1,N}},t::Tensor{S,T2,N})=copy!(similar(t,T1),t)
# Base.convert{S,T}(::Type{Tensor{S,T}},t::Tensor{S,T})=t
# Base.convert{S,T1,T2}(::Type{Tensor{S,T1}},t::Tensor{S,T2})=copy!(similar(t,T1),t)
#
# Base.float{S,T<:FloatingPoint}(t::Tensor{S,T})=t
# Base.float(t::Tensor)=tensor(float(t.data),space(t))
#
# Base.real{S,T<:Real}(t::Tensor{S,T})=t
# Base.real(t::Tensor)=tensor(real(t.data),space(t))
# Base.complex{S,T<:Complex}(t::Tensor{S,T})=t
# Base.complex(t::Tensor)=tensor(complex(t.data),space(t))
#
# for (f,T) in ((:float32,    Float32),
#               (:float64,    Float64),
#               (:complex64,  Complex64),
#               (:complex128, Complex128))
#     @eval (Base.$f){S}(t::Tensor{S,$T}) = t
#     @eval (Base.$f)(t::Tensor) = tensor(($f)(t.data),space(t))
# end
#
# Basic vector space methods:
# ---------------------------
function Base.scale!(t1::TensorMap, t2::TensorMap, a::Number)
    (codomain(t1)==codomain(t2) && domain(t1) == domain(t2)) || throw(SpaceError())
    t1.data .= a .* t2.data
    return t1
end

function add!(t1::TensorMap, t2::TensorMap, a::Number)
    (codomain(t1)==codomain(t2) && domain(t1) == domain(t2)) || throw(SpaceError())
    t1.data .+= a .* t2.data
end

# Indexing
#----------
# # linear indexing using ProductBasisVector
Base.getindex(t::TensorMap, indices...)=getindex(t.data, indices...)
Base.setindex!(t::TensorMap, value, indices...)=setindex!(t.data, value, indices...)

# # Index methods
# #---------------
# @eval function insertind{S}(t::Tensor{S},ind::Int,V::S)
#     N=numind(t)
#     0<=ind<=N || throw(IndexError("index out of range"))
#     iscnumber(V) || throw(SpaceError("can only insert index with c-number index space"))
#     spacet=space(t)
#     newspace=spacet[1:ind] ⊗ V ⊗ spacet[ind+1:N]
#     return tensor(t.data,newspace)
# end
# @eval function deleteind(t::Tensor,ind::Int)
#     N=numind(t)
#     1<=ind<=N || throw(IndexError("index out of range"))
#     iscnumber(space(t,ind)) || throw(SpaceError("can only delete index with c-number index space"))
#     spacet=space(t)
#     newspace=spacet[1:ind-1] ⊗ spacet[ind+1:N]
#     return tensor(t.data,newspace)
# end
#
# for (S,TT) in ((CartesianSpace,CartesianTensor),(ComplexSpace,ComplexTensor))
#     @eval function fuseind(t::$TT,ind1::Int,ind2::Int,V::$S)
#         N=numind(t)
#         ind2==ind1+1 || throw(IndexError("only neighbouring indices can be fused"))
#         1<=ind1<=N-1 || throw(IndexError("index out of range"))
#         fuse(space(t,ind1),space(t,ind2),V) || throw(SpaceError("index spaces $(space(t,ind1)) and $(space(t,ind2)) cannot be fused to $V"))
#         spacet=space(t)
#         newspace=spacet[1:ind1-1]*V*spacet[ind2+1:N]
#         return tensor(t.data,newspace)
#     end
#     @eval function splitind(t::$TT,ind::Int,V1::$S,V2::$S)
#         1<=ind<=numind(t) || throw(IndexError("index out of range"))
#         fuse(V1,V2,space(t,ind)) || throw(SpaceError("index space $(space(t,ind)) cannot be split into $V1 and $V2"))
#         spacet=space(t)
#         newspace=spacet[1:ind-1]*V1*V2*spacet[ind+1:numind(t)]
#         return tensor(t.data,newspace)
#     end
# end
#


## NOTE: Do we want this?
# # tensors from concatenation
# function tensorcat{S}(catind, X::Tensor{S}...)
#     catind = collect(catind)
#     isempty(catind) && error("catind should not be empty")
#     # length(unique(catdims)) != length(catdims) && error("every dimension should appear only once")
#
#     nargs = length(X)
#     numindX = map(numind, X)
#
#     all(n->(n == numindX[1]), numindX) || throw(SpaceError("all tensors should have the same number of indices for concatenation"))
#
#     numindC = numindX[1]
#     ncatind = setdiff(1:numindC,catind)
#     spaceCvec = Array(S, numindC)
#     for n = 1:numindC
#         spaceCvec[n] = space(X[1],n)
#     end
#     for i = 2:nargs
#         for n in catind
#             spaceCvec[n] = directsum(spaceCvec[n], space(X[i],n))
#         end
#         for n in ncatind
#             spaceCvec[n] == space(X[i],n) || throw(SpaceError("space mismatch for index $n"))
#         end
#     end
#     spaceC = ⊗(spaceCvec...)
#     typeC = mapreduce(eltype, promote_type, X)
#     dataC = zeros(typeC, map(dim,spaceC))
#
#     offset = zeros(Int,numindC)
#     for i=1:nargs
#         currentdims=ntuple(numindC,n->dim(space(X[i],n)))
#         currentrange=[offset[n]+(1:currentdims[n]) for n=1:numindC]
#         dataC[currentrange...] = X[i].data
#         for n in catind
#             offset[n]+=currentdims[n]
#         end
#     end
#     return tensor(dataC,spaceC)
# end
