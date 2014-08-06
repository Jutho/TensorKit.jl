type LinearCombination{S,P,T}<:AbstractTensorMap{S,P,T}
    maps::Vector{AbstractTensorMap{S,P}}
    coeffs::Vector{T}
    function LinearCombination(maps::Vector{AbstractTensorMap{S,P}},coeffs::Vector{T})
        N=length(maps)
        N==length(coeffs) || throw(ArgumentError("Number of coefficients doesn't match number of terms"))
        dom=domain(maps[1])
        codom=codomain(maps[1])
        for n=1:N
            domain(maps[n])==dom || throw(SpaceError("Domain mismatch"))
            codomain(maps[n])==codom || throw(SpaceError("Codomain mismatch"))
            promote_type(T,eltype(maps[n]))==T || throw(InexactError())
        end
        new(maps,coeffs)
    end
end

# basic methods
Base.size(A::LinearCombination,n)=size(A.maps[1],n)
Base.size(A::LinearCombination)=size(A.maps[1])
Base.isreal(A::LinearCombination)=all(isreal,A.maps) && all(isreal,A.coeffs) # sufficient but not necessary
Base.issym(A::LinearCombination)=all(issym,A.maps) # sufficient but not necessary
Base.ishermitian(A::LinearCombination)=all(ishermitian,A.maps) && all(isreal,A.coeffs) # sufficient but not necessary
Base.isposdef(A::LinearCombination)=all(isposdef,A.maps) && all(isposdef,A.coeffs) # sufficient but not necessary

# adding linear maps
function +{S,P}(A1::LinearCombination{S,P},A2::LinearCombination{S,P})
    domain(A1)==domain(A2) || throw(SpaceError("Domain mismatch"))
    codomain(A1)==codomain(A2) || throw(SpaceError("Coomain mismatch"))
    T=promote_type(eltype(A1),eltype(A2))
    return LinearCombination{S,P,T}(AbstractTensorMap{S,P}[A1.maps...,A2.maps...],T[A1.coeffs...,A2.coeffs...])
end
function +{S,P}(A1::AbstractTensorMap{S,P},A2::LinearCombination{S,P})
    domain(A1)==domain(A2) || throw(SpaceError("Domain mismatch"))
    codomain(A1)==codomain(A2) || throw(SpaceError("Coomain mismatch"))
    T=promote_type(eltype(A1),eltype(A2))
    return LinearCombination{S,P,T}(AbstractTensorMap{S,P}[A1,A2.maps...],T[one(T),A2.coeffs...])
end
+{S,P}(A1::LinearCombination{S,P},A2::AbstractTensorMap{S,P})=+{S,P}(A2,A1)
function +{S,P}(A1::AbstractTensorMap{S,P},A2::AbstractTensorMap{S,P})
    domain(A1)==domain(A2) || throw(SpaceError("Domain mismatch"))
    codomain(A1)==codomain(A2) || throw(SpaceError("Coomain mismatch"))
    T=promote_type(eltype(A1),eltype(A2))
    return LinearCombination{S,P,T}(AbstractTensorMap{S,P}[A1,A2],T[one(T),one(T)])
end
function -{S,P}(A1::LinearCombination{S,P},A2::LinearCombination{S,P})
    domain(A1)==domain(A2) || throw(SpaceError("Domain mismatch"))
    codomain(A1)==codomain(A2) || throw(SpaceError("Coomain mismatch"))
    T=promote_type(eltype(A1),eltype(A2))
    return LinearCombination{S,P,T}(AbstractTensorMap{S,P}[A1.maps...,A2.maps...],T[A1.coeffs...,map(-,A2.coeffs)...])
end
function -{S,P}(A1::AbstractTensorMap{S,P},A2::LinearCombination{S,P})
    domain(A1)==domain(A2) || throw(SpaceError("Domain mismatch"))
    codomain(A1)==codomain(A2) || throw(SpaceError("Coomain mismatch"))
    T=promote_type(eltype(A1),eltype(A2))
    return LinearCombination{S,P,T}(AbstractTensorMap{S,P}[A1,A2.maps...],T[one(T),map(-,A2.coeffs)...])
end
function -{S,P}(A1::LinearCombination{S,P},A2::AbstractTensorMap{S,P})
    domain(A1)==domain(A2) || throw(SpaceError("Domain mismatch"))
    codomain(A1)==codomain(A2) || throw(SpaceError("Coomain mismatch"))
    T=promote_type(eltype(A1),eltype(A2))
    return LinearCombination{S,P,T}(AbstractTensorMap{S,P}[A1.maps...,A2],T[A1.coeffs...,-one(T)])
end
function -{S,P}(A1::AbstractTensorMap{S,P},A2::AbstractTensorMap{S,P})
    domain(A1)==domain(A2) || throw(SpaceError("Domain mismatch"))
    codomain(A1)==codomain(A2) || throw(SpaceError("Coomain mismatch"))
    T=promote_type(eltype(A1),eltype(A2))
    return LinearCombination{S,P,T}(AbstractTensorMap{S,P}[A1,A2],T[one(T),-one(T)])
end

# scalar multiplication
-{S,P,T}(A::AbstractTensorMap{S,P,T})=LinearCombination{S,P,T}(AbstractTensorMap{S,P}[A],[-one(T)])
-(A::LinearCombination)=LinearCombination(A.maps,-A.coeffs)

function *{S,P}(alpha::Number,A::AbstractTensorMap{S,P})
    T=promote_type(eltype(alpha),eltype(A))
    return LinearCombination{S,P,T}(AbstractTensorMap{S,P}[A],T[alpha])
end
*(A::AbstractTensorMap,alpha::Number)=*(alpha,A)

*(alpha::Number,A::LinearCombination)=LinearCombination(A.maps,alpha*A.coeffs)
*(A::LinearCombination,alpha::Number)=*(alpha,A)

\(alpha::Number,A::AbstractTensorMap)=*(1/alpha,A)
/(A::AbstractTensorMap,alpha::Number)=*(1/alpha,A)

# comparison of LinearCombination objects
==(A::LinearCombination,B::LinearCombination)=(eltype(A)==eltype(B) && A.maps==B.maps && A.coeffs==B.coeffs)

# special transposition behavior
transpose(A::LinearCombination)=LinearCombination{eltype(A)}(AbstractLinearMap[transpose(l) for l in A.maps],A.coeffs)
ctranspose(A::LinearCombination)=LinearCombination{eltype(A)}(AbstractLinearMap[ctranspose(l) for l in A.maps],conj(A.coeffs))

# multiplication with vectors
function Base.A_mul_B!{S,P}(y::AbstractTensor{S,P},A::LinearCombination,x::AbstractTensor{S,P})
    # no size checking, will be done by individual maps
    Base.A_mul_B!(y,A.maps[1],x)
    scale!(A.coeffs[1],y)
    if length(A.maps)>1
        z=similar(y)
    end
    for n=2:length(A.maps)
        Base.A_mul_B!(z,A.maps[n],x)
        Base.axpy!(A.coeffs[n],z,y)
    end
    return y
end
function Base.At_mul_B!{S,P}(y::AbstractTensor{S,P},A::LinearCombination,x::AbstractTensor{S,P})
    # no size checking, will be done by individual maps
    Base.At_mul_B!(y,A.maps[1],x)
    scale!(A.coeffs[1],y)
    if length(A.maps)>1
        z=similar(y)
    end
    for n=2:length(A.maps)
        Base.At_mul_B!(z,A.maps[n],x)
        Base.axpy!(A.coeffs[n],z,y)
    end
    return y
end
function Base.Ac_mul_B!{S,P}(y::AbstractTensor{S,P},A::LinearCombination,x::AbstractTensor{S,P})
    # no size checking, will be done by individual maps
    Base.Ac_mul_B!(y,A.maps[1],x)
    scale!(conj(A.coeffs[1]),y)
    if length(A.maps)>1
        z=similar(y)
    end
    for n=2:length(A.maps)
        Base.Ac_mul_B!(z,A.maps[n],x)
        Base.axpy!(conj(A.coeffs[n]),z,y)
    end
    return y
end
