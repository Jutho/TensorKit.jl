type CompositeMap{S,P,T}<:AbstractTensorMap{S,P,T}
    maps::Vector{AbstractTensorMap{S,P}} # stored in order of application to vector
    function CompositeMap(maps::Vector{AbstractTensorMap})
        N=length(maps)
        for n=2:N
            domain(maps[n])==codomain(maps[n-1]) || throw(SpaceError("Mismatch between domain of map $n and codomain of previous map"))
        end
        for n=1:N
            spacetype(maps[n])==S || throw(SpaceError("Incompatible spacetype for map $n"))
            tensortype(maps[n])==P || throw(SpaceError("Incompatible tensortype for map $n"))
            promote_type(T,eltype(maps[n]))==T || throw(InexactError())
        end
        new(maps)
    end
end

# basic methods
domain(A::CompositeMap)=domain(A.maps[1])
codomain(A::CompositeMap)=codomain(A.maps[end])
Base.isreal(A::CompositeMap)=all(isreal,A.maps) # sufficient but not necessary

# the following rules are sufficient but not necessary
function Base.issym(A::CompositeMap)
    N=length(A.maps)
    if isodd(N)
        issym(A.maps[div(N+1,2)]) || return false
    end
    for n=1:div(N,2)
        A.maps[n]==transpose(A.maps[N-n+1]) || return false
    end
    return true
end
function Base.ishermitian(A::CompositeMap)
    N=length(A.maps)
    if isodd(N)
        ishermitian(A.maps[div(N+1,2)]) || return false
    end
    for n=1:div(N,2)
        A.maps[n]==ctranspose(A.maps[N-n+1]) || return false
    end
    return true
end
function Base.isposdef(A::CompositeMap)
    N=length(A.maps)
    if isodd(N)
        isposdef(A.maps[div(N+1,2)]) || return false
    end
    for n=1:div(N,2)
        A.maps[n]==ctranspose(A.maps[N-n+1]) || return false
    end
    return true
end

# composition of linear maps
function *{S,P,T1,T2}(A1::CompositeMap{S,P,T1},A2::CompositeMap{S,P,T2})
    domain(A1)==codomain(A2) || throw(SpaceError())
    T=promote_type(T1,T2)
    return CompositeMap{S,P,T}(AbstractLinearMap{S,P}[A2.maps...,A1.maps...])
end
function *{S,P,T1,T2}(A1::AbstractTensorMap{S,P,T1},A2::CompositeMap{S,P,T2})
    domain(A1)==codomain(A2) || throw(SpaceError())
    T=promote_type(T1,T2)
    return CompositeMap{S,P,T}(AbstractLinearMap{S,P}[A2.maps...,A1])
end
function *{S,P,T1,T2}(A1::CompositeMap{S,P,T1},A2::AbstractTensorMap{S,P,T2})
    domain(A1)==codomain(A2) || throw(SpaceError())
    T=promote_type(T1,T2)
    return CompositeMap{S,P,T}(AbstractLinearMap{S,P}[A2,A1.maps...])
end
function *{S,P,T1,T2}(A1::AbstractTensorMap{S,P,T1},A2::AbstractTensorMap{S,P,T2})
    domain(A1)==codomain(A2) || throw(SpaceError())
    T=promote_type(T1,T2)
    return CompositeMap{S,P,T}(AbstractTensorMap{S,P}[A2,A1])
end

# comparison of CompositeMap objects
==(A::CompositeMap,B::CompositeMap)=(eltype(A)==eltype(B) && A.maps==B.maps)

# special transposition behavior
transpose{S,P}(A::CompositeMap{S,P})=CompositeMap{S,P,eltype(A)}(AbstractTensorMap{S,P}[transpose(M) for M in reverse(A.maps)])
ctranspose{S,P}(A::CompositeMap{S,P})=CompositeMap{S,P,eltype(A)}(AbstractTensorMap{S,P}[ctranspose(M) for M in reverse(A.maps)])

# multiplication with tensors
function Base.A_mul_B!{S,P}(y::AbstractTensor{S,P},A::CompositeMap{S,P},x::AbstractTensor{S,P})
    # no domain checking, will be done by individual maps
    N=length(A.maps)
    if N==1
        Base.A_mul_B!(y,A.maps[1],x)
    else
        T=promote_type(eltype(A),eltype(x))
        dest=Array(T,dim(domain(A.maps[1])))
        w=tensor(dest,dual(domain(A.maps[1])))
        Base.At_mul_B!(w,A.maps[1],x)
        v=w
        source=dest
        if N>2
            dest=Array(T,dim(domain(A.maps[2])))
        end
        for n=2:N-1
            resize!(dest,dim(domain(A.maps[n])))
            w=tensor(dest,dual(domain(A.maps[1])))
            Base.At_mul_B!(w,A.maps[n],v)
            v=w
            dest,source=source,dest # alternate dest and source
        end
        Base.A_mul_B!(y,A.maps[N],v)
    end
    return y
end
function Base.A_mul_B!{S,P}(y::AbstractTensor{S,P},A::CompositeMap{S,P},x::AbstractTensor{S,P})
    # no domain checking, will be done by individual maps
    N=length(A.maps)
    if N==1
        Base.Ac_mul_B!(y,A.maps[1],x)
    else
        T=promote_type(eltype(A),eltype(x))
        dest=Array(T,dim(domain(A.maps[1])))
        w=tensor(dest,dual(conj(domain(A.maps[1]))))
        Base.A_mul_B!(w,A.maps[1],x)
        v=w
        source=dest
        if N>2
            dest=Array(T,dim(codomain(A.maps[2])))
        end
        for n=2:N-1
            resize!(dest,dim(domain(A.maps[n])))
            w=tensor(dest,dual(conj(domain(A.maps[1]))))
            Base.A_mul_B!(w,A.maps[n],v)
            v=w
            dest,source=source,dest # alternate dest and source
        end
        Base.Ac_mul_B!(y,A.maps[N],v)
    end
    return y
end

