# lightweight poly for abstract cost estimation
abstract AbstractPoly{D,T<:Number}

function Base.show{D,T<:Real}(io::IO,p::AbstractPoly{D,T})
    N=degree(p)
    separated=true
    for i=N:-1:0
        if p[i]!=0
            if !separated
                print(io,p[i]<0 ? " - " : " + ")
            end
            abs(p[i])==1 || print(io,"$(abs(p[i]))*")
            i==0 && print(io,"$(abs(p[i]))")
            i>0 && print(io,"$D")
            i>1 && print(io,"^$i")
            separated=false
        end
    end
end
function Base.show{D,T<:Complex}(io::IO,p::AbstractPoly{D,T})
    N=degree(p)
    separated=true
    for i=N:-1:0
        if p[i]!=0
            separated || print(io," + ")
            p[i]==1 || print(io,"($(p[i]))*")
            i==0 && print(io,"($(p[i]))")
            i>0 && print(io,"$D")
            i>1 && print(io,"^$i")
            separated=false
        end
    end
end
immutable Power{D,T} <: AbstractPoly{D,T}
    coeff::T
    N::Int
end
degree(p::Power)=p.N
Base.getindex{D,T}(p::Power{D,T},i::Int)=(i==p.N ? p.coeff : zero(T))
Base.call{D,T}(::Type{Power{D}},coeff::T,N::Int=0)=Power{D,T}(coeff,N)
Base.convert{D}(::Type{Power{D}},coeff::Number)=Power{D,T}(coeff,0)
Base.convert{D,T}(::Type{Power{D,T}},coeff::Number)=Power{D,T}(coeff,0)
function Base.show{D,T}(io::IO,p::Power{D,T})
    if p.coeff==1
    elseif p.coeff==-1
        print(io,"-")
    elseif isa(p.coeff,Complex)
        print(io,"($(p.coeff))")
    else
        print(io,"$(p.coeff)")
    end
    p.coeff==1 || p.coeff==-1 || p.N==0 || print(io,"*")
    p.N==0 && (p.coeff==1 || p.coeff==-1) && print(io,"1")
    p.N>0 && print(io,"$D")
    p.N>1 && print(io,"^$(p.N)")
end

*{D}(p1::Power{D},p2::Power{D})=Power{D}(p1.coeff*p2.coeff,degree(p1)+degree(p2))
*{D}(p::Power{D},s::Number)=Power{D}(p.coeff*s,degree(p))
*(s::Number,p::Power)=*(p,s)
^{D}(p::Power{D},n::Int)=Power{D}(p.coeff^n,n*degree(p))

immutable Poly{D,T} <: AbstractPoly{D,T}
    coeffs::Vector{T}
end
degree(p::Poly)=length(p.coeffs)-1
Base.getindex{D,T}(p::Poly{D,T},i::Int)=(0<=i<=degree(p) ? p.coeffs[i+1] : zero(T))
Base.call{D,T}(::Type{Poly{D}},coeffs::Vector{T})=Poly{D,T}(coeffs)
Base.call{D,T<:Number}(::Type{Poly{D}},c0::T)=Poly{D,T}([c0])
Base.call{D,T}(::Type{Poly{D}},p::Power{D,T})=Poly{D,T}(vcat(zeros(T,p.N),p.coeff))
Base.call{D,T}(::Type{Poly{D,T}},c0::Number)=Poly{D,T}([T(c0)])
Base.call{D,T1,T2}(::Type{Poly{D,T1}},p::Power{D,T2})=Poly{D,T1}(vcat(zeros(T1,p.N),T1(p.coeff)))

+{D}(p::Poly{D},s::Number)=Poly{D}([p[i]+(i==0 ? s : zero(s)) for i=0:degree(p)])
+(s::Number,p::Poly)=+(p,s)
+{D,T1,T2}(p1::Power{D,T1},p2::Power{D,T2})=begin
    T=promote_type(T1,T2)
    coeffs=zeros(T,max(degree(p1),degree(p2))+1)
    coeffs[p1.N+1]=p1.coeff
    coeffs[p2.N+1]+=p2.coeff
    return Poly{D,T}(coeffs)
end
+{D,T1,T2}(p1::Power{D,T1},p2::Poly{D,T2})=begin
    T=promote_type(T1,T2)
    coeffs=zeros(T,max(degree(p1),degree(p2))+1)
    coeffs[(0:degree(p2))+1]=p2.coeffs
    coeffs[p1.N+1]+=p1.coeff
    return Poly{D,T}(coeffs)
end
+{D}(p1::Poly{D},p2::Power{D})=+(p2,p1)
+{D,T1,T2}(p1::Poly{D,T1},p2::Poly{D,T2})=begin
    T=promote_type(T1,T2)
    coeffs=zeros(T,max(degree(p1),degree(p2))+1)
    coeffs[(0:degree(p1))+1]=p1.coeffs
    for j=0:degree(p2)
        coeffs[j+1]+=p2.coeffs[j+1]
    end
    return Poly{D,T}(coeffs)
end


-{D}(p::Poly{D})=Poly{D}(-p.coeffs)
-{D}(p::Poly{D},s::Number)=Poly{D}([p[i]+(i==0 ? s : zero(s)) for i=0:degree(p)])
-{D}(s::Number,p::Poly{D})=Poly{D}([-p[i]+(i==0 ? s : zero(s)) for i=0:degree(p)])
-{D}(p1::Union(Power{D},Poly{D}),p2::Union(Power{D},Poly{D}))=Poly{D}([p1[i]-p2[i] for i=0:max(degree(p1),degree(p2))])

*{D}(p1::Power{D},p2::Poly{D})=Poly{D}([p1.coeff*p2[n-degree(p1)] for n=0:degree(p1)+degree(p2)])
*{D}(p1::Poly{D},p2::Power{D})=*(p2,p1)
*{D}(p::Poly{D},s::Number)=Poly{D}(s*p.coeffs)
*(s::Number,p::Poly)=*(p,s)
*{D}(p1::Poly{D},p2::Poly{D})=begin
    N=degree(p1)+degree(p2)
    s=p1[0]*p2[0]
    coeffs=zeros(typeof(s),N+1)
    for i=0:degree(p1)
        for j=0:degree(p2)
            coeffs[i+j+1]+=p1[i]*p2[j]
        end
    end
    return Poly{D}(coeffs)
end

Base.promote_rule{D,T1<:Number,T2<:Number}(::Type{Power{D,T1}},::Type{Power{D,T2}})=Power{D,promote_type(T1,T2)}
Base.promote_rule{D,T1<:Number,T2<:Number}(::Type{Power{D,T1}},::Type{T2})=Power{D,promote_type(T1,T2)}
Base.promote_rule{D,T1<:Number,T2<:Number}(::Type{Poly{D,T1}},::Type{Poly{D,T2}})=Poly{D,promote_type(T1,T2)}
Base.promote_rule{D,T1<:Number,T2<:Number}(::Type{Poly{D,T1}},::Type{T2})=Poly{D,promote_type(T1,T2)}
Base.convert{D,T1}(::Type{Power{D,T1}},x::Number)=Power{D}(T1(x))
Base.convert{D,T1}(::Type{Poly{D,T1}},x::Number)=Poly{D}([T1(x)])

function =={D}(p1::AbstractPoly{D},p2::AbstractPoly{D})
    for i=max(degree(p1),degree(p2)):-1:0
        p1[i]==p2[i] || return false
    end
    return true
end
function <{D}(p1::AbstractPoly{D},p2::AbstractPoly{D})
    for i=max(degree(p1),degree(p2)):-1:0
        p1[i]<p2[i] && return true
        p1[i]>p2[i] && return false
    end
    return false
end

# user function to create symbolic variable
bonddimension(D::Symbol=:D)=Power{D,Int}(1,1)