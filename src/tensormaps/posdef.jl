immutable PosDefTensorOperator{S<:ElementaryHilbertSpace,P,T}<:AbstractTensorMap{S,P,T}
    map::AbstractTensorMap{S,P,T}
    function PosDefTensorOperator(A::AbstractTensorMap{S,P,T})
        domain(A)==conj(dual(codomain(A))) || throw(SpaceError("Not an operator, i.e. domain != codomain"))
        new(A)
    end
end
posdef{S,P,T}(A::AbstractTensorMap{S,P,T})=PosDefTensorOperator{S,P,T}(A::AbstractTensorMap{S,P,T})

# properties
domain(A::PosDefTensorOperator)=domain(A.map)
codomain(A::PosDefTensorOperator)=codomain(A.map)

Base.isreal(A::PosDefTensorOperator)=isreal(A.map)
Base.issym(A::PosDefTensorOperator)=isreal(A.map)
Base.ishermitian(A::PosDefTensorOperator)=true
Base.isposdef(A::PosDefTensorOperator)=true

# comparison
==(A::PosDefTensorOperator,B::PosDefTensorOperator)=A.map==B.map

# multiplication with vector
Base.A_mul_B!{S,P}(y::AbstractTensor{S,P},A::PosDefTensorOperator{S,P},x::AbstractTensor{S,P})=Base.A_mul_B!(y,A.map,x)
Base.Ac_mul_B!{S,P}(y::AbstractTensor{S,P},A::PosDefTensorOperator{S,P},x::AbstractTensor{S,P})=Base.A_mul_B!(y,A.map,x)
