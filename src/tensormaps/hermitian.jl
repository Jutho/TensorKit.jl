immutable HermitianTensorOperator{S<:HilbertSpace,P,T}<:AbstractTensorMap{S,P,T}
    map::AbstractTensorMap{S,P,T}
    function HermitianTensorOperator(A::AbstractTensorMap{S,P,T})
        domain(A)==dual(conj(codomain(A))) || throw(SpaceError("Not an operator, i.e. domain != codomain"))
        new(A)
    end
end
hermitian{S,P,T}(A::AbstractTensorMap{S,P,T})=HermitianTensorOperator{S,P,T}(A::AbstractTensorMap{S,P,T})

# properties
domain(A::HermitianTensorOperator)=domain(A.map)
codomain(A::HermitianTensorOperator)=codomain(A.map)

Base.isreal(A::HermitianTensorOperator)=isreal(A.map)
Base.issym(A::HermitianTensorOperator)=isreal(A.map)
Base.ishermitian(A::HermitianTensorOperator)=true
Base.isposdef(A::HermitianTensorOperator)=isposdef(A.map)

# comparison
==(A::HermitianTensorOperator,B::HermitianTensorOperator)=A.map==B.map

# multiplication with vector
Base.A_mul_B!{S,P}(y::AbstractTensor{S,P},A::HermitianTensorOperator{S,P},x::AbstractTensor{S,P})=Base.A_mul_B!(y,A.map,x)
Base.Ac_mul_B!{S,P}(y::AbstractTensor{S,P},A::HermitianTensorOperator{S,P},x::AbstractTensor{S,P})=Base.A_mul_B!(y,A.map,x)
