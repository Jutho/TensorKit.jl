immutable ShiftedTensorOperator{S,P,T}<:AbstractTensorMap{S,P,T}
    map::AbstractTensorMap{S,P}
    shift::T
    function ShiftedTensorOperator(A::AbstractTensorMap{S,P},shift::T)
        domain(A)==codomain(A) || throw(SpaceError("Not an operator, i.e. domain != codomain"))
        promote_type(eltype(A),T)==T || throw(InexactError())
        new(A,shift)
    end
end
ShiftedTensorOperator{S,P,T1,T2}(A::AbstractTensorMap{S,P,T1},shift::T2)=ShiftedTensorOperator{S,P,promote_type(T1,T2)}(A,convert(promote_type(T1,T2),shift))

# create from adding scalars to tensormaps
+(A::ShiftedTensorOperator,shift::Number)=ShiftedTensorOperator(A.map,A.shift+shift)
+(shift::Number,A::ShiftedTensorOperator)=ShiftedTensorOperator(A.map,A.shift+shift)
+(A::AbstractTensorMap,shift::Number)=ShiftedTensorOperator(A,shift)
+(shift::Number,A::AbstractTensorMap)=ShiftedTensorOperator(A,shift)
-(A::ShiftedTensorOperator,shift::Number)=ShiftedTensorOperator(A.map,A.shift-shift)
-(shift::Number,A::ShiftedTensorOperator)=ShiftedTensorOperator(-A.map,-A.shift+shift)
-(A::AbstractTensorMap,shift::Number)=ShiftedTensorOperator(A,-shift)
-(shift::Number,A::AbstractTensorMap)=ShiftedTensorOperator(-A,shift)

+(A::AbstractTensorMap,shift::UniformScaling)=ShiftedTensorOperator(A,shift[1,1])
+(shift::UniformScaling,A::AbstractTensorMap)=ShiftedTensorOperator(A,shift[1,1])
-(A::AbstractTensorMap,shift::UniformScaling)=ShiftedTensorOperator(A,-shift[1,1])
-(shift::UniformScaling,A::AbstractTensorMap)=ShiftedTensorOperator(-A,shift[1,1])


# properties
domain(A::ShiftedTensorOperator)=domain(A.map)
codomain(A::ShiftedTensorOperator)=codomain(A.map)

Base.isreal(A::ShiftedTensorOperator)=isreal(A.map) && isreal(A.shift)
Base.issym(A::ShiftedTensorOperator)=issym(A.map)
Base.ishermitian(A::ShiftedTensorOperator)=ishermitian(A.map)
Base.isposdef(A::ShiftedTensorOperator)=isposdef(A.map) && isposdef(A.shift)

# comparison
# ==(A::ShiftedTensorOperator,B::ShiftedTensorOperator)=(A.map==B.map && A.shift==B.shift)

# multiplication with vector
Base.A_mul_B!{S,P}(y::AbstractTensor{S,P},A::ShiftedTensorOperator{S,P},x::AbstractTensor{S,P})=(Base.A_mul_B!(y,A.map,x);Base.axpy!(A.shift,x,y))
Base.At_mul_B!{S,P}(y::AbstractTensor{S,P},A::ShiftedTensorOperator{S,P},x::AbstractTensor{S,P})=(Base.At_mul_B!(y,A.map,x);Base.axpy!(A.shift,x,y))
Base.Ac_mul_B!{S,P}(y::AbstractTensor{S,P},A::ShiftedTensorOperator{S,P},x::AbstractTensor{S,P})=(Base.Ac_mul_B!(y,A.map,x);Base.axpy!(conj(A.shift),x,y))
