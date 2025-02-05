function ChainRulesCore.rrule(::typeof(TensorOperations.tensoradd!),
                              C::AbstractTensorMap,
                              A::AbstractTensorMap, pA::Index2Tuple, conjA::Bool,
                              α::Number, β::Number, ba...)
    C′ = tensoradd!(copy(C), A, pA, conjA, α, β, ba...)

    projectA = ProjectTo(A)
    projectC = ProjectTo(C)
    projectα = ProjectTo(α)
    projectβ = ProjectTo(β)

    function pullback(ΔC′)
        ΔC = unthunk(ΔC′)
        dC = @thunk projectC(scale(ΔC, conj(β)))
        dA = @thunk let
            ipA = invperm(linearize(pA))
            pdA = _repartition(ipA, A)
            TA = promote_add(ΔC, α)
            # TODO: allocator
            _dA = tensoralloc_add(TA, ΔC, pdA, conjA, Val(false))
            _dA = tensoradd!(_dA, ΔC, pdA, conjA, conjA ? α : conj(α), Zero(), ba...)
            return projectA(_dA)
        end
        dα = @thunk let
            # TODO: this is an inner product implemented as a contraction
            # for non-symmetric tensors this might be more efficient like this,
            # but for symmetric tensors an intermediate object will anyways be created
            # and then it might be more efficient to use an addition and inner product
            tΔC = twist(ΔC, filter(x -> isdual(space(ΔC, x)), allind(ΔC)))
            _dα = tensorscalar(tensorcontract(A, ((), linearize(pA)),
                                              !conjA, tΔC,
                                              (trivtuple(TO.numind(pA)), ()), false,
                                              ((), ()), One(), ba...))
            return projectα(_dα)
        end
        dβ = @thunk projectβ(inner(C, ΔC))
        dba = map(_ -> NoTangent(), ba)
        return NoTangent(), dC, dA, NoTangent(), NoTangent(), dα, dβ, dba...
    end

    return C′, pullback
end

function ChainRulesCore.rrule(::typeof(TensorOperations.tensorcontract!),
                              C::AbstractTensorMap,
                              A::AbstractTensorMap, pA::Index2Tuple, conjA::Bool,
                              B::AbstractTensorMap, pB::Index2Tuple, conjB::Bool,
                              pAB::Index2Tuple,
                              α::Number, β::Number, ba...)
    C′ = tensorcontract!(copy(C), A, pA, conjA, B, pB, conjB, pAB, α, β, ba...)

    projectA = ProjectTo(A)
    projectB = ProjectTo(B)
    projectC = ProjectTo(C)
    projectα = ProjectTo(α)
    projectβ = ProjectTo(β)

    function pullback(ΔC′)
        ΔC = unthunk(ΔC′)
        ipAB = invperm(linearize(pAB))
        pΔC = _repartition(ipAB, TO.numout(pA))

        dC = @thunk projectC(scale(ΔC, conj(β)))
        dA = @thunk let
            ipA = _repartition(invperm(linearize(pA)), A)
            conjΔC = conjA
            conjB′ = conjA ? conjB : !conjB
            TA = promote_contract(scalartype(ΔC), scalartype(B), scalartype(α))
            # TODO: allocator
            tB = twist(B,
                       TupleTools.vcat(filter(x -> !isdual(space(B, x)), pB[1]),
                                       filter(x -> isdual(space(B, x)), pB[2])))
            _dA = tensoralloc_contract(TA, ΔC, pΔC, conjΔC, tB, reverse(pB), conjB′, ipA,
                                       Val(false))
            _dA = tensorcontract!(_dA,
                                  ΔC, pΔC, conjΔC,
                                  tB, reverse(pB), conjB′, ipA,
                                  conjA ? α : conj(α), Zero(), ba...)
            return projectA(_dA)
        end
        dB = @thunk let
            ipB = _repartition(invperm(linearize(pB)), B)
            conjΔC = conjB
            conjA′ = conjB ? conjA : !conjA
            TB = promote_contract(scalartype(ΔC), scalartype(A), scalartype(α))
            # TODO: allocator
            tA = twist(A,
                       TupleTools.vcat(filter(x -> isdual(space(A, x)), pA[1]),
                                       filter(x -> !isdual(space(A, x)), pA[2])))
            _dB = tensoralloc_contract(TB, tA, reverse(pA), conjA′, ΔC, pΔC, conjΔC, ipB,
                                       Val(false))
            _dB = tensorcontract!(_dB,
                                  tA, reverse(pA), conjA′,
                                  ΔC, pΔC, conjΔC, ipB,
                                  conjB ? α : conj(α), Zero(), ba...)
            return projectB(_dB)
        end
        dα = @thunk let
            # TODO: this result should be AB = (C′ - βC) / α as C′ = βC + αAB
            AB = tensorcontract(A, pA, conjA, B, pB, conjB, pAB, One(), ba...)
            return projectα(inner(AB, ΔC))
        end
        dβ = @thunk projectβ(inner(C, ΔC))
        dba = map(_ -> NoTangent(), ba)
        return NoTangent(), dC,
               dA, NoTangent(), NoTangent(),
               dB, NoTangent(), NoTangent(),
               NoTangent(),
               dα, dβ, dba...
    end
    return C′, pullback
end

function ChainRulesCore.rrule(::typeof(TensorOperations.tensortrace!),
                              C::AbstractTensorMap, A::AbstractTensorMap,
                              p::Index2Tuple, q::Index2Tuple, conjA::Bool,
                              α::Number, β::Number, ba...)
    C′ = tensortrace!(copy(C), A, p, q, conjA, α, β, ba...)

    projectA = ProjectTo(A)
    projectC = ProjectTo(C)
    projectα = ProjectTo(α)
    projectβ = ProjectTo(β)

    function pullback(ΔC′)
        ΔC = unthunk(ΔC′)
        dC = @thunk projectC(scale(ΔC, conj(β)))
        dA = @thunk let
            ip = invperm((linearize(p)..., q[1]..., q[2]...))
            pdA = _repartition(ip, A)
            E = one!(TO.tensoralloc_add(scalartype(A), A, q, conjA))
            twist!(E, filter(x -> !isdual(space(E, x)), codomainind(E)))
            pE = ((), trivtuple(TO.numind(q)))
            pΔC = (trivtuple(TO.numind(p)), ())
            TA = promote_scale(ΔC, α)
            # TODO: allocator
            _dA = tensoralloc_contract(TA, ΔC, pΔC, conjA, E, pE, conjA, pdA, Val(false))
            _dA = tensorproduct!(_dA, ΔC, pΔC, conjA, E, pE, conjA, pdA,
                                 conjA ? α : conj(α), Zero(), ba...)
            return projectA(_dA)
        end
        dα = @thunk let
            # TODO: this result might be easier to compute as:
            # C′ = βC + α * trace(A) ⟹ At = (C′ - βC) / α
            At = tensortrace(A, p, q, conjA)
            return projectα(inner(At, ΔC))
        end
        dβ = @thunk projectβ(inner(C, ΔC))
        dba = map(_ -> NoTangent(), ba)
        return NoTangent(), dC, dA, NoTangent(), NoTangent(), NoTangent(), dα, dβ, dba...
    end

    return C′, pullback
end

function ChainRulesCore.rrule(::typeof(TensorKit.scalar), t::AbstractTensorMap)
    val = scalar(t)
    function scalar_pullback(Δval)
        dt = similar(t)
        first(blocks(dt))[2][1] = unthunk(Δval)
        return NoTangent(), dt
    end
    return val, scalar_pullback
end
