function ext._rrule_tensoradd!(C::AbstractTensorMap, A::AbstractTensorMap, pA::Index2Tuple,
                               conjA::Bool, α::Number, β::Number, ba)
    C′ = tensoradd!(copy(C), A, pA, conjA, α, β, ba...)

    projectA = ProjectTo(A)
    projectC = ProjectTo(C)
    projectα = ProjectTo(α)
    projectβ = ProjectTo(β)

    function pullback(ΔC′)
        ΔC = unthunk(ΔC′)
        dC = @thunk projectC(scale(ΔC, conj(β)))
        dA = @thunk begin
            ipA = invperm(linearize(pA))
            _dA = zerovector(A, promote_add(ΔC, α))
            _dA = tensoradd!(_dA, ΔC, (ipA, ()), conjA, conjA ? α : conj(α), Zero(), ba...)
            return projectA(_dA)
        end
        dα = @thunk begin
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
        return NoTangent(), dC, dA, NoTangent(), NoTangent(), dα, dβ
    end

    return C′, pullback
end

function ext._rrule_tensorcontract!(C::AbstractTensorMap, A::AbstractTensorMap,
                                    pA::Index2Tuple,
                                    conjA::Bool, B::AbstractTensorMap, pB::Index2Tuple,
                                    conjB::Bool, pAB::Index2Tuple, α::Number, β::Number, ba)
    C′ = tensorcontract!(copy(C), A, pA, conjA, B, pB, conjB, pAB, α, β, ba...)

    projectA = ProjectTo(A)
    projectB = ProjectTo(B)
    projectC = ProjectTo(C)
    projectα = ProjectTo(α)
    projectβ = ProjectTo(β)

    function pullback(ΔC′)
        ΔC = unthunk(ΔC′)
        ipAB = invperm(linearize(pAB))
        pΔC = (TupleTools.getindices(ipAB, trivtuple(TO.numout(pA))),
               TupleTools.getindices(ipAB, TO.numout(pA) .+ trivtuple(TO.numin(pB))))

        dC = @thunk projectC(scale(ΔC, conj(β)))
        dA = @thunk begin
            ipA = (invperm(linearize(pA)), ())
            conjΔC = conjA
            conjB′ = conjA ? conjB : !conjB
            _dA = zerovector(A,
                             promote_contract(scalartype(ΔC), scalartype(B), scalartype(α)))
            tB = twist(B,
                       TupleTools.vcat(filter(x -> !isdual(space(B, x)), pB[1]),
                                       filter(x -> isdual(space(B, x)), pB[2])))
            _dA = tensorcontract!(_dA,
                                  ΔC, pΔC, conjΔC,
                                  tB, reverse(pB), conjB′, ipA,
                                  conjA ? α : conj(α), Zero(), ba...)
            return projectA(_dA)
        end
        dB = @thunk begin
            ipB = (invperm(linearize(pB)), ())
            conjΔC = conjB
            conjA′ = conjB ? conjA : !conjA
            _dB = zerovector(B,
                             promote_contract(scalartype(ΔC), scalartype(A), scalartype(α)))
            tA = twist(A,
                       TupleTools.vcat(filter(x -> isdual(space(A, x)), pA[1]),
                                       filter(x -> !isdual(space(A, x)), pA[2])))
            _dB = tensorcontract!(_dB,
                                  tA, reverse(pA), conjA′,
                                  ΔC, pΔC, conjΔC, ipB,
                                  conjB ? α : conj(α), Zero(), ba...)
            return projectB(_dB)
        end
        dα = @thunk begin
            # TODO: this result should be AB = (C′ - βC) / α as C′ = βC + αAB
            AB = tensorcontract(A, pA, conjA, B, pB, conjB, pAB, One(), ba...)
            return projectα(inner(AB, ΔC))
        end
        dβ = @thunk projectβ(inner(C, ΔC))
        return NoTangent(), dC,
               dA, NoTangent(), NoTangent(),
               dB, NoTangent(), NoTangent(),
               NoTangent(), dα, dβ
    end
    return C′, pullback
end

function ext._rrule_tensortrace!(C::AbstractTensorMap, A::AbstractTensorMap, p::Index2Tuple,
                                 q::Index2Tuple, conjA::Bool, α::Number, β::Number, ba)
    C′ = tensortrace!(copy(C), A, p, q, conjA, α, β, ba...)

    projectA = ProjectTo(A)
    projectC = ProjectTo(C)
    projectα = ProjectTo(α)
    projectβ = ProjectTo(β)

    function pullback(ΔC′)
        ΔC = unthunk(ΔC′)
        dC = @thunk projectC(scale(ΔC, conj(β)))
        dA = @thunk begin
            ip = invperm((linearize(p)..., q[1]..., q[2]...))
            E = one!(TO.tensoralloc_add(scalartype(A), A, q, conjA))
            twist!(E, filter(x -> !isdual(space(E, x)), codomainind(E)))
            _dA = zerovector(A, promote_scale(ΔC, α))
            _dA = tensorproduct!(_dA, ΔC,
                                 (trivtuple(TO.numind(p)), ()), conjA, E,
                                 ((), trivtuple(TO.numind(q))), conjA, (ip, ()),
                                 conjA ? α : conj(α), Zero(), ba...)
            return projectA(_dA)
        end
        dα = @thunk begin
            # TODO: this result might be easier to compute as:
            # C′ = βC + α * trace(A) ⟹ At = (C′ - βC) / α
            At = tensortrace(A, p, q, conjA)
            return projectα(inner(At, ΔC))
        end
        dβ = @thunk projectβ(inner(C, ΔC))
        return NoTangent(), dC, dA, NoTangent(), NoTangent(), NoTangent(), dα, dβ
    end

    return C′, pullback
end
