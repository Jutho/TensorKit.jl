# Additional postprocessors for @planar and @plansor

# Temporaries were explicitly created by _decompose_planar_contractions and were thus
# instantiated as if they were new output tensors rather than temporary tensors; we need
# to correct for this by adding the `istemp = true` flag.
function _annotate_temporaries(ex, temporaries)
    if isexpr(ex, :(=)) && isexpr(ex.args[2], :call) &&
            ex.args[2].args[1] ∈ (:tensoralloc_add, :tensoralloc_contract)
        lhs = ex.args[1]
        i = findfirst(==(lhs), temporaries)
        if i !== nothing
            rhs = ex.args[2]
            # add `istemp = true` flag
            newrhs = Expr(:call, rhs.args[1:end-1]..., true)
            return Expr(:(=), lhs, newrhs)
        end
    elseif ex isa Expr
        return Expr(ex.head, [_annotate_temporaries(a, temporaries) for a in ex.args]...)
    end
    return ex
end

function _contains_arg(ex, arg)
    if ex isa Expr
        return any(_contains_arg(a, arg) for a in ex.args)
    else
        return ex == arg
    end
end

function _free_temporaries(ex, temporaries)
    if isexpr(ex, :block)
        newargs = copy(ex.args)
        for t in temporaries
            i = findlast(e -> _contains_arg(e, t), newargs)
            @assert !isnothing(i)
            if i == length(newargs)
                @assert isexpr(newargs[i], :(=))
                lhs = newargs[i].args[1]
                push!(newargs, Expr(:call, :tensorfree!, t))
                push!(newargs, lhs)
            else
                newargs = insert!(newargs, i+1, Expr(:call, :tensorfree!, t))
            end
        end
        return Expr(:block, newargs...)
    end
end

# Replace the tensor operations created by `TO.instantiate` with the corresponding
# planar operations, immediately inserting them with `GlobalRef`.
function _insert_planar_operations(ex)
    if isexpr(ex, :call)
        if ex.args[1] == GlobalRef(TensorOperations, :tensoradd!)
            conjA = popat!(ex.args, 5)
            @assert conjA == :(:N) "conj flag should be `:N` ($conjA)"
            return Expr(ex.head, GlobalRef(TensorKit, Symbol(:planaradd!)),
                        map(_insert_planar_operations, ex.args[2:end])...)
        elseif ex.args[1] == GlobalRef(TensorOperations, :tensorcontract!)
            conjB = popat!(ex.args, 9)
            conjA = popat!(ex.args, 6)
            @assert conjA == conjB == :(:N) "conj flag should be `:N` ($conjA), ($conjB)"
            return Expr(ex.head, GlobalRef(TensorKit, Symbol(:planarcontract!)),
                        map(_insert_planar_operations, ex.args[2:end])...)
        elseif ex.args[1] == GlobalRef(TensorOperations, :tensortrace!)
            conjA = popat!(ex.args, 6)
            @assert conjA == :(:N) "conj flag should be `:N` ($conjA)"
            return Expr(ex.head, GlobalRef(TensorKit, Symbol(:planartrace!)),
                        map(_insert_planar_operations, ex.args[2:end])...)
        elseif ex.args[1] in TensorOperations.tensoroperationsfunctions
            return Expr(ex.head, GlobalRef(TensorOperations, ex.args[1]),
                        map(_insert_planar_operations, ex.args[2:end])...)
        end
    elseif isa(ex, Expr)
        return Expr(ex.head, (_insert_planar_operations(e) for e in ex.args)...)
    end
    return ex
end

const _PLANAR_OPERATIONS = (:planaradd!, :planarcontract!, :planartrace!)

# Mimick `TO.insert_operationbackend` for planar operations.
function insert_operationbackend(ex, backend)
    if isexpr(ex, :call) && ex.args[1] isa GlobalRef &&
       ex.args[1].mod == TensorKit &&
       ex.args[1].name ∈ _PLANAR_OPERATIONS
        b = Backend{backend}()
        return Expr(:call, ex.args..., b)
    elseif isa(ex, Expr)
        return Expr(ex.head, (insert_operationbackend(e, backend) for e in ex.args)...)
    else
        return ex
    end
end