# Additional postprocessors for @planar and @plansor

# since temporaries were taken out in preprocessing by _decompose_planar_contractions, they
# are not identified by the parsing step of TensorOperations, so we have to manually fix this
# Step 1: we have to find the new name that TO.tensorify assigned to these temporaries
# since it parses `tmp[] := a[] * b[]` as `newtmp = similar...; tmp = contract!(.... , newtmp, ...)`

const _PLANAR_OPERATIONS = (:planaradd!, :planarcontract!, :planartrace!)
const _TENSOR_OPERATIONS = (:tensoradd!, :tensorcontract!, :tensortrace!)

_is_tensoroperation(ex::Expr) = ex.head == :call && (ex.args[1] in _TENSOR_OPERATIONS)
_is_tensoroperation(ex) = false

function _update_temporaries(ex, temporaries)
    if ex isa Expr && ex.head == :(=)
        lhs = ex.args[1]
        i = findfirst(==(lhs), temporaries)
        if i !== nothing
            rhs = ex.args[2]
            if _is_tensoroperation(rhs)
                temporaries[i] = rhs.args[2]
            else
                temporaries[i] = rhs
                # error("lhs = $lhs , rhs = $rhs")
            end
        end
    elseif ex isa Expr
        for a in ex.args
            _update_temporaries(a, temporaries)
        end
    end
    return ex
end

# Step 2: we find `newtmp = similar_from_...` and replace with `newtmp = cached_similar_from...`
# this is not necessary anymore?
function _annotate_temporaries(ex, temporaries)
    # if ex isa Expr && ex.head == :(=)
    #     lhs = ex.args[1]
    #     i = findfirst(==(lhs), temporaries)
    #     if i !== nothing
    #         rhs = ex.args[2]
    #         if !(rhs isa Expr && rhs.head == :call && rhs.args[1] == :similar_from_indices)
    #             @error "lhs = $lhs , rhs = $rhs"
    #         end
    #         newrhs = Expr(:call, :cached_similar_from_indices,
    #                         QuoteNode(lhs), rhs.args[2:end]...)
    #         return Expr(:(=), lhs, newrhs)
    #     end
    # elseif ex isa Expr
    #     return Expr(ex.head, [_annotate_temporaries(a, temporaries) for a in ex.args]...)
    # end
    return ex
end

# add correct modules (`GlobalRef`) to various functions
function _add_modules(ex::Expr)
    if ex.head == :call
        if ex.args[1] == :tensoradd!
            conjA = popat!(ex.args, 5)
            @assert conjA == :(:N) "conj flag should be `:N` ($conjA)"
            return Expr(ex.head, GlobalRef(TensorKit, Symbol(:planaradd!)),
                        map(_add_modules, ex.args[2:end])...)
        elseif ex.args[1] == :tensorcontract!
            conjB = popat!(ex.args, 9)
            conjA = popat!(ex.args, 6)
            @assert conjA == conjB == :(:N) "conj flag should be `:N` ($conjA), ($conjB)"
            return Expr(ex.head, GlobalRef(TensorKit, Symbol(:planarcontract!)),
                        map(_add_modules, ex.args[2:end])...)
        elseif ex.args[1] == :tensortrace!
            conjA = popat!(ex.args, 6)
            @assert conjA == :(:N) "conj flag should be `:N` ($conjA)"
            return Expr(ex.head, GlobalRef(TensorKit, Symbol(:planartrace!)),
                        map(_add_modules, ex.args[2:end])...)
        elseif ex.args[1] in TensorOperations.tensoroperationsfunctions
            return Expr(ex.head, GlobalRef(TensorOperations, ex.args[1]),
                        map(_add_modules, ex.args[2:end])...)
        end
    end
    return Expr(ex.head, (_add_modules(e) for e in ex.args)...)
end
_add_modules(ex) = ex
