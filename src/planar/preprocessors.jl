@noinline not_planar_err() = throw(ArgumentError("not a planar diagram expression"))
@noinline not_planar_err(ex) = throw(ArgumentError("not a planar diagram expression: $ex"))

# Preprocessors used by `@planar` and `@plansor`
function _conj_to_adjoint(ex)
    if isexpr(ex, :call) && ex.args[1] == :conj && TO.istensor(ex.args[2])
        obj, leftind, rightind = TO.decomposetensor(ex.args[2])
        return Expr(:typed_vcat, Expr(TO.prime, obj),
                    Expr(:tuple, rightind...), Expr(:tuple, leftind...))
    elseif ex isa Expr
        return Expr(ex.head, [_conj_to_adjoint(a) for a in ex.args]...)
    else
        return ex
    end
end

# replacement of TensorOperations functionality:
# adds checks for matching number of domain and codomain indices
# special cases adjoints so that t and t' are considered the same object
# ignore braiding tensors
function _extract_tensormap_objects(ex)
    inputtensors = collect(filter(!=(:τ), _remove_adjoint.(TO.getinputtensorobjects(ex))))
    outputtensors = _remove_adjoint.(TO.getoutputtensorobjects(ex))
    newtensors = TO.getnewtensorobjects(ex)
    (any(==(:τ), newtensors) || any(==(:τ), outputtensors)) &&
        throw(ArgumentError("The name τ is reserved for the braiding, and should not be assigned to."))
    @assert !any(_is_adjoint, newtensors)
    existingtensors = unique!(vcat(inputtensors, outputtensors))
    alltensors = unique!(vcat(existingtensors, newtensors))
    tensordict = Dict{Any,Any}(a => gensym(string(a))
                               for a in alltensors if !(a isa Symbol))
    pre = Expr(:block,
               [Expr(:(=), tensordict[a], a) for a in existingtensors if !(a isa Symbol)]...)
    pre2 = Expr(:block)
    ex = TO.replacetensorobjects(ex) do obj, leftind, rightind
        _is_adj = _is_adjoint(obj)
        if _is_adj
            leftind, rightind = rightind, leftind
            obj = _remove_adjoint(obj)
        end
        newobj = get(tensordict, obj, obj)
        if (obj in existingtensors)
            nl = length(leftind)
            nr = length(rightind)
            tensor_inds = :((numout($newobj)), (numin($newobj)))
            errorstr = Expr(:string,
                            "Incorrect number of input-output indices for $obj: ($nl, $nr) instead of (",
                            tensor_inds, ").")
            checksize = quote
                (numout($newobj) == $nl && numin($newobj) == $nr) ||
                    throw(IndexError($errorstr))
            end
            push!(pre2.args, checksize)
        end
        return _is_adj ? _add_adjoint(newobj) : newobj
    end
    post = Expr(:block,
                [Expr(:(=), a, tensordict[a]) for a in newtensors if !(a isa Symbol)]...)
    pre = Expr(:macrocall, Symbol("@notensor"),
               LineNumberNode(@__LINE__, Symbol(@__FILE__)), pre)
    pre2 = Expr(:macrocall, Symbol("@notensor"),
                LineNumberNode(@__LINE__, Symbol(@__FILE__)), pre2)
    post = Expr(:macrocall, Symbol("@notensor"),
                LineNumberNode(@__LINE__, Symbol(@__FILE__)), post)
    return Expr(:block, pre, pre2, ex, post)
end
_is_adjoint(ex) = isexpr(ex, TO.prime)
_remove_adjoint(ex) = _is_adjoint(ex) ? ex.args[1] : ex
_add_adjoint(ex) = Expr(TO.prime, ex)

# used by `@planar`: identify braiding tensors (corresponding to name τ) and discover their 
# spaces from the rest of the expression. Construct the explicit BraidingTensor objects and
# insert them in the expression.
function _construct_braidingtensors(ex::Expr)
    if TO.isdefinition(ex) || TO.isassignment(ex)
        lhs, rhs = TO.getlhs(ex), TO.getrhs(ex)
        if !TO.istensorexpr(rhs)
            return ex
        end
        preargs = Vector{Any}()
        indexmap = Dict{Any,Any}()
        if TO.isassignment(ex) && TO.istensor(lhs)
            obj, leftind, rightind = TO.decomposetensor(lhs)
            for (i, l) in enumerate(leftind)
                indexmap[l] = Expr(:call, :space, _add_adjoint(obj), i)
            end
            for (i, l) in enumerate(rightind)
                indexmap[l] = Expr(:call, :space, _add_adjoint(obj), length(leftind) + i)
            end
        end
        newrhs, success = _construct_braidingtensors!(rhs, preargs, indexmap)
        success ||
            throw(ArgumentError("cannot determine the spaces of all braiding tensors in $ex"))
        pre = Expr(:macrocall, Symbol("@notensor"),
                   LineNumberNode(@__LINE__, Symbol(@__FILE__)), Expr(:block, preargs...))
        return Expr(:block, pre, Expr(ex.head, lhs, newrhs))
    elseif TO.istensorexpr(ex)
        preargs = Vector{Any}()
        indexmap = Dict{Any,Any}()
        newex, success = _construct_braidingtensors!(ex, preargs, indexmap)
        success ||
            throw(ArgumentError("cannot determine the spaces of all braiding tensors in $ex"))
        pre = Expr(:macrocall, Symbol("@notensor"),
                   LineNumberNode(@__LINE__, Symbol(@__FILE__)), Expr(:block, preargs...))
        return Expr(:block, pre, newex)
    else
        return Expr(ex.head, map(_construct_braidingtensors, ex.args)...)
    end
end
_construct_braidingtensors(x) = x

function _construct_braidingtensors!(ex, preargs, indexmap) # ex is guaranteed to be a single tensor expression
    if TO.istensor(ex)
        obj, leftind, rightind = TO.decomposetensor(ex)
        if _remove_adjoint(obj) == :τ
            # try to construct a braiding tensor
            length(leftind) == length(rightind) == 2 ||
                throw(ArgumentError("The name τ is reserved for the braiding, and should have two input and two output indices."))
            if _is_adjoint(obj)
                i1b, i2b, = leftind
                i2a, i1a, = rightind
            else
                i2b, i1b, = leftind
                i1a, i2a, = rightind
            end

            foundV1, foundV2 = false, false
            if haskey(indexmap, i1a)
                V1 = indexmap[i1a]
                foundV1 = true
            elseif haskey(indexmap, i1b)
                V1 = Expr(:call, :dual, indexmap[i1b])
                foundV1 = true
            end
            if haskey(indexmap, i2a)
                V2 = indexmap[i2a]
                foundV2 = true
            elseif haskey(indexmap, i2b)
                V2 = Expr(:call, :dual, indexmap[i2b])
                foundV2 = true
            end
            if foundV1 && foundV2
                s = gensym(:τ)
                constructex = Expr(:call, GlobalRef(TensorKit, :BraidingTensor), V1, V2)
                push!(preargs, Expr(:(=), s, constructex))
                obj = _is_adjoint(obj) ? _add_adjoint(s) : s
                success = true
            else
                success = false
            end
            newex = Expr(:typed_vcat, obj, Expr(:tuple, leftind...),
                         Expr(:tuple, rightind...))
        else
            newex = ex
            success = true
        end
        # add spaces of the tensor to the indexmap
        for (i, l) in enumerate(leftind)
            if !haskey(indexmap, l)
                indexmap[l] = Expr(:call, :space, obj, i)
            end
        end
        for (i, l) in enumerate(rightind)
            if !haskey(indexmap, l)
                indexmap[l] = Expr(:call, :space, obj, length(leftind) + i)
            end
        end
        return newex, success
    elseif TO.isgeneraltensor(ex)
        args = ex.args
        newargs = Vector{Any}(undef, length(args))
        success = true
        for i in 1:length(ex.args)
            newargs[i], successa = _construct_braidingtensors!(args[i], preargs, indexmap)
            success = success && successa
        end
        newex = Expr(ex.head, newargs...)
        return newex, success
    elseif isexpr(ex, :call) && ex.args[1] == :*
        args = ex.args
        newargs = Vector{Any}(undef, length(args))
        newargs[1] = args[1]
        successes = map(i -> false, args)
        successes[1] = true
        numsuccess = 1
        while !all(successes)
            for i in 2:length(ex.args)
                successes[i] && continue
                newargs[i], successa = _construct_braidingtensors!(args[i], preargs,
                                                                   indexmap)
                successes[i] = successa
            end
            if numsuccess == count(successes)
                break
            end
            numsuccess = count(successes)
        end
        success = numsuccess == length(successes)
        newex = Expr(ex.head, newargs...)
        return newex, success
    elseif isexpr(ex, :call) && ex.args[1] ∈ (:+, :-)
        args = ex.args
        newargs = Vector{Any}(undef, length(args))
        newargs[1] = args[1]
        success = true
        indices = TO.getindices(ex)
        for i in 2:length(ex.args)
            indexmapa = copy(indexmap)
            newargs[i], successa = _construct_braidingtensors!(args[i], preargs, indexmapa)
            for l in indices[i]
                if !haskey(indexmap, l) && haskey(indexmapa, l)
                    indexmap[l] = indexmapa[l]
                end
            end
            success = success && successa
        end
        newex = Expr(ex.head, newargs...)
        return newex, success
    else
        @show("huh?")
        return ex, true
    end
end

# used by non-planar parser of `@plansor`: remove explicit braiding tensors
function _remove_braidingtensors(ex)
    ex isa Expr || return ex
    outgoing = []

    if TO.isdefinition(ex) || TO.isassignment(ex)
        lhs, rhs = TO.getlhs(ex), TO.getrhs(ex)
        if TO.istensorexpr(rhs)
            list = TO.gettensors(_conj_to_adjoint(rhs))
            if TO.istensor(lhs)
                obj, l, r = TO.decomposetensor(lhs)
                outgoing = [l; r]
            end
        else
            return ex
        end
    elseif TO.istensorexpr(ex)
        list = TO.gettensors(_conj_to_adjoint(ex))
    else
        return Expr(ex.head, map(_remove_braidingtensors, ex.args)...)
    end

    τs = Any[]
    i = 1
    while i <= length(list)
        t = list[i]
        if _remove_adjoint(TO.gettensorobject(t)) == :τ
            push!(τs, t)
            deleteat!(list, i)
        else
            i += 1
        end
    end

    indexmap = Dict{Any,Any}()
    # to remove the braidingtensors, we need to map certain indices to other indices
    for t in τs
        obj, leftind, rightind = TO.decomposetensor(t)
        length(leftind) == length(rightind) == 2 ||
            throw(ArgumentError("The name τ is reserved for the braiding, and should have two input and two output indices."))
        if _is_adjoint(obj)
            i1b, i2b, = leftind
            i2a, i1a, = rightind
        else
            i2b, i1b, = leftind
            i1a, i2a, = rightind
        end

        i1a = get(indexmap, i1a, i1a)
        i1b = get(indexmap, i1b, i1b)
        i2a = get(indexmap, i2a, i2a)
        i2b = get(indexmap, i2b, i2b)

        obj_and_pos1a = _findindex(i1a, list)
        obj_and_pos2a = _findindex(i2a, list)
        obj_and_pos1b = _findindex(i1b, list)
        obj_and_pos2b = _findindex(i2b, list)

        if i1a in outgoing
            indexmap[i1a] = i1a
            indexmap[i1b] = i1a
        elseif i1b in outgoing
            indexmap[i1a] = i1b
            indexmap[i1b] = i1b
        else
            if i1a isa Int && i1b isa Int
                indexmap[i1a] = max(i1a, i1b)
                indexmap[i1b] = max(i1a, i1b)
            else
                indexmap[i1a] = i1a
                indexmap[i1b] = i1a
            end
        end

        if i2a in outgoing
            indexmap[i2a] = i2a
            indexmap[i2b] = i2a
        elseif i2b in outgoing
            indexmap[i2a] = i2b
            indexmap[i2b] = i2b
        else
            if i2a isa Int && i2b isa Int
                indexmap[i2a] = max(i2a, i2b)
                indexmap[i2b] = max(i2a, i2b)
            else
                indexmap[i2a] = i2a
                indexmap[i2b] = i2a
            end
        end
    end

    # simple loop that tries to simplify the indicemaps (a=>b,b=>c -> a=>c,b=>c)
    changed = true
    while changed == true
        changed = false
        i = 1
        for (k, v) in indexmap
            if v in keys(indexmap) && indexmap[v] != v
                changed = true
                indexmap[k] = indexmap[v]
            end
        end
    end

    ex = TO.replaceindices(i -> get(indexmap, i, i), ex)
    return _purge_braidingtensors(ex)
end

function _purge_braidingtensors(ex) # actually remove the braidingtensors
    ex isa Expr || return ex
    args = collect(filter(ex.args) do a
                       if isexpr(a, :call) && a.args[1] == :conj
                           a = a.args[2]
                       end
                       if a isa Expr && TO.istensor(a) &&
                          _remove_adjoint(TO.gettensorobject(a)) == :τ
                           _, leftind, rightind = TO.decomposetensor(a)
                           (leftind[1] == rightind[2] && leftind[2] == rightind[1]) ||
                               throw(ArgumentError("unable to remove braiding tensor $a"))
                           return false
                       end
                       return true
                   end)

    # multiplication with only a single argument is (rightfully) seen as invalid syntax
    if isexpr(ex, :call) && args[1] == :* && length(args) == 2
        return _purge_braidingtensors(args[2])
    else
        return Expr(ex.head, map(_purge_braidingtensors, args)...)
    end
end

function _check_planarity(ex)
    ex isa Expr || return ex
    if isexpr(ex, :macrocall) && ex.args[1] == Symbol("@notensor")
        return ex
    elseif TO.isassignment(ex) || TO.isdefinition(ex)
        lhs, rhs = TO.getlhs(ex), TO.getrhs(ex)
        if TO.istensorexpr(rhs)
            if TO.istensorexpr(lhs)
                @assert TO.istensor(lhs)
                indlhs = only(get_possible_planar_indices(lhs)) # should have only one element
            else
                indlhs = Any[]
            end
            indsrhs = get_possible_planar_indices(rhs)
            isempty(indsrhs) && not_planar_err(rhs)
            i = findfirst(ind -> iscyclicpermutation(ind, indlhs), indsrhs)
            i === nothing && not_planar_err(ex)
        end
    else
        foreach(ex.args) do a
            return _check_planarity(a)
        end
    end
    return ex
end

# decompose contraction trees in order to fix index order of temporaries
# to ensure that planarity is guaranteed
_decompose_planar_contractions(ex, temporaries) = ex
function _decompose_planar_contractions(ex::Expr, temporaries)
    if isexpr(ex, :macrocall) && ex.args[1] == Symbol("@notensor")
        return ex
    end
    if TO.isassignment(ex) || TO.isdefinition(ex)
        lhs, rhs = TO.getlhs(ex), TO.getrhs(ex)
        if TO.istensorexpr(rhs)
            pre = Vector{Any}()
            if TO.istensor(lhs)
                rhs = _extract_contraction_pairs(rhs, lhs, pre, temporaries)
                return Expr(:block, pre..., Expr(ex.head, lhs, rhs))
            else
                lhssym = gensym(string(lhs))
                lhstensor = Expr(:typed_vcat, lhssym, Expr(:tuple), Expr(:tuple))
                rhs = _extract_contraction_pairs(rhs, lhstensor, pre, temporaries)
                push!(temporaries, lhssym)
                return Expr(:block, pre..., Expr(:(:=), lhstensor, rhs),
                            Expr(:(=), lhs, lhstensor))
            end
        else
            return ex
        end
    end
    if TO.istensorexpr(ex)
        pre = Vector{Any}()
        rhs = _extract_contraction_pairs(ex, (Any[], Any[]), pre, temporaries)
        return Expr(:block, pre..., rhs)
    end
    if isexpr(ex, :block)
        return Expr(ex.head,
                    [_decompose_planar_contractions(a, temporaries) for a in ex.args]...)
    end
    return ex
end

# decompose a contraction into elementary binary contractions of tensors without inner traces
# if lhs is an expression, it contains the existing lhs and thus the index order
# if lhs is a tuple, the result is a temporary object and the tuple (lind, rind) gives a suggestion for the preferred index order
function _extract_contraction_pairs(rhs, lhs, pre, temporaries)
    if TO.isscalarexpr(rhs)
        return rhs
    elseif TO.isgeneraltensor(rhs)
        if TO.hastraceindices(rhs) && lhs isa Tuple
            s = gensym()
            newlhs = Expr(:typed_vcat, s, Expr(:tuple, lhs[1]...), Expr(:tuple, lhs[2]...))
            push!(temporaries, s)
            push!(pre, Expr(:(:=), newlhs, rhs))
            return newlhs
        else
            return rhs
        end
    elseif isexpr(rhs, :call) && rhs.args[1] == :*
        # @assert length(rhs.args) == 3 # has already been checked in _check_planarity

        if lhs isa Expr
            _, leftind, rightind = TO.decomposetensor(lhs)
        else
            leftind, rightind = lhs
        end
        lhs_ind = vcat(leftind, reverse(rightind))

        # find possible planar order
        rhs_inds = Any[]
        for ind1 in get_possible_planar_indices(rhs.args[2])
            for ind2 in get_possible_planar_indices(rhs.args[3])
                for (oind1, oind2, cind1, cind2) in possible_planar_complements(ind1, ind2)
                    if iscyclicpermutation(vcat(oind1, oind2), lhs_ind)
                        push!(rhs_inds, (ind1, ind2, oind1, oind2, cind1, cind2))
                    end
                    isempty(rhs_inds) || break
                end
                isempty(rhs_inds) || break
            end
            isempty(rhs_inds) || break
        end
        ind1, ind2, oind1, oind2, cind1, cind2 = only(rhs_inds) # inds_rhs should hold exactly one match
        if all(in(leftind), oind2) || all(in(rightind), oind1) # reverse order
            a1 = _extract_contraction_pairs(rhs.args[3], (oind2, reverse(cind2)), pre,
                                            temporaries)
            a2 = _extract_contraction_pairs(rhs.args[2], (cind1, reverse(oind1)), pre,
                                            temporaries)
            oind1, oind2 = oind2, oind1
            cind1, cind2 = cind2, cind1
        else
            a1 = _extract_contraction_pairs(rhs.args[2], (oind1, reverse(cind1)), pre,
                                            temporaries)
            a2 = _extract_contraction_pairs(rhs.args[3], (cind2, reverse(oind2)), pre,
                                            temporaries)
        end
        # @show a1, a2, oind1, oind2

        if TO.isscalarexpr(a1) || TO.isscalarexpr(a2)
            rhs = Expr(:call, :*, a1, a2)
            s = gensym()
            newlhs = Expr(:typed_vcat, s, Expr(:tuple, oind1...),
                          Expr(:tuple, reverse(oind2)...))
            push!(temporaries, s)
            push!(pre, Expr(:(:=), newlhs, rhs))
            return newlhs
        end

        # note that index order in `lhs` is only a suggestion, now we have actual index order
        _, l1, r1, = TO.decomposegeneraltensor(a1)
        _, l2, r2, = TO.decomposegeneraltensor(a2)
        if all(in(r1), oind1) && all(in(l2), oind2) # reverse order
            a1, a2 = a2, a1
            ind1, ind2 = ind2, ind1
            oind1, oind2 = oind2, oind1
        end
        # @show a1, a2, oind1, oind2
        if lhs isa Tuple
            rhs = Expr(:call, :*, a1, a2)
            s = gensym()
            newlhs = Expr(:typed_vcat, s, Expr(:tuple, oind1...),
                          Expr(:tuple, reverse(oind2)...))
            push!(temporaries, s)
            push!(pre, Expr(:(:=), newlhs, rhs))
            return newlhs
        else
            if leftind == oind1 && rightind == reverse(oind2)
                rhs = Expr(:call, :*, a1, a2)
                return rhs
            else
                rhs = Expr(:call, :*, a1, a2)
                s = gensym()
                newlhs = Expr(:typed_vcat, s, Expr(:tuple, oind1...),
                              Expr(:tuple, reverse(oind2)...))
                push!(temporaries, s)
                push!(pre, Expr(:(:=), newlhs, rhs))
                return newlhs
            end
        end
    elseif isexpr(rhs, :call) && rhs.args[1] ∈ (:+, :-)
        args = [_extract_contraction_pairs(a, lhs, pre, temporaries)
                for
                a in rhs.args[2:end]]
        return Expr(rhs.head, rhs.args[1], args...)
    else
        throw(ArgumentError("unknown tensor expression"))
    end
end

function _findindex(i, list) # finds an index i in a list of tensor expressions
    for t in list
        obj, l, r = TO.decomposetensor(t)
        pos = findfirst(==(i), l)
        isnothing(pos) || return (obj, pos)
        pos = findfirst(==(i), r)
        isnothing(pos) || return (obj, pos + length(l))
    end
    return nothing
end
