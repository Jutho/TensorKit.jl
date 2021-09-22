# NEW MACROS: @planar and @plansor
macro planar(ex::Expr)
    return esc(planar_parser(ex))
end

function planar_parser(ex::Expr)
    parser = TO.TensorParser()

    braidingtensors = Vector{Any}()

    pop!(parser.preprocessors) # remove TO.extracttensorobjects
    push!(parser.preprocessors, _conj_to_adjoint)
    push!(parser.preprocessors, _extract_tensormap_objects)
    push!(parser.preprocessors, _construct_braidingtensors)
    treebuilder = parser.contractiontreebuilder
    treesorter = parser.contractiontreesorter
    push!(parser.preprocessors, ex->TO.processcontractions(ex, treebuilder, treesorter))
    push!(parser.preprocessors, ex->_check_planarity(ex))
    temporaries = Vector{Symbol}()
    push!(parser.preprocessors, ex->_decompose_planar_contractions(ex, temporaries))

    pop!(parser.postprocessors) # remove TO.addtensoroperations
    push!(parser.postprocessors, ex->_update_temporaries(ex, temporaries))
    push!(parser.postprocessors, ex->_annotate_temporaries(ex, temporaries))
    push!(parser.postprocessors, _add_modules)
    return parser(ex)
end

macro plansor(ex::Expr)
    return esc(plansor_parser(ex))
end

function plansor_parser(ex)
    inputtensors = TO.getinputtensorobjects(ex)
    newtensors = TO.getnewtensorobjects(ex)

    # find the first non-braiding tensor to determine the braidingstyle
    targetobj = inputtensors[findfirst(x->x != :τ, inputtensors)]
    targetsym = gensym()

    ex = TO.replacetensorobjects(ex) do obj, leftind, rightind
        obj == targetobj ? targetsym : obj
    end

    defaultparser = TO.TensorParser()
    insert!(defaultparser.preprocessors, 3, _remove_braidingtensors)
    defaultex = defaultparser(ex)
    planarex = planar_parser(ex)

    ex = Expr(:block)
    push!(ex.args, Expr(:(=), targetsym, targetobj))
    push!(ex.args, :(if BraidingStyle(sectortype($targetsym)) isa Bosonic
                $(defaultex)
            else
                $(planarex)
            end))
    if targetobj in newtensors
        push!(ex.args, Expr(:(=), targetobj, targetsym))
        push!(ex.args, newtensors[end])
    end
    return ex
end
