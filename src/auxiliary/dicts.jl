struct SingletonDict{K,V} <: AbstractDict{K,V}
    key::K
    value::V
end
SingletonDict(p::Pair{K,V}) where {K,V} = SingletonDict{K,V}(p.first, p.second)

Base.length(::SingletonDict) = 1
Base.keys(d::SingletonDict) = (d.key,)
Base.values(d::SingletonDict) = (d.value,)
Base.haskey(d::SingletonDict, key) = d.key == key
Base.getindex(d::SingletonDict, key) = d.key == key ? d.value : throw(KeyError(key))
Base.get(dict::SingletonDict, key, default) = d.key == key ? d.value : default

Base.start(::SingletonDict) = false
Base.next(d::SingletonDict, s) = (d.key => d.value), true
Base.done(d::SingletonDict, s) = s

struct VectorDict{K,V} <: AbstractDict{K,V}
    keys::Vector{K}
    values::Vector{V}
end
VectorDict{K,V}() where {K,V} = VectorDict{K,V}(Vector{K}(), Vector{V}())
function VectorDict{K,V}(kv) where {K,V}
    keys = Vector{K}()
    values = Vector{V}()
    if IteratorSize(kv) != SizeUnknown()
        sizehint!(keys, length(kv))
        sizehint!(values, length(kv))
    end
    for (k,v) in kv
        push!(keys, k)
        push!(values, v)
    end
    return VectorDict{K,V}(keys, values)
end
VectorDict(kv::Pair{K,V}...) where {K,V} = VectorDict{K,V}(kv)
function VectorDict(g::Base.Generator)
    v = collect(g)
    VectorDict(first.(v), last.(v))
end

Base.length(d::VectorDict) = length(d.keys)
Base.sizehint!(d::VectorDict, newsz) = (sizehint!(d.keys, newsz); sizehint!(d.values, newsz); return d)

@propagate_inbounds getpair(d::VectorDict, i::Integer) = d.keys[i] => d.values[i]

Base.copy(d::VectorDict) = VectorDict(copy(d.keys), copy(d.values))
empty(::VectorDict, ::Type{K}, ::Type{V}) where {K, V} = VectorDict{K, V}()
Base.empty!(d::VectorDict) = (empty!(d.keys); empty!(d.values); return d)

function Base.delete!(d::VectorDict, key)
    i = findfirst(equalto(key), d.keys)
    if !(i == nothing || i == 0)
        deleteat!(d.keys  , i)
        deleteat!(d.values, i)
    end
    return d
end

Base.keys(d::VectorDict) = d.keys
Base.values(d::VectorDict) = d.values
Base.haskey(d::VectorDict, key) = key in d.keys
function Base.getindex(d::VectorDict, key)
    i = findfirst(equalto(key), d.keys)
    @inbounds begin
        return !(i == nothing || i == 0) ? d.values[i] : throw(KeyError(key))
    end
end
function Base.setindex!(d::VectorDict, v, key)
    i = findfirst(equalto(key), d.keys)
    if i == nothing || i == 0
        push!(d.keys, key)
        push!(d.values, v)
    else
        d.values[i] = v
    end
    return d
end

function Base.get(d::VectorDict, key, default)
    i = findfirst(equalto(key), d.keys)
    @inbounds begin
        return !(i == nothing || i == 0) ? d.values[i] : default
    end
end

Base.start(::VectorDict) = 1
Base.next(d::VectorDict, s) = (d.keys[s] => d.values[s]), s+1
Base.done(d::VectorDict, s) = s > length(d)
