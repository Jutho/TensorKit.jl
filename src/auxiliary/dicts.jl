struct SingletonDict{K,V} <: AbstractDict{K,V}
    key::K
    value::V
end
SingletonDict(p::Pair{K,V}) where {K,V} = SingletonDict{K,V}(p.first,p.second)

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
    VectorDict{K,V}() where {K,V} = new{K,V}(Vector{K}(), Vector{V}())
end

Base.length(d::VectorDict) = length(d.keys)
Base.keys(d::VectorDict) = d.keys
Base.values(d::VectorDict) = d.values
Base.haskey(d::VectorDict, key) = key in d.keys
function Base.getindex(d::VectorDict, key)
    i = findfirst(equalto(key), d.keys)
    @inbounds begin
        return i != 0 ? d.values[i] : throw(KeyError(key))
    end
end
function Base.setindex!(d::VectorDict, v, key)
    i = findfirst(equalto(key), d.keys)
    if i == 0
        push!(d.keys, key)
        push!(d.values, v)
    else
        d.values[i] = v
    end
    return d
end
function Base.push!(d::VectorDict, p::Pair)
    push!(d.keys, p[1])
    push!(d.values, p[2])
    return d
end

function Base.get(d::VectorDict, key, default)
    i = findfirst(equalto(key), d.keys)
    @inbounds begin
        return i != 0 ? d.values[i] : default
    end
end

Base.start(::VectorDict) = 1
Base.next(d::VectorDict, s) = (d.keys[s] => d.values[s]), s+1
Base.done(d::VectorDict, s) = s > length(d)
