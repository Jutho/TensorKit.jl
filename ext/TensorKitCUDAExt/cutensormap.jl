# constants
# ---------
const _CuMatOrDict{I,T} = Union{CuMatrix{T},SectorDict{I,CuMatrix{T}}}

const CuTensorMap{T,S,N₁,N₂,I,A<:_CuMatOrDict{I,T},F₁,F₂} = TensorMap{T,S,N₁,N₂,I,A,F₁,F₂}
const CuTensor{T,S,N,I,A<:_CuMatOrDict{I,T},F₁,F₂} = CuTensorMap{T,S,N,0,I,A,F₁,F₂}
const TrivialCuTensorMap{T,S,N₁,N₂,A<:CuMatrix} = TrivialTensorMap{T,S,N₁,N₂,A}
const TrivialCuTensor{T,S,N,A<:CuMatrix} = TrivialTensor{T,S,N,A}

# constructors
# ------------
function CuTensorMap{T}(::UndefInitializer, V::TensorMapSpace{S,N₁,N₂}) where {T,S,N₁,N₂}
    A = CuMatrix{T,CUDA.DeviceMemory}
    TT = tensormaptype(S, N₁, N₂, A)
    return TT(undef, codomain(V), domain(V))
end
function CuTensorMap{T}(::UndefInitializer, codomain::TensorSpace{S},
                        domain::TensorSpace{S}) where {T,S}
    return CuTensorMap{T}(undef, codomain ← domain)
end
function CuTensor{T}(::UndefInitializer, V::TensorSpace{S}) where {T,S}
    return CuTensorMap{T}(undef, V ← one(V))
end

for (fname, felt) in ((:zeros, :zero), (:ones, :one))
    @eval begin
        function CUDA.$fname(codomain::TensorSpace{S},
                             domain::TensorSpace{S}=one(codomain)) where {S<:IndexSpace}
            return CUDA.$fname(codomain ← domain)
        end
        function CUDA.$fname(::Type{T}, codomain::TensorSpace{S},
                             domain::TensorSpace{S}=one(codomain)) where {T,S<:IndexSpace}
            return CUDA.$fname(T, codomain ← domain)
        end
        CUDA.$fname(V::TensorMapSpace) = CUDA.$fname(Float64, V)
        function CUDA.$fname(::Type{T}, V::TensorMapSpace) where {T}
            t = CuTensorMap{T}(undef, V)
            fill!(t, $felt(T))
            return t
        end
    end
end

for randfun in (:rand, :randn)
    randfun! = Symbol(randfun, :!)
    @eval begin
        # converting `codomain` and `domain` into `HomSpace`
        function CUDA.$randfun(codomain::TensorSpace{S},
                               domain::TensorSpace{S}) where {S<:IndexSpace}
            return CUDA.$randfun(codomain ← domain)
        end
        function CUDA.$randfun(::Type{T}, codomain::TensorSpace{S},
                               domain::TensorSpace{S}) where {T,S<:IndexSpace}
            return CUDA.$randfun(T, codomain ← domain)
        end
        function CUDA.$randfun(rng::Random.AbstractRNG, ::Type{T},
                               codomain::TensorSpace{S},
                               domain::TensorSpace{S}) where {T,S<:IndexSpace}
            return CUDA.$randfun(rng, T, codomain ← domain)
        end

        # accepting single `TensorSpace`
        CUDA.$randfun(codomain::TensorSpace) = CUDA.$randfun(codomain ← one(codomain))
        function CUDA.$randfun(::Type{T}, codomain::TensorSpace) where {T}
            return CUDA.$randfun(T, codomain ← one(codomain))
        end
        function CUDA.$randfun(rng::Random.AbstractRNG, ::Type{T},
                               codomain::TensorSpace) where {T}
            return CUDA.$randfun(rng, T, codomain ← one(domain))
        end

        # filling in default eltype
        CUDA.$randfun(V::TensorMapSpace) = CUDA.$randfun(Float64, V)
        function CUDA.$randfun(rng::Random.AbstractRNG, V::TensorMapSpace)
            return CUDA.$randfun(rng, Float64, V)
        end

        # filling in default rng
        function CUDA.$randfun(::Type{T}, V::TensorMapSpace) where {T}
            return CUDA.$randfun(Random.default_rng(), T, V)
        end

        # implementation
        function CUDA.$randfun(rng::Random.AbstractRNG, ::Type{T},
                               V::TensorMapSpace) where {T}
            t = CuTensorMap{T}(undef, V)
            CUDA.$randfun!(rng, t)
            return t
        end
    end
end

# converters
# ----------
function Base.convert(::Type{CuTensorMap}, d::Dict{Symbol,Any})
    try
        codomain = eval(Meta.parse(d[:codomain]))
        domain = eval(Meta.parse(d[:domain]))
        data = SectorDict(eval(Meta.parse(c)) => CuArray(b) for (c, b) in d[:data])
        return TensorMap(data, codomain, domain)
    catch e # sector unknown in TensorKit.jl; user-defined, hopefully accessible in Main
        codomain = Base.eval(Main, Meta.parse(d[:codomain]))
        domain = Base.eval(Main, Meta.parse(d[:domain]))
        data = SectorDict(Base.eval(Main, Meta.parse(c)) => CuArray(b)
                          for (c, b) in d[:data])
        return TensorMap(data, codomain, domain)
    end
end
