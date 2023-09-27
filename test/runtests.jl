using Test

const GROUP = uppercase(get(ENV, "GROUP", "ALL"))

@testset verbose=true begin
    if GROUP == "ALL" || GROUP == "SECTORS"
        @testset "Sectors" verbose = true begin
            include("sectors.jl")
        end
    end

    if GROUP == "ALL" || GROUP == "SPACES"
        @testset "Spaces" verbose = true begin
            include("spaces.jl")
        end
    end

    if GROUP == "ALL" || GROUP == "FUSIONTREES"
        @testset "Fusiontrees" verbose = true begin
            include("fusiontrees.jl")
        end
    end

    if GROUP == "ALL" || GROUP == "TENSORS"
        @testset "Tensors" verbose = true begin
            include("tensors.jl")
        end
    end
    
    if GROUP == "ALL" || GROUP == "PLANAR"
        @testset "Planar" verbose = true begin
            include("planar.jl")
        end
    end
end
