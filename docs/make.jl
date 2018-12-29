using Documenter
using TensorKit

makedocs(modules=[TensorKit],
            sitename="TensorKit.jl",
            authors = "Jutho Haegeman",
            pages = [
                "Home" => "index.md",
                "Manual" => ["man/intro.md", "man/spaces.md", "man/sectors.md", "man/tensors.md"],
                "Library" => ["lib/spaces.md"]
            ])

deploydocs(repo = "github.com/Jutho/TensorKit.jl.git")
