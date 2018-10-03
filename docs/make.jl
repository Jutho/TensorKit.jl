using Documenter
using TensorKit

makedocs(modules=[TensorKit],
            format=:html,
            sitename="TensorKit.jl",
            pages = [
                "Home" => "index.md",
                "Manual" => ["man/intro.md", "man/spaces.md", "man/sectors.md"]
            ])

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    deps = nothing,
    make = nothing,
    target = "build",
    repo = "github.com/Jutho/TensorKit.jl.git",
    julia = "1.0"
)
