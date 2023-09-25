<picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/Jutho/TensorKit.jl/blob/master/docs/src/assets/logo-dark.svg">
    <img alt="TensorKit.jl logo" src="https://github.com/Jutho/TensorKit.jl/blob/master/docs/src/assets/logo.svg" width="150">
</picture>

# TensorKit.jl

A Julia package for large-scale tensor computations, with a hint of category theory.

| **Build Status** | **Coverage** | **Quality assurance** | **Downloads** |
|:----------------:|:------------:|:---------------------:|:--------------|
| [![CI][ci-img]][ci-url] | [![Codecov][codecov-img]][codecov-url] | [![Aqua QA][aqua-img]][aqua-url] | [![TensorKit Downloads][genie-img]][genie-url] |

[github-img]: https://github.com/Jutho/TensorKit.jl/workflows/CI/badge.svg
[github-url]: https://github.com/Jutho/TensorKit.jl/actions?query=workflow%3ACI

[ci-img]: https://github.com/Jutho/TensorKit.jl/workflows/CI/badge.svg
[ci-url]: https://github.com/Jutho/TensorKit.jl/actions?query=workflow%3ACI

[codecov-img]: https://codecov.io/gh/Jutho/TensorKit.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/Jutho/TensorKit.jl

[aqua-img]: https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg
[aqua-url]: https://github.com/JuliaTesting/Aqua.jl

[genie-img]:
    https://shields.io/endpoint?url=https://pkgs.genieframework.com/api/v1/badge/TensorKit
[genie-url]: https://pkgs.genieframework.com?packages=TensorKit

Install via the package manager.

Check out the [tutorial](https://jutho.github.io/TensorKit.jl/stable/man/tutorial/) and the full [documentation](https://jutho.github.io/TensorKit.jl/stable).


While most common symmetries are already shipped with TensorKit.jl, there exist several extensions: [SUNRepresentations.jl](https://github.com/maartenvd/SUNRepresentations.jl) provides support for SU(N), while [CategoryData.jl](https://github.com/lkdvos/CategoryData.jl) incorporates a large collection of *small* fusion categories.
Additionally, for libraries that implement tensor network algorithms on top of TensorKit.jl, check out [MPSKit.jl](https://github.com/maartenvd/MPSKit.jl), [MERAKit.jl](https://github.com/mhauru/MERAKit.jl) and [PEPSKit.jl](https://github.com/quantumghent/PEPSKit.jl).
