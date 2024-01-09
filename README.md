<picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/Jutho/TensorKit.jl/blob/master/docs/src/assets/logo-dark.svg">
    <img alt="TensorKit.jl logo" src="https://github.com/Jutho/TensorKit.jl/blob/master/docs/src/assets/logo.svg" width="150">
</picture>

# TensorKit.jl

A Julia package for large-scale tensor computations, with a hint of category theory.

| **Documentation** | **Digital Object Identifier** | **Downloads** |
|:-----------------:|:-----------------------------:|:-------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] | [![DOI][doi-img]][doi-url] | [![TensorOperations Downloads][downloads-img]][downloads-url] |
<!-- | [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] | [![DOI][doi-img]][doi-url] | [![TensorOperations Downloads][downloads-img]][downloads-url] | -->

| **Build Status** | **PkgEval** | **Coverage** | **Quality assurance** |
|:----------------:|:------------:|:------------:|:---------------------:|
| [![CI][ci-img]][ci-url] | [![PkgEval][pkgeval-img]][pkgeval-url] | [![Codecov][codecov-img]][codecov-url] | [![Aqua QA][aqua-img]][aqua-url] |


[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://jutho.github.io/TensorKit.jl/stable

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://jutho.github.io/TensorKit.jl/latest

[doi-img]: https://zenodo.org/badge/DOI/10.5281/zenodo.8421339.svg
[doi-url]: https://doi.org/10.5281/zenodo.8421339

[downloads-img]:
  https://shields.io/endpoint?url=https://pkgs.genieframework.com/api/v1/badge/TensorKit
[downloads-url]: https://pkgs.genieframework.com?packages=TensorKit

[ci-img]: https://github.com/Jutho/TensorKit.jl/actions/workflows/CI.yml/badge.svg
[ci-url]: https://github.com/Jutho/TensorKit.jl/actions/workflows/CI.yml

[pkgeval-img]: https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/T/TensorKit.svg
[pkgeval-url]: https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/T/TensorKit.html

[codecov-img]:
  https://codecov.io/gh/Jutho/TensorKit.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/Jutho/TensorKit.jl

[aqua-img]: https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg
[aqua-url]: https://github.com/JuliaTesting/Aqua.jl

Install via the package manager.

Check out the [tutorial](https://jutho.github.io/TensorKit.jl/stable/man/tutorial/) and the full [documentation](https://jutho.github.io/TensorKit.jl/stable).


While most common symmetries are already shipped with TensorKit.jl, there exist several extensions: [SUNRepresentations.jl](https://github.com/maartenvd/SUNRepresentations.jl) provides support for SU(N), while [CategoryData.jl](https://github.com/lkdvos/CategoryData.jl) incorporates a large collection of *small* fusion categories.
Additionally, for libraries that implement tensor network algorithms on top of TensorKit.jl, check out [MPSKit.jl](https://github.com/maartenvd/MPSKit.jl), [MERAKit.jl](https://github.com/mhauru/MERAKit.jl) and [PEPSKit.jl](https://github.com/quantumghent/PEPSKit.jl).
