# TensorKit.jl

*A Julia package for large-scale tensor computations, with a hint of category theory.*

```@meta
CurrentModule = TensorKit
```

## Package summary

TensorKit.jl aims to be a generic package for working with tensors as they appear throughout
the physical sciences. TensorKit implements a parametric type [`Tensor`](@ref) (which is
actually a specific case of the type [`TensorMap`](@ref)) and defines for these types a
number of vector space operations (scalar multiplication, addition, norms and inner
products), index operations (permutations) and linear algebra operations (multiplication,
factorizations). Finally, tensor contractions can be performed using the `@tensor` macro
from [TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl).

Currently, most effort is oriented towards tensors as they appear in the context of quantum
many body physics and in particular the field of tensor networks. Such tensors often have
large dimensions and take on a specific structure when symmetries are present. To deal with
generic symmetries, we employ notations and concepts from category theory all the way down
to the definition of a tensor.

At the same time, TensorKit.jl focusses on computational efficiency and performance. The
underlying storage of a tensor's data can be any `DenseArray`. Currently, certain operations
are already multithreaded, either by distributing the different blocks in case of a
structured tensor (i.e. with symmetries) or by using multithreading provided by the package
[Strided.jl](https://github.com/Jutho/Strided.jl). In the future, we also plan to
investigate using `CuArray`s as underlying storage for the tensors data, so as to leverage
GPUs for the different operations defined on tensors.

## Contents of the manual

```@contents
Pages = ["man/intro.md", "man/categories.md", "man/spaces.md", "man/sectors.md", "man/tensors.md"]
Depth = 3
```

## Library outline

```@contents
Pages = ["lib/spaces.md","lib/sectors.md","lib/tensors.md"]
Depth = 2
```

## Publications using MPSKit

Below you can find a list of publications that have made use of TensorKit. If you have used
this package and wish to have your publication added to this list, please open a pull
request or an issue on the [GitHub repository](https://github.com/Jutho/TensorKit.jl/).

- G. Giudice et al., *“Temporal Entanglement, Quasiparticles, and the Role of Interactions,”* Phys. Rev. Lett., vol. 128, no. 22, p. 220401, Jun. 2022, doi: 10.1103/PhysRevLett.128.220401.
- G. Giudice, F. M. Surace, H. Pichler, and G. Giudici, *“Trimer states with ${\mathbb{Z}}_{3}$ topological order in Rydberg atom arrays,”* Phys. Rev. B, vol. 106, no. 19, p. 195155, Nov. 2022, doi: 10.1103/PhysRevB.106.195155.
- J. Huang, X. Qian, and M. Qin, *“On the Magnetization of the $120^\circ$ order of the Spin-1/2 Triangular Lattice Heisenberg Model: a DMRG revisit.”* arXiv, Oct. 18, 2023. doi: 10.48550/arXiv.2310.11774.
- C. Mc Keever and M. Lubasch, *“Classically optimized Hamiltonian simulation,”* Phys. Rev. Res., vol. 5, no. 2, p. 023146, Jun. 2023, doi: 10.1103/PhysRevResearch.5.023146.
- J. Naumann, E. L. Weerda, M. Rizzi, J. Eisert, and P. Schmoll, *“variPEPS -- a versatile tensor network library for variational ground state simulations in two spatial dimensions.”* arXiv, Aug. 23, 2023. doi: 10.48550/arXiv.2308.12358.
- S. Niu, J. Hasik, J.-Y. Chen, and D. Poilblanc, *“Chiral spin liquids on the kagome lattice with projected entangled simplex states,”* Phys. Rev. B, vol. 106, no. 24, p. 245119, Dec. 2022, doi: 10.1103/PhysRevB.106.245119.
- X. Qian and M. Qin, *“Augmenting Density Matrix Renormalization Group with Disentanglers,”* Chinese Phys. Lett., vol. 40, no. 5, p. 057102, Apr. 2023, doi: 10.1088/0256-307X/40/5/057102.
- X. Qian and M. Qin, *“Absence of Spin Liquid Phase in the $J_1-J_2$ Heisenberg model on the Square Lattice.”* arXiv, Sep. 24, 2023. doi: 10.48550/arXiv.2309.13630.
- G. Roose, N. Bultinck, L. Vanderstraeten, F. Verstraete, K. Van Acoleyen, and J. Haegeman, *“Lattice regularisation and entanglement structure of the Gross-Neveu model,”* J. High Energ. Phys., vol. 2021, no. 7, p. 207, Jul. 2021, doi: 10.1007/JHEP07(2021)207.
- R. Vanhove et al., *“Critical Lattice Model for a Haagerup Conformal Field Theory,”* Phys. Rev. Lett., vol. 128, no. 23, p. 231602, Jun. 2022, doi: 10.1103/PhysRevLett.128.231602.
