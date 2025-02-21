# ParametrisedConvexApproximators
[ParametrisedConvexApproximators.jl](https://github.com/JinraeKim/ParametrisedConvexApproximators.jl)
is a Julia package providing predefined parametrised convex approximators and related functionalities.
An official package of [^3].


## Installation
To install ParametrisedConvexApproximator,
please open Julia's interactive session (a.k.a REPL) and press `]` key
in the REPL to use the package mode, then type the following command

```julia
pkg> add ParametrisedConvexApproximator
```

### Notes
- For PLSE(plus), the differentiation of the minimiser is now available via implicit differentiation.
- The benchmark result was reported in [ParametrisedConvexApproximator.jl v0.1.1](https://github.com/JinraeKim/ParametrisedConvexApproximators.jl/tree/v0.1.1) [^3].


## Quick Start
ParametrisedConvexApproximators.jl focuses on providing predefined approximators
including parameterized convex approximators.
Note that when approximators receive two arguments, the first and second arguments correspond to
parameter and optimization variable, usually denoted by `x` and `u`, respectively.

Note that the terms of parameter `x` and optimization variable `u` are often referred to as condition and decision from the decision-making point of view [^3].

Applications include amortized optimization (learning-based parametric optimization) [^5].

### Network construction
```julia
using ParametrisedConvexApproximators
using Flux
using Random  # to reproduce the following result

# construction
seed = 2023
Random.seed!(seed)
n, m = 3, 2
i_max = 20
T = 1.0
h_array = [64, 64]
act = Flux.leakyrelu
network = PLSE(n, m, i_max, T, h_array, act)  # parametrised log-sum-exp (PLSE) network
x, u = rand(n), rand(m)
f̂ = network(x, u)
@show f̂
```

```julia
f̂ = [3.029994811790289]
```

### Prepare dataset
```julia
min_condition = -ones(n)
max_condition = +ones(n)
min_decision = -ones(m)
max_decision = +ones(m)
target_function_name = :quadratic
target_function = example_target_function(target_function_name)  # f(x, u) = x'*x + u'*u
N = 5_000

dataset = DecisionMakingDataset(
    target_function;
    target_function_name=:quadratic,  # just for metadata
    N=N, n=n, m=m, seed=seed,
    min_condition=min_condition,
    max_condition=max_condition,
    min_decision=min_decision,
    max_decision=max_decision,
)
```

### Network training
```julia
trainer = SupervisedLearningTrainer(dataset, network; optimiser=Adam(1e-4))

@show get_loss(trainer.network, trainer.dataset[:train], trainer.loss)
@show get_loss(trainer.network, trainer.dataset[:validate], trainer.loss)
best_network = Flux.train!(trainer; epochs=200)
@show get_loss(best_network, trainer.dataset[:test], trainer.loss)
```

```julia

...

epoch: 199/200
loss_train = 0.0001664964550015733
loss_validate = 0.0003002414225961646
Best network found!
minimum_loss_validate = 0.0003002414225961646
epoch: 200/200
loss_train = 0.0001647995673689787
loss_validate = 0.00029825480495257375
Best network found!
minimum_loss_validate = 0.00029825480495257375

```

### Find a minimizer `u` for given parameter `x`
```julia
# optimization
Random.seed!(seed)
x = rand(n)  # any value
minimiser = minimise(network, x; u_min=min_decision, u_max=max_decision)  # box-constrained minimization; you can define your own optimization problem manually.
@show minimiser
@show network(x, minimiser)
@show dataset[:train].metadata.target_function(x, minimiser)
```

```julia
minimiser = [-0.003060366520019827, 0.007150205329478883]
network(x, minimiser) = [0.9629849722035002]
(dataset[:train]).metadata.target_function(x, minimiser) = 0.9666740244969058
```



## Documentation
### Types
- `AbstractApproximator` is an abstract type of approximator.
- `ParametrisedConvexApproximator <: AbstractApproximator` is an abstract type of parametrised convex approximator.
- `ConvexApproximator <: ParametrisedConvexApproximator` is an abstract type of convex approximator.
- `DifferenceOfConvexApproximator <: AbstractApproximator` is an abstract type of difference of convex approximator.

### Approximators
- All approximators in ParametrisedConvexApproximators.jl receive two arguments, namely, `x` and `u`.
When `x` and `u` are vectors whose lengths are `n` and `m`, respectively,
the output of an approximator is **one-length vector**.
    - Note that `x` and `u` can be matrices, whose sizes are `(n, d)` and `(m, d)`,
    for evaluations of `d` pairs of `x`'s and `u`'s.
    In this case, the output's size is `(1, d)`.

- The list of predefined approximators:
    - `FNN::AbstractApproximator`: feedforward neural network
    - `MA::ConvexApproximator`: max-affine (MA) network [^1]
    - `LSE::ConvexApproximator`: log-sum-exp (LSE) network [^1]
    - `PICNN::ParametrisedConvexApproximator`: partially input-convex neural network (PICNN) [^2]
    - `PMA::ParametrisedConvexApproximator`: parametrised MA (PMA) network [^3]
    - `PLSE::ParametrisedConvexApproximator`: parametrised LSE (PLSE) network [^3]
        - The default setting is `strict = false`.
        - `PLSEPlus` = `PLSE` with `strict=true`
    - `DLSE::DifferenceOfConvexApproximator`: difference of LSE (DLSE) network [^4]

### Interface
- `(nn::approximator)(x, u)` provides the approximate function value.
- `minimiser = minimise(approximator, x; u_min=nothing, u_max=nothing)` provides the minimiser for given parameter `x`
considering box constraints of `u >= u_min` and `u <= u_max` (element-wise).
    - The parameter `x` can be a vector, i.e., `size(x) = (n,)`,
    or a matrix for multiple parameters via multi-threading, i.e., `size(x) = (n, d)`.

### Dataset
- `DecisionMakingDataset`

### Trainer
- `SupervisedLearningTrainer`


## Gallery
### PMA and PLSE networks illustration
See `./examples/visualization.jl`.


#### MA network construction in theory
- The following illustration shows the construction of MA network for given convex function.
- See [^1].
- **NOTICE**: the following illustration does not show the training progress.

<img src=./figures/anim_ma.gif width=50% height=50%>

#### PMA network construction in theory
- The following illustration shows the construction of PMA network for given parameterized convex function.
- See [^3], Theorem 3.
- **NOTICE**: the following illustration does not show the training progress.

<img src=./figures/anim_pma.gif width=50% height=50%>

#### PLSE construction in theory
- The following illustration shows the PLSE network with different temperature for the corresponding PMA network constructed above.
- See [^3], Corollary 1.

<img src=./figures/anim_plse.gif width=50% height=50%>


### Comparison between MA and PMA networks
#### Subgradient selection in MA network
- To construct an MA network[^1],
any subgradient can arbitrarily be selected.

<img src=./figures/anim_ma_subgrad.gif width=50% height=50%>

#### Subgradient function selection in PMA network
- To construct an PMA network[^1],
the subgradient function, a function of parameter `x`,
should carefully be considered so that it can be continuous and represent (approximate)
the subgradient function well.
    - As shown in the following, the subgradient function may be multivalued.

<img src=./figures/anim_pma_subgrad.gif width=50% height=50%>

The subgradient function can be approximated by a continuous approximate selection.

##### Notion of continuous approximate selection
- Given multivalued function $f:X \to Y$,
a single-valued function $g: X \to Y$ is said to be a *continuous approximate selection*
if $\textup{Graph}(g) \subset \textup{Graph}(B(f, \epsilon))$.
    - The following figure adopts $L_{1}$-norm for illustration.
<img src=./figures/continuous_approximate_selection.png width=50% height=50%>


## References
[^1]: [G. C. Calafiore, S. Gaubert, and C. Possieri, “Log-Sum-Exp Neural Networks and Posynomial Models for Convex and Log-Log-Convex Data,” IEEE Transactions on Neural Networks and Learning Systems, vol. 31, no. 3, pp. 827–838, Mar. 2020, doi: 10.1109/TNNLS.2019.2910417.](https://ieeexplore.ieee.org/abstract/document/8715799?casa_token=ptHxee1NJ30AAAAA:etAIY0UkR0yg6YK7mgtEzCzHavM0d6Cos1VNzpn0cw5hbiEnFnAxNDm1rflWjDAOa-iO6xU5Lg)
[^2]: [B. Amos, L. Xu, and J. Z. Kolter, “Input Convex Neural Networks,” in Proceedings of the 34th International Conference on Machine Learning, Sydney, Australia, Jul. 2017, pp. 146–155.](http://proceedings.mlr.press/v70/amos17b.html)
[^3]: [J. Kim and Y. Kim, “Parameterized Convex Universal Approximators for Decision-Making Problems,” IEEE Trans. Neural Netw. Learning Syst., accepted for publication, 2022, doi: 10.1109/TNNLS.2022.3190198.](https://ieeexplore.ieee.org/document/9833537)
[^4]: [G. C. Calafiore, S. Gaubert, and C. Possieri, “A Universal Approximation Result for Difference of Log-Sum-Exp Neural Networks,” IEEE Transactions on Neural Networks and Learning Systems, vol. 31, no. 12, pp. 5603–5612, Dec. 2020, doi: 10.1109/TNNLS.2020.2975051.](https://ieeexplore.ieee.org/abstract/document/9032340)
[^5]: [J. Kim and Y. Kim, “Parameterized Convex Minorant for Objective Function Approximation in Amortized Optimization.” arXiv, Oct. 03, 2023. arXiv:2310.02519 (submitted to Journal of Machine Learning Research)](https://arxiv.org/abs/2310.02519)
