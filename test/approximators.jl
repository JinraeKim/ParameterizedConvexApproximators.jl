using ParametrisedConvexApproximators
const PCA = ParametrisedConvexApproximators
using Test

using Flux
using Transducers
using Convex
using ForwardDiff
using Plots
using UnPack
using Statistics
using BenchmarkTools


"""
test function
"""
function f(x, u)
    0.5 * (-x'*x + u'*u)
end

function supervised_learning!(normalised_approximator, xuf_data)
    @show normalised_approximator
    xuf_data_train, xuf_data_test = PCA.partitionTrainTest(xuf_data)
    train_approximator!(normalised_approximator, xuf_data_train, xuf_data_test;
                        loss=SupervisedLearningLoss(normalised_approximator),
                            epochs=100,
                           )
    println("No error while training the normalised_approximator")
end

function basic_test(normalised_approximator, _xs, _us)
    @show normalised_approximator
    @show number_of_params(normalised_approximator)
    d = size(_xs)[2]
    m = size(_us)[1]
    _x = _xs[:, 1]
    _u = _us[:, 1]
    u_convex = Convex.Variable(length(_us[:, 1]))
    u_convex.value = _u
    @testset "infer size check" begin
        @test normalised_approximator(_xs, _us) |> size == (1, d)
        @test normalised_approximator(_x, _u) |> size == (1,)
    end
    @testset "Convex.Variable evaluation check" begin
        if normalised_approximator.approximator |> typeof <: ParametrisedConvexApproximator
            @test normalised_approximator(_x, u_convex.value) ≈ normalised_approximator(_x, _u)
            @test evaluate.(normalised_approximator(_x, u_convex)) ≈ normalised_approximator(_x, _u)
        else
            @warn("test ignored; the approximator is not parametrised convex")
        end
    end
end

function generate_approximators(xuf_data)
    n, m, d = length(xuf_data.x[1]), length(xuf_data.u[1]), xuf_data.d
    i_max = 20
    h_array = [16, 16, 16]
    h_array_fnn = [16, 16, 16]
    T = 1e-1
    act = Flux.leakyrelu
    u_is = range(-1, 1, length=i_max) |> Map(_u_i -> [_u_i]) |> collect  # to make it a matrix
    # u_star_is = u_is |> Map(u_i -> (x -> f_partial_u(x, u_i))) |> collect
    u_star_is = u_is |> Map(u_i -> (x -> ForwardDiff.gradient(u -> f(x, u), u_i))) |> collect
    α_is = 1:i_max |> Map(i -> rand(n+m)) |> collect
    β_is = 1:i_max |> Map(i -> rand(1)) |> collect

    # test
    fnn = FNN(n, m, h_array_fnn, act)
    ma = MA(α_is, β_is; n=n, m=m)
    lse = LSE(α_is, β_is, T; n=n, m=m)
    pma_basic = PMA(n, m, i_max, h_array, act)
    # pma_theoretical = PMA(n, m, u_is, u_star_is, f)
    plse = PLSE(n, m, i_max, T, h_array, act)
    approximators = (;
                     # ma=NormalisedApproximator(ma, MinMaxNormaliser(xuf_data)),
                     # lse=NormalisedApproximator(lse, MinMaxNormaliser(xuf_data)),
                     # pma_basic=NormalisedApproximator(pma_basic, MinMaxNormaliser(xuf_data)),  # Note: MinMaxNormaliser is better than StandardNormalDistributionNormaliser
                     # pma_basic=NormalisedApproximator(pma_basic, StandardNormalDistributionNormaliser(xuf_data)),
                     # pma_theoretical=pma_theoretical,  # TODO: make it compatible with Flux.jl's auto-diff
                     # plse=NormalisedApproximator(plse, IdentityNormaliser()),  # Note: MinMaxNormaliser is better than StandardNormalDistributionNormaliser
                     # plse=NormalisedApproximator(plse, StandardNormalDistributionNormaliser(xuf_data)),
                     # fnn=NormalisedApproximator(fnn, MinMaxNormaliser(xuf_data)),  # Note: MinMaxNormaliser is better than StandardNormalDistributionNormaliser
                     fnn=NormalisedApproximator(fnn, IdentityNormaliser()),  # Note: MinMaxNormaliser is better than StandardNormalDistributionNormaliser
                     # ma=NormalisedApproximator(ma, MinMaxNormaliser(xuf_data)),  # Note: MinMaxNormaliser is better than StandardNormalDistributionNormaliser
                     # lse=NormalisedApproximator(lse, MinMaxNormaliser(xuf_data)),  # Note: MinMaxNormaliser is better than StandardNormalDistributionNormaliser
                     # pma_basic=NormalisedApproximator(pma_basic, MinMaxNormaliser(xuf_data)),  # Note: MinMaxNormaliser is better than StandardNormalDistributionNormaliser
                     # plse=NormalisedApproximator(plse, MinMaxNormaliser(xuf_data)),  # Note: MinMaxNormaliser is better than StandardNormalDistributionNormaliser
                     plse=NormalisedApproximator(plse, IdentityNormaliser()),  # Note: MinMaxNormaliser is better than StandardNormalDistributionNormaliser
                    )  # NT
    _approximators = Dict(zip(keys(approximators), values(approximators)))  # Dict
end

function generate_data(n, m, d, xlim, ulim)
    xs = 1:d |> Map(i -> xlim[1] .+ (xlim[2]-xlim[1]) .* rand(n)) |> collect
    us = 1:d |> Map(i -> ulim[1] .+ (ulim[2]-ulim[1]) .* rand(m)) |> collect
    fs = zip(xs, us) |> MapSplat((x, u) -> f(x, u)) |> collect
    xuf_data = PCA.xufData(xs, us, fs)
end

function infer_test(normalised_approximator::NormalisedApproximator,
        n, m, xlim, ulim)
    @show normalised_approximator
    @testset "infer_test" begin
        @warn("The true minimiser is realised ad hoc;
              you may have to manually change the true minimiser
              when changing the target funciton, namely, `f`.")
        d = 100  # inference test
        xuf_data = generate_data(n, m, d, xlim, ulim)
        _xs = hcat(xuf_data.x...)
        _minimiser_true = zeros(m, d)
        _optval_true = hcat((1:d |> Map(i -> f(_xs[:, i], _minimiser_true[:, i])) |> collect)...)
        @time _res = solve!(normalised_approximator, _xs; lim=(ulim[1]*ones(m), ulim[2]*ones(m)))
        errors_minimiser = 1:d |> Map(i -> norm(_res.minimiser[:, i] - _minimiser_true[:, i])) |> collect
        @show mean(errors_minimiser)
        errors_optval = 1:d |> Map(i -> abs(_res.optval[i] - _optval_true[i])) |> collect
        @show mean(errors_optval)
    end
end

@testset "approximators" begin
    dir_log = "figures/test"
    mkpath(dir_log)
    n, m, d = 10, 10, 1_000
    @show n, m, d
    xlim = (-1, 1)
    ulim = (-1, 1)
    xuf_data = generate_data(n, m, d, xlim, ulim)
    _approximators = generate_approximators(xuf_data)
    approximators = (; _approximators...)  # NT
    _xs = hcat(xuf_data.x...)
    _us = hcat(xuf_data.u...)
    # test
    println("Testing basic functionality...")
    approximators |> Map(approx -> basic_test(approx, _xs, _us)) |> collect
    # training
    println("Testing supervised_learning...")
    approximators |> Map(approx -> supervised_learning!(approx, xuf_data)) |> collect
    # inference
    println("Testing inference...")
    approximators |> Map(approx -> infer_test(approx, n, m, xlim, ulim)) |> collect
    # figures
    println("Printing figures...")
    if n == 1 && m == 1
        figs_true = 1:length(approximators) |> Map(approx -> plot(;
                                                                    xlim=(-1, 1), ylim=(-1, 1), zlim=(-25, 25),
                                                                   )) |> collect
        _ = zip(figs_true, approximators) |> MapSplat((fig, approx) -> plot_approx!(fig, (x, u) -> [f(x, u)], xlim, ulim)) |> collect
        figs_approx = 1:length(approximators) |> Map(approx -> plot(;
                                                                    xlim=(-1, 1), ylim=(-1, 1), zlim=(-25, 25),
                                                                   )) |> collect
        _ = zip(figs_approx, approximators) |> MapSplat((fig, approx) -> plot_approx!(fig, approx, xlim, ulim)) |> collect
        # subplots
        figs = zip(figs_approx, figs_true) |> MapSplat((fig_approx, fig_true) -> plot(fig_approx, fig_true; layout=(1, 2),)) |> collect
        # save figs
        _ = zip(figs, _approximators) |> MapSplat((fig, _approx) -> savefig(fig, joinpath(dir_log, "$(_approx[1]).png"))) |> collect
    else
        @warn("printing figures ignored; dimension (`n` and `m`) should be `1` and `1`, resp.")
    end
end
