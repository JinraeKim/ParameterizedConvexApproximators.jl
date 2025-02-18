using ParametrisedConvexApproximators
using Flux
using Plots


seed = 2022
n, m = 1, 1
N = 5_000
h_array = [64, 64]
act = Flux.leakyrelu
i_max = 20
T = 1.0
# dataset
min_condition = -ones(n)
max_condition = +ones(n)
min_decision = -ones(m)
max_decision = +ones(m)


function composite_loss(pred, f)
    tmp = max.(0, pred .- f)
    l = Flux.Losses.mse(pred, f) + Flux.Losses.mae(tmp, zeros(size(tmp)))  # the last term is for pred >= f
    return l
end


function main(epochs=2)
    plse = PLSE(n, m, i_max, T, h_array, act)
    # eplse = EPLSE(
    #               PLSE(n, m, i_max, T, h_array, act),
    #               FNN(n, m, h_array, act),
    #               min_decision,
    #               max_decision,
    #              )

    networks = Dict(
                    :PLSE => plse,
                    # :EPLSE => eplse,
                   )

    target_function = example_target_function(:quadratic_sin_sum)
    dataset = generate_dataset(
        target_function;
        N,
        min_condition,
        max_condition,
        min_decision,
        max_decision,
    )
    for (name, model) in networks
        trainer = SupervisedLearningTrainer(
            dataset, model;
            loss=composite_loss,
            optimiser=Flux.Adam(1e-3),
        )
        Flux.train!(
            trainer;
            batchsize=16,
            epochs=200,
        )
    end
    c_plot = range(min_condition, stop=max_condition; length=100)
    d_plot = range(min_decision, stop=max_decision; length=100)
    fig = plot(c_plot, d_plot, target_function; st=:surface)
    plot!(c_plot, d_plot, plse; st=:surface)
    display(fig)
end
