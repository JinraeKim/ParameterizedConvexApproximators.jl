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
min_condition = -1*ones(n)
max_condition = +1*ones(n)
min_decision = -1*ones(m)
max_decision = +1*ones(m)


function composite_loss(pred, f)
    tmp = max.(0, pred .- f)
    l = 0.0
    l += Flux.Losses.mse(pred, f)
    l += 100*Flux.Losses.mae(tmp, zeros(size(tmp)))  # the last term is for pred >= f
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
    conditions, decisions, costs, metadata = generate_dataset(
        target_function;
        N,
        min_condition,
        max_condition,
        min_decision,
        max_decision,
    )
    dataset = DecisionMakingDataset(
        conditions, decisions, costs;
        metadata, seed=2023,
        ratio1=0.7, ratio2=0.2,
    )

    for (name, model) in networks
        trainer = SupervisedLearningTrainer(
            dataset, model;
            loss=composite_loss,
            optimiser=Flux.Adam(1e-3),
        )

        function callback()
            c_plot = range(min_condition[1], stop=max_condition[1]; length=100)
            d_plot = range(min_decision[1], stop=max_decision[1]; length=100)
            fig = plot(c_plot, d_plot, (c, d) -> target_function([c], [d]); st=:surface, alpha=0.5)
            plot!(c_plot, d_plot, (c, d) -> plse([c], [d])[1]; st=:surface, alpha=0.5)
            display(fig)
        end
        Flux.train!(
            trainer;
            batchsize=16,
            epochs=50,
            callback,
        )
    end
end
