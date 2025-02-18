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
    model = PLSE(n, m, i_max, T, h_array, act)

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

    trainer = SupervisedLearningTrainer(
        dataset, model;
        loss=composite_loss,
        optimiser=Flux.Adam(1e-3),
    )

    anim=Animation()

    function callback()
        c_plot = range(min_condition[1], stop=max_condition[1]; length=100)
        d_plot = range(min_decision[1], stop=max_decision[1]; length=100)
        fig = plot(c_plot, d_plot, (c, d) -> target_function([c], [d]); st=:surface, alpha=0.5)
        plot!(c_plot, d_plot, (c, d) -> model([c], [d])[1]; st=:surface, alpha=0.5)
        display(fig)
        frame(anim)
    end
    Flux.train!(
        trainer;
        batchsize=16,
        epochs=50,
        callback,
    )
    gif(anim, "plse_with_composite_loss.gif", fps=10)
end
