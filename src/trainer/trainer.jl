abstract type AbstractTrainer end


struct SupervisedLearningTrainer <: AbstractTrainer
    network::AbstractApproximator
    dataset::DecisionMakingDataset
    loss
    optimiser
    scheduler
    function SupervisedLearningTrainer(
        dataset, network;
        normalisation=nothing,
        loss=Flux.Losses.mse,
        optimiser=Flux.Adam(1e-3),
        scheduler=nothing,
    )
        network = retrieve_normalised_network(network, dataset, normalisation)
        @assert dataset.split == :full
        new(network, dataset, loss, optimiser, scheduler)
    end
end


function retrieve_normalised_network(network::AbstractApproximator, dataset::DecisionMakingDataset, normalisation)
    if isnothing(normalisation)
        normalised_network = network
    elseif normalisation == :max_abs
        normalised_network = MaxAbsNormalisedApproximator(network, dataset)
    else
        error("Invalid normalisation method $(normalisation)")
    end
    return normalised_network
end


"""
You must explicitly give "the network to be evaluated".
"""
function get_loss(network, dataset, loss)
    l = loss(network(hcat(dataset.conditions...), hcat(dataset.decisions...)), hcat(dataset.costs...))
    return l
end


function Flux.train!(
    trainer::SupervisedLearningTrainer;
    batchsize=16,
    epochs=200,
    rng=Random.default_rng(),
    callback=nothing,
)
    (; network, dataset, loss, optimiser, scheduler) = trainer
    data_train = Flux.DataLoader(
        (
            hcat(dataset[:train].conditions...),
            hcat(dataset[:train].decisions...),
            hcat(dataset[:train].costs...),
        );
        batchsize=batchsize,
        shuffle=true,
        rng=rng,
    )
    opt_state = Flux.setup(optimiser, network)

    losses_train = []
    losses_validate = []
    loss_train = nothing
    loss_validate = nothing
    minimum_loss_validate = Inf
    best_network = nothing
    if isnothing(scheduler)
        scheduler = [optimiser.eta for _ in 1:epochs]
    end
    scheduler = Iterators.Stateful(scheduler)
    eta = nothing
    for epoch in 0:epochs
        if epoch != 0
            eta, _ = iterate(scheduler)
            Flux.Optimisers.adjust!(opt_state, eta)
            if !isnothing(callback)
                callback(epoch)
            end
            loss_train = 0.0
            batch_size = 0
            @showprogress for (x, u, f) in data_train
                val, grads = Flux.withgradient(network) do _network
                    pred = _network(x, u)
                    loss(pred, f)
                end
                loss_train += val
                batch_size += 1
                if !any(isnan, getall(grads[1], AccessorsExtra.RecursiveOfType(Number)))
                    # This will give an warning
                    # https://github.com/gdalle/ImplicitDifferentiation.jl/issues/92
                    # https://discourse.julialang.org/t/julia-nan-check-for-namedtuple/102583/4?u=ihany
                    opt_state, network = Flux.update!(opt_state, network, grads[1])
                end
                if typeof(network) == PICNN  # TODO: an automated solution required
                    project_nonnegative!(network)
                end
            end
            loss_train = loss_train / batch_size
        else
            loss_train = get_loss(trainer.network, trainer.dataset[:train], trainer.loss)
        end
        push!(losses_train, loss_train)
        loss_validate = get_loss(trainer.network, trainer.dataset[:validate], trainer.loss)
        push!(losses_validate, loss_validate)
        println("epoch: $(epoch)/$(epochs), train loss: $(Printf.@sprintf("%.4e", loss_train)), valid loss: $(Printf.@sprintf("%.4e", loss_validate)) (learning rate: $(eta))")
        if loss_validate < minimum_loss_validate
            println("Best network found!")
            minimum_loss_validate = loss_validate
            best_network = deepcopy(network)
        end
    end
    loss_test = get_loss(trainer.network, trainer.dataset[:test], trainer.loss)
    info = Dict()
    info["train_loss"] = losses_train
    info["valid_loss"] = losses_validate
    info["test_loss"] = loss_test
    return best_network, info
end
