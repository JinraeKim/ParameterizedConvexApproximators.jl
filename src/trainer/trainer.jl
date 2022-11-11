abstract type AbstractTrainer end


struct SupervisedLearningTrainer <: AbstractTrainer
    loss
    network::AbstractApproximator
    optimizer
    dataset
    function SupervisedLearningTrainer(
        dataset, network;
        loss=(x, u, f) -> Flux.mse(network(x, u), f),
        optimizer=Adam(1e-4),
    )
        @assert dataset.split == :full
        new(loss, network, optimizer, dataset)
    end
end


function get_loss(trainer::SupervisedLearningTrainer, split::Symbol)
    @assert split ∈ (:train, :validate, :test)
    dataset = trainer.dataset[split]
    (; loss) = trainer
    l = loss(hcat(dataset.conditions...), hcat(dataset.decisions...), hcat(dataset.costs...))
    return l
end


function Flux.train!(
        trainer::SupervisedLearningTrainer;
        batchsize=16,
        throttle_time=1,  # [s]
        epochs=200,
    )
    (; loss, network, optimizer, dataset) = trainer
    parameters = Flux.params(network)
    data_train = Flux.DataLoader((
        hcat(dataset[:train].conditions...),
        hcat(dataset[:train].decisions...),
        hcat(dataset[:train].costs...),
    ); batchsize=batchsize)

    loss_train = nothing
    loss_validate = nothing
    minimum_loss_validate = Inf
    best_network = nothing
    cb = Flux.throttle(throttle_time) do
        loss_train = get_loss(trainer, :train)
        println("loss_train: $(loss_train)")
        loss_validate = get_loss(trainer, :validate)
        @show loss_validate
        @show minimum_loss_validate
        println("loss_validate: $(loss_validate)")
        if loss_validate < minimum_loss_validate
            println("Best network found!")
            minimum_loss_validate = loss_validate
            best_network = deepcopy(network)
        end
    end

    for epoch in 1:epochs
        println("epoch: $(epoch)/$(epochs)")
        Flux.train!(loss, parameters, data_train, optimizer; cb=cb)
    end
    return best_network
end