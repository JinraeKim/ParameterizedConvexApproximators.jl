struct DecisionMakingDataset
    metadata::NamedTuple
    split::Symbol
    conditions::AbstractVector
    decisions::AbstractVector
    costs::AbstractVector
end

function generate_dataset(
        target_function;
        N,
        min_condition,
        max_condition,
        min_decision,
        max_decision,
        seed=2023,
        kwargs...,
    )
    @assert all(min_condition .<= max_condition)
    @assert all(min_decision .<= max_decision)
    f = target_function
    conditions = sample_from_bounds(N, min_condition, max_condition, seed)
    decisions = sample_from_bounds(N, min_decision, max_decision, seed)
    costs = Vector(undef, N)
    p = Progress(N, "Generating dataset...")
    Threads.@threads for i in 1:N
        c = conditions[i]
        d = decisions[i]
        costs[i] = f(c, d)
        next!(p)
    end
    finish!(p)
    # costs = [f(c, d) for (c, d) in zip(conditions, decisions)]
    metadata = (;
                target_function=f,
                min_condition=min_condition,
                max_condition=max_condition,
                min_decision=min_decision,
                max_decision=max_decision,
                kwargs...,
               )
    return conditions, decisions, costs, metadata
end


function DecisionMakingDataset(
        conditions, decisions, costs;
        metadata=(;),  # prior metadata
        name=nothing,
        seed=2023,
        ratio1=0.7, ratio2=0.2,
    )
    N = length(conditions)
    @assert length(decisions) == N
    @assert length(costs) == N
    # split indicies
    Random.seed!(seed)
    train_idx, validate_idx, test_idx = split_data3(collect(1:N), ratio1, ratio2)
    # get data
    metadata = (;
                metadata...,
                name=name,
                split_ratio=(;
                             train=ratio1,
                             validate=ratio2,
                             test=1-(ratio1+ratio2),
                            ),
                train_idx=train_idx,
                validate_idx=validate_idx,
                test_idx=test_idx,
               )
    return DecisionMakingDataset(metadata, :full, conditions, decisions, costs)
end



function Base.getindex(dataset::DecisionMakingDataset, split)
    (; metadata, conditions, decisions, costs) = dataset
    @assert split in (:train, :validate, :test, :full)
    if split == :full
        dataset_ = dataset
    else
        if split == :train
            idx = metadata.train_idx
        elseif split == :validate
            idx = metadata.validate_idx
        else
            idx = metadata.test_idx
        end
        dataset_ = DecisionMakingDataset(metadata, split, conditions[idx], decisions[idx], costs[idx])
    end
    return dataset_
end


"""
    split_data2(dataset, ratio)

Split a dataset into train and test datasets (array).
"""
function split_data2(dataset, ratio; seed=2022)
    Random.seed!(seed)
    @assert ratio >= 0.0
    @assert ratio <= 1.0
    n = length(dataset)
    idx = Random.shuffle(1:n)
    train_idx = view(idx, 1:floor(Int, ratio*n))
    test_idx = view(idx, (floor(Int, ratio*n)+1):n)
    dataset[train_idx], dataset[test_idx]
end


"""
    split_data3(dataset, ratio1, ratio2)

Split a dataset into train, validate, and test datasets (array).
"""
function split_data3(dataset, ratio1, ratio2; seed=2022)
    Random.seed!(seed)
    @assert ratio1 >= 0.0
    @assert ratio2 >= 0.0
    @assert ratio1 + ratio2 <= 1.0
    dataset_train, dataset_valtest = split_data2(dataset, ratio1)
    dataset_validate, dataset_test = split_data2(dataset_valtest, ratio2/(1.0-ratio1))
    return dataset_train, dataset_validate, dataset_test
end


function sample_from_bounds(N, min_value, max_value, seed)
    samples = []
    for i in 1:N
        sampled_value = (
            min_value
            + (max_value - min_value) .* rand(Xoshiro(seed), size(min_value)...)
        )
        push!(samples, sampled_value)
    end
    return samples
end


"""
    target_function(name)

Get a target function.

# References
[1] J. Kim and Y. Kim, “Parameterized Convex Universal Approximators for Decision-Making Problems,” IEEE Trans. Neural Netw. Learning Syst., 2022, doi: 10.1109/TNNLS.2022.3190198.
[2] G. C. Calafiore, S. Gaubert, and C. Possieri, “A Universal Approximation Result for Difference of Log-Sum-Exp Neural Networks,” IEEE Transactions on Neural Networks and Learning Systems, vol. 31, no. 12, pp. 5603–5612, Dec. 2020, doi: 10.1109/TNNLS.2020.2975051.
"""
function example_target_function(name)
    if typeof(name) == Symbol
        if name == :quadratic
            func = (x::Vector, u::Vector) -> x'*x + u'*u
        elseif name == :parameterized_convex_basic  # [1]
            func = (x::Vector, u::Vector) -> -x'*x + u'*u
        elseif name == :quadratic_sin_sum  # [2, Example 3] is modified
            func = (x::Vector, u::Vector) -> x'*x + u'*u + sum(sin.(2*pi*u))
        else
            error("Undefined simple function")
        end
    elseif typeof(name) <: Function
        func = name
    else
        error("Invalid target function type $(typeof(name))")
    end
    return func
end
