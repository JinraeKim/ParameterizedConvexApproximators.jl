using Test
using Flux
using ParametrisedConvexApproximators
using Transducers
using Random


# common parameters
n, m = 3, 2
d = 10
h_array = [64, 64]
z_array = [64, 64]
u_array = vcat(64, z_array...)
act = Flux.leakyrelu
i_max = 20
T = 1.0
α_is = 1:i_max |> Map(i -> Flux.glorot_uniform(n+m)) |> collect
β_is = 1:i_max |> Map(i -> Flux.glorot_uniform(1)) |> collect
u_min = -ones(m)
u_max = ones(m)


function generate_networks()
    fnn = FNN(n, m, h_array, act)
    ma = MA(α_is, β_is; n=n, m=m)
    lse = LSE(α_is, β_is, T; n=n, m=m)
    picnn = PICNN(n, m, u_array, z_array, act, act)
    pma = PMA(n, m, i_max, h_array, act)
    plse = PLSE(n, m, i_max, T, h_array, act)

    networks = Dict(
                    "FNN" => fnn,
                    "MA" => ma,
                    "LSE" => lse,
                    "PICNN" => picnn,
                    "PMA" => pma,
                    "PLSE" => plse,
                   )  # TODO: add more networks
    return networks
end


function test_infer(network; seed=2022)
    println("test_infer")
    Random.seed!(seed)
    x = rand(n)
    u = rand(m)
    f̂ = network(x, u)
    @test size(f̂) == (1,)
end


function test_optimise(network; seed=2022)
    println("test_optimise")
    Random.seed!(seed)
    x = rand(n)
    (; minimiser, optval) = optimise(network, x; u_min=u_min, u_max=u_max)
    @test size(minimiser) == (m,)
    @test size(optval) == (1,)
end


function test_optimise_multiple(network; seed=2022)
    println("test_optimise_multiple")
    Random.seed!(seed)
    x = rand(n, d)
    (; minimiser, optval) = optimise(network, x; u_min=u_min, u_max=u_max)
    @test size(minimiser) == (m, d)
    @test size(optval) == (1, d)
end


function test_infer_multiple(network; seed=2022)
    println("test_infer_multiple")
    Random.seed!(seed)
    x = rand(n, d)
    u = rand(m, d)
    f̂ = network(x, u)
    @test size(f̂) == (1, d)
end


function test_network(network)
    test_infer(network)
    test_infer_multiple(network)
    test_optimise(network)
    test_optimise_multiple(network)
end


function main()
    networks = generate_networks()
    for (name, network) in networks
        println("Tested network: " * name)
        test_network(network)
    end
end


@testset "networks" begin
    main()
end