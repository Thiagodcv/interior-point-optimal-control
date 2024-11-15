include("../optimizer.jl")
using Test


@testset "test_pdip_qp" begin
    """
    Test for solving a 2d problem with 1 equality and 4 inequality (box) constraints.
    """
    n = 2
    p = 4
    m = 1

    Q = Matrix{Float64}(I, 2, 2)
    q = [2.; 2.]
    A = [1. -1.]
    b = [0.]
    G = [1. 0.; -1. 0.; 0. 1.; 0. -1.]
    h = [1.; 0.; 1.; 0.]

    param_dict = Dict("Q" => Q, "q" => q, "A" => A, "b" => b, "G" => G, "h" => h)
    x0 = [0.3; 0.8]
    
    results = pdip_qp(param_dict, x0)
    print("results: ", results)
end

