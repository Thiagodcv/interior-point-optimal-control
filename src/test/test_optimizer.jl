include("../optimizer/optimizer.jl")
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
    A = [0.8 -1.]
    b = [-0.2]
    G = [1. 0.; -1. 0.; 0. 1.; 0. -1.]
    h = [1.; 0.; 1.; 0.]

    param_dict = Dict("Q" => Q, "q" => q, "A" => A, "b" => b, "G" => G, "h" => h)
    x0 = [0.3; 0.8]
    
    results = pdip_qp(param_dict, x0)
    print("results: ", results)
end

@testset "test_initial_step_size_less_than_1" begin
    u = [3; 2; 5; 1]
    d_u = [-5; -1; 3; 5]
    s_max = initial_step_size(u, d_u)
    s_max_true = 3/5

    tol = 1e-6
    @test abs(s_max - s_max_true) < tol
end
