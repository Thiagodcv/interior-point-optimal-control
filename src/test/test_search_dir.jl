include("../search_dir.jl")
using Test

@testset "test_compute_kkt_residual" begin
    """
    Test for computing the KKT residual for a 2d problem with 1 equality and 4 inequality (box) constraints.
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
    x = [0.5; 0.5]
    lambda = [1.; 2.; 3.; 4.]
    nu = [5.]
    s = [3.; 1.; 3.; 1.]
    
    F = zeros((n + 2*p + m,))
    compute_kkt_residual(param_dict, F, x, lambda, nu, s)
    true_F = [13/2; -7/2; 3.; 2.; 9.; 4.; 5/2; 1/2; 5/2; 1/2; 0.]

    print("F: ", F)
    print("true_F: ", true_F)
    tol = 1e-7
    @test norm(F - true_F) < tol
end