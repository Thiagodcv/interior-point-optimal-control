include("../../optimizers/qp/search_dir.jl")
using Test

@testset "test_compute_kkt_residual" begin
    """
    Test for computing the KKT residual for a 2d problem with 1 equality and 4 inequality (box) constraints.
    Test both cases for when res_mat is specified by user, and when it isn't.
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
    compute_kkt_residual(param_dict, x, lambda, nu, s, F)
    F_new = compute_kkt_residual(param_dict, x, lambda, nu, s)  # F not passed in.
    true_F = [13/2; -7/2; 3.; 2.; 9.; 4.; 5/2; 1/2; 5/2; 1/2; 0.]

    print("F: ", F)
    print("true_F: ", true_F)
    print("F_new: ", F_new)
    tol = 1e-7
    @test norm(F - true_F) < tol
    @test norm(F_new - true_F) < tol
end

@testset "test_compute_kkt_jacobian" begin
    """
    Test for computing the Jacobian of the KKT residual for a 2d problem with 1 equality and 4 inequality (box) constraints.
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
    lambda = [1.; 2.; 3.; 4.]
    s = [3.; 1.; 3.; 1.]
    
    J = compute_kkt_jacobian(param_dict, lambda, s)
    J_true = [1 0 0 0 0 0 1 -1 0 0 1; 
            0 1 0 0 0 0 0 0 1 -1 -1; 
            0 0 1 0 0 0 3 0 0 0 0;
            0 0 0 2 0 0 0 1 0 0 0;
            0 0 0 0 3 0 0 0 3 0 0;
            0 0 0 0 0 4 0 0 0 1 0;
            1 0 1 0 0 0 0 0 0 0 0;
            -1 0 0 1 0 0 0 0 0 0 0;
            0 1 0 0 1 0 0 0 0 0 0;
            0 -1 0 0 0 1 0 0 0 0 0;
            1 -1 0 0 0 0 0 0 0 0 0]

    tol = 1e-7
    @test norm(J - J_true) < tol

    lambda = [2.; 5.; 4.; 3.]
    s = [1.; 1.; 2.; 2.]
    compute_kkt_jacobian(param_dict, lambda, s, J)
    J_new_true = [1 0 0 0 0 0 1 -1 0 0 1; 
                 0 1 0 0 0 0 0 0 1 -1 -1; 
                0 0 2 0 0 0 1 0 0 0 0;
                0 0 0 5 0 0 0 1 0 0 0;
                0 0 0 0 4 0 0 0 2 0 0;
                0 0 0 0 0 3 0 0 0 2 0;
                1 0 1 0 0 0 0 0 0 0 0;
                -1 0 0 1 0 0 0 0 0 0 0;
                0 1 0 0 1 0 0 0 0 0 0;
                0 -1 0 0 0 1 0 0 0 0 0;
                1 -1 0 0 0 0 0 0 0 0 0]
    
    @test norm(J - J_new_true) < tol
end

@testset "test_compute_affine_scaling_dir" begin
    """
    Test for computing the affine scaling direction.
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
    
    F = compute_kkt_residual(param_dict, x, lambda, nu, s)
    J = compute_kkt_jacobian(param_dict, lambda, s)

    aff_step = compute_affine_scaling_dir(F, J, n, p, m)
    aff_step_vec = vcat(aff_step["x"], aff_step["s"])
    aff_step_vec = vcat(aff_step_vec, aff_step["lambda"])
    aff_step_vec = vcat(aff_step_vec, aff_step["nu"])
 
    tol = 1e-7
    @test norm(J * aff_step_vec + F) < tol
end

@testset "test_centering_plus_corrector_dir" begin
    """
    Test for computing the centering-plus-corrector direction.
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
    
    F = compute_kkt_residual(param_dict, x, lambda, nu, s)
    J = compute_kkt_jacobian(param_dict, lambda, s)

    aff_step = compute_affine_scaling_dir(F, J, n, p, m)

    sigma = 2
    mu = 2.5
    cc_step = compute_centering_plus_corrector_dir(J, aff_step["s"], aff_step["lambda"], sigma, mu, n, p, m)

    cc_rhs_vec = zeros((n+2*p+m,))
    cc_rhs_vec[n+1:n+p] = sigma * mu * ones((p,)) - Diagonal(aff_step["s"]) * aff_step["lambda"]

    cc_dir = vcat(cc_step["x"], cc_step["s"])
    cc_dir = vcat(cc_dir, cc_step["lambda"])
    cc_dir = vcat(cc_dir, cc_step["nu"])

    tol = 1e-7
    @test norm(J * cc_dir - cc_rhs_vec) < tol
end
