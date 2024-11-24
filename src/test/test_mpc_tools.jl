include("../examples/mpc_tools.jl")


@testset "test_mpc_to_qp_hessian" begin
    """
    Test to see if Hessian matrix is computed correctly.
    """
    Q = [1. 0. 0.;
         0. 2. 0.;
         0. 0. 3.]
    R = [4. 0.; 
         0. 5.]
    S = [2. 0.;
         0. 1.]
    Q_T = [5. 0. 0.;
           0. 6. 0.;
           0. 0. 1.]
    cost_dict = Dict("Q" => Q, "R" => R, "S" => S, "Q_T" => Q_T)
    n = 3
    m = 2
    T = 2
    H_ret = mpc_to_qp_hessian(cost_dict, n, m, T)

    H = [8. 0. 0. 0. 0. -2. 0. 0. 0. 0.;
         0. 7. 0. 0. 0. 0. -1. 0. 0. 0.;
         0. 0. 1. 0. 0. 0. 0. 0. 0. 0.;
         0. 0. 0. 2. 0. 0. 0. 0. 0. 0.;
         0. 0. 0. 0. 3. 0. 0. 0. 0. 0.;
         -2. 0. 0. 0. 0. 6. 0. 0. 0. 0.;
         0. -1. 0. 0. 0. 0. 6. 0. 0. 0.;
         0. 0. 0. 0. 0. 0. 0. 5. 0. 0.;
         0. 0. 0. 0. 0. 0. 0. 0. 6. 0.;
         0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]

    tol = 1e-6
    @test norm(H - H_ret) < tol
end