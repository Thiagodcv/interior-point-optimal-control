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


@testset "test_mpc_to_qp_ineq_mat" begin
    """
    Test to see if inequality matrix is computed correctly.
    """
    F_u = [1. 0.;
           0. 1.;
           2. 1.]
    F_du = [2. 2.]
    F_x = [1. 2. 0.;
           0. 3. 1.]
    F_T = [3. 1. 0.;
           2. 0. 1.;
           0. 5. 6.]
    

    constraint_dict = Dict("F_u" => F_u, "F_du" => F_du, "F_x" => F_x, "F_T" => F_T)
    n = 3
    m = 2
    T = 2
    P_ret = mpc_to_qp_ineq_mat(constraint_dict, n, m, T)

    P = [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.;
         0. 1. 0. 0. 0. 0. 0. 0. 0. 0.;
         2. 1. 0. 0. 0. 0. 0. 0. 0. 0.;
         2. 2. 0. 0. 0. 0. 0. 0. 0. 0;
         0. 0. 1. 2. 0. 0. 0. 0. 0. 0.;
         0. 0. 0. 3. 1. 0. 0. 0. 0. 0.;
         0. 0. 0. 0. 0. 1. 0. 0. 0. 0.;
         0. 0. 0. 0. 0. 0. 1. 0. 0. 0.;
         0. 0. 0. 0. 0. 2. 1. 0. 0. 0.;
         -2. -2. 0. 0. 0. 2. 2. 0. 0. 0.;
         0. 0. 0. 0. 0. 0. 0. 3. 1. 0.;
         0. 0. 0. 0. 0. 0. 0. 2. 0. 1.;
         0. 0. 0. 0. 0. 0. 0. 0. 5. 6.]

    tol = 1e-6
    println(P_ret)
    @test norm(P - P_ret) < tol
end
