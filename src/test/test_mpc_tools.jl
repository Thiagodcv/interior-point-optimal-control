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
    # println(P_ret)
    @test norm(P - P_ret) < tol
end


@testset "test_mpc_to_qp_eq_mat" begin
    """
    Test to see if equality matrix is computed correctly.
    """
    A = [4. 0. 0.;
         5. 6. 0.;
         0. 1. 1.]
    B = [1. 0.;
         0. 1.;
         0. 2.]
    system_dict = Dict("A" => A, "B" => B)
    n = 3
    m = 2
    T = 2
    C_ret = mpc_to_qp_eq_mat(system_dict, n, m, T)

    C = [-1. 0. 1. 0. 0. 0. 0. 0. 0. 0.;
         0. -1. 0. 1. 0. 0. 0. 0. 0. 0.;
         0. -2. 0. 0. 1. 0. 0. 0. 0. 0.;
         0. 0. -4. 0. 0. -1. 0. 1. 0. 0.;
         0. 0. -5. -6. 0. 0. -1. 0. 1. 0.;
         0. 0. 0. -1. -1. 0. -2. 0. 0. 1.]

    tol = 1e-6
    # println(C_ret)
    @test norm(C - C_ret) < tol
end


@testset "test_mpc_to_qp_linear_term" begin
    """
    Test to see if linear term of the QP cost is computed correctly.
    """
    S = [2. 0.;
         0. 1.]
    r = [5; 4]
    q = [3; 2; 1]
    q_T = [6; 1; 2]
    u_latest = [1; 3]

    cost_dict = Dict("S" => S, "r" => r, "q" => q, "q_T" => q_T)
    n = 3
    m = 2
    T = 2
    g_ret = mpc_to_qp_linear_term(cost_dict, n, m, u_latest, T)
    g = [1; -2; 3; 2; 1; 5; 4; 6; 1; 2]
    
    tol = 1e-6
    println(g_ret)
    @test norm(g - g_ret) < tol
end


@testset "test_mpc_to_qp_ineq_vec" begin
    """
    Test to see if the inequality vector is computed correctly.
    """
    F_du = [2. 2.]
    f_u = [1.; 3.; 2.]
    f_du = [5.]
    f_x = [7.; 3.]
    f_T = [2.; 1.; 0.]

    T = 2
    u_latest = [1.; 3.]

    constraint_dict = Dict("F_du" => F_du, "f_u" => f_u, "f_du" => f_du, "f_x" => f_x, "f_T" => f_T)
    h_ret = mpc_to_qp_ineq_vec(constraint_dict, u_latest, T)
    h = [1.; 3.; 2.; 13.; 7.; 3.; 1.; 3.; 2.; 5.; 2.; 1.; 0.]
    
    tol = 1e-6
    # println(h_ret)
    @test norm(h - h_ret) < tol
end


@testset "test_mpc_to_qp_eq_vec" begin
    """
    Test to see if the equality vector is computed correctly.
    """
    T = 2
    n = 3
    w = [2; 1; 3]
    A = [4. 0. 0.; 
         5. 6. 0.; 
         0. 1. 1.]
    B = [1. 0.;
         0. 1.;
         0. 2.]
    x0 = [1.; 1.; 5.]

    system_dict = Dict("A" => A, "B" => B, "x0" => x0, "w" => w)
    b_ret = mpc_to_qp_eq_vec(system_dict, x0, n, T)
    b = [6.; 12.; 9.; 2.; 1.; 3.]
    
    tol = 1e-6
    @test norm(b - b_ret) < tol
end
