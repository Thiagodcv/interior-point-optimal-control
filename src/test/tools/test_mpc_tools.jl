include("../../tools/mpc_tools.jl")
include("../../optimizers/qp/optimizer.jl")


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
    # println(g_ret)
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


@testset "test_box_constraints" begin
    """
    Test to see if correct QP inequality parameters are returned.
    """
    x_lb = [-1.; -2.; -3.]
    x_ub = [3.; 2.; 1.]
    u_lb = [-1.; -5.]
    u_ub = [3.; -1.]
    du_lb = [-0.5; 0.5]
    du_ub = [1.; 1.]
    x_T_lb = [-1.; 0.; 2.]
    x_T_ub = [3.; 1.; 5.]

    limit_dict = Dict("x_lb" => x_lb, "x_ub" => x_ub, "u_lb" => u_lb, "u_ub" => u_ub, 
                      "du_lb" => du_lb, "du_ub" => du_ub, "x_T_lb" => x_T_lb, "x_T_ub" => x_T_ub)
    box_dict = box_constraints(limit_dict)

    F_x = [1. 0. 0.; 
           0. 1. 0.;
           0. 0. 1.;
           -1. 0. 0.;
           0. -1. 0.;
           0. 0. -1.]
    F_T = F_x
    F_u = [1. 0.;
           0. 1.;
           -1. 0.;
           0. -1.]
    F_du = F_u
    f_x = [3.; 2.; 1.; 1.; 2.; 3.]
    f_u = [3.; -1.; 1.; 5.]
    f_du = [1.; 1.; 0.5; -0.5]
    f_T = [3.; 1.; 5.; 1.; 0; -2.]
    
    tol = 1e-6
    @test norm(F_x - box_dict["F_x"]) < tol
    @test norm(F_u - box_dict["F_u"]) < tol
    @test norm(F_du - box_dict["F_du"]) < tol
    @test norm(F_T - box_dict["F_T"]) < tol
    @test norm(f_x - box_dict["f_x"]) < tol
    @test norm(f_u - box_dict["f_u"]) < tol
    @test norm(f_du - box_dict["f_du"]) < tol
    @test norm(f_T - box_dict["f_T"]) < tol
end


# @testset "test_lti_problem" begin
#     """
#     Test to see if can solve an optimal control problem on an LTI system.
#     """
#     n = 3
#     m = 2

#     d = [5; 4; 1]

#     # Cost parameters
#     cost_dict = Dict()
#     cost_dict["Q"] = Matrix{Float64}(I, n, n)
#     cost_dict["q"] = -2*d
#     cost_dict["R"] = 0.01*Matrix{Float64}(I, m, m)
#     cost_dict["r"] = zeros((m,))
#     cost_dict["S"] = cost_dict["R"]
#     cost_dict["Q_T"] = 2*cost_dict["Q"]
#     cost_dict["q_T"] = 2*cost_dict["q"]

#     # Parameters for variable limits
#     big_num = 10_000
#     limit_dict = Dict()
#     limit_dict["x_ub"] = [4.; 2.; big_num]
#     limit_dict["x_lb"] = -limit_dict["x_ub"]

#     limit_dict["u_ub"] = [big_num; big_num]
#     limit_dict["u_lb"] = -limit_dict["u_ub"]

#     limit_dict["du_ub"] = [0.5; 0.5]
#     limit_dict["du_lb"] = -limit_dict["du_ub"]

#     limit_dict["x_T_ub"] = [4.; 2.; big_num]
#     limit_dict["x_T_lb"] = -limit_dict["x_T_ub"]
#     constraint_dict = box_constraints(limit_dict)
    
#     # Parameters for LTI system
#     system_dict = Dict()
#     system_dict["A"] = Matrix{Float64}(I, n, n)
#     system_dict["B"] = [1. 0.;
#                         0. 1.;
#                         1. 1.]
#     system_dict["w"] = zeros((n,))

#     # initial state, the latest input, and the time horizon
#     x0 = [0.; 0.; 0.]
#     u_latest = [0.; 0.]
#     T = 10

#     qp_dict = mpc_to_qp(cost_dict, constraint_dict, system_dict, x0, u_latest, T)

#     x_init = zeros((T*(n+m),))
#     ret = pdip_qp(qp_dict, x_init)
#     println("iters: ", ret["iters"])
#     println("solution: ", ret["x"])
# end


@testset "test_separate_solution" begin
     """
    Test to see if separate_solution() returns the correct result.
    """
     sol = [11; 12; 1; 2; 3; 13; 14; 4; 5; 6]
     u_latest = [9; 10]
     n = 3
     m = 2
     T = 2
     sep_sol = separate_solution(sol, n, m, u_latest, T)

     x = [1; 2; 3; 4; 5; 6]
     u = [11; 12; 13; 14]
     du = [2; 2; 2; 2]

     tol = 1e-6
     @test norm(sep_sol["x"] - x) < tol
     @test norm(sep_sol["u"] - u) < tol
     @test norm(sep_sol["du"] - du) < tol
end


@testset "test_double_integrator" begin
    """
    Test to see if can solve the double integrator optimal control problem on an LTI system.
    """
    n = 2
    m = 1

    d = [0; 0]

    # Cost parameters
    cost_dict = Dict()
    cost_dict["Q"] = Matrix{Float64}(I, n, n)
    cost_dict["q"] = -2*d
    cost_dict["R"] = 0.01*Matrix{Float64}(I, m, m)
    cost_dict["r"] = zeros((m,))
    cost_dict["S"] = cost_dict["R"]
    cost_dict["Q_T"] = cost_dict["Q"]
    cost_dict["q_T"] = cost_dict["q"]

    # Parameters for variable limits
    big_num = 10_000
    limit_dict = Dict()
    limit_dict["x_ub"] = [5.; 2.]
    limit_dict["x_lb"] = -limit_dict["x_ub"]

    limit_dict["u_ub"] = [1.]
    limit_dict["u_lb"] = -limit_dict["u_ub"]

    limit_dict["du_ub"] = [0.5]
    limit_dict["du_lb"] = -limit_dict["du_ub"]

    limit_dict["x_T_ub"] = [5.; 2.]
    limit_dict["x_T_lb"] = -limit_dict["x_T_ub"]
    constraint_dict = box_constraints(limit_dict)
    
    # Parameters for LTI system
    system_dict = Dict()
    freq = 0.5
    system_dict["A"] = [1. freq;
                        0. 1.]
    system_dict["B"] = [0.; freq]
    system_dict["w"] = zeros((n,))

    # initial state, the latest input, and the time horizon
    x0 = [4.; 1.]
    u_latest = [0.]
    T = 20

    qp_dict = mpc_to_qp(cost_dict, constraint_dict, system_dict, x0, u_latest, T)

    x_init = zeros((T*(n+m),))
    ret = pdip_qp(qp_dict, x_init)
    ret_separate = separate_solution(ret["x"], n, m, u_latest, T)
    println("iters: ", ret["iters"])
    # println("solution: ", ret["x"])
    println("state solution: ", ret_separate["x"])
    println("input solution: ", ret_separate["u"])
    println("diff input solution: ", ret_separate["du"])
end


@testset "test_nonlinear_eq_constraint" begin
     """
     Test to see if nonlinear_eq_constraint returns the correct answer.
     """
     x0 = [1.; 2.]
     x1 = [2.; 3.]
     x2 = [3.; 4.]
     x3 = [4.; 5.]

     u0 = [5.; 6.; 7.]
     u1 = [7.; 8.; 9.]
     u2 = [9.; 10.; 11.]

     z = vcat(u0, x1, u1, x2, u2, x3)
     n_x = 2
     n_u = 3
     T = 3

     function f(x, u)
          return norm(u)^2 * [sin(x[1])*x[2]; sin(x[2])*x[1]]
     end

     constraint_vec = nonlinear_eq_constraint(z, x0, n_x, n_u, T, f)

     expected_vec = x1 - f(x0, u0)
     expected_vec = vcat(expected_vec, x2 - f(x1, u1))
     expected_vec = vcat(expected_vec, x3 - f(x2, u2))

     tol = 1e-7
     # println(constraint_vec)
     # println(expected_vec)
     @test norm(constraint_vec - expected_vec) < tol

     x0 = [1.1; 2.2]
     x1 = [2.2; 3.3]
     x2 = [3.3; 4.4]
     x3 = [4.4; 5.5]
     z = vcat(u0, x1, u1, x2, u2, x3)
     nonlinear_eq_constraint(z, x0, n_x, n_u, T, f, constraint_vec)

     expected_vec = x1 - f(x0, u0)
     expected_vec = vcat(expected_vec, x2 - f(x1, u1))
     expected_vec = vcat(expected_vec, x3 - f(x2, u2))
     # println(constraint_vec)
     # println(expected_vec)
     @test norm(constraint_vec - expected_vec) < tol
 end
 

 @testset "test_nonlinear_eq_jacobian" begin
     """
     Test to see if nonlinear_eq_jacobian returns the correct answer.
     """
     x0 = [1.; 2.]
     x1 = [2.; 3.]
     x2 = [3.; 4.]
     x3 = [4.; 5.]

     u0 = [5.; 6.; 7.]
     u1 = [7.; 8.; 9.]
     u2 = [9.; 10.; 11.]

     z = vcat(u0, x1, u1, x2, u2, x3)
     n_x = 2
     n_u = 3
     T = 3

     function f(x, u)
          return norm(u)^2 * [sin(x[1])*x[2]; sin(x[2])*x[1]]
     end

     function fx(x, u)
          mat = [cos(x[1])*x[2] sin(x[1]); 
                 sin(x[2]) cos(x[2])*x[1]]
          return norm(u)^2 * mat
     end

     function fu(x, u)
          return 2*[sin(x[1])*x[2]; sin(x[2])*x[1]] * transpose(u)
     end

     epsilon = 1e-8
     fx1 = (f(x0 + [epsilon; 0], u0) - f(x0, u0)) / epsilon
     fx2 = (f(x0 + [0; epsilon], u0) - f(x0, u0)) / epsilon
     fx_hat = hcat(fx1, fx2)

     fu1 = (f(x0, u0 + [epsilon; 0; 0]) - f(x0, u0)) / epsilon
     fu2 = (f(x0, u0 + [0; epsilon; 0]) - f(x0, u0)) / epsilon
     fu3 = (f(x0, u0 + [0; 0; epsilon]) - f(x0, u0)) / epsilon
     fu_hat = hcat(fu1, fu2, fu3)


     # Ensure my Jacobians fx and fu were computed right.
     tol = 1e-5
     @test norm(fx(x0, u0) - fx_hat) < tol
     @test norm(fu(x0, u0) - fu_hat) < tol

     id = Matrix{Float64}(I, 2, 2)
     test_jac = zeros((3*2, 3*(2+3)))
     test_jac[1:2, 1:3] = -fu(x0, u0) 
     test_jac[1:2, 4:5] = id

     test_jac[3:4, 4:5] = -fx(x1, u1)
     test_jac[3:4, 6:8] = -fu(x1, u1) 
     test_jac[3:4, 9:10] = id 

     test_jac[5:6, 9:10] = -fx(x2, u2)
     test_jac[5:6, 11:13] = -fu(x2, u2)
     test_jac[5:6, 14:15] = id

     jac = nonlinear_eq_jacobian(z, x0, n_x, n_u, T, fx, fu)
     @test norm(jac - test_jac) < tol
     # println(jac)
     # println(test_jac)

     # Now perturb x1 and u2 by a lot. Update jac.
     x1 += [1., 2.]
     u2 += [4., 1., 5.]
     z = vcat(u0, x1, u1, x2, u2, x3)
     nonlinear_eq_jacobian(z, x0, n_x, n_u, T, fx, fu, jac)

     # Update test_jac.
     test_jac[3:4, 4:5] = -fx(x1, u1)
     test_jac[3:4, 6:8] = -fu(x1, u1) 
     test_jac[5:6, 9:10] = -fx(x2, u2)
     test_jac[5:6, 11:13] = -fu(x2, u2)
     @test norm(jac - test_jac) < tol
     # println(jac)
     # println(test_jac)
 end
 