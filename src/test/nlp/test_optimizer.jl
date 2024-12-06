using Test
include("../../tools/mpc_tools.jl")
include("../../optimizers/nlp/optimizer.jl")


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

    param = Dict()
    param["H"] = qp_dict["Q"]
    param["g"] = qp_dict["q"]
    param["P"] = qp_dict["G"]
    param["h"] = qp_dict["h"]

    function eq_vec(z)
        return qp_dict["A"] * z - qp_dict["b"]
    end

    function eq_jac(z)
        return qp_dict["A"]
    end

    eq_consts = Dict("vec" => eq_vec, "jac" => eq_jac)
    z_init = zeros((T*(n+m),))

    ret = pdip_nlp(param, eq_consts, z_init)
    ret_separate = separate_solution(ret["z"], n, m, u_latest, T)

    println("iters: ", ret["iters"])
    # println("solution: ", ret["x"])
    println("state solution: ", ret_separate["x"])
    println("input solution: ", ret_separate["u"])
    println("diff input solution: ", ret_separate["du"])
end


@testset "test_pendulum" begin
    """
    Test to see if NLP optimizer can solve the pendulum problem.
    """
    g = 9.8
    dt = 0.01

    # The continuous-time dynamics function
    function f(x, u)
        return [x[2]; -g*sin(x[1]) - x[2] + u]
    end

    # Jacobian of f w.r.t. x
    function f_x(x, u)
        return [0. 1.; -g*cos(x[1]) -1.]
    end

    # Jacobian of f w.r.t. u
    function f_u(x, u)
        return [0.; 1.]
    end

    # The discrete-time dynamics function
    function h(x, u)
        return x + f(x, u)*dt
    end

    # Jacobian of h w.r.t. x
    function h_x(x, u)
        return [1. 0.; 0. 1.] + f_x(x, u)*dt
    end

    # Jacobian of h w.r.t. u
    function h_u(x, u)
        return f_u(x, u)*dt
    end

    n_x = 2
    n_u = 1
    T = 10
    x0 = [0.2; 0.2]

    constraint_vec = zeros((T*n_x,))
    jac = zeros((T*n_x, T*(n_x + n_u)))

    function eq_const_func(z)
        return nonlinear_eq_constraint(z, x0, n_x, n_u, T, h, constraint_vec)
    end

    function eq_jac_func(z)
        return nonlinear_eq_jacobian(z, x0, n_x, n_u, T, h_x, h_u, jac)
    end

end
