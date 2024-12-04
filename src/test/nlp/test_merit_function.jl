include("../../optimizers/nlp/merit_function.jl")
using Test


@testset "test_merit_func_finite_diff" begin
    """
    Test merit_func, dmerit_dz, and dmerit_ds using finite differencing.
    """
    H = [1. 0. 0.;
         0. 2. 0.;
         0. 0. 4.]
    g = [1.; 2.; 3.]

    P = [1. 3. 2;
         6. 6. 4.]
    h = [2.; 8.]

    function eq_vec(x)
        return [x[1] - x[2]; x[2] - x[3]]
    end

    function eq_jac(x)
        return [1. -1. 0.; 0. 1. -1.]
    end

    mu = 2.3
    rho = 3.1
    z = [5.5; 4.3; 1.2]
    s = [2.3; 5.5]

    params = Dict("H" => H, "g" => g, "P" => P, "h" => h, 
                  "eq_vec" => eq_vec(z), "eq_jac" => eq_jac(z))

    mer_orig = merit_func(z, s, mu, params, rho)
    epsilon = 1e-6

    z1 = z + [epsilon; 0.; 0.]
    params["eq_vec"] = eq_vec(z1)
    params["eq_jac"] = eq_jac(z1)
    mer_z1 = merit_func(z1, s, mu, params, rho)

    z2 = z + [0.; epsilon; 0.]
    params["eq_vec"] = eq_vec(z2)
    params["eq_jac"] = eq_jac(z2)
    mer_z2 = merit_func(z2, s, mu, params, rho)

    z3 = z + [0.; 0.; epsilon]
    params["eq_vec"] = eq_vec(z3)
    params["eq_jac"] = eq_jac(z3)
    mer_z3 = merit_func(z3, s, mu, params, rho)

    params["eq_vec"] = eq_vec(z)
    params["eq_jac"] = eq_jac(z)

    dmerit_dz1 = (mer_z1 - mer_orig)/epsilon
    dmerit_dz2 = (mer_z2 - mer_orig)/epsilon
    dmerit_dz3 = (mer_z3 - mer_orig)/epsilon

    dmerit_dz_true = [dmerit_dz1; dmerit_dz2; dmerit_dz3]
    dmerit_dz_test = dmerit_dz(z, s, params, rho)

    tol = 1e-6
    println("true: ", dmerit_dz_true)
    println("test: ", dmerit_dz_test)
    @test norm(dmerit_dz_true - dmerit_dz_test) < tol

    # s1 = s + [epsilon; 0.]
    # mer_s1 = merit_func(z, s1, mu, params, rho)

    # s2 = s + [epsilon; 0.]
    # mer_s2 = merit_func(z, s2, mu, params, rho)
end
