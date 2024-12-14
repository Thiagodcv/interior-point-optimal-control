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

    tol = 1e-5
    # println("true: ", dmerit_dz_true)
    # println("test: ", dmerit_dz_test)
    @test norm(dmerit_dz_true - dmerit_dz_test) < tol

    s1 = s + [epsilon; 0.]
    mer_s1 = merit_func(z, s1, mu, params, rho)

    s2 = s + [0.; epsilon]
    mer_s2 = merit_func(z, s2, mu, params, rho)

    dmerit_ds1 = (mer_s1 - mer_orig) / epsilon
    dmerit_ds2 = (mer_s2 - mer_orig) / epsilon

    dmerit_ds_true = [dmerit_ds1; dmerit_ds2]
    dmerit_ds_test = dmerit_ds(z, s, mu, params, rho)

    # println("true: ", dmerit_ds_true)
    # println("test: ", dmerit_ds_test)
    @test norm(dmerit_ds_true - dmerit_ds_test) < tol

    p_z = [1.; 2.; 3.]
    p_s = [0.1; 2.5]

    alpha = 1e-7
    params["eq_vec"] = eq_vec(z)
    params["eq_jac"] = eq_jac(z)
    direc_dmerit = p_z' * dmerit_dz(z, s, params, rho) + p_s' * dmerit_ds(z, s, mu, params, rho)

    params_next = copy(params)
    params_next["eq_vec"] = eq_vec(z+alpha*p_z)
    params_next["eq_jac"] = eq_jac(z+alpha*p_z)
    direc_dmerit_test = merit_func(z+alpha*p_z, s+alpha*p_s, mu, params_next, rho) - merit_func(z, s, mu, params, rho)
    direc_dmerit_test = direc_dmerit_test/alpha

    println("direc_dmerit: ", direc_dmerit)
    println("direc_dmerit_test: ", direc_dmerit_test)
    # println("diff between params: ", norm(params_next["eq_jac"] - params["eq_jac"]))
    # println(merit_func(z+alpha*p_z, s+alpha*p_s, mu, params_next, rho) - merit_func(z+alpha*p_z, s+alpha*p_s, mu, params, rho))
    @test norm(direc_dmerit - direc_dmerit_test) < tol
end
