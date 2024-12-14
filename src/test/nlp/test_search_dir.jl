include("../../optimizers/nlp/search_dir.jl")
using Test


@testset "test_kkt_residual_nlp" begin
    """
    Ensure kkt_residual_nlp returns the correct solution.
    """
    # obj
    H = [1. 2. 3.;
         5. 7. 6.;
         8. 3. 6.]
    g = [1.; 2.; 7.]
    z = [2., 1., -2.]

    # linear constraints
    P = [1. -3. 6.;
         8. -1. 3.]
    h = [-4.; 3.]
    s = [1.; 2.]
    lambda = [1.; 5.]

    # equality constraints
    eq_vec = [-1.; 0.4; -0.2]
    eq_jac = [-1. 2. 3.;
              5. -7. 6.;
              -8. -3. 6.]
    nu = [1.; -4.; 3.4]

    sol = zeros((10,))
    sol[1:3] = 2*H*z + g + transpose(P)*lambda + transpose(eq_jac)*nu 
    sol[4:5] = Diagonal(s) * lambda
    sol[6:7] = P*z - h + s 
    sol[8:10] = eq_vec

    param = Dict("H" => H, "g" => g, "P" => P, "h" => h, "eq_vec" => eq_vec, "eq_jac" => eq_jac)
    res_vec = kkt_residual_nlp(z, lambda, nu, s, param)

    tol = 1e-6
    println(sol)
    println(res_vec)
    @test norm(sol - res_vec) < tol

    z = [3., 3., -3.]
    s = [0.5; 3.]
    lambda = [1.6; 5.7]
    nu = [1.; -4.; 3.4]
    eq_vec = [1.6; -0.4; -0.2]
    eq_jac = [-1. 2. 3.;
              5. 7. 6.;
              -8. 3. 6.]

    sol[1:3] = 2*H*z + g + transpose(P)*lambda + transpose(eq_jac)*nu 
    sol[4:5] = Diagonal(s) * lambda
    sol[6:7] = P*z - h + s 
    sol[8:10] = eq_vec

    param["eq_vec"] = eq_vec
    param["eq_jac"] = eq_jac

    kkt_residual_nlp(z, lambda, nu, s, param, res_vec)
    @test norm(sol - res_vec) < tol
end


@testset "test_kkt_jacobian_nlp" begin
    """
    Ensure kkt_jacobian_nlp returns the correct solution.
    """
    # obj
    H = [1. 2. 3.;
         5. 7. 6.;
         8. 3. 6.]
    g = [1.; 2.; 7.]
    z = [2., 1., -2.]

    # linear constraints
    P = [1. -3. 6.;
         8. -1. 3.]
    h = [-4.; 3.]
    s = [1.; 2.]
    lambda = [1.; 5.]

    # equality constraints
    eq_vec = [-1.; 0.4; -0.2]
    eq_jac = [-1. 2. 3.;
              5. -7. 6.;
              -8. -3. 6.]
    nu = [1.; -4.; 3.4]

    # BFGS estimate
    B = 2.1*Matrix{Float64}(I, 3, 3)

    sol = zeros((10,10))

    sol[1:3, 1:3] = B
    sol[1:3, 6:7] = transpose(P)
    sol[1:3, 8:10] = transpose(eq_jac)

    sol[4:5, 4:5] = Diagonal(lambda)
    sol[4:5, 6:7] = Diagonal(s)

    sol[6:7, 1:3] = P
    sol[6:7, 4:5] = Matrix{Float64}(I, 2, 2)

    sol[8:10, 1:3] = eq_jac

    param = Dict("H" => H, "g" => g, "P" => P, "h" => h, "eq_vec" => eq_vec, "eq_jac" => eq_jac)
    jac = kkt_jacobian_nlp(lambda, s, B, param)

    tol = 1e-6
    @test norm(sol - jac) < tol

    s = [3.; 6.]
    lambda = [1.; 5.]
    param["eq_vec"] = [1.; -0.4; 0.2]
    param["eq_jac"] = [-1. 2. 3.;
                        5. 7. 6.; 
                        8.7 -3. 6.]
    B = 0.5*Matrix{Float64}(I, 3, 3)

    sol[1:3, 1:3] = B
    sol[1:3, 8:10] = transpose(param["eq_jac"])
    sol[4:5, 4:5] =  Diagonal(lambda)
    sol[4:5, 6:7] = Diagonal(s)
    sol[8:10, 1:3] = param["eq_jac"]

    kkt_jacobian_nlp(lambda, s, B, param, jac)
    @test norm(sol - jac) < tol
end
