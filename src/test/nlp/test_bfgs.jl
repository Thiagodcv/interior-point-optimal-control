include("../../optimizers/nlp/bfgs.jl")
using Test


@testset "test_damped_bfgs_update" begin
    """
    Ensure damped_bfgs_update can run without failing.

    NOTE: this test does NOT test for correctness. 
    """
    z_curr = [1.; 2.; 3.]
    z_next = [4.; 5.; 0.]
    H = Matrix{Float64}(I, 3, 3)

    eq_jac_curr = [1. 2. 3.;
                   -1. 3. 5.]   
    eq_jac_next = [0.3 -2. -3.;
                   -1. 10. -5.]           
    nu = [1., -2.]
    B = Matrix{Float64}(I, 3, 3)
    B_next = damped_bfgs_update(z_curr, z_next, eq_jac_curr, eq_jac_next, H, nu, B)
    @test all(eigvals(B_next) .> 0)

    eq_jac_next = [0.3 -2. -3.;
                   -1. -10. -5.]           
    B_next = damped_bfgs_update(z_curr, z_next, eq_jac_curr, eq_jac_next, H, nu, B)
    @test all(eigvals(B_next) .> 0)
end
