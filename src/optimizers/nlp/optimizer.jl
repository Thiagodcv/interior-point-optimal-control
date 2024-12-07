include("./search_dir.jl")
include("./bfgs.jl")
include("../qp/search_dir.jl")
include("../qp/optimizer.jl")


"""
    pdip_nlp(param, eq_consts, x0)

Primal-dual interior point NLP with Mehrotra correction. This implementation
is specifically tailored to optimal control problems.

# Arguments
- `param::Dict{String, Array}`: the parameters of the NLP problem. Contains key-value pairs
        - "H"::Array: the Hessian of the NLP,
        - "g"::Array: the linear term of the NLP,
        - "P"::Array: the inequality constraint matrix,
        - "h"::Array: the inequality constraint vector.
- `eq_consts::Dict{String, Function}`: the residual vector and its Jacobian of equality constraints. Contains key-value pairs
        - "vec"::Function: the residual vector as a function of z,
        - "jac"::Function: the Jacobian as a function of z.
- `z0::Array`: the initial primal iterate. Must satisfy inequality constraints. 

# Returns
- `Dict{String, Array}`: primal dual solution and optimization details.
"""
function pdip_nlp(param, eq_consts, z0)
    epsilon = 1e-8
    max_iters = 100

    # Evaluate equality constraints at z0
    param["eq_vec"] = eq_consts["vec"](z0)
    param["eq_jac"] = eq_consts["jac"](z0)

    # Size of primal variable and constraints
    n_z = size(z0)[1]
    n_lam = size(param["P"])[1]
    n_nu = size(param["eq_vec"])[1]

    # Set initial primal and dual iterates
    z = z0
    s = param["h"] - param["P"]*z
    lambda = 1 ./ s
    nu = ones((n_nu,))

    # Barrier parameter and initial estimate of Hessian
    mu = transpose(s) * lambda / n_lam
    B = copy(param["H"])

    # Initial kkt residual and its Jacobian
    kkt_res = kkt_residual_nlp(z, lambda, nu, s, param)
    kkt_jac = kkt_jacobian_nlp(lambda, s, B, param)

    for iter in 1:max_iters
        if norm(kkt_res) < epsilon
            return Dict("z" => z, "s" => s, "lambda" => lambda, "nu" => nu, "iters" => iter)
        end

        aff_dir = compute_affine_scaling_dir(kkt_res, kkt_jac, n_z, n_lam, n_nu)

        # Compute sigma
        alpha_sigma = sigma_step(s, lambda, aff_dir["s"], aff_dir["lambda"])  # Figure this out.
        sigma = transpose(s + alpha_sigma * aff_dir["s"]) * (lambda + alpha_sigma * aff_dir["lambda"]) / (transpose(s) * lambda)
        sigma = sigma^3

        cc_dir = centering_plus_corrector_dir_nlp(kkt_jac, aff_dir["s"], aff_dir["lambda"], sigma, s, mu, n_z, n_lam, n_nu)

        # Update approximate Hessian using BFGS
        alpha = primal_dual_step(s, lambda, aff_dir["s"], aff_dir["lambda"])
        z_next = z + alpha * (aff_dir["x"] + cc_dir["x"])
        eq_jac_next = eq_consts["jac"](z_next)
        B = damped_bfgs_update(z, z_next, param["eq_jac"], eq_jac_next, param["H"], nu, B)

        # println("min_abs_eval(kkt_jac): ", minimum(abs.(eigvals(kkt_jac))))
        # println("min_eval(B): ", minimum(eigvals(B)))
        # println("min Sigma: ", minimum(Diagonal(1 ./ s) * lambda))
        println("norm(kkt_res): ", norm(kkt_res))

        # Update iterates
        z = z_next
        s = s + alpha * (aff_dir["s"] + cc_dir["s"])
        lambda = lambda + alpha * (aff_dir["lambda"] + cc_dir["lambda"])
        nu = nu + alpha * (aff_dir["nu"] + cc_dir["nu"])

        # Update equality constraints
        param["eq_vec"] = eq_consts["vec"](z_next)
        param["eq_jac"] = eq_jac_next

        # Update KKT residual and its Jacobian
        kkt_residual_nlp(z, lambda, nu, s, param, kkt_res)
        kkt_jacobian_nlp(lambda, s, B, param, kkt_jac)
    end

    error("Did not converge within max_iters = ", max_iters, " iterations.")
end
