include("./search_dir.jl")
include("./bfgs.jl")
include("./merit_function.jl")
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
    epsilon = 1e-3
    max_iters = 100
    max_iners = 1

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

    # Initial estimate of Hessian
    B = 2*copy(param["H"])

    # Initial mu
    mu = transpose(s) * lambda / n_lam
    sigma = 0.2

    # Initial kkt residual and its Jacobian
    kkt_res = kkt_residual_nlp(z, lambda, nu, s, param, nothing, mu)
    kkt_jac = kkt_jacobian_nlp(lambda, s, B, param)

    # Fraction to boundary parameter
    tau = 0.995

    for iter in 1:max_iters
        if norm(kkt_res) < epsilon
            return Dict("z" => z, "s" => s, "lambda" => lambda, "nu" => nu, "iters" => iter)
        end
        println("norm(kkt_res): ", norm(kkt_res))

        for iner in 1:max_iners # (max_iners+1)
            if norm(kkt_res) < mu
                break
            end
            println("norm(kkt_res): ", norm(kkt_res))

            # if iner > max_iners
            #     error("Did not converge within max_iners = ", max_iners, " iterations.")
            # end

            p_dir = compute_affine_scaling_dir(kkt_res, kkt_jac, n_z, n_lam, n_nu)  # Actually, the full direction
            alpha_s_max = frac_to_boundary(s, p_dir["s"], tau)
            alpha_lam_max = frac_to_boundary(lambda, p_dir["lambda"], tau)

            # Find optimal step size
            alpha_s = armijo_linesearch(z, s, p_dir["x"], p_dir["s"], alpha_s_max, mu, param, eq_consts, B)
            println("alpha_s: ", alpha_s)
            alpha_lam = alpha_s  # alpha_lam_max
            println("alpha_lam: ", alpha_lam)

            # Update Hessian approximation
            z_next = z + alpha_s * p_dir["x"]
            eq_jac_next = eq_consts["jac"](z_next)
            B = damped_bfgs_update(z, z_next, param["eq_jac"], eq_jac_next, param["H"], nu, B)

            # Update iterates
            z = z_next
            s = s + alpha_s * p_dir["s"]
            lambda = lambda + alpha_lam * p_dir["lambda"]
            nu = nu + alpha_lam * p_dir["nu"]

            # Update equality constraint residual and its Jacobian
            param["eq_vec"] = eq_consts["vec"](z)
            param["eq_jac"] = eq_jac_next

            # Update KKT residual and its Jacobian
            kkt_residual_nlp(z, lambda, nu, s, param, kkt_res, mu)
            kkt_jacobian_nlp(lambda, s, B, param, kkt_jac)
        end

        mu = sigma * mu
    end

    error("Did not converge within max_iters = ", max_iters, " iterations.")
end


"""
    frac_to_boundary(u, p_u, tau)

Finds a step size using the fraction to boundary rule. 

# Arguments
- `u::Array`: the slack variable.
- `p_u::Array`: the step in the slack variable.
- `tau::Float64`: the parameter for frac_to_boundary

# Returns
- `Float64`: the step size found using the fraction to boundary rule.

"""
function frac_to_boundary(u, p_u, tau=0.995)
    neg_idx = findall(x -> x < 0, p_u)

    if isempty(neg_idx)
        alpha_max = 1
    else
        alpha_max = minimum(-tau * u[neg_idx] ./ p_u[neg_idx])
    end
end
