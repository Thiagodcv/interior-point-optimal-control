include("search_dir.jl")


"""
    pdip_qp(param_dict, x0)

Primal-dual interior point QP with Mehrotra correction.

# Arguments
- `param_dict::Dict{String, Array}`: the parameters which define the QP problem. Contains key-value pairs
        - "Q"::Array: the Hessian of the QP (n,n),
        - "q"::Array: the linear term of the QP (n,),
        - "G"::Array: the linear inequality constraint matrix (p,n),
        - "h"::Array: the linear inequality constraint limits (p,),
        - "A"::Array: the linear equality constraint matrix (m,n),
        - "b"::Array: the linear equality constraint limits (m,).
- `x0::Array`: the initial primal iterate (n,). Must satisfy inequality constraints.

# Returns
- `Dict{String, Array}`: primal dual solution and optimization details.
"""
function pdip_qp(param_dict, x0)
    epsilon_feas = 1e-6
    epsilon = 1e-7
    max_iters = 100

    # Find dimensions
    n = size(x0)[1]
    p = size(param_dict["G"])[1]
    m = size(param_dict["A"])[1]

    # Set initial primal and dual iterates
    x = x0
    s = param_dict["h"] - param_dict["G"] * x
    lambda = 1 ./ s
    nu = ones((m,))  # Is this a good initial guess for nu?

    # Other parameters
    mu = transpose(s) * lambda / p

    # Initial kkt residual and its Jacobian
    kkt_res = compute_kkt_residual(param_dict, x, lambda, nu, s)
    kkt_jac = compute_kkt_jacobian(param_dict, lambda, s)

    for iter in 1:max_iters
        # TODO: Update the stopping criteria
        if norm(kkt_res) < epsilon
            return Dict("x" => x, "s" => s, "lambda" => lambda, "nu" => nu, "iters" => iter)
        end

        aff_dir = compute_affine_scaling_dir(kkt_res, kkt_jac, n, p, m)
        
        # Compute sigma
        alpha_sigma = sigma_step(s, lambda, aff_dir["s"], aff_dir["lambda"])
        sigma = transpose(s + alpha_sigma * aff_dir["s"]) * (lambda + alpha_sigma * aff_dir["lambda"]) / (transpose(s) * lambda)
        sigma = sigma^3

        cc_dir = compute_centering_plus_corrector_dir(kkt_jac, aff_dir["s"], aff_dir["lambda"], sigma, mu, n, p, m)
        
        # Update primal and dual iterates
        alpha = primal_dual_step(s, lambda, aff_dir["s"], aff_dir["lambda"])
        x = x + alpha * (aff_dir["x"] + cc_dir["x"])
        s = s + alpha * (aff_dir["s"] + cc_dir["s"])
        lambda = lambda + alpha * (aff_dir["lambda"] + cc_dir["lambda"])
        nu = nu + alpha * (aff_dir["nu"] + cc_dir["nu"])

        # Update kkt residual and its Jacobian
        compute_kkt_residual(param_dict, x, lambda, nu, s, kkt_res)
        compute_kkt_jacobian(param_dict, lambda, s, kkt_jac)
    end

    error("Did not converge within max_iters = ", max_iters, " iterations.")
end


"""
    sigma_step(s, lambda, d_s_aff, d_lambda_aff)

Find the optimal step size alpha for computing the sigma parameter.

# Arguments
- `s::Array`: the slack variable for the inequality constraint.
- `lambda::Array`: the dual variable associated with the inequality constraint.
- `d_s_aff::Array`: the affine scaling direction for s.
- `d_lambda_aff::Array`: the affine scaling direction for lambda.

# Returns
- `Float64`: the step size
"""
function sigma_step(s, lambda, d_s_aff, d_lambda_aff)
    alpha_s = initial_step_size(s, d_s_aff)
    alpha_lambda = initial_step_size(lambda, d_lambda_aff)
    return min(1, alpha_s, alpha_lambda)
end


"""
    primal_dual_step(s, lambda, d_s_aff, d_lambda_aff)

Find the optimal step size alpha for updating the primal and dual iterates.

# Arguments
- `s::Array`: the slack variable for the inequality constraint.
- `lambda::Array`: the dual variable associated with the inequality constraint.
- `d_s_aff::Array`: the affine scaling direction for s.
- `d_lambda_aff::Array`: the affine scaling direction for lambda.

# Returns
- `Float64`: the step size
"""
function primal_dual_step(s, lambda, d_s_aff, d_lambda_aff)
    alpha_s = initial_step_size(s, d_s_aff)
    alpha_lambda = initial_step_size(lambda, d_lambda_aff)
    return min(1, 0.99*alpha_s, 0.99*alpha_lambda)
end


"""
    max_step_size(u, d_u)

Find the largest positive step length (not exceeding 1) that ensures u_next is 
positive (element-wise).

# Arguments
- `u::Array`: an arbitrary variable.
- `d_u::Array`: a step for the arbitrary variable u.

# Returns
- `Float64`: max step size.
"""
function initial_step_size(u, d_u)
    neg_idx = findall(x -> x < 0, d_u)
    if isempty(neg_idx)
        alpha_max = 1
    else
        alpha_max = min(1, minimum(-u[neg_idx] ./ d_u[neg_idx]))
    end

    return alpha_max
end
