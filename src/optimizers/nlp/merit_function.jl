using LinearAlgebra
include("../../tools/mpc_tools.jl")


"""
    merit_func(z, s, mu, params, rho)

Evaluate the merit function corresponding to an interior point primal dual method.

# Arguments
- `z::Array`: the primal variable.
- `s::Array`: the slack variable.
- `mu::Float64`: the barrier parameter.
- `params::Dict{String, Array}`: the parameters of the NLP problem. Contains key-value pairs
        - "H"::Array: the Hessian of the NLP,
        - "g"::Array: the linear term of the NLP,
        - "P"::Array: the inequality constraint matrix,
        - "h"::Array: the inequality constraint vector,
        - "eq_vec"::Array: the equality constraint residual vector evaluated at z.
- `rho::Float64`: the parameter of the merit function.

# Returns
- `Float64`: the merit function evaluated at (z,s) parameterized by (mu, rho).
"""
function merit_func(z, s, mu, params, rho)
    barrier_obj = transpose(z)*params["H"]*z + transpose(params["g"])*z - mu*sum(log.(s))
    constraint_cost = rho*norm(params["eq_vec"]) + rho*norm(params["P"]*z - params["h"] + s)
    return barrier_obj + constraint_cost
end


"""
    dmerit_dz(z, s, params, rho)

Evaluate the gradient of the merit function with respect to z.

# Arguments
- `z::Array`: the primal variable.
- `s::Array`: the slack variable.
- `params::Dict{String, Array}`: the parameters of the NLP problem. Contains key-value pairs
        - "H"::Array: the Hessian of the NLP,
        - "g"::Array: the linear term of the NLP,
        - "P"::Array: the inequality constraint matrix,
        - "h"::Array: the inequality constraint vector,
        - "eq_vec"::Array: the equality constraint residual vector evaluated at z.
        - "eq_jac"::Array: the Jacobian of the equality constraint residual vector evaluated at z.
- `rho::Float64`: the parameter of the merit function.

# Returns
- `Array`: the gradient of the merit function with respect to z.
"""
function dmerit_dz(z, s, params, rho)
    obj_part = 2*params["H"] * z + params["g"] 

    eq_vec = params["eq_vec"]
    eq_jac = params["eq_jac"]
    fudge = 1e-10
    eq_part = rho / (norm(eq_vec) + fudge) * transpose(eq_jac) * eq_vec

    P = params["P"]
    h = params["h"]
    ineq_part = rho / (norm(P*z - h + s) + fudge) * transpose(P) * (P*z - h + s)

    return obj_part + eq_part + ineq_part
end


"""
    dmerit_ds(z, s, mu, params, rho)

Evaluate the gradient of the merit function with respect to s.

# Arguments
- `z::Array`: the primal variable.
- `s::Array`: the slack variable.
- `mu::Float64`: the barrier parameter.
- `params::Dict{String, Array}`: the parameters of the NLP problem. Contains key-value pairs
        - "P"::Array: the inequality constraint matrix,
        - "h"::Array: the inequality constraint vector,
- `rho::Float64`: the parameter of the merit function.

# Returns
- `Array`: the gradient of the merit function with respect to s.
"""
function dmerit_ds(z, s, mu, params, rho)
    P = params["P"]
    h = params["h"]

    barrier_part = -mu ./ s 
    fudge = 1e-10
    ineq_part = rho / (norm(P*z - h + s) + fudge) * (P*z - h + s)

    return barrier_part + ineq_part
end


"""
    armijo_linesearch(z, s, p_z, p_s, alpha_max, mu, params, eq_consts, dd_L)

Armijo linesearch over the primal and slack variables conducted on the merit function.

# Arguments
- `z::Array`: the primal variable.
- `s::Array`: the slack variable.
- `p_z::Array`: a step in the primal variable.
- `p_s::Array`: a step in the slack variable.
- `alpha_max::Float64`: the maximum step size to start linesearch.
- `mu::Float64`: the barrier parameter.
- `params::Dict{String, Array}`: the parameters of the NLP problem. Contains key-value pairs
        - "H"::Array: the Hessian of the NLP,
        - "g"::Array: the linear term of the NLP,
        - "P"::Array: the inequality constraint matrix,
        - "h"::Array: the inequality constraint vector,
        - "eq_vec"::Array: the equality constraint residual vector evaluated at z.
        - "eq_jac"::Array: the Jacobian of the equality constraint residual vector evaluated at z.
- `eq_consts::Dict{String, Function}`: the residual vector and its Jacobian of equality constraints. Contains key-value pairs
        - "vec"::Function: the residual vector as a function of z,
        - "jac"::Function: the Jacobian as a function of z.
- `dd_L:Array`: the Hessian of the Lagrangian, approximated or exact. Assumed to be positive definite.

# Returns
- `Float64`: a step size.
"""
function armijo_linesearch(z, s, p_z, p_s, alpha_max, mu, params, eq_consts, dd_L)
    max_iters = 100
    heta = 10^(-4)  # in (0, 1). Slope dampener. 
    c = 0.5  # in (0, 1). Contraction factor for decreasing step length.
    rho = 1000 # compute_rho(z, p_z, params, dd_L)  # merit function parameter

    alpha = alpha_max
    merit_curr = merit_func(z, s, mu, params, rho)
    d_merit = dmerit_dz(z, s, params, rho)' * p_z + dmerit_ds(z, s, mu, params, rho)' * p_s

    params_step = copy(params)
    for iter in 1:max_iters
        z_next = z + alpha * p_z
        s_next = s + alpha * p_s

        params_step["eq_vec"] = eq_consts["vec"](z_next)
        params_step["eq_jac"] = eq_consts["jac"](z_next)
        merit_next = merit_func(z_next, s_next, mu, params_step, rho)

        if merit_next <= merit_curr + heta * alpha * d_merit
            return alpha
        end
        
        alpha = c * alpha
    end

    error("Armijo line search did not converge within max_iters = ", max_iters, " iterations.")
end


"""
Computes the parameter for the merit function assuming the first block
diagonal of the KKT matrix is positive definite. Based off of (Nocedal, 18.36).

NOTE: this function only looks at the primal variable and not the slack variable. 
This may cause issues.

# Arguments
- `z::Array`: the primal variable.
- `p_z::Array`: a step in the primal variable.
- `params::Dict{String, Array}`: the parameters of the NLP problem. Contains key-value pairs
        - "H"::Array: the Hessian of the NLP,
        - "g"::Array: the linear term of the NLP,
        - "P"::Array: the inequality constraint matrix,
        - "h"::Array: the inequality constraint vector,
        - "eq_vec"::Array: the equality constraint residual vector evaluated at z.
        - "eq_jac"::Array: the Jacobian of the equality constraint residual vector evaluated at z.
- `dd_L:Array`: the Hessian of the Lagrangian, approximated or exact. Assumed to be positive definite.

# Returns
- `Float64`: the merit function parameter.
"""
function compute_rho(z, p_z, params, dd_L)
    df = params["H"]*z + params["g"]
    sigma = 1
    lb_param = 0.5  # Not too sure what to set this to.

    rho_lb = transpose(df) * p_z + (sigma/2)*transpose(p_z) * dd_L * p_z
    rho_lb = rho_lb/((1-lb_param)*norm(params["eq_vec"]))
    rho = min((1.01)*rho_lb, 0.999)

    return rho
end
