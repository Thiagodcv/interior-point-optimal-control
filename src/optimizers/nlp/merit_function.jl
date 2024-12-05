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
    eq_part = rho / norm(eq_vec) * transpose(eq_jac) * eq_vec

    P = params["P"]
    h = params["h"]
    ineq_part = rho / norm(P*z - h + s) * transpose(P) * (P*z - h + s)

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
    ineq_part = rho / norm(P*z - h + s) * (P*z - h + s)

    return barrier_part + ineq_part
end


"""
    armijo_linesearch(z, s, p_z, p_s, alpha_max, rho, params)

Armijo linesearch over the primal and slack variables conducted on the merit function.

# Arguments
- `z::Array`: the primal variable.
- `s::Array`: the slack variable.
- `p_z::Array`: a step in the primal variable.
- `p_s::Array`: a step in the slack variable.
- `alpha_max::Float64`: the maximum step size to start linesearch.
- `rho::Float64`: the parameter of the merit function.
- `params::Dict{String, Array}`: the parameters of the NLP problem. Contains key-value pairs
        - "H"::Array: the Hessian of the NLP,
        - "g"::Array: the linear term of the NLP,
        - "P"::Array: the inequality constraint matrix,
        - "h"::Array: the inequality constraint vector,
        - "eq_vec"::Array: the equality constraint residual vector evaluated at z.
        - "eq_jac"::Array: the Jacobian of the equality constraint residual vector evaluated at z.

# Returns
- `Float64`: a step size.
"""
function armijo_linesearch(z, s, p_z, p_s, alpha_max, rho, params)
    max_iter = 100
    heta = 1e^(-4)  # in (0, 1). Slope dampener. 
    c = 0.5  # in (0, 1). Contraction factor for decreasing step length.
    mu = 1  # how to set this?

    alpha = alpha_max
    merit_curr = merit_func(z, s, mu, params, rho)
    d_merit = dmerit_dz(z, s, params, rho) * p_z + dmerit_ds(z, s, mu, params, rho) * p_s

    for iter in 1:max_iter
        z_next = z + alpha * p_z
        s_next = s + alpha * p_s

        merit_next = merit_func(z_next, s_next, mu, params, rho)

        if merit_next <= merit_curr + heta * alpha * d_merit
            return alpha
        end
        
        alpha = c * alpha
    end

    error("Armijo line search did not converge within max_iters = ", max_iters, " iterations.")
end
