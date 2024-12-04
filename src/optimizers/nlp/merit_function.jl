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
