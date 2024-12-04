using LinearAlgebra


"""
    merit_func(z, s, mu, params, rho)

Evaluate the merit function corresponding to an interior point primal dual method.

# Arguments
- `z::Array`: the primal variable.
- `s::Array`: the slack variable.
- `mu::Float64`: the barrier parameter.
- `params::Dict{String, Array}`: the parameters of the NLP problem.
- `rho::Float64`: the parameter of the merit function.

# Returns
- `Float65`: the merit function evaluated at (z,s) parameterized by (mu, rho).
"""
function merit_func(z, s, mu, params, rho)
    return -1
end
