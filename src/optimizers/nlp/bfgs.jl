using LinearAlgebra


"""
Updates the approximation of the Lagrangian Hessian according to damped BFGS update rules.

# Arguments
- `z_curr::Array`: the primal variable before the Newton step.
- `z_next::Array`: the primal variable after the Newton step.
- `H::Array`: the Hessian of the NLP.
- `eq_jac::Array`: the Jacobian of the equality constraint residual vector evaluated at z.
- `nu::Array`: the dual vector corresponding to the equality constraints AFTER the Newton step.
- `B::Array`: the BFGS approximation from the previous timestep.
"""
function damped_bfgs_update(z_curr, z_next, eq_jac_curr, eq_jac_next, H, nu, B)
    y = H*(z_next - z_curr) + transpose(eq_jac_next - eq_jac_curr) * nu
    dz = z_next - z_curr 

    quad_form = dz' * B * dz
    if transpose(dz) * y >= 0.2*quad_form
        # println(1)
        theta = 1
    else
        # println(2)
        theta = (0.8*quad_form)/(quad_form - transpose(dz) * y)
    end
    
    r = theta*y + (1-theta)*B*dz
    B_next = B - (B*dz * transpose(B*dz))/quad_form + (r * transpose(r))/(transpose(dz) * r)

    return B_next
end
