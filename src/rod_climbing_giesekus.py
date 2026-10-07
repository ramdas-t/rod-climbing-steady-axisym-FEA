#!/usr/bin/env python3
"""
rod_climbing_giesekus.py

Steady, axisymmetric, finite-Reynolds-number rod climbing of a Giesekus
fluid in a finite-radius cylindrical bath.

Numerical formulation
---------------------
* 2-D meridional (r,z) mesh with an axisymmetric coordinate system.
* The azimuthal velocity u_theta is retained as an independent unknown.
* Giesekus is solved in the Fattal-Kupferman log-conformation formulation.
* ALE free surface: the interface is a moving mesh boundary.
* Surface tension is imposed in weak Young-Laplace form.
* The liquid volume is enforced with a pressure/volume Lagrange multiplier.
* Infinite depth is approximated by a finite depth H.  DEPTH_CONVERGENCE
  below can be used to demonstrate convergence as H is increased.
* The pressure variable solved by NavierStokesEquations is the physical
  pressure p.  The modified pressure used in the manuscript is
      Phi = p + G*z
  where
      G = rho*g*a/(eta0*Omega).
  Thus the code is algebraically equivalent to the stated
      Re (u.grad)u = -grad(Phi) + div(sigma)
  formulation.

The dimensionless scales are
    length       : a
    velocity     : a*Omega
    time         : 1/Omega
    stress       : eta0*Omega

so that
    Re       = rho*a^2*Omega/eta0
    Wi_G     = lambda*Omega
    Ca       = eta0*Omega*a/gamma
    Bo       = rho*g*a^2/gamma
    G        = rho*g*a/(eta0*Omega) = Re/Fr^2

The Giesekus model is
    tau_p = eta_p/lambda * (C-I)
    C^triangledown =
        -(1/lambda)[(C-I) + alpha (C-I)^2].

In dimensionless variables this is implemented by pyoomph as
    relaxation_time = Wi_G
    polymer_viscosity = eta_p/eta0
and formulation="log-conf".

IMPORTANT:
The pyoomph version targeted here is 0.2.x.  The code deliberately uses
pyoomph's built-in axisymmetric log-conformation machinery rather than
reimplementing the Fattal-Kupferman eigenframe algebra by hand.  The one
custom constitutive/flow coupling below is the polymer contribution to the
separate azimuthal momentum equation.

Run:
    python rod_climbing_giesekus.py

The code writes VTU/PVD output into OUTPUT_DIRECTORY.
"""

from __future__ import annotations

import math
from pathlib import Path

from pyoomph import *

from pyoomph.equations.navier_stokes import (
    NavierStokesEquations,
    NavierStokesFreeSurface,
    NavierStokesContactAngle,
    NavierStokesAzimuthalComponent,
    NoSlipBC,
)

from pyoomph.equations.viscoelastic import ViscoelasticEquations, Giesekus
from pyoomph.equations.ALE import PseudoElasticMesh, EnforceVolumeByPressure
from pyoomph.meshes.simplemeshes import RectangularQuadMesh
from pyoomph.expressions import *
# from pyoomph.expressions.coordsys import axisymmetric
from pyoomph.expressions.units import degree

# =============================================================================
# USER PARAMETERS
# =============================================================================
#
# All physical inputs below are dimensional numerical values in a consistent
# unit system.  The solver itself is dimensionless.  The default values are
# deliberately modest and are intended as a starting point, not as a benchmark.
#

# Geometry
ROD_RADIUS = 1.0
OUTER_RADIUS = 10.0

# Rotation
OMEGA_TARGET = 1.0                 # rod angular velocity
OMEGA_INITIAL = 0.02               # continuation starting point
N_OMEGA_STEPS = 8

# Fluid
RHO = 1.0                          # density
ETA_S = 0.5                        # solvent viscosity
ETA_P = 0.5                        # polymer viscosity
LAMBDA = 1.0                       # Giesekus relaxation time
GIESEKUS_ALPHA = 0.1               # 0 <= alpha <= 1/2

# External physics
GRAVITY = 1.0                      # gravitational acceleration
SURFACE_TENSION = 1.0              # gamma
ATMOSPHERIC_PRESSURE = 0.0         # gauge atmospheric pressure

# Contact angle
CONTACT_ANGLE_DEG = 90.0

# Numerical depth used to approximate z -> -infinity.
# Increase this until h(r) and stresses are insensitive to DEPTH.
DEPTH = 25.0

# Mesh
NR = 60
NZ = 120

# Finite-element choices
VELOCITY_MODE = "TH"               # Taylor-Hood
VELOCITY_SPACE = "C2"
PRESSURE_SPACE = "C1"
LOGCONF_SPACE = "C2"

# Constitutive stabilization.
# Start at zero.  After the first converged solution, ramping this toward 1
# is usually safer than starting with full SUPG at the rest state.
USE_SUPG = True
SUPG_INITIAL = 0.0
SUPG_FINAL = 1.0

# Mesh motion
# PseudoElasticMesh is substantially more robust than pure Laplace smoothing
# when the free surface develops a significant deformation.
MESH_ELASTIC_NU = 0.3
MESH_ELASTIC_E = 1.0

# Optional remeshing is intentionally disabled by default.  First establish
# depth/mesh convergence on a fixed topology.
USE_REMESHING = False

# Nonlinear solver controls
NEWTON_TOL = 1.0e-8
MAX_NEWTON_ITER = 50

# Output
OUTPUT_DIRECTORY = "rod_climbing_output"

# Run controls
RUN_CONTINUATION = True
WRITE_EVERY_STEP = True

# Optional depth convergence study.  Set to e.g. [15, 25, 40, 60] after the
# main solver works.
DEPTH_CONVERGENCE = False
DEPTH_LIST = [15.0, 25.0, 40.0, 60.0]


# =============================================================================
# DERIVED DIMENSIONLESS GROUPS
# =============================================================================

ETA0 = ETA_S + ETA_P
BETA_S = ETA_S / ETA0
BETA_P = ETA_P / ETA0

if ETA0 <= 0.0:
    raise ValueError("ETA_S + ETA_P must be positive.")

if not (0.0 <= GIESEKUS_ALPHA <= 0.5):
    raise ValueError("GIESEKUS_ALPHA should satisfy 0 <= alpha <= 1/2.")

if OUTER_RADIUS <= ROD_RADIUS:
    raise ValueError("OUTER_RADIUS must exceed ROD_RADIUS.")

if DEPTH <= 0.0:
    raise ValueError("DEPTH must be positive.")

CONTACT_ANGLE = CONTACT_ANGLE_DEG * math.pi / 180.0

RE_TARGET = RHO * ROD_RADIUS**2 * OMEGA_TARGET / ETA0
WI_G_TARGET = LAMBDA * OMEGA_TARGET
CA_TARGET = ETA0 * OMEGA_TARGET * ROD_RADIUS / SURFACE_TENSION
BO = RHO * GRAVITY * ROD_RADIUS**2 / SURFACE_TENSION

print("=" * 72)
print("Rod-climbing Giesekus FEM")
print("=" * 72)
print(f"Re(target)       = {RE_TARGET:g}")
print(f"Wi_G(target)     = {WI_G_TARGET:g}")
print(f"Ca(target)       = {CA_TARGET:g}")
print(f"Bo               = {BO:g}")
print(f"eta_s/eta0       = {BETA_S:g}")
print(f"eta_p/eta0       = {BETA_P:g}")
print(f"Giesekus alpha   = {GIESEKUS_ALPHA:g}")
print(f"R/a              = {OUTER_RADIUS / ROD_RADIUS:g}")
print(f"H/a              = {DEPTH / ROD_RADIUS:g}")
print("=" * 72)


# =============================================================================
# CUSTOM AZIMUTHAL POLYMER-STRESS COUPLING
# =============================================================================

class PolymerAzimuthalMomentum(Equations):
    """
    Add the polymer contribution to the separately-defined axisymmetric
    azimuthal momentum equation.

    pyoomph's axisymmetric Navier-Stokes formulation stores u_r and u_z in
    the ordinary velocity vector and u_theta as velocity_phi.  The standard
    NavierStokesAzimuthalComponent supplies the Newtonian and inertial pieces.

    The polymer stress contribution is

        div(tau_p)_theta

    whose weak form, for an axisymmetric symmetric tensor, is represented by

        ∫ tau_{r theta} [d v_theta/dr - v_theta/r]
          + tau_{z theta} d v_theta/dz

    with the axisymmetric integration measure.

    This is the part that must be added explicitly because the azimuthal
    velocity has its own scalar equation in pyoomph.
    """

    def __init__(self, polymer_stress_name="polymer_stress",
                 velocity_name="velocity"):
        super().__init__()
        self.polymer_stress_name = polymer_stress_name
        self.velocity_name = velocity_name

    def define_residuals(self):
        utheta, utheta_test = var_and_test(
            self.velocity_name + "_phi"
        )

        tau = var(self.polymer_stress_name)

        tau_rtheta = tau[0, 2]
        tau_ztheta = tau[1, 2]

        r = var("coordinate_x")

        # The cylindrical gradient of an azimuthal test function contains
        #
        #     d(v_theta)/dr - v_theta/r
        #
        # in the r-theta slot.
        self.add_residual(
            weak(tau_rtheta, partial_x(utheta_test))
            - weak(tau_rtheta, utheta_test / r)
            + weak(tau_ztheta, partial_y(utheta_test))
        )


# =============================================================================
# OUTPUT/DIAGNOSTIC EQUATIONS
# =============================================================================

class RodClimbingDiagnostics(Equations):
    """
    Add useful local output quantities.

    Tensor ordering used by pyoomph's axisymmetric tensor representation is
    interpreted here as

        0 = r
        1 = z
        2 = theta.

    Hence
        N1 = sigma_theta_theta - sigma_rr
        N2 = sigma_rr - sigma_zz.

    The solved pressure is physical p.  The Joseph-style modified pressure is
        Phi = p + G*z.
    """

    def __init__(self, gravity_factor):
        super().__init__()
        self.gravity_factor = gravity_factor

    def define_residuals(self):
        u = var("velocity")
        p = var("pressure")
        polymer = var("polymer_stress")

        # Solvent contribution.
        tau_s = 2.0 * BETA_S * sym(grad(u))

        total_extra = tau_s + polymer

        z = var("coordinate_y")
        phi = p + self.gravity_factor * z

        self.add_local_function("modified_pressure", phi)
        self.add_local_function("total_extra_stress", total_extra)
        self.add_local_function("N1_total",
                                total_extra[2, 2] - total_extra[0, 0])
        self.add_local_function("N2_total",
                                total_extra[0, 0] - total_extra[1, 1])
        self.add_local_function("N1_polymer",
                                polymer[2, 2] - polymer[0, 0])
        self.add_local_function("N2_polymer",
                                polymer[0, 0] - polymer[1, 1])

        self.add_local_function("u_theta", var("velocity_phi"))
        self.add_local_function("r_coordinate", var("coordinate_x"))
        self.add_local_function("z_coordinate", var("coordinate_y"))


# =============================================================================
# MAIN PROBLEM
# =============================================================================

class RodClimbingGiesekusProblem(Problem):

    def __init__(self, depth=DEPTH):
        super().__init__()

        self.depth = float(depth)

        # A dimensionless continuation parameter.  Actual angular velocity is
        #
        #     Omega = OMEGA_TARGET * omega_factor.
        #
        # The target solution is omega_factor=1.
        self.omega_factor = self.define_global_parameter(
            omega_factor=OMEGA_INITIAL / OMEGA_TARGET
        )

        # SUPG is also a continuation parameter.
        self.supg_factor = self.define_global_parameter(
            supg_factor=SUPG_INITIAL
        )

    # -------------------------------------------------------------------------
    # Dimensionless parameters as symbolic pyoomph expressions
    # -------------------------------------------------------------------------

    def omega(self):
        return OMEGA_TARGET * self.omega_factor

    def re_expr(self):
        return RHO * ROD_RADIUS**2 * self.omega() / ETA0

    def wi_expr(self):
        return LAMBDA * self.omega()

    def ca_expr(self):
        return ETA0 * self.omega() * ROD_RADIUS / SURFACE_TENSION

    def gravity_stress_ratio_expr(self):
        # G = rho*g*a/(eta0*Omega)
        return RHO * GRAVITY * ROD_RADIUS / (ETA0 * self.omega())

    def froude_inverse_squared_expr(self):
        # 1/Fr^2 = g/(a*Omega^2)
        return GRAVITY / (ROD_RADIUS * self.omega()**2)

    def atmospheric_pressure_expr(self):
        # Pressure scale is eta0*Omega.  The user-facing atmospheric pressure
        # is dimensional; convert it to the current dimensionless pressure.
        return ATMOSPHERIC_PRESSURE / (ETA0 * self.omega())

    # -------------------------------------------------------------------------
    # Mesh and equations
    # -------------------------------------------------------------------------

    def define_problem(self):

        self.set_coordinate_system("axisymmetric")

        # The spatial scale is a.  The velocity scale is a*Omega_target.
        # The equations themselves use omega_factor explicitly, so changing
        # omega_factor changes Re, Wi_G, Ca, and gravity consistently.
        self.set_scaling(
            spatial=ROD_RADIUS,
            velocity=ROD_RADIUS * OMEGA_TARGET,
            pressure=ETA0 * OMEGA_TARGET,
        )

        # Computational domain:
        #
        #       z = free surface
        #       ----------------
        #       |              |
        # r=1   |              | r=b
        #       |              |
        #       ----------------
        #       z = -H
        #
        # In the physical problem H -> infinity.  Here H is deliberately
        # explicit so depth convergence can be demonstrated.
        mesh = RectangularQuadMesh(
            lower_left=[ROD_RADIUS, -self.depth * ROD_RADIUS],
            size=[
                (OUTER_RADIUS - ROD_RADIUS),
                self.depth * ROD_RADIUS,
            ],
            N=[NR, NZ],
        )

        self.add_mesh(mesh)

        # ---------------------------------------------------------------------
        # Gravity and flow equations
        # ---------------------------------------------------------------------

        # Navier-Stokes is written in physical pressure p:
        #
        # Re (u.grad)u = -grad(p) + div(tau) + Re/Fr^2 e_z
        #
        # Since mass_density=Re and gravity=g/(a Omega^2), their product is
        # exactly G=rho*g*a/(eta0*Omega).
        Re = self.re_expr()
        Fr_inv2 = self.froude_inverse_squared_expr()

        ns = NavierStokesEquations(
            dynamic_viscosity=BETA_S,
            mass_density=Re,
            gravity=vector(0, -Fr_inv2, 0),
            mode=VELOCITY_MODE,
            velocity_name="velocity",
            pressure_name="pressure",
            GCL=True,
        )

        # Remove the pressure nullspace by fixing one pressure degree of
        # freedom.  At sufficiently deep H the lower-left corner approaches
        # the hydrostatic far-field pressure
        #
        # p_far = p_a + rho*g*H.
        #
        # In current pressure units:
        G = self.gravity_stress_ratio_expr()
        p_far = self.atmospheric_pressure_expr() + G * self.depth

        ns = ns.with_pressure_fixation(nondim_p_value=p_far)

        eqs = ns

        # ---------------------------------------------------------------------
        # Azimuthal velocity equation
        # ---------------------------------------------------------------------

        # This supplies u_theta and its Newtonian/inertial cylindrical
        # residual.  The custom equation immediately below adds the polymer
        # r-theta and z-theta stress divergence.
        eqs += NavierStokesAzimuthalComponent()
        eqs += PolymerAzimuthalMomentum()

        # ---------------------------------------------------------------------
        # Giesekus log-conformation
        # ---------------------------------------------------------------------

        #
        #   C^triangledown =
        #      -(1/Wi_G)[(C-I)+alpha(C-I)^2]
        #
        #   tau_p = beta_p/Wi_G (C-I)
        #
        # with Psi=log(C).
        #
        # pyoomph's axisymmetric implementation retains the full 3x3
        # conformation tensor, including the r-theta and z-theta entries.
        #
        giesekus = Giesekus(alpha=GIESEKUS_ALPHA)

        visco = ViscoelasticEquations(
            model=giesekus,
            relaxation_time=self.wi_expr(),
            polymer_viscosity=BETA_P,
            formulation="log-conf",
            field_name="log_conformation",
            velocity_name="velocity",
            wind=var("velocity"),
            stabilization="SUPG" if USE_SUPG else None,
            supg_factor=self.supg_factor,
            eigen_epsilon=1.0e-8,
            use_subexpression=True,
            output_conformation=True,
        )

        eqs += visco

        # ---------------------------------------------------------------------
        # ALE mesh
        # ---------------------------------------------------------------------

        eqs += PseudoElasticMesh(
            E=MESH_ELASTIC_E,
            nu=MESH_ELASTIC_NU,
            coordsys=axisymmetric,
        )

        # ---------------------------------------------------------------------
        # Free surface
        # ---------------------------------------------------------------------

        #
        # The solved pressure is physical p, so the dimensional normal
        # traction condition is naturally
        #
        #   n.T.n = -p_a - gamma*kappa
        #
        # in pyoomph's sign convention.
        #
        # This is equivalent to the user's modified-pressure form because
        #
        #   p = Phi - rho*g*z
        #
        # and therefore
        #
        #   -Phi + sigma_nn + G*h
        #
        # appears in the nondimensional free-surface equation.
        #
        sigma_hat = 1.0 / self.ca_expr()
        p_a_hat = self.atmospheric_pressure_expr()

        free_surface = NavierStokesFreeSurface(
            surface_tension=sigma_hat,
            additional_normal_traction=p_a_hat,
            static_interface=False,
        )

        eqs += free_surface @ "top"

        # ---------------------------------------------------------------------
        # Contact angle
        # ---------------------------------------------------------------------

        # The free surface intersects the inner and outer vertical walls.
        #
        # Inner wall: liquid lies toward +r.
        # Outer wall: liquid lies toward -r.
        #
        # At beta=90 degrees both reduce to horizontal free-surface tangency.
        #
        # pyoomph's contact-angle condition is a weak capillary/contact-line
        # condition and is therefore preferable to differentiating h twice.
        inner_wall_normal = vector(1, 0, 0)
        outer_wall_normal = vector(-1, 0, 0)

        # Wall tangent is chosen along +z.  With respect to the wall tangent,
        # pyoomph's contact-angle convention gives the required equilibrium
        # angle at each end.
        wall_tangent = vector(0, 1, 0)

        eqs += NavierStokesContactAngle(
            contact_angle=CONTACT_ANGLE,
            wall_normal=inner_wall_normal,
            wall_tangent=wall_tangent,
            with_respect_to_tangent=True,
        ) @ "top/left"

        eqs += NavierStokesContactAngle(
            contact_angle=CONTACT_ANGLE,
            wall_normal=outer_wall_normal,
            wall_tangent=wall_tangent,
            with_respect_to_tangent=True,
        ) @ "top/right"

        # ---------------------------------------------------------------------
        # No-slip walls
        # ---------------------------------------------------------------------

        # Inner cylinder:
        #
        #   u_r     = 0
        #   u_z     = 0
        #   u_theta = 1
        #
        # Outer cylinder:
        #
        #   u_r = u_z = u_theta = 0
        #
        eqs += DirichletBC(
            velocity_x=0,
            velocity_y=0,
            velocity_phi=1,
            mesh_x=True,
        ) @ "left"

        eqs += DirichletBC(
            velocity_x=0,
            velocity_y=0,
            velocity_phi=0,
            mesh_x=True,
        ) @ "right"

        # Bottom truncation of the infinite-depth domain:
        #
        #   u = 0
        #   Psi = 0
        #   mesh fixed
        #
        # At sufficiently large H this represents the z -> -infinity
        # condition.
        eqs += DirichletBC(
            velocity_x=0,
            velocity_y=0,
            velocity_phi=0,
            mesh_x=True,
            mesh_y=True,
            log_conformation_xx=0,
            log_conformation_xy=0,
            log_conformation_yy=0,
            log_conformation_xz=0,
            log_conformation_yz=0,
            log_conformation_zz=0,
        ) @ "bottom"

        # ---------------------------------------------------------------------
        # Volume constraint
        # ---------------------------------------------------------------------

        #
        # The initial flat interface is z=0, so the target liquid volume is
        #
        # V0 = 2*pi * integral_1^b r*H dr
        #    = pi*H*(b^2-1)
        #
        # in units of a^3.
        #
        V0 = math.pi * self.depth * (
            OUTER_RADIUS**2 - ROD_RADIUS**2
        )

        eqs += EnforceVolumeByPressure(volume=V0)

        # ---------------------------------------------------------------------
        # Diagnostics/output
        # ---------------------------------------------------------------------

        G_expr = self.gravity_stress_ratio_expr()
        eqs += RodClimbingDiagnostics(G_expr)

        eqs += MeshFileOutput()

        # ---------------------------------------------------------------------
        # Initial conditions
        # ---------------------------------------------------------------------

        # Start from the Newtonian Couette azimuthal velocity
        #
        #   u_theta = A r + B/r
        #
        # with u_theta(1)=1 and u_theta(b)=0.
        #
        r = var("lagrangian_x")
        b = OUTER_RADIUS / ROD_RADIUS

        A = -1.0 / (b**2 - 1.0)
        B = b**2 / (b**2 - 1.0)

        utheta0 = A * r + B / r

        eqs += InitialCondition(velocity_phi=utheta0)

        # Meridional flow starts from zero.
        eqs += InitialCondition(velocity_x=0)
        eqs += InitialCondition(velocity_y=0)

        # Flat interface / undeformed mesh.
        X, Y = var(["lagrangian_x", "lagrangian_y"])
        eqs += InitialCondition(mesh_x=X)
        eqs += InitialCondition(mesh_y=Y)

        # Start the polymer at equilibrium C=I, hence Psi=0.
        eqs += InitialCondition(
            log_conformation_xx=0,
            log_conformation_xy=0,
            log_conformation_yy=0,
            log_conformation_xz=0,
            log_conformation_yz=0,
            log_conformation_zz=0,
        )

        # A useful initial pressure field:
        #
        # p = p_a - G*z + centrifugal contribution.
        #
        # The exact centrifugal integral for the Newtonian Couette initial
        # velocity is included to give Newton a substantially better starting
        # point than p=0.
        rr = X
        zz = Y

        I = (
            A**2 * rr**2 / 2.0
            + 2.0 * A * B * log(rr)
            - B**2 / (2.0 * rr**2)
        )

        Ib = (
            A**2 * b**2 / 2.0
            + 2.0 * A * B * log(b)
            - B**2 / (2.0 * b**2)
        )

        centrifugal = Re * (I - Ib)
        pressure0 = (
            self.atmospheric_pressure_expr()
            - G * zz
            + centrifugal
        )

        eqs += InitialCondition(pressure=pressure0)

        self.add_equations(eqs @ "domain")

    # -------------------------------------------------------------------------
    # Solve helpers
    # -------------------------------------------------------------------------

    def solve_at(self, omega_factor, supg_factor=None, output=True):
        self.omega_factor.value = omega_factor

        if supg_factor is not None:
            self.supg_factor.value = supg_factor

        print(
            f"\nSolving omega/Omega_target = {omega_factor:g}; "
            f"Re = {RE_TARGET * omega_factor:g}; "
            f"Wi_G = {WI_G_TARGET * omega_factor:g}"
        )

        self.solve(
            max_newton_iterations=MAX_NEWTON_ITER,
            tolerance=NEWTON_TOL,
        )

        if output:
            self.output()

    def solve_continuation(self):
        if not RUN_CONTINUATION:
            self.solve_at(1.0, supg_factor=SUPG_FINAL)
            return

        # Geometric-ish continuation is much better for viscoelastic startup
        # than one large jump.
        start = OMEGA_INITIAL / OMEGA_TARGET

        if N_OMEGA_STEPS <= 1:
            omega_values = [1.0]
        else:
            # Avoid zero because Wi_G, Ca and gravity/stress scale all contain
            # powers of Omega.
            omega_values = [
                start + (1.0 - start) * i / (N_OMEGA_STEPS - 1)
                for i in range(N_OMEGA_STEPS)
            ]

        # First solve without SUPG.  pyoomph's log-conformation documentation
        # notes that full SUPG at the rest state can worsen the initial
        # Newton Jacobian.
        self.solve_at(
            omega_values[0],
            supg_factor=SUPG_INITIAL,
            output=WRITE_EVERY_STEP,
        )

        # Once a nontrivial solution exists, ramp SUPG.
        if USE_SUPG and SUPG_FINAL > SUPG_INITIAL:
            self.solve_at(
                omega_values[0],
                supg_factor=SUPG_FINAL,
                output=WRITE_EVERY_STEP,
            )

        for omega_factor in omega_values[1:]:
            self.solve_at(
                omega_factor,
                supg_factor=SUPG_FINAL if USE_SUPG else 0.0,
                output=WRITE_EVERY_STEP,
            )

    def output_summary(self):
        print("\nFinal dimensionless groups:")
        omega_factor = self.omega_factor.value
        omega = OMEGA_TARGET * omega_factor

        re = RHO * ROD_RADIUS**2 * omega / ETA0
        wi = LAMBDA * omega
        ca = ETA0 * omega * ROD_RADIUS / SURFACE_TENSION
        gstress = RHO * GRAVITY * ROD_RADIUS / (ETA0 * omega)

        print(f"Omega/Omega_target = {omega_factor:g}")
        print(f"Re                  = {re:g}")
        print(f"Wi_G                = {wi:g}")
        print(f"Ca                  = {ca:g}")
        print(f"Bo                  = {BO:g}")
        print(f"G                   = {gstress:g}")
        print(f"eta_s/eta0          = {BETA_S:g}")
        print(f"eta_p/eta0          = {BETA_P:g}")
        print(f"alpha               = {GIESEKUS_ALPHA:g}")
        print(f"depth/a             = {self.depth / ROD_RADIUS:g}")


# =============================================================================
# DRIVER
# =============================================================================

def run_single_depth(depth):
    print("\n" + "#" * 72)
    print(f"RUNNING DEPTH = {depth:g} a")
    print("#" * 72)

    with RodClimbingGiesekusProblem(depth=depth) as problem:
        problem.solve_continuation()
        problem.output_summary()


def main():

    Path(OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)

    # pyoomph writes output relative to the current working directory.
    # Keep the executable self-contained; users can change cwd or OUTPUT
    # naming above as desired.
    if DEPTH_CONVERGENCE:
        for H in DEPTH_LIST:
            run_single_depth(H)
    else:
        run_single_depth(DEPTH)


if __name__ == "__main__":
    main()
