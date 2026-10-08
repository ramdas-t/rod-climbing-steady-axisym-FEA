#!/usr/bin/env python3
"""
rod_climbing_giesekus.py

Steady, axisymmetric, finite-Reynolds-number rod climbing of a Giesekus
fluid in a finite-radius cylindrical bath.

Numerical formulation
---------------------
* 2-D meridional (r,z) mesh with an axisymmetric coordinate system.
* The azimuthal velocity u_theta is an independent unknown
  ("velocity_phi") with its own scalar momentum equation.
* The Giesekus fluid is solved in the CONFORMATION-TENSOR formulation with
  all six independent components of the symmetric 3x3 tensor
      C_rr, C_zz, C_thth, C_rz, C_rth, C_zth
  as unknowns, plus optional SUPG stabilisation.

  Why not pyoomph's built-in ViscoelasticEquations / log-conformation?
  In axisymmetric coordinates pyoomph's built-in implementation is
  swirl-free: its tensor has only the unknowns xx, yy, xy and aa (= theta
  theta).  The r-theta and z-theta components do not exist, so the
  azimuthal shear that DRIVES rod climbing never reaches the polymer and no
  normal stresses are produced.  The swirl terms therefore have to be
  written explicitly, which is what GiesekusSwirlConformation below does.
  (A log-conformation version would additionally need a 3x3 eigen-
  decomposition with analytic Jacobians, which pyoomph does not provide.)
* ALE free surface: the interface is a moving mesh boundary.
* Surface tension is imposed in weak Young-Laplace form.
* The liquid volume is enforced with a global volume Lagrange multiplier.
  The pressure level is deliberately NOT pinned at a node: the volume
  multiplier together with the free-surface level already fixes it, and a
  pinned node would force the flat-interface hydrostatic value at a point
  where the true (rotating-flow) pressure is different, which contaminates
  the solution locally around that node.
* Infinite depth is approximated by a finite depth H.  DEPTH_CONVERGENCE
  below can be used to demonstrate convergence as H is increased.
* The pressure variable solved by NavierStokesEquations is the physical
  pressure p.  The modified pressure used in the manuscript is
      Phi = p + G*z
  where
      G = rho*g*a/(eta0*Omega_target).

The dimensionless scales are
    length       : a
    velocity     : a*Omega_target
    time         : 1/Omega_target
    stress       : eta0*Omega_target

so that
    Re       = rho*a^2*Omega/eta0
    Wi_G     = lambda*Omega
    Ca       = eta0*Omega*a/gamma
    Bo       = rho*g*a^2/gamma
    G        = rho*g*a/(eta0*Omega) = Re/Fr^2

The rod speed is ramped through the factor s = Omega/Omega_target
(continuation parameter "omega_factor"), which only multiplies the rod
boundary condition u_theta(r=1) = s.  All dimensionless groups stay at their
TARGET values, so nothing is divided by a small s during the continuation.

The Giesekus model is
    tau_p = eta_p/lambda * (C-I)
    C^triangledown = -(1/lambda)[(C-I) + alpha (C-I)^2].

Run:
    python rod_climbing_giesekus.py

The code writes VTU/PVD output into OUTPUT_DIRECTORY.
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np

from pyoomph import *

from pyoomph.equations.navier_stokes import (
    NavierStokesEquations,
    NavierStokesFreeSurface,
    NavierStokesContactAngle,
    NavierStokesAzimuthalComponent,
    NavierStokesSlipLength,
)

from pyoomph.equations.ALE import PseudoElasticMesh, EnforceVolumeByPressure
from pyoomph.expressions import *
from pyoomph.expressions.coordsys import AxisymmetricCoordinateSystem
from pyoomph.meshes.simplemeshes import RectangularQuadMesh

# =============================================================================
# USER PARAMETERS
# =============================================================================
#
# All physical inputs below are dimensional numerical values in a consistent
# unit system.  The solver itself is dimensionless (see the scales above).
# The default values are deliberately modest and are intended as a starting
# point, not as a benchmark.
#

# Geometry
ROD_RADIUS = 4e-3
OUTER_RADIUS = 10*4e-3

# Rotation
OMEGA_TARGET = 5                 # rod angular velocity
OMEGA_INITIAL = 0.1               # continuation starting point
N_OMEGA_STEPS = 11

# Fluid
RHO = 997                          # density
ETA_S = 1e-3                        # solvent viscosity
ETA_P = 10                        # polymer viscosity
LAMBDA = 0.2                      # Giesekus relaxation time
GIESEKUS_ALPHA = 0.1               # 0 <= alpha <= 1/2

# External physics
GRAVITY = 9.81                     # gravitational acceleration
SURFACE_TENSION = 72e-3              # gamma
ATMOSPHERIC_PRESSURE = 0.0         # gauge atmospheric pressure

# Contact angle (liquid side, measured from the wall into the liquid)
CONTACT_ANGLE_DEG = 90.0

# Numerical depth used to approximate z -> -infinity.
# Increase this until h(r) and stresses are insensitive to DEPTH.
DEPTH = 15*ROD_RADIUS

# Meridional slip length (in units of the rod radius a) on the two vertical
# walls.  The azimuthal velocity stays no-slip; the (r,z) velocity gets
# u_r = 0 (no penetration) plus a Navier slip law for u_z.  This is needed
# because with a fully pinned contact-line velocity the contact-angle force
# has nothing to act on, the contact-line height is undetermined, and the
# Jacobian is singular.  In the steady solution the contact-line velocity is
# still zero, so the slip law does not enter the contact-angle balance;
# keep the value small so that the walls remain effectively no-slip.
WALL_SLIP_LENGTH = 1.0e-2

# Bottom boundary condition of the truncated domain:
#   "farfield" : u_z = 0, everything else natural (d/dz = 0 for u_r, u_theta).
#                This is the z -> -infinity condition (z-independent
#                Couette-type flow) and has no corner singularity.
#   "noslip"   : u = 0 on the bottom (a solid bottom plate; the rod then
#                ends flush on the plate, which is a singular corner).
BOTTOM_CONDITION = "farfield"

# Mesh
NR = 256
NZ = 256

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

# Nonlinear solver controls
NEWTON_TOL = 1.0e-8
MAX_NEWTON_ITER = 50

# Output
OUTPUT_DIRECTORY = "lev16_0.2s_relTime_logSpacedOmega_realisticVals_rod_climbing_output"

# Run controls
RUN_CONTINUATION = True
WRITE_EVERY_STEP = True

# Optional depth convergence study.  Set DEPTH_CONVERGENCE = True to loop over
# DEPTH_LIST after the main solver works.
DEPTH_CONVERGENCE = False
DEPTH_LIST = [15.0, 25.0, 40.0, 60.0]


# =============================================================================
# DERIVED DIMENSIONLESS GROUPS (all at the TARGET rotation rate)
# =============================================================================

ETA0 = ETA_S + ETA_P
if ETA0 <= 0.0:
    raise ValueError("ETA_S + ETA_P must be positive.")

BETA_S = ETA_S / ETA0
BETA_P = ETA_P / ETA0

if LAMBDA <= 0.0:
    raise ValueError("LAMBDA must be positive (use ETA_P = 0 for a Newtonian test).")

if not (0.0 <= GIESEKUS_ALPHA <= 0.5):
    raise ValueError("GIESEKUS_ALPHA should satisfy 0 <= alpha <= 1/2.")

if OUTER_RADIUS <= ROD_RADIUS:
    raise ValueError("OUTER_RADIUS must exceed ROD_RADIUS.")

if DEPTH <= 0.0:
    raise ValueError("DEPTH must be positive.")

if BOTTOM_CONDITION not in ("farfield", "noslip"):
    raise ValueError("BOTTOM_CONDITION must be 'farfield' or 'noslip'.")

CONTACT_ANGLE = CONTACT_ANGLE_DEG * math.pi / 180.0

RE_TARGET = RHO * ROD_RADIUS**2 * OMEGA_TARGET / ETA0
WI_G_TARGET = LAMBDA * OMEGA_TARGET
CA_TARGET = ETA0 * OMEGA_TARGET * ROD_RADIUS / SURFACE_TENSION
BO = RHO * GRAVITY * ROD_RADIUS**2 / SURFACE_TENSION

# G = rho*g*a/(eta0*Omega) = Re/Fr^2 ;  1/Fr^2 = g/(a*Omega^2)
G_TARGET = RHO * GRAVITY * ROD_RADIUS / (ETA0 * OMEGA_TARGET)
FR_INV2_TARGET = GRAVITY / (ROD_RADIUS * OMEGA_TARGET**2)

# Dimensionless gauge atmospheric pressure (pressure scale eta0*Omega_target)
P_ATM_HAT = ATMOSPHERIC_PRESSURE / (ETA0 * OMEGA_TARGET)

# Dimensionless geometry (length scale a)
B_OUT = OUTER_RADIUS / ROD_RADIUS

print("=" * 72)
print("Rod-climbing Giesekus FEM")
print("=" * 72)
print(f"Re(target)       = {RE_TARGET:g}")
print(f"Wi_G(target)     = {WI_G_TARGET:g}")
print(f"Ca(target)       = {CA_TARGET:g}")
print(f"Bo               = {BO:g}")
print(f"G(target)        = {G_TARGET:g}")
print(f"eta_s/eta0       = {BETA_S:g}")
print(f"eta_p/eta0       = {BETA_P:g}")
print(f"Giesekus alpha   = {GIESEKUS_ALPHA:g}")
print(f"R/a              = {B_OUT:g}")
print(f"H/a              = {DEPTH / ROD_RADIUS:g}")
print("=" * 72)


# =============================================================================
# GIESEKUS CONFORMATION TENSOR WITH AZIMUTHAL (SWIRL) COMPONENTS
# =============================================================================

class GiesekusSwirlConformation(Equations):
    """
    Conformation tensor C of a Giesekus fluid in axisymmetric coordinates
    WITH swirl, together with the polymer stress contributions to all three
    momentum equations.

    Component ordering of every 3x3 tensor is (r, z, theta) = (0, 1, 2),
    which is pyoomph's axisymmetric ordering (x = r, y = z).  Unknowns
    (suffix -> entry):
        xx -> rr    yy -> zz    aa -> theta theta
        xy -> rz    xa -> r theta    ya -> z theta

    Evolution equation (upper-convected, steady or transient):

        dC/dt + (u_m . grad) C + (u_theta/r) F(C) - L.C - C.L^T
            + (1/Wi) [ (C-I) + alpha (C-I).(C-I) ] = 0

    with
      * u_m = (u_r, u_z) the meridional velocity,
      * L_ij = d u_i / d x_j the cylindrical velocity gradient WITH swirl,
            [ du_r/dr    du_r/dz   -u_th/r ]
        L = [ du_z/dr    du_z/dz    0      ]
            [ du_th/dr   du_th/dz   u_r/r  ]
      * F(C) = R.C + C.R^T the rotation of the cylindrical basis vectors
        (d e_r/d theta = e_theta, d e_theta/d theta = -e_r) acting on the
        components of C, taken from
        AxisymmetricCoordinateSystem.azimuthal_frame_rotation_of_tensor.

    Polymer stress: tau_p = (beta_p/Wi) (C - I).  It enters
      * the (r,z) momentum equations through weak(tau_p, grad(v)), which
        contains tau_rr, tau_rz, tau_zz and the hoop stress tau_thth,
      * the theta momentum equation through
            weak(tau_rth, dv/dr - v/r) + weak(tau_zth, dv/dz).
    """

    # (row, col) in (r, z, theta) ordering -> field-name suffix
    COMPONENTS = {
        (0, 0): "xx",
        (1, 1): "yy",
        (2, 2): "aa",
        (0, 1): "xy",
        (0, 2): "xa",
        (1, 2): "ya",
    }

    def __init__(self, *, alpha, relaxation_time, polymer_viscosity,
                 supg_factor=0, velocity_name="velocity",
                 field_name="conformation", space="C2"):
        super().__init__()
        self.alpha = alpha
        self.relaxation_time = relaxation_time
        self.polymer_viscosity = polymer_viscosity
        self.supg_factor = supg_factor
        self.velocity_name = velocity_name
        self.field_name = field_name
        self.space = space

    # ------------------------------------------------------------------ helpers

    def _C(self):
        n = self.field_name
        c = lambda s: var(n + "_" + s)
        return matrix([
            [c("xx"), c("xy"), c("xa")],
            [c("xy"), c("yy"), c("ya")],
            [c("xa"), c("ya"), c("aa")],
        ])

    def _polymer_stress(self):
        prefactor = self.polymer_viscosity / self.relaxation_time
        if isinstance(prefactor, (int, float)) and prefactor == 0:
            # Newtonian limit (ETA_P = 0): 0 * matrix would collapse to a scalar.
            return matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        return prefactor * (self._C() - identity_matrix(3))

    # ------------------------------------------------------------------ fields

    def define_fields(self):
        if not isinstance(self.get_coordinate_system(),
                          AxisymmetricCoordinateSystem):
            raise RuntimeError(
                "GiesekusSwirlConformation requires axisymmetric coordinates."
            )
        for sfx in self.COMPONENTS.values():
            name = self.field_name + "_" + sfx
            self.define_scalar_field(name, self.space)
            # Rest state: C = I
            self.set_initial_condition(name, 1 if sfx in ("xx", "yy", "aa") else 0)

        self.define_field_by_substitution("polymer_stress",
                                          self._polymer_stress())

    # --------------------------------------------------------------- residuals

    def define_residuals(self):
        n = self.field_name
        u = var(self.velocity_name)                  # (u_r, u_z, 0)
        ut = var(self.velocity_name + "_phi")        # u_theta
        r = var("coordinate_x")

        C = self._C()
        I3 = identity_matrix(3)

        # Velocity gradient L_ij = du_i/dx_j with swirl.  grad(u) is
        # swirl-free in axisymmetry: [[u_r,r u_r,z 0],[u_z,r u_z,z 0],[0 0 u_r/r]].
        swirl = matrix([
            [0, 0, -ut / r],
            [0, 0, 0],
            [partial_x(ut), partial_y(ut), 0],
        ])
        L = grad(u) + swirl

        stretch = matproduct(L, C) + matproduct(C, transpose(L))
        frame_rot = AxisymmetricCoordinateSystem.azimuthal_frame_rotation_of_tensor(
            C, 0, 2
        )

        CmI = C - I3
        relax = (CmI + self.alpha * matproduct(CmI, CmI)) / self.relaxation_time

        # SUPG: transport velocity is the meridional ALE velocity.
        supg = None
        if self.supg_factor is not None:
            wind = u
            if self.get_current_code_generator()._coordinates_as_dofs:
                wind = u - mesh_velocity()
            h = var("cartesian_element_length_h")
            # tau -> h/(2|u|) where advection dominates, -> Wi where slow.
            tau = self.supg_factor / square_root(
                4 * dot(wind, wind) / h**2 + (1 / self.relaxation_time) ** 2
            )
            supg = (subexpression(tau), wind)

        for (i, j), sfx in self.COMPONENTS.items():
            fname = n + "_" + sfx
            transport = material_derivative(var(fname), u, ALE="auto")
            res = (
                transport
                + (ut / r) * frame_rot[i][j]
                - stretch[i, j]
                + relax[i, j]
            )
            self.add_residual(weak(res, testfunction(fname)))
            if supg is not None:
                tau_s, wind = supg
                self.add_residual(
                    weak(res, tau_s * dot(wind, grad(testfunction(fname))))
                )

        # ---- polymer stress in the momentum equations -----------------------
        tau_p = self._polymer_stress()

        # (r, z) momentum: pyoomph's axisymmetric grad(v) is swirl-free, so this
        # picks up tau_rr, tau_rz, tau_zz and tau_thth (hoop term v_r/r).
        self.add_residual(weak(tau_p, grad(testfunction(self.velocity_name))))

        # theta momentum: weak form of (div tau)_theta.
        vt_test = testfunction(self.velocity_name + "_phi")
        self.add_residual(
            weak(tau_p[0, 2], partial_x(vt_test))
            - weak(tau_p[0, 2], vt_test / r)
            + weak(tau_p[1, 2], partial_y(vt_test))
        )

        # ---- output ----------------------------------------------------------
        self.add_local_function("conformation_trace", trace(C))
        # (N1, N2 of the polymer and of the total extra stress are output by
        # RodClimbingDiagnostics.)


# =============================================================================
# OUTPUT/DIAGNOSTIC EQUATIONS
# =============================================================================

class RodClimbingDiagnostics(Equations):
    """
    Add useful local output quantities.

    Tensor ordering used by pyoomph's axisymmetric tensors is
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

        # Solvent contribution (diagonal entries are all that N1, N2 use).
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

    def __init__(self, depth=DEPTH, output_directory=OUTPUT_DIRECTORY):
        super().__init__()

        self.depth = float(depth)                 # dimensional depth
        self.depth_nd = self.depth / ROD_RADIUS   # in units of a

        self.set_output_directory(str(output_directory))

        # Dimensionless continuation parameter s = Omega/Omega_target.
        # It multiplies the rod velocity only; all groups stay at target.
        self.omega_factor = self.define_global_parameter(
            omega_factor=OMEGA_INITIAL / OMEGA_TARGET
        )

        # SUPG is also a continuation parameter.
        self.supg_factor = self.define_global_parameter(
            supg_factor=SUPG_INITIAL
        )

    # -------------------------------------------------------------------------
    # Mesh and equations
    # -------------------------------------------------------------------------

    def define_problem(self):

        self.set_coordinate_system("axisymmetric")

        # Everything below is written in the dimensionless variables listed in
        # the module docstring, so pyoomph's internal scales stay at 1.

        # Computational domain (lengths in units of a):
        #
        #       z = free surface (z = 0 initially)
        #       ----------------
        #       |              |
        # r=1   |              | r=b
        #       |              |
        #       ----------------
        #       z = -H
        mesh = RectangularQuadMesh(
            lower_left=[1.0, -self.depth_nd],
            size=[B_OUT - 1.0, self.depth_nd],
            N=[NR, NZ],
        )
        self.add_mesh(mesh)

        # ---------------------------------------------------------------------
        # Gravity and flow equations
        # ---------------------------------------------------------------------

        # Re (u.grad)u = -grad(p) + div(tau) - G e_z, with
        # mass_density=Re and gravity=-g/(a Omega^2) e_z so that their
        # product is exactly G = rho*g*a/(eta0*Omega).
        #
        # NOTE: no pressure pin here.  The pressure level is determined by the
        # free-surface normal stress together with the volume constraint
        # below.  Pinning one node to the flat-interface hydrostatic value is
        # wrong in the rotating flow (the true pressure there differs by an
        # O(N1) amount) and pollutes the solution near the pinned node.
        ns = NavierStokesEquations(
            dynamic_viscosity=BETA_S,
            mass_density=RE_TARGET,
            gravity=vector(0, -FR_INV2_TARGET, 0),
            mode="TH",
            velocity_name="velocity",
            pressure_name="pressure",
        )
        eqs = ns

        # ---------------------------------------------------------------------
        # Azimuthal velocity equation (Newtonian + inertial part, centrifugal
        # force in the radial equation)
        # ---------------------------------------------------------------------
        eqs += NavierStokesAzimuthalComponent()

        # ---------------------------------------------------------------------
        # Giesekus conformation tensor incl. swirl and the polymer stress in
        # the r, z and theta momentum equations
        # ---------------------------------------------------------------------
        eqs += GiesekusSwirlConformation(
            alpha=GIESEKUS_ALPHA,
            relaxation_time=WI_G_TARGET,
            polymer_viscosity=BETA_P,
            supg_factor=self.supg_factor if USE_SUPG else None,
            velocity_name="velocity",
            field_name="conformation",
        )

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
        # The solved pressure is the physical p, so the dimensionless normal
        # traction condition is
        #
        #   n.T.n = -p_a - (1/Ca) kappa
        #
        # Using p = Phi - G z, this is equivalent to the modified-pressure
        # form  -Phi + sigma_nn + G h = ...
        free_surface = NavierStokesFreeSurface(
            surface_tension=1.0 / CA_TARGET,
            additional_normal_traction=P_ATM_HAT,
            static_interface=False,
        )
        eqs += free_surface @ "top"

        # ---------------------------------------------------------------------
        # Contact angle
        # ---------------------------------------------------------------------
        #
        # pyoomph's convention: wall_normal points from the wall into the
        # liquid, wall_tangent points along the wall from the contact line
        # into the wetted part of the wall, i.e. DOWN (-z) at both walls.
        # The contact-line force acts along
        #     m = sin(theta) wall_normal + cos(theta) wall_tangent.
        #
        # Inner wall (rod, r=1): liquid lies toward +r.
        # Outer wall (r=b)     : liquid lies toward -r.
        wall_tangent = vector(0, -1, 0)

        eqs += NavierStokesContactAngle(
            contact_angle=CONTACT_ANGLE,
            wall_normal=vector(1, 0, 0),
            wall_tangent=wall_tangent,
            with_respect_to_tangent=True,
        ) @ "top/left"

        eqs += NavierStokesContactAngle(
            contact_angle=CONTACT_ANGLE,
            wall_normal=vector(-1, 0, 0),
            wall_tangent=wall_tangent,
            with_respect_to_tangent=True,
        ) @ "top/right"

        # ---------------------------------------------------------------------
        # Walls
        # ---------------------------------------------------------------------

        # Inner cylinder: u_r = 0, u_theta = s, small meridional slip for u_z.
        # The mesh may slide vertically along the wall (only mesh_x is pinned).
        eqs += DirichletBC(
            velocity_x=0,
            velocity_phi=self.omega_factor,
            mesh_x=True,
        ) @ "left"
        eqs += NavierStokesSlipLength(WALL_SLIP_LENGTH) @ "left"

        # Outer cylinder: no penetration, no azimuthal slip, small meridional slip.
        eqs += DirichletBC(
            velocity_x=0,
            velocity_phi=0,
            mesh_x=True,
        ) @ "right"
        eqs += NavierStokesSlipLength(WALL_SLIP_LENGTH) @ "right"

        # Bottom truncation of the infinite-depth domain.  The mesh is fixed.
        # No boundary condition is imposed on the conformation tensor: the
        # transport equation has no inflow here (u.n = 0), and imposing C = I
        # would contradict the sheared far-field state below the rod.
        if BOTTOM_CONDITION == "farfield":
            eqs += DirichletBC(
                velocity_y=0,
                mesh_x=True,
                mesh_y=True,
            ) @ "bottom"
        else:
            eqs += DirichletBC(
                velocity_x=0,
                velocity_y=0,
                velocity_phi=0,
                mesh_x=True,
                mesh_y=True,
            ) @ "bottom"

        # ---------------------------------------------------------------------
        # Volume constraint
        # ---------------------------------------------------------------------
        #
        # The initial flat interface is z=0, so the target liquid volume is
        #
        # V0 = 2*pi * integral_1^b r*H dr = pi*H*(b^2-1)      (units of a^3)
        V0 = math.pi * self.depth_nd * (B_OUT**2 - 1.0)
        eqs += EnforceVolumeByPressure(volume=V0)

        # ---------------------------------------------------------------------
        # Diagnostics/output
        # ---------------------------------------------------------------------
        eqs += RodClimbingDiagnostics(G_TARGET)
        eqs += MeshFileOutput()

        # ---------------------------------------------------------------------
        # Initial conditions
        # ---------------------------------------------------------------------

        s = self.omega_factor

        # Newtonian Couette azimuthal velocity u_theta = s (A r + B/r) with
        # u_theta(1)=s and u_theta(b)=0.
        r = var("lagrangian_x")
        b = B_OUT

        A = -1.0 / (b**2 - 1.0)
        B = b**2 / (b**2 - 1.0)

        eqs += InitialCondition(velocity_phi=s * (A * r + B / r))

        # Meridional flow starts from zero.
        eqs += InitialCondition(velocity_x=0)
        eqs += InitialCondition(velocity_y=0)

        # Flat interface / undeformed mesh.
        X, Y = var(["lagrangian_x", "lagrangian_y"])
        eqs += InitialCondition(mesh_x=X)
        eqs += InitialCondition(mesh_y=Y)

        # (The conformation tensor starts at C = I; see GiesekusSwirlConformation.)

        # Initial pressure: p = p_a - G*z + centrifugal contribution of the
        # Newtonian Couette flow, which gives Newton a much better starting
        # point than p = 0.
        I_r = (
            A**2 * r**2 / 2.0
            + 2.0 * A * B * log(r)
            - B**2 / (2.0 * r**2)
        )
        I_b = (
            A**2 * b**2 / 2.0
            + 2.0 * A * B * log(b)
            - B**2 / (2.0 * b**2)
        )
        centrifugal = RE_TARGET * s**2 * (I_r - I_b)
        pressure0 = P_ATM_HAT - G_TARGET * Y + centrifugal

        eqs += InitialCondition(pressure=pressure0)

        self.add_equations(eqs @ "domain")

    # -------------------------------------------------------------------------
    # Post-processing helpers
    # -------------------------------------------------------------------------

    def free_surface_profile(self):
        """Return (r, z) arrays of the free surface nodes, sorted by r."""
        mesh = self.get_mesh("domain/top")
        pts = []
        for node in mesh.nodes():
            pts.append((node.x(0), node.x(1)))
        pts.sort()
        return [p[0] for p in pts], [p[1] for p in pts]

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
            newton_solver_tolerance=NEWTON_TOL,
        )

        if output:
            self.output()

    def solve_continuation(self):
        if not RUN_CONTINUATION:
            self.solve_at(1.0, supg_factor=SUPG_FINAL if USE_SUPG else 0.0)
            return

        start = OMEGA_INITIAL / OMEGA_TARGET

        if N_OMEGA_STEPS <= 1:
            omega_values = [1.0]
        else:
            # Linear steps from the starting value to 1.
            # omega_values = [
            #     start + (1.0 - start) * i / (N_OMEGA_STEPS - 1)
            #     for i in range(N_OMEGA_STEPS)
            # ]
            omega_values = np.geomspace(start, 1, N_OMEGA_STEPS)

        # First solve without SUPG: at the rest state full SUPG can worsen
        # the initial Newton Jacobian.
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
        print(f"depth/a             = {self.depth_nd:g}")

        try:
            r, z = self.free_surface_profile()
            print(f"h(r=a)/a            = {z[0]:g}   (rod)")
            print(f"h(r=b)/a            = {z[-1]:g}   (outer wall)")
        except Exception as exc:  # diagnostics must never kill a finished run
            print(f"(free-surface profile not available: {exc})")


# =============================================================================
# DRIVER
# =============================================================================

def run_single_depth(depth):
    print("\n" + "#" * 72)
    print(f"RUNNING DEPTH = {depth:g}")
    print("#" * 72)

    outdir = Path(OUTPUT_DIRECTORY)
    if DEPTH_CONVERGENCE:
        outdir = outdir / f"depth_{depth:g}"

    with RodClimbingGiesekusProblem(depth=depth,
                                    output_directory=outdir) as problem:
        problem.solve_continuation()
        problem.output_summary()


def main():
    Path(OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)

    if DEPTH_CONVERGENCE:
        for H in DEPTH_LIST:
            run_single_depth(H)
    else:
        run_single_depth(DEPTH)


if __name__ == "__main__":
    main()
