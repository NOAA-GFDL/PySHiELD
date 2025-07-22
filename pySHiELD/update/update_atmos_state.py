from typing import Optional

import pyFV3
from ndsl import QuantityFactory, StencilFactory, orchestrate
from ndsl.constants import X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_INTERFACE_DIM
from ndsl.dsl.gt4py import BACKWARD, FORWARD, PARALLEL, computation, interval, log, min, max, exp
from ndsl.dsl.typing import Float, FloatField, FloatFieldIJ
from ndsl.grid import DriverGridData, GridData
from ndsl.typing import Communicator
import ndsl.constants as constants
from pyFV3.stencils import fv_subgridz
from pySHiELD.update.fv_update_phys import ApplyPhysicsToDycore


# TODO: when this file is not importable from physics or pyFV3, import
#       PhysicsState and DycoreState and use them to type hint below

QMIN = 1.0e-10
P00 = 1.e5
PK0INV = (1.0 / P00)**constants.KAPPA

def fill_gfs_delp(delp: FloatField, q: FloatField, q_min: Float):
    with computation(BACKWARD):
        with interval(0, -2):
            if q[0, 0, 1] < q_min:
                q = q[0, 0, 0] + (q[0, 0, 1] - q_min) * delp[0, 0, 1] / delp[0, 0, 0]

    with computation(PARALLEL), interval(1, -1):
        if q[0, 0, 0] < q_min:
            q = q_min

    with computation(FORWARD), interval(1, -1):
        if q[0, 0, -1] < 0.0:
            q = q[0, 0, 0] + q[0, 0, -1] * (delp[0, 0, -1]) / (delp[0, 0, 0])

    with computation(FORWARD), interval(0, -1):
        if q[0, 0, 0] < 0.0:
            q = 0.0


def prepare_tendencies_and_update_tracers(
    u_dt: FloatField,
    v_dt: FloatField,
    pt_dt: FloatField,
    u_t1: FloatField,
    v_t1: FloatField,
    physics_updated_pt: FloatField,
    physics_updated_specific_humidity: FloatField,
    physics_updated_qliquid: FloatField,
    physics_updated_qrain: FloatField,
    physics_updated_qsnow: FloatField,
    physics_updated_qice: FloatField,
    physics_updated_qgraupel: FloatField,
    u_t0: FloatField,
    v_t0: FloatField,
    pt_t0: FloatField,
    qvapor_t0: FloatField,
    qliquid_t0: FloatField,
    qrain_t0: FloatField,
    qsnow_t0: FloatField,
    qice_t0: FloatField,
    qgraupel_t0: FloatField,
    prsi: FloatField,
    delp: FloatField,
    rdt: Float,
):
    """Gather tendencies and adjust dycore tracers values
    GFS total air mass = dry_mass + water_vapor (condensate excluded)
    GFS mixing ratios  = tracer_mass / (dry_mass + vapor_mass)
    FV3 total air mass = dry_mass + [water_vapor + condensate ]
    FV3 mixing ratios  = tracer_mass / (dry_mass+vapor_mass+cond_mass)
    """
    with computation(PARALLEL), interval(0, -1):
        u_dt += (u_t1 - u_t0) * rdt
        v_dt += (v_t1 - v_t0) * rdt
        pt_dt += (physics_updated_pt - pt_t0) * rdt
        dp = prsi[0, 0, 1] - prsi[0, 0, 0]
        qwat_qv = dp * physics_updated_specific_humidity
        qwat_ql = dp * physics_updated_qliquid
        qwat_qr = dp * physics_updated_qrain
        qwat_qs = dp * physics_updated_qsnow
        qwat_qi = dp * physics_updated_qice
        qwat_qg = dp * physics_updated_qgraupel
        qt = qwat_qv + qwat_ql + qwat_qr + qwat_qs + qwat_qi + qwat_qg
        q_sum = qvapor_t0 + qliquid_t0 + qrain_t0 + qsnow_t0 + qice_t0 + qgraupel_t0
        q0 = delp * (1.0 - q_sum) + qt
        delp = q0
        qvapor_t0 = qwat_qv / q0
        qliquid_t0 = qwat_ql / q0
        qrain_t0 = qwat_qr / q0
        qsnow_t0 = qwat_qs / q0
        qice_t0 = qwat_qi / q0
        qgraupel_t0 = qwat_qg / q0


def copy_dycore_to_physics(
    qvapor_in: FloatField,
    qliquid_in: FloatField,
    qrain_in: FloatField,
    qsnow_in: FloatField,
    qice_in: FloatField,
    qgraupel_in: FloatField,
    qo3mr_in: FloatField,
    qsgs_tke_in: FloatField,
    qcld_in: FloatField,
    pt_in: FloatField,
    delp_in: FloatField,
    delz_in: FloatField,
    ua_in: FloatField,
    va_in: FloatField,
    w_in: FloatField,
    omga_in: FloatField,
    qvapor_out: FloatField,
    qliquid_out: FloatField,
    qrain_out: FloatField,
    qsnow_out: FloatField,
    qice_out: FloatField,
    qgraupel_out: FloatField,
    qo3mr_out: FloatField,
    qsgs_tke_out: FloatField,
    qcld_out: FloatField,
    pt_out: FloatField,
    delp_out: FloatField,
    delz_out: FloatField,
    ua_out: FloatField,
    va_out: FloatField,
    w_out: FloatField,
    omga_out: FloatField,
):
    with computation(PARALLEL), interval(0, -1):
        qvapor_out = qvapor_in
        qliquid_out = qliquid_in
        qrain_out = qrain_in
        qsnow_out = qsnow_in
        qice_out = qice_in
        qgraupel_out = qgraupel_in
        qo3mr_out = qo3mr_in
        qsgs_tke_out = qsgs_tke_in
        qcld_out = qcld_in
        pt_out = pt_in
        delp_out = delp_in
        delz_out = delz_in
        ua_out = ua_in
        va_out = va_in
        w_out = w_in
        omga_out = omga_in

def SHiELD_atm_conversion(
    delp: FloatField,
    delz: FloatField,
    qvap: FloatField,
    qliquid: FloatField,
    qice: FloatField,
    qrain: FloatField,
    qsnow: FloatField,
    qgraupel: FloatField,
    qo3mr: FloatField,
    qtke: FloatField,
    qcld: FloatField,
    pt: FloatField,
    ua: FloatField,
    va: FloatField,
    omega: FloatField,
    ptop: FloatFieldIJ,
    prsl: FloatField,
    prsi: FloatField,
    prsik: FloatField,
    prslk: FloatField,
    phii: FloatField,
    qgrs_vap: FloatField,
    qgrs_liquid: FloatField,
    qgrs_ice: FloatField,
    qgrs_rain: FloatField,
    qgrs_snow: FloatField,
    qgrs_graupel: FloatField,
    qgrs_o3mr: FloatField,
    tgrs: FloatField,
    ugrs: FloatField,
    vgrs: FloatField,
    vvl: FloatField,
    qgrs_tke: FloatField,
    qgrs_cld: FloatField,
):
    """
    Converts fields from FV3 to SHiELD 
    Primarily this converts pressures to be used in physics calculations
    Assuming k=0 is the surface for all fields...
    """
    from __externals__ import hydrostatic, use_hydro_pressures
    with computation(PARALLEL), interval(0, -1):
        tgrs = pt
        ugrs = ua
        vgrs = va
        vvl = omega
        prsl = delp
        qgrs_vap = qvap * prsl
        qgrs_liquid = qliquid * prsl
        qgrs_ice = qice * prsl
        qgrs_rain = qrain * prsl
        qgrs_snow = qsnow * prsl
        qgrs_graupel = qgraupel * prsl
        qgrs_o3mr = qo3mr * prsl
        qgrs_tke = qtke
        qgrs_cld = qcld

        # Remove the contribution of condensates to delp (mass):
        # TODO: extend for variable tracers
        prsl = prsl - qgrs_liquid - qgrs_ice - qgrs_rain - qgrs_snow - qgrs_graupel

    with computation(BACKWARD):
        # Re-compute pressure (dry_mass + water_vapor) derived fields:
        with interval(-1, None):
            prsi = ptop
            prsk = log(ptop)

        with interval(1, -1):
            prsi = prsi[0, 0, 1] + prsl
            prsik = log(prsi)
            qgrs_vap = qgrs_vap / prsl
            qgrs_liquid = qgrs_liquid / prsl
            qgrs_ice = qgrs_ice / prsl
            qgrs_rain = qgrs_rain / prsl
            qgrs_snow = qgrs_snow / prsl
            qgrs_graupel = qgrs_graupel / prsl
            qgrs_o3mr = qgrs_o3mr / prsl

        with interval(0, 0):
            prsi = prsi[0, 0, 1] + prsl
            prsik = log(prsi)
            qgrs_vap = qgrs_vap / prsl
            qgrs_liquid = qgrs_liquid / prsl
            qgrs_ice = qgrs_ice / prsl
            qgrs_rain = qgrs_rain / prsl
            qgrs_snow = qgrs_snow / prsl
            qgrs_graupel = qgrs_graupel / prsl
            qgrs_o3mr = qgrs_o3mr / prsl
            pgr = prsi
            phii = 0.0

    with computation(FORWARD):
        with interval(1, None):
            if not (hydrostatic or use_hydro_pressures):
                phii = phii[0, 0, -1] - delz[0, 0, -1] * constants.GRAV

    # Layer mean pressure by perfect gas law:
    with computation(PARALLEL):
        with interval(0, -1):
            qgrs_rad = max(QMIN, qgrs_vap)
            rTv = constants.RDGAS * tgrs * (1.0 + constants.ZVIR * qgrs_rad)
            dm = prsl
            prsl = dm * rTv / (phii[0, 0, 1] - phii)
            # Ensure subgrid MONOTONICITY of Pressure: SJL 09/11/2016
            if not hydrostatic:
                # If violated, replaces it with hydrostatic pressure
                prsl = min(prsl, prsi - 0.01 * dm)
                prsl = max(prsl, prsi[0, 0, 1] + 0.01 * dm)
            
            # Exner function layer center:
            # large sensitivity to non-hydro runs with moist kappa
            prslk = exp(constants.KAPPA * log(prsl / P00))
            # layer center geopotential; geometric midpoint
            phil = 0.5 * (phii + phii[0, 0, 1])

            # Compute Exner function at layer "interfaces"
            prsik = exp(constants.KAPPA * prsik) * PK0INV
        with interval(-1, None):
            prsik = (ptop/P00)**constants.KAPPA



class DycoreToPhysics:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        dycore_config: pyFV3.DynamicalCoreConfig,
        do_dry_convective_adjust: bool,
        dycore_only: bool,
    ):
        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
            dace_compiletime_args=["dycore_state", "physics_state", "tendency_state"],
        )

        self._copy_dycore_to_physics = stencil_factory.from_dims_halo(
            copy_dycore_to_physics,
            compute_dims=[
                X_INTERFACE_DIM,
                Y_INTERFACE_DIM,
                Z_INTERFACE_DIM,
            ],
            compute_halos=(0, 0),
        )
        self._do_dry_convective_adjustment = do_dry_convective_adjust
        self._dycore_only = dycore_only
        if self._do_dry_convective_adjustment:
            self._fv_subgridz = fv_subgridz.DryConvectiveAdjustment(
                stencil_factory=stencil_factory,
                quantity_factory=quantity_factory,
                nwat=dycore_config.nwat,
                fv_sg_adj=dycore_config.fv_sg_adj,
                n_sponge=dycore_config.n_sponge,
                hydrostatic=dycore_config.hydrostatic,
            )

    def __call__(
        self,
        dycore_state,
        physics_state,
        tendency_state=None,
        timestep: Optional[float] = None,
    ):
        if self._do_dry_convective_adjustment:
            self._fv_subgridz(
                state=dycore_state,
                u_dt=tendency_state.u_dt,
                v_dt=tendency_state.v_dt,
                timestep=timestep,
            )
        if not self._dycore_only:
            self._copy_dycore_to_physics(
                qvapor_in=dycore_state.qvapor,
                qliquid_in=dycore_state.qliquid,
                qrain_in=dycore_state.qrain,
                qsnow_in=dycore_state.qsnow,
                qice_in=dycore_state.qice,
                qgraupel_in=dycore_state.qgraupel,
                qo3mr_in=dycore_state.qo3mr,
                qsgs_tke_in=dycore_state.qsgs_tke,
                qcld_in=dycore_state.qcld,
                pt_in=dycore_state.pt,
                delp_in=dycore_state.delp,
                delz_in=dycore_state.delz,
                ua_in=dycore_state.ua,
                va_in=dycore_state.va,
                w_in=dycore_state.w,
                omga_in=dycore_state.omga,
                qvapor_out=physics_state.qvapor,
                qliquid_out=physics_state.qliquid,
                qrain_out=physics_state.qrain,
                qsnow_out=physics_state.qsnow,
                qice_out=physics_state.qice,
                qgraupel_out=physics_state.qgraupel,
                qo3mr_out=physics_state.qo3mr,
                qsgs_tke_out=physics_state.qsgs_tke,
                qcld_out=physics_state.qcld,
                pt_out=physics_state.pt,
                delp_out=physics_state.delp,
                delz_out=physics_state.delz,
                ua_out=physics_state.ua,
                va_out=physics_state.va,
                w_out=physics_state.w,
                omga_out=physics_state.omga,
            )


class UpdateAtmosphereState:
    """Fortran name is atmosphere_state_update
    This is an API to apply tendencies and compute a consistent prognostic state.
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        grid_data: GridData,
        namelist,
        comm: Communicator,
        grid_info: DriverGridData,
        state: pyFV3.DycoreState,
        quantity_factory: QuantityFactory,
        dycore_only: bool,
        apply_tendencies: bool,
        tendency_state,
    ):
        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
            dace_compiletime_args=[
                "dycore_state",
                "phy_state",
            ],
        )

        grid_indexing = stencil_factory.grid_indexing
        self.namelist = namelist
        self._rdt = 1.0 / Float(self.namelist.dt_atmos)

        self._prepare_tendencies_and_update_tracers = (
            stencil_factory.from_origin_domain(
                prepare_tendencies_and_update_tracers,
                origin=grid_indexing.origin_compute(),
                domain=grid_indexing.domain_compute(add=(0, 0, 1)),
            )
        )

        self._fill_GFS_delp = stencil_factory.from_origin_domain(
            fill_gfs_delp,
            origin=grid_indexing.origin_full(),
            domain=grid_indexing.domain_full(add=(0, 0, 1)),
        )

        self._apply_physics_to_dycore = ApplyPhysicsToDycore(
            stencil_factory,
            quantity_factory,
            grid_data,
            self.namelist,
            comm,
            grid_info,
            state,
            tendency_state.u_dt,
            tendency_state.v_dt,
        )
        self._dycore_only = dycore_only
        # apply_tendencies when we have run physics or fv_subgridz
        # if neither of those are true, we still need to run
        # fill_GFS_delp
        self._apply_tendencies = apply_tendencies

    # [DaCe] Parsing limit: accessing a quantity withing a dataclass more than
    # one-level down in the call stack is forbidden for now due to the quantity
    # being resolved early has an array (loose of type of the object leads
    # to bad inference lower down the stack)
    def __call__(
        self,
        dycore_state,
        phy_state,
        u_dt,
        v_dt,
        pt_dt,
        dt: float,
    ):
        if self._dycore_only:
            self._fill_GFS_delp(dycore_state.delp, dycore_state.qvapor, 1.0e-9)
        else:
            self._fill_GFS_delp(
                dycore_state.delp, phy_state.physics_updated_specific_humidity, 1.0e-9
            )
            self._prepare_tendencies_and_update_tracers(
                u_dt,
                v_dt,
                pt_dt,
                phy_state.physics_updated_ua,
                phy_state.physics_updated_va,
                phy_state.physics_updated_pt,
                phy_state.physics_updated_specific_humidity,
                phy_state.physics_updated_qliquid,
                phy_state.physics_updated_qrain,
                phy_state.physics_updated_qsnow,
                phy_state.physics_updated_qice,
                phy_state.physics_updated_qgraupel,
                phy_state.ua,
                phy_state.va,
                phy_state.pt,
                dycore_state.qvapor,
                dycore_state.qliquid,
                dycore_state.qrain,
                dycore_state.qsnow,
                dycore_state.qice,
                dycore_state.qgraupel,
                phy_state.prsi,
                dycore_state.delp,
                self._rdt,
            )
        if self._apply_tendencies:
            self._apply_physics_to_dycore(
                dycore_state,
                u_dt,
                v_dt,
                pt_dt,
                dt=dt,
            )
