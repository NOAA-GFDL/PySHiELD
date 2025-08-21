import datetime

import numpy as np

import ndsl.constants as constants
import pyshield.constants as physcons
from ndsl import QuantityFactory, StencilFactory, orchestrate
from ndsl.constants import X_DIM, Y_DIM, Z_DIM, Z_INTERFACE_DIM
from ndsl.dsl.gt4py import BACKWARD, FORWARD, PARALLEL, computation, cos, exp
from ndsl.dsl.gt4py import function as gtfunction
from ndsl.dsl.gt4py import interval, log, sin
from ndsl.dsl.typing import Float, FloatField, FloatFieldIJ, Int, IntFieldK
from ndsl.grid import GridData
from ndsl.logging import ndsl_log
from pyshield._config import PHYSICS_PACKAGES, PhysicsConfig
from pyshield.physics_state import PhysicsState, SurfaceState
from pyshield.radiation.rte_rrtmgp import (
    RadiationConfig,
    RadiationState,
    RTE_RRTMGPDriver,
)
from pyshield.stencils.get_phi_fv3 import get_phi_fv3
from pyshield.stencils.get_prs_fv3 import get_prs_fv3
from pyshield.stencils.microphysics import Microphysics


def calc_sigma(ak: np.ndarray, bk: np.ndarray, k_toa: int):
    return (ak + bk * physcons.P_REF - ak[k_toa]) / (physcons.P_REF - ak[k_toa])


def flip_field_k(
    in_field: FloatField,
    out_field: FloatField,
    k_flip: IntFieldK,
):
    with computation(PARALLEL), interval(...):
        out_field = in_field[0, 0, k_flip]


def set_sst(tsea, gridlat):
    from __externals__ import tmax

    with computation(FORWARD), interval(0, 1):
        tsea = tmax * (1.0 - sin(gridlat) ** 2)


def calc_p_lay_hydro(
    p_level: FloatField,
    p_layer: FloatField,
):
    """
    stencil to calculate hydrostatic layer mean pressure
    from level (interface) pressure
    """
    with computation(PARALLEL), interval(...):
        p_layer = (p_level[0, 0, 1] - p_level) / log(p_level[0, 0, 1] / p_level)


def calc_p_lay_nonhydro(
    delp: FloatField,
    delz: FloatField,
    t_layer: FloatField,
    qvapor: FloatField,
    p_layer: FloatField,
):
    """
    stencil to calculate nonhydrostatic layer mean pressure
    Assumes delp has condensates subtracted out
    """
    with computation(PARALLEL), interval(0, -1):
        tmp = constants.RDGAS * t_layer * (1 + constants.ZVIR * qvapor)
        p_layer = delp / (constants.GRAV * delz) * tmp


def copy_to_radiation(
    prsi: FloatField,
    prsl: FloatField,
    pt: FloatField,
    tsfc: FloatFieldIJ,
    qvapor: FloatField,
    qliquid: FloatField,
    qice: FloatField,
    qo3mr: FloatField,
    qcld: FloatField,
    rad_prsi: FloatField,
    rad_prsl: FloatField,
    rad_tlyr: FloatField,
    rad_tsfc: FloatFieldIJ,
    rad_qvapor: FloatField,
    rad_qliquid: FloatField,
    rad_qice: FloatField,
    rad_qo3mr: FloatField,
    rad_qcld: FloatField,
    layer_flip: IntFieldK,
    level_flip: IntFieldK,
):
    with computation(FORWARD):
        with interval(0, 1):
            rad_tsfc = tsfc
            rad_prsi = prsi[0, 0, level_flip]
            rad_prsl = prsl[0, 0, layer_flip]
            rad_tlyr = pt[0, 0, layer_flip]
            # Convert gases to molar mixing ratio
            qv = qvapor[0, 0, layer_flip]
            rad_qvapor = (qv / (1.0 - qv)) * (physcons.MMDRY / physcons.MMVAP)
            qo3 = qo3mr[0, 0, layer_flip]
            rad_qo3mr = qo3 * physcons.MMDRY / physcons.MMO3
            rad_qliquid = qliquid[0, 0, layer_flip]
            rad_qice = qice[0, 0, layer_flip]

            rad_qcld = qcld[0, 0, layer_flip]
        with interval(1, None):
            rad_prsi = prsi[0, 0, level_flip]
            rad_prsl = prsl[0, 0, layer_flip]
            rad_tlyr = pt[0, 0, layer_flip]
            # Convert gases to molar mixing ratio
            qv = qvapor[0, 0, layer_flip]
            rad_qvapor = (qv / (1.0 - qv)) * (physcons.MMDRY / physcons.MMVAP)
            qo3 = qo3mr[0, 0, layer_flip]
            rad_qo3mr = qo3 * physcons.MMDRY / physcons.MMO3
            rad_qliquid = qliquid[0, 0, layer_flip]
            rad_qice = qice[0, 0, layer_flip]
            rad_qcld = qcld[0, 0, layer_flip]


# TODO: combine with interpolate_radiation stencil
def copy_from_radiation(
    rad_htrsw: FloatField,
    rad_htrlw: FloatField,
    rad_swflux_up: FloatField,
    rad_swflux_down: FloatField,
    rad_lwflux_up: FloatField,
    rad_lwflux_down: FloatField,
    htrsw: FloatField,
    htrlw: FloatField,
    swflux_up: FloatField,
    swflux_down: FloatField,
    lwflux_up: FloatField,
    lwflux_down: FloatField,
    layer_flip: IntFieldK,
    level_flip: IntFieldK,
):
    with computation(PARALLEL), interval(...):
        htrsw = rad_htrsw[0, 0, layer_flip]
        htrlw = rad_htrlw[0, 0, layer_flip]
        swflux_up = rad_swflux_up[0, 0, level_flip]
        swflux_down = rad_swflux_down[0, 0, level_flip]
        lwflux_up = rad_lwflux_up[0, 0, level_flip]
        lwflux_down = rad_lwflux_down[0, 0, level_flip]


def interpolate_radiation(
    sinlat: FloatFieldIJ,
    coslat: FloatFieldIJ,
    xlon: FloatFieldIJ,
    coszen: FloatFieldIJ,
    t_sea: FloatFieldIJ,
    t_surface: FloatFieldIJ,
    t_surface_longwave: FloatFieldIJ,
    sfcemis: FloatFieldIJ,
    sfcdlw: FloatFieldIJ,
    sfcnsw: FloatFieldIJ,
    sfcdsw: FloatFieldIJ,
    sfcnirbmu: FloatFieldIJ,
    sfcnirdfu: FloatFieldIJ,
    sfcvisbmu: FloatFieldIJ,
    sfcvisdfu: FloatFieldIJ,
    sfcnirbmd: FloatFieldIJ,
    sfcnirdfd: FloatFieldIJ,
    sfcvisbmd: FloatFieldIJ,
    sfcvisdfd: FloatFieldIJ,
    swh: FloatField,
    swhc: FloatField,
    hlw: FloatField,
    hlwc: FloatField,
    dtdt: FloatField,
    dtdtc: FloatField,
    adjsfcdlw: FloatFieldIJ,
    adjsfculw: FloatFieldIJ,
    adjsfcnsw: FloatFieldIJ,
    adjsfcdsw: FloatFieldIJ,
    adjnirbmu: FloatFieldIJ,
    adjnirdfu: FloatFieldIJ,
    adjvisbmu: FloatFieldIJ,
    adjvisdfu: FloatFieldIJ,
    adjnirbmd: FloatFieldIJ,
    adjnirdfd: FloatFieldIJ,
    adjvisbmd: FloatFieldIJ,
    adjvisdfd: FloatFieldIJ,
    xcosz: FloatFieldIJ,
    xmu: FloatFieldIJ,
    solhr: Float,
    slag: Float,
    sdec: Float,
    cdec: Float,
):
    """
        fits radiative fluxes and heating rates from a coarse radiation
        calc time interval into model's more frequent time steps.
        Assumes k=0 is the surface
        Fortran name is dcyc2t3
        !  ====================  defination of variables  ====================  !
    !                                                                       !
    !  inputs:                                                              !
    !     solhr        - real, forecast time in 24-hour form (hr)           !
    !     slag         - real, equation of time in radians                  !
    !     sdec, cdec   - real, sin and cos of the solar declination angle   !
    !     sinlat(im), coslat(im):                                           !
    !                  - real, sin and cos of latitude                      !
    !     xlon   (im)  - real, longitude in radians                         !
    !     coszen (im)  - real, avg of cosz over daytime sw call interval    !
    !     tsea   (im)  - real, ground surface temperature (k)               !
    !     tf     (im)  - real, surface air (layer 1) temperature (k)        !
    !     sfcemis(im)  - real, surface emissivity (fraction)                !
    !     tsflw  (im)  - real, sfc air (layer 1) temp in k saved in lw call !
    !     sfcdsw (im)  - real, total sky sfc downward sw flux ( w/m**2 )    !
    !     sfcnsw (im)  - real, total sky sfc net sw into ground (w/m**2)    !
    !     sfcdlw (im)  - real, total sky sfc downward lw flux ( w/m**2 )    !
    !     swh(ix,levs) - real, total sky sw heating rates ( k/s )           !
    !     swhc(ix,levs) - real, clear sky sw heating rates ( k/s )          !
    !     hlw(ix,levs) - real, total sky lw heating rates ( k/s )           !
    !     hlwc(ix,levs) - real, clear sky lw heating rates ( k/s )          !
    !     sfcnirbmu(im)- real, tot sky sfc nir-beam sw upward flux (w/m2)   !
    !     sfcnirdfu(im)- real, tot sky sfc nir-diff sw upward flux (w/m2)   !
    !     sfcvisbmu(im)- real, tot sky sfc uv+vis-beam sw upward flux (w/m2)!
    !     sfcvisdfu(im)- real, tot sky sfc uv+vis-diff sw upward flux (w/m2)!
    !     sfcnirbmd(im)- real, tot sky sfc nir-beam sw downward flux (w/m2) !
    !     sfcnirdfd(im)- real, tot sky sfc nir-diff sw downward flux (w/m2) !
    !     sfcvisbmd(im)- real, tot sky sfc uv+vis-beam sw dnward flux (w/m2)!
    !     sfcvisdfd(im)- real, tot sky sfc uv+vis-diff sw dnward flux (w/m2)!
    !     ix, im       - integer, horiz. dimention and num of used points   !
    !     levs         - integer, vertical layer dimension                  !
    !                                                                       !
    !  input/output:                                                        !
    !     dtdt(im,levs)- real, model time step adjusted total radiation     !
    !                          heating rates ( k/s )                        !
    !     dtdtc(im,levs)- real, model time step adjusted clear sky radiation!
    !                          heating rates ( k/s )                        !
    !                                                                       !
    !  outputs:                                                             !
    !     adjsfcdsw(im)- real, time step adjusted sfc dn sw flux (w/m**2)   !
    !     adjsfcnsw(im)- real, time step adj sfc net sw into ground (w/m**2)!
    !     adjsfcdlw(im)- real, time step adjusted sfc dn lw flux (w/m**2)   !
    !     adjsfculw(im)- real, sfc upward lw flux at current time (w/m**2)  !
    !     adjnirbmu(im)- real, t adj sfc nir-beam sw upward flux (w/m2)     !
    !     adjnirdfu(im)- real, t adj sfc nir-diff sw upward flux (w/m2)     !
    !     adjvisbmu(im)- real, t adj sfc uv+vis-beam sw upward flux (w/m2)  !
    !     adjvisdfu(im)- real, t adj sfc uv+vis-diff sw upward flux (w/m2)  !
    !     adjnirbmd(im)- real, t adj sfc nir-beam sw downward flux (w/m2)   !
    !     adjnirdfd(im)- real, t adj sfc nir-diff sw downward flux (w/m2)   !
    !     adjvisbmd(im)- real, t adj sfc uv+vis-beam sw dnward flux (w/m2)  !
    !     adjvisdfd(im)- real, t adj sfc uv+vis-diff sw dnward flux (w/m2)  !
    !     xmu   (im)   - real, time step zenith angle adjust factor for sw  !
    !     xcosz (im)   - real, cosine of zenith angle at current time step  !
    !                                                                       !
    !  ====================    end of description    =====================  !
    """
    from __externals__ import daily_mean

    with computation(FORWARD), interval(0, 1):
        cns = constants.PI * (solhr - 12.0) / 12.0 + slag

        # adjust sfc downward lw flux to account for t changes in layer 1
        tem1 = (t_surface / t_surface_longwave) ** 2
        adjsfcdlw = sfcdlw * tem1 * tem1

        # compute sfc upward lw flux from current sfc temp
        adjsfculw = sfcemis * constants.SBC * (t_sea) ** 4 + (1.0 - sfcemis) * adjsfcdlw

        # sw time-step adjustment
        ss = sinlat * sdec
        cc = coslat * cdec
        ch = cc * cos(xlon + cns)
        xcosz = ch + ss

        if daily_mean:
            # replace cosz with daily mean value
            xcosz = coszen

        if (xcosz > physcons.F_EPS) and (coszen > physcons.F_EPS):
            xmu = xcosz / coszen
        else:
            xmu = 0.0

        # adjust sfc net and downward sw fluxes for zenith angle changes
        adjsfcnsw = sfcnsw * xmu
        adjsfcdsw = sfcdsw * xmu
        adjnirbmu = sfcnirbmu * xmu
        adjnirdfu = sfcnirdfu * xmu
        adjvisbmu = sfcvisbmu * xmu
        adjvisdfu = sfcvisdfu * xmu
        adjnirbmd = sfcnirbmd * xmu
        adjnirdfd = sfcnirdfd * xmu
        adjvisbmd = sfcvisbmd * xmu
        adjvisdfd = sfcvisdfd * xmu

    with computation(FORWARD), interval(...):
        # adjust sw heating rates with zenith angle change and add with
        # lw heating to temperature tendency
        dtdt = dtdt + swh * xmu + hlw
        dtdtc = dtdtc + swhc * xmu + hlwc


def atmos_phys_driver_statein(
    prsik: FloatField,
    phii: FloatField,
    prsi: FloatField,
    delz: FloatField,
    delp: FloatField,
    qvapor: FloatField,
    qliquid: FloatField,
    qrain: FloatField,
    qice: FloatField,
    qsnow: FloatField,
    qgraupel: FloatField,
    qo3mr: FloatField,
    qsgs_tke: FloatField,
    qcld: FloatField,
    pt: FloatField,
    dm: FloatField,
):
    from __externals__ import nwat, pk0inv, pktop, ptop

    with computation(BACKWARD), interval(...):
        phii = 0.0

    with computation(BACKWARD), interval(0, -1):
        phii = phii[0, 0, 1] - delz * constants.GRAV

    with computation(PARALLEL), interval(...):
        prsik = 1.0e25
        qvapor = qvapor * delp
        qliquid = qliquid * delp
        qrain = qrain * delp
        qice = qice * delp
        qsnow = qsnow * delp
        qgraupel = qgraupel * delp
        qo3mr = qo3mr * delp
    # The following needs to execute after the above (TODO)
    with computation(PARALLEL), interval(...):
        if nwat == 6:
            delp = delp - qliquid - qrain - qice - qsnow - qgraupel

    with computation(PARALLEL), interval(0, 1):
        prsi = ptop

    with computation(FORWARD), interval(1, None):
        prsi = prsi[0, 0, -1] + delp[0, 0, -1]

    with computation(PARALLEL), interval(0, -1):
        prsik = log(prsi)
        qvapor = qvapor / delp
        qliquid = qliquid / delp
        qrain = qrain / delp
        qice = qice / delp
        qsnow = qsnow / delp
        qgraupel = qgraupel / delp
        qo3mr = qo3mr / delp
        qsgs_tke = qsgs_tke / delp

    with computation(PARALLEL), interval(-1, None):
        prsik = log(prsi)

    with computation(PARALLEL), interval(0, 1):
        prsik = log(ptop)

    with computation(PARALLEL), interval(0, -1):
        qmin = 1.0e-10  # set it here since externals cannot be 2D
        qgrs_rad = max(qmin, qvapor)
        rTv = constants.RDGAS * pt * (1.0 + constants.ZVIR * qgrs_rad)
        dm = delp[0, 0, 0]
        delp = dm * rTv / (phii[0, 0, 0] - phii[0, 0, 1])
        delp = min(delp, prsi[0, 0, 1] - 0.01 * dm)
        delp = max(delp, prsi + 0.01 * dm)

    with computation(PARALLEL), interval(-1, None):
        prsik = exp(constants.KAPPA * prsik) * pk0inv

    with computation(PARALLEL), interval(0, 1):
        prsik = pktop


def prepare_microphysics(
    dz: FloatField,
    phii: FloatField,
    wmp: FloatField,
    omga: FloatField,
    qvapor: FloatField,
    pt: FloatField,
    delp: FloatField,
    u_dt: FloatField,
    v_dt: FloatField,
    pt_dt: FloatField,
    qv_dt: FloatField,
    ql_dt: FloatField,
    qr_dt: FloatField,
    qi_dt: FloatField,
    qs_dt: FloatField,
    qg_dt: FloatField,
    qa_dt: FloatField,
):
    with computation(BACKWARD), interval(...):
        dz = (phii[0, 0, 1] - phii[0, 0, 0]) * constants.RGRAV
        wmp = (
            -omga
            * (1.0 + constants.ZVIR * qvapor)
            * pt
            / delp
            * (constants.RDGAS * constants.RGRAV)
        )
    with computation(PARALLEL), interval(...):
        u_dt = 0.0
        v_dt = 0.0
        pt_dt = 0.0
        qv_dt = 0.0
        ql_dt = 0.0
        qr_dt = 0.0
        qi_dt = 0.0
        qs_dt = 0.0
        qg_dt = 0.0
        qa_dt = 0.0


@gtfunction
def forward_euler(q_t0, q_dt, dt):
    return q_t0 + q_dt * dt


def update_physics_state_with_tendencies(
    qvapor: FloatField,
    qliquid: FloatField,
    qrain: FloatField,
    qice: FloatField,
    qsnow: FloatField,
    qgraupel: FloatField,
    qcld: FloatField,
    pt: FloatField,
    ua: FloatField,
    va: FloatField,
    qv_dt: FloatField,
    ql_dt: FloatField,
    qr_dt: FloatField,
    qi_dt: FloatField,
    qs_dt: FloatField,
    qg_dt: FloatField,
    qa_dt: FloatField,
    pt_dt: FloatField,
    udt: FloatField,
    vdt: FloatField,
    physics_updated_specific_humidity: FloatField,
    physics_updated_qliquid: FloatField,
    physics_updated_qrain: FloatField,
    physics_updated_qice: FloatField,
    physics_updated_qsnow: FloatField,
    physics_updated_qgraupel: FloatField,
    physics_updated_cloud_fraction: FloatField,
    physics_updated_pt: FloatField,
    physics_updated_ua: FloatField,
    physics_updated_va: FloatField,
    dt: Float,
):
    with computation(PARALLEL), interval(...):
        physics_updated_specific_humidity = forward_euler(qvapor, qv_dt, dt)
        physics_updated_qliquid = forward_euler(qliquid, ql_dt, dt)
        physics_updated_qrain = forward_euler(qrain, qr_dt, dt)
        physics_updated_qice = forward_euler(qice, qi_dt, dt)
        physics_updated_qsnow = forward_euler(qsnow, qs_dt, dt)
        physics_updated_qgraupel = forward_euler(qgraupel, qg_dt, dt)
        physics_updated_cloud_fraction = forward_euler(qcld, qa_dt, dt)
        physics_updated_pt = forward_euler(pt, pt_dt, dt)
        physics_updated_ua = forward_euler(ua, udt, dt)
        physics_updated_va = forward_euler(va, vdt, dt)


class Physics:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        grid_data: GridData,
        namelist: PhysicsConfig,
        rad_config: RadiationConfig,
        pre_radiation=False,
        hydro_delp=False,
    ):
        schemes = [scheme.value for scheme in namelist.schemes]
        for scheme in schemes:
            if scheme not in PHYSICS_PACKAGES:  # type: ignore
                raise NotImplementedError(
                    f"{scheme} is not an implemented physics parameterization"
                )
            ndsl_log.info(f"{scheme} enabled")
        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
            dace_compiletime_args=["physics_state"],
        )

        grid_indexing = stencil_factory.grid_indexing
        nz = grid_indexing.domain[2]
        npz = nz + 1
        self._setup_statein()
        self._ptop = grid_data.ptop
        self._pktop = (self._ptop / self._p00) ** constants.KAPPA
        self._pk0inv = (1.0 / self._p00) ** constants.KAPPA
        self._pre_radiation = pre_radiation
        self._prescribe_sst = namelist.prescribe_sst
        self._gridlon = grid_data.lon_agrid
        self._gridlat = grid_data.lat_agrid
        self._nsteps = 0
        self._nsswr = namelist.nsswr
        self._nslwr = namelist.nslwr
        self._hydro_delp = hydro_delp

        def make_quantity():
            return quantity_factory.zeros(dims=[X_DIM, Y_DIM, Z_DIM], units="unknown")

        self._prsik = make_quantity()
        self._dm3d = make_quantity()
        self._del_gz = make_quantity()

        self._level_flip = quantity_factory.zeros(
            dims=[Z_INTERFACE_DIM], units="", dtype=Int
        )
        self._layer_flip = quantity_factory.zeros(
            dims=[Z_INTERFACE_DIM], units="", dtype=Int
        )
        for k in range(npz):
            self._level_flip.data[k] = npz - 1 - 2 * k
            if k < nz:
                self._layer_flip.data[k] = nz - 1 - 2 * k

        self._flip_field_k = stencil_factory.from_origin_domain(
            func=flip_field_k,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(add=(0, 0, 1)),
        )

        if self._hydro_delp:
            self._calc_p_lay_hydro = stencil_factory.from_origin_domain(
                func=calc_p_lay_hydro,
                origin=grid_indexing.origin_compute(),
                domain=grid_indexing.domain_compute(),
            )
        if self._prescribe_sst:
            self._set_sst = stencil_factory.from_origin_domain(
                func=set_sst,
                externals={"tmax": namelist.peak_sst},
                origin=grid_indexing.origin_compute(),
                domain=grid_indexing.domain_compute(),
            )
        self._get_prs_fv3 = stencil_factory.from_origin_domain(
            func=get_prs_fv3,
            origin=grid_indexing.origin_full(),
            domain=grid_indexing.domain_full(add=(0, 0, 1)),
        )
        self._get_phi_fv3 = stencil_factory.from_origin_domain(
            func=get_phi_fv3,
            origin=grid_indexing.origin_full(),
            domain=grid_indexing.domain_full(add=(0, 0, 1)),
        )
        self._atmos_phys_driver_statein = stencil_factory.from_origin_domain(
            func=atmos_phys_driver_statein,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(add=(0, 0, 1)),
            externals={
                "nwat": self._nwat,
                "ptop": self._ptop,
                "pk0inv": self._pk0inv,
                "pktop": self._pktop,
            },
        )
        if not self._pre_radiation:
            self._interpolate_radiation = stencil_factory.from_origin_domain(
                func=interpolate_radiation,
                externals={
                    "daily_mean": namelist.daily_mean,
                },
                origin=grid_indexing.origin_compute(),
                domain=grid_indexing.domain_compute(),
            )
        if "RTE_RRTMGP" in schemes:
            self._rterrtmgp = True
            sigma = calc_sigma(grid_data.ak.data, grid_data.bk.data, 0)
            self._copy_to_radiation = stencil_factory.from_origin_domain(
                func=copy_to_radiation,
                origin=grid_indexing.origin_full(),
                domain=grid_indexing.domain_full(),
            )
            self._copy_from_radiation = stencil_factory.from_origin_domain(
                func=copy_from_radiation,
                origin=grid_indexing.origin_full(),
                domain=grid_indexing.domain_full(),
            )
            self._radiation = RTE_RRTMGPDriver(
                rad_config,
                self._gridlon,
                self._gridlat,
                sigma[::-1],
                quantity_factory,
                stencil_factory,
            )
        else:
            self._rterrtmgp = False
        if "GFS_microphysics" in schemes:
            self._gfs_microphysics = True
            self._prepare_microphysics = stencil_factory.from_origin_domain(
                func=prepare_microphysics,
                origin=grid_indexing.origin_compute(),
                domain=grid_indexing.domain_compute(),
            )
            self._update_physics_state_with_tendencies = (
                stencil_factory.from_origin_domain(
                    func=update_physics_state_with_tendencies,
                    origin=grid_indexing.origin_compute(),
                    domain=grid_indexing.domain_compute(),
                )
            )
            self._microphysics = Microphysics(
                stencil_factory, quantity_factory, grid_data, namelist=namelist
            )
        else:
            self._gfs_microphysics = False

    def _setup_statein(self):
        self._NQ = 8  # state.nq_tot - spec.namelist.dnats
        self._dnats = 1  # spec.namelist.dnats
        self._nwat = 6  # spec.namelist.nwat
        self._p00 = 1.0e5

    def __call__(
        self,
        physics_state: PhysicsState,
        radiation_state: RadiationState,
        sfc_state: SurfaceState,
        date: datetime.datetime,
        timestep: float,
    ):
        do_radiation = (self._nsteps % self._nsswr == 0) or (
            self._nsteps % self._nslwr == 0
        )
        self._atmos_phys_driver_statein(
            self._prsik,
            physics_state.phii,
            physics_state.prsi,
            physics_state.delz,
            physics_state.delp,
            physics_state.qvapor,
            physics_state.qliquid,
            physics_state.qrain,
            physics_state.qice,
            physics_state.qsnow,
            physics_state.qgraupel,
            physics_state.qo3mr,
            physics_state.qsgs_tke,
            physics_state.qcld,
            physics_state.pt,
            self._dm3d,
        )

        if self._hydro_delp:
            self._calc_p_lay_hydro(physics_state.prsi, physics_state.delp)

        self._get_prs_fv3(
            physics_state.phii,
            physics_state.prsi,
            physics_state.pt,
            physics_state.qvapor,
            physics_state.delprsi,
            self._del_gz,
        )
        if self._prescribe_sst:
            self._set_sst(physics_state.tsfc, self._gridlat)
        # If PBL scheme is present, physics_state should be updated here
        self._get_phi_fv3(
            physics_state.pt,
            physics_state.qvapor,
            self._del_gz,
            physics_state.phii,
            physics_state.phil,
        )

        # Call radiation if timestep is right
        if do_radiation and self._rterrtmgp:
            self._copy_to_radiation(
                physics_state.prsi,
                physics_state.delp,
                physics_state.pt,
                physics_state.tsfc,
                physics_state.qvapor,
                physics_state.qliquid,
                physics_state.qice,
                physics_state.qo3mr,
                physics_state.qcld,
                radiation_state.prsi,
                radiation_state.prsl,
                radiation_state.tlyr,
                radiation_state.tsfc,
                radiation_state.qvapor,
                radiation_state.qliquid,
                radiation_state.qice,
                radiation_state.qo3mr,
                radiation_state.qcld,
                self._layer_flip,
                self._level_flip,
            )
            ndsl_log.info("Entering radiation")
            self._radiation.step_radiation(radiation_state, sfc_state, date)
            self._copy_from_radiation(
                radiation_state.hrtsw,
                radiation_state.hrtlw,
                radiation_state.fswu,
                radiation_state.fswd,
                radiation_state.flwu,
                radiation_state.flwd,
                physics_state.hrtsw,
                physics_state.hrtlw,
                physics_state.fswu,
                physics_state.fswd,
                physics_state.flwu,
                physics_state.flwd,
                self._layer_flip,
                self._level_flip,
            )

        # Do physics schemes here:
        if self._gfs_microphysics:
            ndsl_log.info("calling GFS microphysics")
            self._prepare_microphysics(
                physics_state.dz,
                physics_state.phii,
                physics_state.wmp,
                physics_state.omga,
                physics_state.qvapor,
                physics_state.pt,
                physics_state.delp,
                physics_state.microphysics.udt,
                physics_state.microphysics.vdt,
                physics_state.microphysics.pt_dt,
                physics_state.microphysics.qv_dt,
                physics_state.microphysics.ql_dt,
                physics_state.microphysics.qr_dt,
                physics_state.microphysics.qi_dt,
                physics_state.microphysics.qs_dt,
                physics_state.microphysics.qg_dt,
                physics_state.microphysics.qa_dt,
            )
            self._microphysics(physics_state.microphysics, timestep=timestep)
            # Fortran uses IPD interface, here we use physics_updated_<var> to denote
            # the updated field
            self._update_physics_state_with_tendencies(
                physics_state.qvapor,
                physics_state.qliquid,
                physics_state.qrain,
                physics_state.qice,
                physics_state.qsnow,
                physics_state.qgraupel,
                physics_state.qcld,
                physics_state.pt,
                physics_state.ua,
                physics_state.va,
                physics_state.microphysics.qv_dt,
                physics_state.microphysics.ql_dt,
                physics_state.microphysics.qr_dt,
                physics_state.microphysics.qi_dt,
                physics_state.microphysics.qs_dt,
                physics_state.microphysics.qg_dt,
                physics_state.microphysics.qa_dt,
                physics_state.microphysics.pt_dt,
                physics_state.microphysics.udt,
                physics_state.microphysics.vdt,
                physics_state.physics_updated_specific_humidity,
                physics_state.physics_updated_qliquid,
                physics_state.physics_updated_qrain,
                physics_state.physics_updated_qice,
                physics_state.physics_updated_qsnow,
                physics_state.physics_updated_qgraupel,
                physics_state.physics_updated_cloud_fraction,
                physics_state.physics_updated_pt,
                physics_state.physics_updated_ua,
                physics_state.physics_updated_va,
                timestep,
            )

    def test_flip_layer(self, q_in, q_out):
        self._flip_field_k(q_in, q_out, self._layer_flip)

    def test_flip_level(self, q_in, q_out):
        self._flip_field_k(q_in, q_out, self._level_flip)
