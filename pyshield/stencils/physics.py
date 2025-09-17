import ndsl.constants as constants
import pyshield.constants as physcons
from ndsl import QuantityFactory, StencilFactory, orchestrate
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.dsl.gt4py import BACKWARD, FORWARD, PARALLEL, computation, cos, exp
from ndsl.dsl.gt4py import function as gtfunction
from ndsl.dsl.gt4py import interval, log
from ndsl.dsl.typing import Float, FloatField, FloatFieldIJ
from ndsl.grid import GridData
from ndsl.logging import ndsl_log
from pyshield._config import PHYSICS_PACKAGES, PhysicsConfig
from pyshield.physics_state import PhysicsState
from pyshield.stencils.get_phi_fv3 import get_phi_fv3
from pyshield.stencils.get_prs_fv3 import get_prs_fv3
from pyshield.stencils.gfdl_cld_microphysics import (
    GFDLCloudMicrophysics,
    GFDLCloudMicrophysicsState,
    GFDLCloudMPConfig,
)
from pyshield.stencils.gfs_microphysics import GFSMicrophysics


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


def prepare_gfs_microphysics(
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


def prepare_gfdl_cld_microphysics(
    delp: FloatField,
    delz: FloatField,
    ua: FloatField,
    va: FloatField,
    wa: FloatField,
    pt: FloatField,
    qvapor: FloatField,
    qliquid: FloatField,
    qrain: FloatField,
    qice: FloatField,
    qsnow: FloatField,
    qgraupel: FloatField,
    qcld: FloatField,
    physics_delp: FloatField,
    physics_delz: FloatField,
    physics_ua: FloatField,
    physics_va: FloatField,
    physics_wa: FloatField,
    physics_pt: FloatField,
    physics_qvapor: FloatField,
    physics_qliquid: FloatField,
    physics_qrain: FloatField,
    physics_qice: FloatField,
    physics_qsnow: FloatField,
    physics_qgraupel: FloatField,
    physics_qcld: FloatField,
    water: FloatFieldIJ,
    rain: FloatFieldIJ,
    ice: FloatFieldIJ,
    snow: FloatFieldIJ,
    graupel: FloatFieldIJ,
    prefluxw: FloatFieldIJ,
    prefluxr: FloatFieldIJ,
    prefluxi: FloatFieldIJ,
    prefluxs: FloatFieldIJ,
    prefluxg: FloatFieldIJ,
    qnl1: FloatFieldIJ,
    qni1: FloatFieldIJ,
):
    with computation(FORWARD), interval(0, 1):
        water = 0.0
        rain = 0.0
        ice = 0.0
        snow = 0.0
        graupel = 0.0
        prefluxw = 0.0
        prefluxr = 0.0
        prefluxi = 0.0
        prefluxs = 0.0
        prefluxg = 0.0
        qnl1 = 0.0
        qni1 = 0.0
    with computation(PARALLEL), interval(...):
        delp = physics_delp
        delz = physics_delz
        ua = physics_ua
        va = physics_va
        wa = physics_wa
        pt = physics_pt
        qvapor = physics_qvapor
        qliquid = physics_qliquid
        qrain = physics_qrain
        qice = physics_qice
        qsnow = physics_qsnow
        qgraupel = physics_qgraupel
        qcld = physics_qcld


def post_shield_mp(
    delp: FloatField,
    delz: FloatField,
    ua: FloatField,
    va: FloatField,
    wa: FloatField,
    pt: FloatField,
    qvapor: FloatField,
    qliquid: FloatField,
    qrain: FloatField,
    qice: FloatField,
    qsnow: FloatField,
    qgraupel: FloatField,
    qcloud: FloatField,
    physics_delp: FloatField,
    physics_delz: FloatField,
    physics_ua: FloatField,
    physics_va: FloatField,
    physics_wa: FloatField,
    physics_pt: FloatField,
    physics_qvapor: FloatField,
    physics_qliquid: FloatField,
    physics_qrain: FloatField,
    physics_qice: FloatField,
    physics_qsnow: FloatField,
    physics_qgraupel: FloatField,
    physics_qcloud: FloatField,
    water: FloatFieldIJ,
    rain: FloatFieldIJ,
    ice: FloatFieldIJ,
    snow: FloatFieldIJ,
    graupel: FloatFieldIJ,
    rain1: FloatFieldIJ,
):
    from __externals__ import dtp

    with computation(FORWARD), interval(0, 1):
        tem = dtp * physcons.CON_P001 * physcons.CON_DAY
        water = water * tem
        rain = rain * tem
        ice = ice * tem
        snow = snow * tem
        graupel = graupel * tem
        rain1 = water + rain + ice + snow + graupel
    with computation(PARALLEL), interval(...):
        physics_delp = delp
        physics_delz = delz
        physics_ua = ua
        physics_va = va
        physics_wa = wa
        physics_pt = pt
        physics_qvapor = qvapor
        physics_qliquid = qliquid
        physics_qrain = qrain
        physics_qice = qice
        physics_qsnow = qsnow
        physics_qgraupel = qgraupel
        physics_qcloud = qcloud


class Physics:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        grid_data: GridData,
        namelist: PhysicsConfig,
        gfdl_cld_mp_config: GFDLCloudMPConfig = None,
        pre_radiation=False,
    ):
        schemes = [scheme.value for scheme in namelist.schemes]
        for scheme in schemes:
            if scheme not in PHYSICS_PACKAGES:  # type: ignore
                raise NotImplementedError(
                    f"{scheme} is not an implemented physics parameterization"
                )
        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
            dace_compiletime_args=["physics_state"],
        )

        grid_indexing = stencil_factory.grid_indexing
        self._setup_statein()
        self._ptop = grid_data.ptop
        self._pktop = (self._ptop / self._p00) ** constants.KAPPA
        self._pk0inv = (1.0 / self._p00) ** constants.KAPPA
        self._pre_radiation = pre_radiation
        self._dt_phys = namelist.dt_atmos

        def make_quantity():
            return quantity_factory.zeros(dims=[X_DIM, Y_DIM, Z_DIM], units="unknown")

        self._rain1 = quantity_factory.zeros(dims=[X_DIM, Y_DIM], units="unknown")
        self._prsik = make_quantity()
        self._dm3d = make_quantity()
        self._del_gz = make_quantity()
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
        if "GFS_microphysics" in schemes:
            if "GFDL_cloud_microphysics" in schemes:
                raise ValueError(
                    f"Multiple microphysics schemes: {schemes}"
                )  # TODO: We should consider a ConfigurationError exception for this
            self._microphysics = "GFS"
            ndsl_log.info("GFS microphysics selected")
            self._prepare_gfs_microphysics = stencil_factory.from_origin_domain(
                func=prepare_gfs_microphysics,
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
            self._gfs_microphysics = GFSMicrophysics(
                stencil_factory, quantity_factory, grid_data, config=namelist
            )
        elif "GFDL_cloud_microphysics" in schemes:
            ndsl_log.info("GFDL Cloud microphysics selected")
            if gfdl_cld_mp_config is None:
                raise ValueError(
                    "Specify a configuration to use the GFDL Cloud Microphysics"
                )
            self._microphysics = "GFDL_CLoud"
            self.microphyics_state = GFDLCloudMicrophysicsState.init_zeros(
                quantity_factory
            )
            self._prepare_gfdl_cld_microphysics = stencil_factory.from_origin_domain(
                func=prepare_gfdl_cld_microphysics,
                origin=grid_indexing.origin_compute(),
                domain=grid_indexing.domain_compute(),
            )
            self._gfdl_cld_microphysics = GFDLCloudMicrophysics(
                stencil_factory, quantity_factory, grid_data, gfdl_cld_mp_config
            )
            self._post_gfdl_cld_microphysics = stencil_factory.from_origin_domain(
                func=post_shield_mp,
                externals={
                    "dtp": self._dt_phys,
                },
                origin=grid_indexing.origin_compute(),
                domain=grid_indexing.domain_compute(),
            )
        else:
            ndsl_log.info("No microphysics selected")
            self._microphysics = None

    def _setup_statein(self):
        self._NQ = 8  # state.nq_tot - spec.config.dnats
        self._dnats = 1  # spec.config.dnats
        self._nwat = 6  # spec.config.nwat
        self._p00 = 1.0e5

    def __call__(self, physics_state: PhysicsState, timestep: float):

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
        self._get_prs_fv3(
            physics_state.phii,
            physics_state.prsi,
            physics_state.pt,
            physics_state.qvapor,
            physics_state.delprsi,
            self._del_gz,
        )
        # If PBL scheme is present, physics_state should be updated here
        self._get_phi_fv3(
            physics_state.pt,
            physics_state.qvapor,
            self._del_gz,
            physics_state.phii,
            physics_state.phil,
        )
        if self._microphysics:
            if self._microphysics == "GFS":
                self._prepare_gfs_microphysics(
                    physics_state.dz,
                    physics_state.phii,
                    physics_state.wmp,
                    physics_state.omga,
                    physics_state.qvapor,
                    physics_state.pt,
                    physics_state.delp,
                    physics_state.gfs_microphysics.udt,
                    physics_state.gfs_microphysics.vdt,
                    physics_state.gfs_microphysics.pt_dt,
                    physics_state.gfs_microphysics.qv_dt,
                    physics_state.gfs_microphysics.ql_dt,
                    physics_state.gfs_microphysics.qr_dt,
                    physics_state.gfs_microphysics.qi_dt,
                    physics_state.gfs_microphysics.qs_dt,
                    physics_state.gfs_microphysics.qg_dt,
                    physics_state.gfs_microphysics.qa_dt,
                )
                self._gfs_microphysics(
                    physics_state.gfs_microphysics,
                    timestep=timestep,
                )
                # Fortran uses IPD interface,
                # here we use physics_updated_<var> to denote the updated field
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
                    physics_state.gfs_microphysics.qv_dt,
                    physics_state.gfs_microphysics.ql_dt,
                    physics_state.gfs_microphysics.qr_dt,
                    physics_state.gfs_microphysics.qi_dt,
                    physics_state.gfs_microphysics.qs_dt,
                    physics_state.gfs_microphysics.qg_dt,
                    physics_state.gfs_microphysics.qa_dt,
                    physics_state.gfs_microphysics.pt_dt,
                    physics_state.gfs_microphysics.udt,
                    physics_state.gfs_microphysics.vdt,
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
            elif self._microphysics == "GFDL_CLoud":
                self._prepare_gfdl_cld_microphysics(
                    self.microphyics_state.delp,
                    self.microphyics_state.delz,
                    self.microphyics_state.ua,
                    self.microphyics_state.va,
                    self.microphyics_state.wa,
                    self.microphyics_state.pt,
                    self.microphyics_state.qvapor,
                    self.microphyics_state.qliquid,
                    self.microphyics_state.qrain,
                    self.microphyics_state.qice,
                    self.microphyics_state.qsnow,
                    self.microphyics_state.qgraupel,
                    self.microphyics_state.qcld,
                    physics_state.delp,
                    physics_state.delz,
                    physics_state.ua,
                    physics_state.va,
                    physics_state.w,
                    physics_state.pt,
                    physics_state.qvapor,
                    physics_state.qliquid,
                    physics_state.qrain,
                    physics_state.qice,
                    physics_state.qsnow,
                    physics_state.qgraupel,
                    physics_state.qcld,
                    self.microphyics_state.column_water,
                    self.microphyics_state.column_rain,
                    self.microphyics_state.column_ice,
                    self.microphyics_state.column_snow,
                    self.microphyics_state.column_graupel,
                    self.microphyics_state.preflux_water,
                    self.microphyics_state.preflux_rain,
                    self.microphyics_state.preflux_ice,
                    self.microphyics_state.preflux_snow,
                    self.microphyics_state.preflux_graupel,
                    self.microphyics_state.qcloud_cond_nuclei,
                    self.microphyics_state.qcloud_ice_nuclei,
                )
                self._gfdl_cld_microphysics(
                    self.microphyics_state,
                    last_step=True,
                )
                self._post_gfdl_cld_microphysics(
                    self.microphyics_state.delp,
                    self.microphyics_state.delz,
                    self.microphyics_state.ua,
                    self.microphyics_state.va,
                    self.microphyics_state.wa,
                    self.microphyics_state.pt,
                    self.microphyics_state.qvapor,
                    self.microphyics_state.qliquid,
                    self.microphyics_state.qrain,
                    self.microphyics_state.qice,
                    self.microphyics_state.qsnow,
                    self.microphyics_state.qgraupel,
                    self.microphyics_state.qcld,
                    physics_state.delp,
                    physics_state.delz,
                    physics_state.ua,
                    physics_state.va,
                    physics_state.w,
                    physics_state.pt,
                    physics_state.qvapor,
                    physics_state.qliquid,
                    physics_state.qrain,
                    physics_state.qice,
                    physics_state.qsnow,
                    physics_state.qgraupel,
                    physics_state.qcld,
                    self.microphyics_state.column_water,
                    self.microphyics_state.column_rain,
                    self.microphyics_state.column_ice,
                    self.microphyics_state.column_snow,
                    self.microphyics_state.column_graupel,
                    self._rain1,
                )
            else:
                raise NotImplementedError(
                    f"{self._microphysics} scheme not implemented"
                )
        else:
            ndsl_log.info("No microphysics selected, skipping...")
