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
from ndsl.stencils.basic_operations import copy_defn
from pyshield._config import (
    PHYSICS_PACKAGES,
    TRACER_DIM,
    FloatFieldTracer,
    PhysicsConfig,
)
from pyshield.physics_state import PhysicsState
from pyshield.stencils.get_phi_fv3 import get_phi_fv3
from pyshield.stencils.get_prs_fv3 import get_prs_fv3
from pyshield.stencils.microphysics import Microphysics
from pyshield.stencils.pbl import PBLConfig, SATMEDMFVDiffState, ScaleAwareTKEMoistEDMF


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
        qgrs_rad = max(physcons.QMIN, qvapor)
        rTv = constants.RDGAS * pt * (1.0 + constants.ZVIR * qgrs_rad)
        dm = delp[0, 0, 0]
        delp = dm * rTv / (phii[0, 0, 0] - phii[0, 0, 1])
        delp = min(delp, prsi[0, 0, 1] - 0.01 * dm)
        delp = max(delp, prsi + 0.01 * dm)

    with computation(PARALLEL), interval(-1, None):
        prsik = exp(constants.KAPPA * prsik) * pk0inv

    with computation(PARALLEL), interval(0, 1):
        prsik = pktop


def start_physics(
    dvdt: FloatField,
    dudt: FloatField,
    dtdt: FloatField,
    dqdt: FloatFieldTracer,
    dusfc: FloatFieldIJ,
    dvsfc: FloatFieldIJ,
    dtsfc: FloatFieldIJ,
    dqsfc: FloatFieldIJ,
):
    with computation(FORWARD), interval(0, 1):
        dusfc = 0.0
        dvsfc = 0.0
        dtsfc = 0.0
        dqsfc = 0.0
    with computation(FORWARD), interval(...):
        dvdt = 0.0
        dudt = 0.0
        dtdt = 0.0
        dqdt = 0.0


def pack_tracers(
    qgrs: FloatFieldTracer,
    qvapor: FloatField,
    qliquid: FloatField,
    qrain: FloatField,
    qice: FloatField,
    qsnow: FloatField,
    qgraupel: FloatField,
    qo3mr: FloatField,
    qsgs_tke: FloatField,
    qcld: FloatField,
):
    from __externals__ import (
        ntcw,
        ntgraupel,
        ntiw,
        ntke,
        ntliquid,
        nto3,
        ntrain,
        ntsnow,
        ntvap,
    )

    with computation(PARALLEL), interval(...):
        qgrs[0, 0, 0][ntvap] = qvapor
        qgrs[0, 0, 0][ntliquid] = qliquid
        qgrs[0, 0, 0][ntrain] = qrain
        qgrs[0, 0, 0][ntiw] = qice
        qgrs[0, 0, 0][ntsnow] = qsnow
        qgrs[0, 0, 0][ntgraupel] = qgraupel
        qgrs[0, 0, 0][nto3] = qo3mr
        qgrs[0, 0, 0][ntke] = qsgs_tke
        qgrs[0, 0, 0][ntcw] = qcld


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
        pre_radiation=False,
        pbl_config: PBLConfig = None,
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
        self._ntracers = namelist.ntracers
        if self._ntracers != 9:
            raise NotImplementedError(
                f"ntracers != 9 has not been implemented, got {self._ntracers}"
            )
        self.TRACER_DIM = TRACER_DIM
        self.quantity_factory = quantity_factory
        self.quantity_factory.set_extra_dim_lengths(
            **{
                self.TRACER_DIM: self._ntracers,
            }
        )

        grid_indexing = stencil_factory.grid_indexing
        self._setup_statein()
        self._ptop = grid_data.ptop
        self._pktop = (self._ptop / self._p00) ** constants.KAPPA
        self._pk0inv = (1.0 / self._p00) ** constants.KAPPA
        if self._ntracers == 9:  # TODO: implement actual tracer handling eventually
            self._ntliquid = 1
            self._ntrain = 2
            self._ntsnow = 4
            self._ntgraupel = 5
            self._nto3 = 6
            self._ntvap = 0
        self._pre_radiation = pre_radiation

        def make_quantity():
            return self.quantity_factory.zeros(
                dims=[X_DIM, Y_DIM, Z_DIM], units="unknown"
            )

        def make_quantity_2d():
            return self.quantity_factory.zeros(dims=[X_DIM, Y_DIM], units="unknown")

        self._prsik = make_quantity()
        self._dm3d = make_quantity()
        self._del_gz = make_quantity()
        self._copy_stencil = stencil_factory.from_origin_domain(
            func=copy_defn,
            origin=grid_indexing.origin_full(),
            domain=grid_indexing.domain_full(add=(0, 0, 1)),
        )
        self._start_physics = stencil_factory.from_origin_domain(
            func=start_physics,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        self._pack_tracers = stencil_factory.from_origin_domain(
            func=pack_tracers,
            externals={
                "ntiw": namelist.ntiw,
                "ntcw": namelist.ntcw,
                "ntke": namelist.ntke,
                "ntliquid": self._ntliquid,
                "ntrain": self._ntrain,
                "ntsnow": self._ntsnow,
                "ntgraupel": self._ntgraupel,
                "nto3": self._nto3,
                "ntvap": self._ntvap,
            },
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
                stencil_factory, self.quantity_factory, grid_data, namelist=namelist
            )
        else:
            self._gfs_microphysics = False

        self._dudt = make_quantity()
        self._dvdt = make_quantity()
        self._dtdt = make_quantity()
        self._dqdt = self.quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM, self.TRACER_DIM],
            units="unknown",
            dtype=Float,
        )
        self._qgrs = self.quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM, self.TRACER_DIM],
            units="unknown",
            dtype=Float,
        )

        self._dusfc = make_quantity_2d()
        self._dvsfc = make_quantity_2d()
        self._dtsfc = make_quantity_2d()
        self._dqsfc = make_quantity_2d()

        if "SATM_EDMF" in schemes:
            ndsl_log.info("SATM EDMF PBL scheme selected")
            if pbl_config is None:
                raise ValueError("Specify a PBL configuration to use SATM_EDMF scheme")
            self.pbl_state = SATMEDMFVDiffState.init_zeros()
            self._satm_edmf = True
            self._pbl = ScaleAwareTKEMoistEDMF(
                stencil_factory,
                self.quantity_factory,
                grid_data,
                pbl_config,
            )
        else:
            self._satm_edmf = False

    def _setup_statein(self):
        self._NQ = 8  # state.nq_tot - spec.namelist.dnats
        self._dnats = 1  # spec.namelist.dnats
        self._nwat = 6  # spec.namelist.nwat
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
        self._start_physics(
            self._dvdt,
            self._dudt,
            self._dtdt,
            self._dqdt,  # FloatField with extra data dimension
            self._dusfc,
            self._dvsfc,
            self._dtsfc,
            self._dqsfc,
        )
        if self._ntracers == 9:
            self._pack_tracers(
                self._qgrs,
                physics_state.qvapor,
                physics_state.qliquid,
                physics_state.qrain,
                physics_state.qice,
                physics_state.qsnow,
                physics_state.qgraupel,
                physics_state.qo3mr,
                physics_state.qsgs_tke,
                physics_state.qcld,
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
        # TODO: Once more things are merged we can update the PBL state
        # and call the PBL scheme
        if self._satm_edmf:
            self._pbl(self.pbl_state)
        #     self._pbl(
        #         physics_state.kpbl,
        #         physics_state.kinver,
        #         self._dvdt,
        #         self._dudt,
        #         self._dtdt,
        #         self._dqdt,  # FloatField with extra data dimension
        #         physics_state.hpbl,
        #         physics_state.ua,
        #         physics_state.va,
        #         physics_state.pt,
        #         self._qgrs,  # FloatField with extra data dimension
        #         physics_state.hsw,
        #         physics_state.hlw,
        #         xmu,
        #         psk,
        #         rbsoil,
        #         zorl,
        #         tsea,
        #         u10m,
        #         v10m,
        #         fm,
        #         fh,
        #         evap,
        #         heat,
        #         stress,
        #         spd1,
        #         prsi,
        #         delta,
        #         prsl,
        #         prslk,
        #         physics_state.phii,
        #         physics_state.phil,
        #         self._dusfc,
        #         self._dvsfc,
        #         self._dtsfc,
        #         self._dqsfc,
        #     )

        #     self._update_physics_state_with_tendencies(
        #         physics_state.qvapor,
        #         physics_state.qliquid,
        #         physics_state.qrain,
        #         physics_state.qice,
        #         physics_state.qsnow,
        #         physics_state.qgraupel,
        #         physics_state.qcld,
        #         physics_state.pt,
        #         physics_state.ua,
        #         physics_state.va,
        #         physics_state.microphysics.qv_dt,
        #         physics_state.microphysics.ql_dt,
        #         physics_state.microphysics.qr_dt,
        #         physics_state.microphysics.qi_dt,
        #         physics_state.microphysics.qs_dt,
        #         physics_state.microphysics.qg_dt,
        #         physics_state.microphysics.qa_dt,
        #         physics_state.microphysics.pt_dt,
        #         physics_state.microphysics.udt,
        #         physics_state.microphysics.vdt,
        #         physics_state.physics_updated_specific_humidity,
        #         physics_state.physics_updated_qliquid,
        #         physics_state.physics_updated_qrain,
        #         physics_state.physics_updated_qice,
        #         physics_state.physics_updated_qsnow,
        #         physics_state.physics_updated_qgraupel,
        #         physics_state.physics_updated_cloud_fraction,
        #         physics_state.physics_updated_pt,
        #         physics_state.physics_updated_ua,
        #         physics_state.physics_updated_va,
        #         timestep,
        #     )
        if self._gfs_microphysics:
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
