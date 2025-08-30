import ndsl.constants as constants
import pyshield.constants as physcons
from ndsl import QuantityFactory, StencilFactory, orchestrate
from ndsl.constants import X_DIM, Y_DIM, Z_DIM, Z_INTERFACE_DIM
from ndsl.dsl.gt4py import BACKWARD, FORWARD, PARALLEL, computation, cos, exp
from ndsl.dsl.gt4py import function as gtfunction
from ndsl.dsl.gt4py import interval, log
from ndsl.dsl.typing import Float, FloatField, FloatFieldIJ, Int, IntFieldK
from ndsl.grid import GridData
from pyshield._config import PHYSICS_PACKAGES, PhysicsConfig
from pyshield.physics_state import PhysicsState
from pyshield.stencils.get_phi_fv3 import get_phi_fv3
from pyshield.stencils.get_prs_fv3 import get_prs_fv3
from pyshield.stencils.microphysics import Microphysics
from pyshield.stencils.surface import SurfaceLayer, SurfaceState


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
    pgr: FloatFieldIJ,
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
        pgr = prsi

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


def flip_fields(
    u: FloatField,
    v: FloatField,
    t: FloatField,
    q: FloatField,
    prsl: FloatField,
    prsi: FloatField,
    prsik: FloatField,
    prslk: FloatField,
    phil: FloatField,
    phii: FloatField,
    u1: FloatField,
    v1: FloatField,
    t1: FloatField,
    q1: FloatField,
    prsl1: FloatField,
    prsi1: FloatField,
    prsik1: FloatField,
    prslk1: FloatField,
    phil1: FloatField,
    phii1: FloatField,
    level_flip: IntFieldK,
    layer_flip: IntFieldK,
):
    with computation(PARALLEL), interval(...):
        u1 = u[0, 0, layer_flip]
        v1 = v[0, 0, layer_flip]
        t1 = t[0, 0, layer_flip]
        q1 = q[0, 0, layer_flip]
        prsl1 = prsl[0, 0, layer_flip]
        prsi1 = prsi[0, 0, level_flip]
        prslk1 = prslk[0, 0, layer_flip]
        prsik1 = prsik[0, 0, level_flip]
        phil1 = phil[0, 0, layer_flip]
        phii1 = phii[0, 0, level_flip]


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
        nz = grid_indexing.domain[2]
        npz = nz + 1
        self._setup_statein()
        self._ptop = grid_data.ptop
        self._pktop = (self._ptop / self._p00) ** constants.KAPPA
        self._pk0inv = (1.0 / self._p00) ** constants.KAPPA
        self._pre_radiation = pre_radiation
        self._timestep = namelist.dt_atmos

        def make_quantity():
            return quantity_factory.zeros(dims=[X_DIM, Y_DIM, Z_DIM], units="unknown")

        self._dm3d = make_quantity()
        self._del_gz = make_quantity()
        self._u1 = make_quantity()
        self._v1 = make_quantity()
        self._t1 = make_quantity()
        self._prsl1 = make_quantity()
        self._prsi1 = quantity_factory.zeros(
            dims=[X_DIM, Y_DIM, Z_INTERFACE_DIM], units="unknown"
        )
        self._prsik1 = quantity_factory.zeros(
            dims=[X_DIM, Y_DIM, Z_INTERFACE_DIM], units="unknown"
        )
        self._prslk1 = make_quantity()
        self._qvapor1 = make_quantity()
        self._phil1 = make_quantity()
        self._phii1 = quantity_factory.zeros(
            dims=[X_DIM, Y_DIM, Z_INTERFACE_DIM], units="unknown"
        )
        self._rb = quantity_factory.zeros(dims=[X_DIM, Y_DIM], units="unknown")
        self._stress = quantity_factory.zeros(dims=[X_DIM, Y_DIM], units="unknown")
        self._hflx = quantity_factory.zeros(dims=[X_DIM, Y_DIM], units="unknown")
        self._adjsfcdlw = quantity_factory.zeros(dims=[X_DIM, Y_DIM], units="unknown")
        self._adjsfcdsw = quantity_factory.zeros(dims=[X_DIM, Y_DIM], units="unknown")
        self._adjsfcnsw = quantity_factory.zeros(dims=[X_DIM, Y_DIM], units="unknown")

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
        self._flip_fields = stencil_factory.from_origin_domain(
            func=flip_fields,
            origin=grid_indexing.origin_full(),
            domain=grid_indexing.domain_full(),
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
        if "SFC_layer" in schemes:
            self._sfc_layer = True
            self._sfc = SurfaceLayer(
                stencil_factory,
                quantity_factory,
                namelist.surface,
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
        timestep: float = 0.0,
        surface_state: SurfaceState = None,
    ):
        if timestep == 0.0:
            timestep = self._timestep
        self._atmos_phys_driver_statein(
            physics_state.prsik,
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
            physics_state.pgr,
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
        self._flip_fields(
            physics_state.ua,
            physics_state.va,
            physics_state.pt,
            physics_state.qvapor,
            physics_state.delp,
            physics_state.prsi,
            physics_state.prsik,
            physics_state.prslk,
            physics_state.phil,
            physics_state.phii,
            self._u1,
            self._v1,
            self._t1,
            self._qvapor1,
            self._prsl1,
            self._prsi1,
            self._prsik1,
            self._prslk1,
            self._phil1,
            self._phii1,
            self._level_flip,
            self._layer_flip,
        )
        if self._sfc_layer:
            if not surface_state:
                raise ValueError("You must pass a surface state to run surface schemes")
            self._sfc(
                surface_state,
                self._u1,
                self._v1,
                self._t1,
                self._prsl1,
                self._prsik1,
                self._prslk1,
                self._qvapor1,
                self._phil1,
                self._rb,
                self._stress,
                physics_state.pgr,
                self._hflx,
                self._adjsfcdlw,
                self._adjsfcdsw,
                self._adjsfcnsw,
            )
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
