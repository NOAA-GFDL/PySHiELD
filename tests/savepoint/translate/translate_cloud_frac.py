from gt4py.cartesian.gtscript import __INLINED, PARALLEL, computation, interval

import pyshield.constants as physcons
import pyshield.stencils.shield_microphysics.physical_functions as physfun
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.dsl.stencil import GridIndexing, StencilFactory
from ndsl.dsl.typing import FloatField, FloatFieldIJ
from ndsl.initialization import SubtileGridSizer
from ndsl.initialization.allocator import QuantityFactory
from ndsl.namelist import Namelist
from pyshield._config import MicroPhysicsConfig, PhysicsConfig
from pyshield.stencils.shield_microphysics.cloud_fraction import (  # noqa
    CloudFraction,
    cloud_scheme_1,
    cloud_scheme_2,
    cloud_scheme_3,
    cloud_scheme_4,
)
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py


def cloud_fraction_test(
    qvapor: FloatField,
    qliquid: FloatField,
    qrain: FloatField,
    qice: FloatField,
    qsnow: FloatField,
    qgraupel: FloatField,
    qa: FloatField,
    temperature: FloatField,
    density: FloatField,
    pz: FloatField,
    te: FloatField,
    h_var: FloatFieldIJ,
    gsize: FloatFieldIJ,
    qsi: FloatField,
    dqidt: FloatField,
    qsw: FloatField,
    dqwdt: FloatField,
):

    from __externals__ import (
        c1_ice,
        c1_liq,
        c1_vap,
        cfflag,
        li00,
        lv00,
        rad_graupel,
        rad_rain,
        rad_snow,
        t_wfr,
    )

    with computation(PARALLEL), interval(...):
        (
            q_liq,
            q_solid,
            cvm,
            te,
            lcpk,
            icpk,
            tcpk,
            tcp3,
        ) = physfun.calc_heat_cap_and_latent_heat_coeff(
            qvapor, qliquid, qrain, qice, qsnow, qgraupel, temperature
        )

        # Combine water species
        ice = q_solid
        q_solid = qice
        if __INLINED(rad_snow):
            q_solid += qsnow
            if __INLINED(rad_graupel):
                q_solid += qgraupel

        liq = q_liq
        q_liq = qliquid
        if __INLINED(rad_rain):
            q_liq += qrain

        q_cond = q_liq + q_solid
        qpz = qvapor + q_cond

        # use the "liquid - frozen water temperature" (tin)
        # to compute saturated specific humidity
        ice = ice - q_solid
        liq = liq - q_liq
        tin = (te - lv00 * qpz + li00 * ice) / (
            1.0 + qpz * c1_vap + liq * c1_liq + ice * c1_ice
        )

        if tin <= t_wfr:
            qstar = qsi
            dqdt = dqidt
        elif tin >= physcons.TICE0:
            qstar = qsw
            dqdt = dqwdt
        else:
            if q_cond > physcons.QCMIN:
                rqi = q_solid / q_cond
            else:
                rqi = (physcons.TICE0 - tin) / (physcons.TICE0 - t_wfr)
            qstar = rqi * qsi + (1.0 - rqi) * qsw
            dqdt = 0.5 * (dqidt + dqwdt)

        # Cloud schemes
        rh = qpz / qstar

        if __INLINED(cfflag == 1):
            qa = cloud_scheme_1(
                qpz,
                qstar,
                q_cond,
                qa,
                pz,
                rh,
                h_var,
            )
        elif __INLINED(cfflag == 2):
            qa = cloud_scheme_2(
                qstar,
                q_cond,
                qa,
                rh,
            )
        elif __INLINED(cfflag == 3):
            qa = cloud_scheme_3(
                q_cond,
                q_liq,
                q_solid,
                qa,
                gsize,
            )
        else:  # cfflag == 4:
            qa = cloud_scheme_4(
                q_cond,
                qa,
                gsize,
            )


class CloudFractionTest:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        config: MicroPhysicsConfig,
    ):
        self._idx: GridIndexing = stencil_factory.grid_indexing

        self._te = quantity_factory.zeros(dims=[X_DIM, Y_DIM, Z_DIM], units="unknown")

        self._cloud_fraction = stencil_factory.from_origin_domain(
            func=cloud_fraction_test,
            externals={
                "c1_ice": config.c1_ice,
                "c1_liq": config.c1_liq,
                "c1_vap": config.c1_vap,
                "cfflag": config.cfflag,
                "li00": config.li00,
                "lv00": config.lv00,
                "li20": config.li20,
                "d1_vap": config.d1_vap,
                "d1_ice": config.d1_ice,
                "rad_graupel": config.rad_graupel,
                "rad_rain": config.rad_rain,
                "rad_snow": config.rad_snow,
                "t_wfr": config.t_wfr,
                "tice": physcons.TICE0,
                "cld_min": config.cld_min,
                "do_cld_adj": config.do_cld_adj,
                "f_dq_m": config.f_dq_m,
                "f_dq_p": config.f_dq_p,
                "icloud_f": config.icloud_f,
                "rh_thres": config.rh_thres,
                "xr_a": config.xr_a,
                "xr_b": config.xr_b,
                "xr_c": config.xr_c,
            },
            origin=self._idx.origin_compute(),
            domain=self._idx.domain_compute(),
        )

    def __call__(
        self,
        qvapor: FloatField,
        qliquid: FloatField,
        qrain: FloatField,
        qice: FloatField,
        qsnow: FloatField,
        qgraupel: FloatField,
        qa: FloatField,
        temperature: FloatField,
        density: FloatField,
        pz: FloatField,
        h_var: FloatFieldIJ,
        gsize: FloatFieldIJ,
        qsi: FloatField,
        dqidt: FloatField,
        qsw: FloatField,
        dqwdt: FloatField,
    ):
        """
        Calculates cloud fraction diagnostic.
        Args:
            qvapor (in):
            qliquid (in):
            qrain (in):
            qice (in):
            qsnow (in):
            qgraupel (in):
            qa (out):
            temperature (in):
            density (in):
            pz (in):
            h_var (in):
            gsize (in):
        """
        self._cloud_fraction(
            qvapor,
            qliquid,
            qrain,
            qice,
            qsnow,
            qgraupel,
            qa,
            temperature,
            density,
            pz,
            self._te,
            h_var,
            gsize,
            qsi,
            dqidt,
            qsw,
            dqwdt,
        )


class TranslateCloudFrac(TranslatePhysicsFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "qvapor": {"serialname": "cf_qv", "mp3": True},
            "qliquid": {"serialname": "cf_ql", "mp3": True},
            "qrain": {"serialname": "cf_qr", "mp3": True},
            "qice": {"serialname": "cf_qi", "mp3": True},
            "qsnow": {"serialname": "cf_qs", "mp3": True},
            "qgraupel": {"serialname": "cf_qg", "mp3": True},
            "qa": {"serialname": "cf_qa", "mp3": True},
            "temperature": {"serialname": "cf_pt", "mp3": True},
            "density": {"serialname": "cf_den", "mp3": True},
            "pz": {"serialname": "cf_pz", "mp3": True},
            "h_var": {"serialname": "cf_h_var", "mp3": True},
            "gsize": {"serialname": "cf_gsize", "mp3": True},
            "qsi": {"serialname": "cf_qsi", "mp3": True},
            "dqidt": {"serialname": "cf_dqidt", "mp3": True},
            "qsw": {"serialname": "cf_qsw", "mp3": True},
            "dqwdt": {"serialname": "cf_dqwdt", "mp3": True},
        }

        self.out_vars = {
            "qa": {"serialname": "cf_qa", "kend": namelist.npz, "mp3": True},
        }

        self.stencil_factory = stencil_factory
        self.grid_indexing = self.stencil_factory.grid_indexing
        pconf = PhysicsConfig.from_namelist(namelist)
        self.config = pconf.microphysics
        self.config.do_mp_table_emulation = True

        sizer = SubtileGridSizer.from_tile_params(
            nx_tile=self.namelist.npx - 1,
            ny_tile=self.namelist.npy - 1,
            nz=self.namelist.npz,
            n_halo=3,
            extra_dim_lengths={},
            layout=self.namelist.layout,
        )

        self.quantity_factory = QuantityFactory.from_backend(
            sizer, self.stencil_factory.backend
        )

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)

        # inputs.pop("qsi")
        # inputs.pop("dqidt")
        # inputs.pop("qsw")
        # inputs.pop("dqwdt")

        compute_func = CloudFractionTest(
            self.stencil_factory,
            self.quantity_factory,
            self.config,
        )

        compute_func(**inputs)

        return self.slice_output(inputs)
