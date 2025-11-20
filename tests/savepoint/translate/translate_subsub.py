import ndsl.stencils.basic_operations as basic  # noqa
import pyshield.stencils.gfdl_cld_microphysics.constants as mpcons
import pyshield.stencils.gfdl_cld_microphysics.physical_functions as physfun  # noqa
from ndsl import GridIndexing, StencilFactory
from ndsl.dsl.gt4py import FORWARD, computation, exp  # noqa
from ndsl.dsl.gt4py import function as gtfunction
from ndsl.dsl.gt4py import interval, log  # noqa
from ndsl.dsl.typing import FloatField, FloatFieldIJ
from pyshield.stencils.gfdl_cld_microphysics import GFDLCloudMPConfig
from pyshield.stencils.gfdl_cld_microphysics.subgrid_z_proc import (  # noqa
    cloud_condensation_evaporation,
    complete_freeze,
    deposit_and_sublimate_graupel,
    deposit_and_sublimate_ice,
    deposit_and_sublimate_snow,
    freeze_bigg,
    perform_instant_processes,
    wegener_bergeron_findeisen,
)
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py


@gtfunction
def perform_instant_processes_test(
    qvapor,
    qliquid,
    qrain,
    qice,
    qsnow,
    qgraupel,
    temperature,
    density,
    delp,
    te,
    rh_adj,
    dep,
    reevap,
    sub,
    qsi,
    dqidt,
    cvm,
    lcpk,
    icpk,
    tcpk,
    tcp3,
):
    """
    Instant processes (include deposition, evaporation, and sublimation)
    Fortran name is pinst
    """
    from __externals__ import c1_ice, c1_liq, c1_vap, li00, lv00, t_min, t_sub

    # Instant deposit all water vapor to cloud ice when temperature is super low
    if temperature < t_min:
        sink = basic.dim(qvapor, mpcons.QCMIN)
        dep += sink * delp

        (
            qvapor,
            qliquid,
            qrain,
            qice,
            qsnow,
            qgraupel,
            cvm,
            temperature,
            lcpk,
            icpk,
            tcpk,
            tcp3,
        ) = physfun.update_hydrometeors_and_temperatures(
            qvapor,
            qliquid,
            qrain,
            qice,
            qsnow,
            qgraupel,
            -sink,
            0.0,
            0.0,
            sink,
            0.0,
            0.0,
            te,
        )

    # Instant evaporation / sublimation of all clouds when rh < rh_adj
    qpz = qvapor + qliquid + qice
    tin = (te - lv00 * qpz + li00 * (qsnow + qgraupel)) / (
        1.0 + qpz * c1_vap + qrain * c1_liq + (qsnow + qgraupel) * c1_ice
    )

    if tin > (t_sub + 6.0):
        # qsi, dqidt = physfun.sat_spec_hum_water_ice(tin, density)
        rh = qpz / qsi

        if rh < rh_adj:
            sink = qliquid
            tmp = qice

            reevap += sink * delp
            sub += tmp * delp

            (
                qvapor,
                qliquid,
                qrain,
                qice,
                qsnow,
                qgraupel,
                cvm,
                temperature,
                lcpk,
                icpk,
                tcpk,
                tcp3,
            ) = physfun.update_hydrometeors_and_temperatures(
                qvapor,
                qliquid,
                qrain,
                qice,
                qsnow,
                qgraupel,
                sink + tmp,
                -sink,
                0.0,
                -tmp,
                0,
                0,
                te,
            )
    return (
        qvapor,
        qliquid,
        qrain,
        qice,
        qsnow,
        qgraupel,
        temperature,
        cvm,
        lcpk,
        icpk,
        tcpk,
        tcp3,
        dep,
        reevap,
        sub,
    )


def vertical_subgrid_processes(
    qvapor: FloatField,
    qliquid: FloatField,
    qrain: FloatField,
    qice: FloatField,
    qsnow: FloatField,
    qgraupel: FloatField,
    temperature: FloatField,
    density: FloatField,
    density_factor: FloatField,
    delp: FloatField,
    cloud_condensation_nuclei: FloatField,
    cloud_ice_nuclei: FloatField,
    te: FloatField,
    cvm: FloatField,
    lcpk: FloatField,
    icpk: FloatField,
    tcpk: FloatField,
    tcp3: FloatField,
    cond: FloatFieldIJ,
    dep: FloatFieldIJ,
    reevap: FloatFieldIJ,
    sub: FloatFieldIJ,
    rh_adj: FloatFieldIJ,
    qsi: FloatField,
    dqidt: FloatField,
    qsw: FloatField,
    dqwdt: FloatField,
):
    """"""
    from __externals__ import do_warm_rain_mp, do_wbf  # noqa

    with computation(FORWARD):
        with interval(-1, None):
            cond = 0
            dep = 0
            reevap = 0
            sub = 0

    with computation(FORWARD):
        with interval(...):
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

            if not do_warm_rain_mp:
                (
                    qvapor,
                    qliquid,
                    qrain,
                    qice,
                    qsnow,
                    qgraupel,
                    temperature,
                    cvm,
                    lcpk,
                    icpk,
                    tcpk,
                    tcp3,
                    dep,
                    reevap,
                    sub,
                ) = perform_instant_processes(
                    qvapor,
                    qliquid,
                    qrain,
                    qice,
                    qsnow,
                    qgraupel,
                    temperature,
                    density,
                    delp,
                    te,
                    rh_adj,
                    dep,
                    reevap,
                    sub,
                )

            (
                qvapor,
                qliquid,
                qrain,
                qice,
                qsnow,
                qgraupel,
                temperature,
                cvm,
                lcpk,
                icpk,
                tcpk,
                tcp3,
                cond,
                reevap,
            ) = cloud_condensation_evaporation(
                qvapor,
                qliquid,
                qrain,
                qice,
                qsnow,
                qgraupel,
                temperature,
                delp,
                density,
                te,
                tcp3,
                cond,
                reevap,
            )

            # if (not do_warm_rain_mp):
            #     (
            #         qvapor,
            #         qliquid,
            #         qrain,
            #         qice,
            #         qsnow,
            #         qgraupel,
            #         temperature,
            #         cvm,
            #         lcpk,
            #         icpk,
            #         tcpk,
            #         tcp3,
            #     ) = complete_freeze(
            #         qvapor,
            #         qliquid,
            #         qrain,
            #         qice,
            #         qsnow,
            #         qgraupel,
            #         temperature,
            #         cvm,
            #         te,
            #         lcpk,
            #         icpk,
            #         tcpk,
            #         tcp3,
            #     )

            #     if (do_wbf):
            #         (
            #             qvapor,
            #             qliquid,
            #             qrain,
            #             qice,
            #             qsnow,
            #             qgraupel,
            #             temperature,
            #             cvm,
            #             lcpk,
            #             icpk,
            #             tcpk,
            #             tcp3,
            #         ) = wegener_bergeron_findeisen(
            #             qvapor,
            #             qliquid,
            #             qrain,
            #             qice,
            #             qsnow,
            #             qgraupel,
            #             temperature,
            #             density,
            #             cvm,
            #             te,
            #             lcpk,
            #             icpk,
            #             tcpk,
            #             tcp3,
            #         )

            #     (
            #         qvapor,
            #         qliquid,
            #         qrain,
            #         qice,
            #         qsnow,
            #         qgraupel,
            #         cloud_condensation_nuclei,
            #         temperature,
            #         cvm,
            #         lcpk,
            #         icpk,
            #         tcpk,
            #         tcp3,
            #     ) = freeze_bigg(
            #         qvapor,
            #         qliquid,
            #         qrain,
            #         qice,
            #         qsnow,
            #         qgraupel,
            #         cloud_condensation_nuclei,
            #         temperature,
            #         density,
            #         cvm,
            #         te,
            #         lcpk,
            #         icpk,
            #         tcpk,
            #         tcp3,
            #     )

            #     (
            #         qvapor,
            #         qliquid,
            #         qrain,
            #         qice,
            #         qsnow,
            #         qgraupel,
            #         cloud_ice_nuclei,
            #         temperature,
            #         cvm,
            #         lcpk,
            #         icpk,
            #         tcpk,
            #         tcp3,
            #         dep,
            #         sub,
            #     ) = deposit_and_sublimate_ice(
            #         qvapor,
            #         qliquid,
            #         qrain,
            #         qice,
            #         qsnow,
            #         qgraupel,
            #         cloud_ice_nuclei,
            #         temperature,
            #         delp,
            #         density,
            #         cvm,
            #         te,
            #         dep,
            #         sub,
            #         lcpk,
            #         icpk,
            #         tcpk,
            #         tcp3,
            #     )

            #     (
            #         qvapor,
            #         qliquid,
            #         qrain,
            #         qice,
            #         qsnow,
            #         qgraupel,
            #         temperature,
            #         cvm,
            #         lcpk,
            #         icpk,
            #         tcpk,
            #         tcp3,
            #         dep,
            #         sub,
            #     ) = deposit_and_sublimate_snow(
            #         qvapor,
            #         qliquid,
            #         qrain,
            #         qice,
            #         qsnow,
            #         qgraupel,
            #         temperature,
            #         delp,
            #         density,
            #         density_factor,
            #         cvm,
            #         te,
            #         dep,
            #         sub,
            #         lcpk,
            #         icpk,
            #         tcpk,
            #         tcp3,
            #     )

            #     (
            #         qvapor,
            #         qliquid,
            #         qrain,
            #         qice,
            #         qsnow,
            #         qgraupel,
            #         temperature,
            #         cvm,
            #         lcpk,
            #         icpk,
            #         tcpk,
            #         tcp3,
            #         dep,
            #         sub,
            #     ) = deposit_and_sublimate_graupel(
            #         qvapor,
            #         qliquid,
            #         qrain,
            #         qice,
            #         qsnow,
            #         qgraupel,
            #         temperature,
            #         delp,
            #         density,
            #         density_factor,
            #         cvm,
            #         te,
            #         dep,
            #         sub,
            #         lcpk,
            #         icpk,
            #         tcpk,
            #         tcp3,
            #     )


class SubSubgridProcesses:
    """
    subgrid_z_proc in Fortran
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        config,
        timestep: float,
    ):
        self._idx: GridIndexing = stencil_factory.grid_indexing

        if config.do_hail:
            mu_g = config.muh
            blin_g = config.blinh
        else:
            mu_g = config.mug
            blin_g = config.bling

        if config.inflag not in [1, 2, 3, 4, 5]:
            raise ValueError(
                f"Ice Nucleation Flag must be an integer from 1 to 5"
                f"not {config.inflag}"
            )

        if config.igflag not in [1, 2, 3, 4]:
            raise ValueError(
                f"Ice Generation Flag must be an int from 1 to 4, got {config.igflag}"
            )

        self._vertical_subgrid_processes = stencil_factory.from_origin_domain(
            func=vertical_subgrid_processes,
            externals={
                "timestep": timestep,
                "c1_ice": config.c1_ice,
                "c1_liq": config.c1_liq,
                "c1_vap": config.c1_vap,
                "d1_ice": config.d1_ice,
                "d1_vap": config.d1_vap,
                "li00": config.li00,
                "li20": config.li20,
                "lv00": config.lv00,
                "t_wfr": config.t_wfr,
                "t_min": config.t_min,
                "t_sub": config.t_sub,
                "do_cond_timescale": config.do_cond_timescale,
                "do_evap_timescale": config.do_evap_timescale,
                "rh_fac_evap": config.rh_fac_evap,
                "rh_fac_cond": config.rh_fac_cond,
                "rhc_cevap": config.rhc_cevap,
                "tau_l2v": config.tau_l2v,
                "tau_v2l": config.tau_v2l,
                "use_rhc_cevap": config.use_rhc_cevap,
                "qi0_crt": config.qi0_crt,
                "tau_wbf": config.tau_wbf,
                "do_psd_water_num": config.do_psd_water_num,
                "muw": config.muw,
                "pcaw": config.pcaw,
                "pcbw": config.pcbw,
                "do_psd_ice_num": config.do_psd_ice_num,
                "igflag": config.igflag,
                "inflag": config.inflag,
                "is_fac": config.is_fac,
                "mui": config.mui,
                "pcai": config.pcai,
                "pcbi": config.pcbi,
                "prog_ccn": config.prog_ccn,
                "qi_lim": config.qi_lim,
                "blins": config.blins,
                "mus": config.mus,
                "cssub_1": config.cssub_1,
                "cssub_2": config.cssub_2,
                "cssub_3": config.cssub_3,
                "cssub_4": config.cssub_4,
                "cssub_5": config.cssub_5,
                "ss_fac": config.ss_fac,
                "cgsub_1": config.cgsub_1,
                "cgsub_2": config.cgsub_2,
                "cgsub_3": config.cgsub_3,
                "cgsub_4": config.cgsub_4,
                "cgsub_5": config.cgsub_5,
                "bling": config.bling,
                "mug": config.mug,
                "gs_fac": config.gs_fac,
                "do_warm_rain_mp": config.do_warm_rain_mp,
                "do_wbf": config.do_wbf,
                "do_mp_table_emulation": config.do_mp_table_emulation,
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
        temperature: FloatField,
        density: FloatField,
        density_factor: FloatField,
        delp: FloatField,
        cloud_condensation_nuclei: FloatField,
        cloud_ice_nuclei: FloatField,
        te: FloatField,
        cond: FloatFieldIJ,
        dep: FloatFieldIJ,
        reevap: FloatFieldIJ,
        sub: FloatFieldIJ,
        rh_adj: FloatFieldIJ,
        cvm: FloatField,
        lcpk: FloatField,
        icpk: FloatField,
        tcpk: FloatField,
        tcp3: FloatField,
        qsi: FloatField,
        dqidt: FloatField,
        qsw: FloatField,
        dqwdt: FloatField,
    ):
        """
        Temperature sentive high vertical resolution processes
        Args:
            qvapor (inout):
            qliquid (inout):
            qrain (inout):
            qice (inout):
            qsnow (inout):
            qgraupel (inout):
            temperature (inout):
            density (in):
            density_factor (in):
            delp (in):
            cloud_condensation_nuclei (inout):
            cloud_ice_nuclei (inout):
            te (in):
            cond (out):
            dep (out):
            reevap (out):
            sub (out):
            rh_adj (in):
        """

        self._vertical_subgrid_processes(
            qvapor,
            qliquid,
            qrain,
            qice,
            qsnow,
            qgraupel,
            temperature,
            density,
            density_factor,
            delp,
            cloud_condensation_nuclei,
            cloud_ice_nuclei,
            te,
            cvm,
            lcpk,
            icpk,
            tcpk,
            tcp3,
            cond,
            dep,
            reevap,
            sub,
            rh_adj,
            qsi,
            dqidt,
            qsw,
            dqwdt,
        )


class TranslateSubgridZSubs(TranslatePhysicsFortranData2Py):
    def __init__(
        self,
        grid,
        namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "qvapor": {"serialname": "szs_qv", "shield": True},
            "qliquid": {"serialname": "szs_ql", "shield": True},
            "qrain": {"serialname": "szs_qr", "shield": True},
            "qice": {"serialname": "szs_qi", "shield": True},
            "qsnow": {"serialname": "szs_qs", "shield": True},
            "qgraupel": {"serialname": "szs_qg", "shield": True},
            "temperature": {"serialname": "szs_pt", "shield": True},
            "density": {"serialname": "szs_den", "shield": True},
            "density_factor": {"serialname": "szs_denfac", "shield": True},
            "delp": {"serialname": "szs_delp", "shield": True},
            "rh_adj": {"serialname": "szs_rh_adj", "shield": True},
            "cloud_condensation_nuclei": {"serialname": "szs_ccn", "shield": True},
            "cloud_ice_nuclei": {"serialname": "szs_cin", "shield": True},
            "te": {"serialname": "szs_te", "shield": True},
            "cond": {"serialname": "szs_cond", "shield": True},
            "dep": {"serialname": "szs_dep", "shield": True},
            "reevap": {"serialname": "szs_reevap", "shield": True},
            "sub": {"serialname": "szs_sub", "shield": True},
            "cvm": {"serialname": "szs_cvm", "shield": True},
            "lcpk": {"serialname": "szs_lcpk", "shield": True},
            "icpk": {"serialname": "szs_icpk", "shield": True},
            "tcpk": {"serialname": "szs_tcpk", "shield": True},
            "tcp3": {"serialname": "szs_tcp3", "shield": True},
            "qsi": {"serialname": "szs_qsi", "shield": True},
            "dqidt": {"serialname": "szs_dqidt", "shield": True},
            "qsw": {"serialname": "szs_qsw", "shield": True},
            "dqwdt": {"serialname": "szs_dqwdt", "shield": True},
        }

        self.in_vars["parameters"] = [
            "dt",
        ]

        self.out_vars = {
            "qvapor": {"serialname": "szs_qv", "kend": namelist.npz, "shield": True},
            "qliquid": {"serialname": "szs_ql", "kend": namelist.npz, "shield": True},
            "qrain": {"serialname": "szs_qr", "kend": namelist.npz, "shield": True},
            "qice": {"serialname": "szs_qi", "kend": namelist.npz, "shield": True},
            "qsnow": {"serialname": "szs_qs", "kend": namelist.npz, "shield": True},
            "qgraupel": {"serialname": "szs_qg", "kend": namelist.npz, "shield": True},
            "temperature": {
                "serialname": "szs_pt",
                "kend": namelist.npz,
                "shield": True,
            },
            "cloud_condensation_nuclei": {
                "serialname": "szs_ccn",
                "kend": namelist.npz,
                "shield": True,
            },
            "cloud_ice_nuclei": {
                "serialname": "szs_cin",
                "kend": namelist.npz,
                "shield": True,
            },
            "cond": {"serialname": "szs_cond", "kend": namelist.npz, "shield": True},
            "dep": {"serialname": "szs_dep", "kend": namelist.npz, "shield": True},
            "reevap": {
                "serialname": "szs_reevap",
                "kend": namelist.npz,
                "shield": True,
            },
            "sub": {"serialname": "szs_sub", "kend": namelist.npz, "shield": True},
            "cvm": {"serialname": "szs_cvm", "kend": namelist.npz, "shield": True},
            "lcpk": {"serialname": "szs_lcpk", "kend": namelist.npz, "shield": True},
            "icpk": {"serialname": "szs_icpk", "kend": namelist.npz, "shield": True},
            "tcpk": {"serialname": "szs_tcpk", "kend": namelist.npz, "shield": True},
            "tcp3": {"serialname": "szs_tcp3", "kend": namelist.npz, "shield": True},
        }

        self.stencil_factory = stencil_factory
        self.grid_indexing = self.stencil_factory.grid_indexing
        self.config = GFDLCloudMPConfig.from_namelist(namelist)
        self.config.do_mp_table_emulation = True

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)

        compute_func = SubSubgridProcesses(
            self.stencil_factory,
            self.config,
            timestep=inputs.pop("dt"),
        )

        compute_func(**inputs)

        return self.slice_output(inputs)
