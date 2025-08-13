import ndsl.constants as constants
from ndsl.dsl.stencil import StencilFactory
from ndsl.dsl.typing import FloatField, FloatFieldIJ
from ndsl.namelist import Namelist
from pyshield import PhysicsConfig
from pyshield.stencils.shield_microphysics.terminal_fall import (
    prep_terminal_fall,
    update_energy_wind_heat_post_fall,
)
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py


class StartFall:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        config,
    ):
        self._idx = stencil_factory.grid_indexing
        self.config = config
        self._prep_terminal_fall = stencil_factory.from_origin_domain(
            func=prep_terminal_fall,
            externals={
                "do_sedi_w": config.do_sedi_w,
                "c1_ice": config.c1_ice,
                "c1_liq": config.c1_liq,
                "c1_vap": config.c1_vap,
                "c_air": config.c_air,
            },
            origin=self._idx.origin_compute(),
            domain=self._idx.domain_compute(),
        )

    def __call__(
        self,
        q_fall: FloatField,
        delp: FloatField,
        qvapor: FloatField,
        qliquid: FloatField,
        qrain: FloatField,
        qice: FloatField,
        qsnow: FloatField,
        qgraupel: FloatField,
        temperature: FloatField,
        dm: FloatField,
        tot_e_initial: FloatFieldIJ,
        no_fall: FloatFieldIJ,
    ):
        self._prep_terminal_fall(
            q_fall,
            delp,
            qvapor,
            qliquid,
            qrain,
            qice,
            qsnow,
            qgraupel,
            temperature,
            dm,
            tot_e_initial,
            no_fall,
        )


class EndFall:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        config,
    ):
        self._idx = stencil_factory.grid_indexing
        self.config = config
        self._update_energy_wind_heat_post_fall = stencil_factory.from_origin_domain(
            func=update_energy_wind_heat_post_fall,
            externals={
                "do_sedi_uv": config.do_sedi_uv,
                "do_sedi_w": config.do_sedi_w,
                "do_sedi_heat": config.do_sedi_heat,
                "cw": constants.C_ICE,
                "c1_ice": config.c1_ice,
                "c1_liq": config.c1_liq,
                "c1_vap": config.c1_vap,
                "c_air": config.c_air,
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
        ua: FloatField,
        va: FloatField,
        wa: FloatField,
        temperature: FloatField,
        delp: FloatField,
        delz: FloatField,
        flux: FloatField,
        dm: FloatField,
        v_terminal: FloatField,
        tmp_energy1: FloatFieldIJ,
        tmp_energy2: FloatFieldIJ,
        column_energy_change: FloatFieldIJ,
        no_fall: FloatFieldIJ,
    ):
        self._update_energy_wind_heat_post_fall(
            qvapor,
            qliquid,
            qrain,
            qice,
            qsnow,
            qgraupel,
            ua,
            va,
            wa,
            temperature,
            delp,
            delz,
            flux,
            dm,
            v_terminal,
            tmp_energy1,
            tmp_energy2,
            column_energy_change,
            no_fall,
        )


class TranslateStartFall(TranslatePhysicsFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "q_fall": {"serialname": "sf_qf", "shield": True},
            "qvapor": {"serialname": "sf_qv", "shield": True},
            "qliquid": {"serialname": "sf_ql", "shield": True},
            "qrain": {"serialname": "sf_qr", "shield": True},
            "qice": {"serialname": "sf_qi", "shield": True},
            "qsnow": {"serialname": "sf_qs", "shield": True},
            "qgraupel": {"serialname": "sf_qg", "shield": True},
            "temperature": {"serialname": "sf_pt", "shield": True},
            "delp": {"serialname": "sf_delp", "shield": True},
            "dm": {"serialname": "sf_dm", "shield": True},
            "tot_e_initial": {"serialname": "sf_e1", "shield": True},
            "no_fall": {"serialname": "sf_nf", "shield": True},
        }

        self.out_vars = {
            "dm": {"serialname": "sf_dm", "kend": namelist.npz, "shield": True},
            "tot_e_initial": {"serialname": "sf_e1", "shield": True},
            "no_fall": {"serialname": "sf_nf", "shield": True},
        }

        self.stencil_factory = stencil_factory
        pconf = PhysicsConfig.from_namelist(namelist)
        self.config = pconf.microphysics

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)

        compute_func = StartFall(
            self.stencil_factory,
            self.config,
        )

        compute_func(**inputs)

        return self.slice_output(inputs)


class TranslateEndFall(TranslatePhysicsFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "qvapor": {"serialname": "ef_qv", "shield": True},
            "qliquid": {"serialname": "ef_ql", "shield": True},
            "qrain": {"serialname": "ef_qr", "shield": True},
            "qice": {"serialname": "ef_qi", "shield": True},
            "qsnow": {"serialname": "ef_qs", "shield": True},
            "qgraupel": {"serialname": "ef_qg", "shield": True},
            "ua": {"serialname": "ef_ua", "shield": True},
            "va": {"serialname": "ef_va", "shield": True},
            "wa": {"serialname": "ef_wa", "shield": True},
            "temperature": {"serialname": "ef_pt", "shield": True},
            "delp": {"serialname": "ef_dp", "shield": True},
            "delz": {"serialname": "ef_dz", "shield": True},
            "flux": {"serialname": "ef_pfi", "shield": True},
            "dm": {"serialname": "ef_dm", "shield": True},
            "v_terminal": {"serialname": "ef_vt", "shield": True},
            "column_energy_change": {"serialname": "ef_dte", "shield": True},
            "tmp_energy1": {"serialname": "ef_ie", "shield": True},
            "tmp_energy2": {"serialname": "ef_fe", "shield": True},
            "no_fall": {"serialname": "ef_nf", "shield": True},
        }

        self.out_vars = {
            "ua": {"serialname": "ef_ua", "kend": namelist.npz, "shield": True},
            "va": {"serialname": "ef_va", "kend": namelist.npz, "shield": True},
            "wa": {"serialname": "ef_wa", "kend": namelist.npz, "shield": True},
            "temperature": {"serialname": "ef_pt", "kend": namelist.npz, "shield": True},
            "tmp_energy1": {
                "serialname": "ef_ie",
                "shield": True,
            },
            "column_energy_change": {"serialname": "ef_dte", "shield": True},
        }

        self.stencil_factory = stencil_factory
        pconf = PhysicsConfig.from_namelist(namelist)
        self.config = pconf.microphysics

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)

        compute_func = EndFall(
            self.stencil_factory,
            self.config,
        )

        compute_func(**inputs)

        return self.slice_output(inputs)
