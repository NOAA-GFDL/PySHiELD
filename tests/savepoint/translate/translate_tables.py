import pyshield.stencils.gfdl_cld_microphysics.constants as mpcons
import pyshield.stencils.gfdl_cld_microphysics.physical_functions as physfun
from ndsl.dsl.gt4py import FORWARD, computation, interval, max, min
from ndsl.dsl.stencil import StencilFactory
from ndsl.dsl.typing import FloatField, IntField
from ndsl.namelist import Namelist
from pyshield.stencils.gfdl_cld_microphysics import GFDLCloudMPConfig
from pyshield.stencils.gfdl_cld_microphysics.humidity_tables import (
    HumiditySaturationTables,
)
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py


def init_tables(
    temp: FloatField,
    table0: FloatField,
    table2: FloatField,
):
    with computation(FORWARD), interval(...):
        it = physfun.temperature_index(temp)
        table0 = physfun.table0(it)
        table2 = physfun.table2(it)


def calc_table_values(
    temp: FloatField,
    den: FloatField,
    iqs: FloatField,
    wqs: FloatField,
    didt: FloatField,
    dwdt: FloatField,
    ap1: FloatField,
    it1: IntField,
    it2: IntField,
):
    from __externals__ import do_mp_table_emulation

    with computation(FORWARD), interval(...):
        if do_mp_table_emulation:
            wqs, dwdt = physfun.wqs(temp, den)
            iqs, didt = physfun.iqs(temp, den)
        else:
            wqs, dwdt = physfun.sat_spec_hum_water(temp, den)
            iqs, didt = physfun.sat_spec_hum_water_ice(temp, den)

        ap1 = 10.0 * max(temp - (mpcons.TICE0 - 160.0), 0.0) + 1
        ap1 = min(ap1, 2621.0)
        it1 = ap1
        it2 = ap1 - 0.5


class CalcTables:
    def __init__(self, stencil_factory: StencilFactory, config):
        self._idx = stencil_factory.grid_indexing
        self.config = config
        self._init_tables = stencil_factory.from_origin_domain(
            func=init_tables,
            origin=self._idx.origin_compute(),
            domain=self._idx.domain_compute(),
        )
        self._calc_table_values = stencil_factory.from_origin_domain(
            func=calc_table_values,
            externals={
                "do_mp_table_emulation": config.do_mp_table_emulation,
            },
            origin=self._idx.origin_compute(),
            domain=self._idx.domain_compute(),
        )

    def __call__(
        self,
        temp: FloatField,
        table0: FloatField,
        table2: FloatField,
        iqs: FloatField,
        wqs: FloatField,
        didt: FloatField,
        dwdt: FloatField,
        temp2: FloatField,
        den: FloatField,
        ap1: FloatField,
        it1: IntField,
        it2: IntField,
    ):
        self._init_tables(
            temp,
            table0,
            table2,
        )
        self._calc_table_values(temp2, den, iqs, wqs, didt, dwdt, ap1, it1, it2)


class LookupPython:
    def __init__(self, length):
        self.sat_tables = HumiditySaturationTables(length)

    def __call__(
        self,
        index: FloatField,
        table0: FloatField,
        table2: FloatField,
        iqs: FloatField,
        wqs: FloatField,
        didt: FloatField,
        dwdt: FloatField,
        temp: FloatField,
        den: FloatField,
    ):
        table0 = self.sat_tables.table0[list(index.astype(int) - 1)]
        table2 = self.sat_tables.table2[list(index.astype(int) - 1)]
        wqs, dwdt = self.sat_tables.sat_water(temp, den)
        iqs, didt = self.sat_tables.sat_ice_water(temp, den)
        return table0, table2, wqs, dwdt, iqs, didt


class TranslatePythonTables(TranslatePhysicsFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)

        self.in_vars["data_vars"] = {
            "index": {"serialname": "tc_index", "shield": True},
            "table0": {"serialname": "tc_t0", "shield": True},
            "table2": {"serialname": "tc_t2", "shield": True},
            "wqs": {"serialname": "tab_wq", "shield": True},
            "dwdt": {"serialname": "tab_dwq", "shield": True},
            "iqs": {"serialname": "tab_iq", "shield": True},
            "didt": {"serialname": "tab_diq", "shield": True},
            "temp": {"serialname": "tab_pt", "shield": True},
            "den": {"serialname": "tab_den", "shield": True},
        }

        self.out_vars = {
            "table0": {"serialname": "tc_t0", "kend": namelist.npz, "shield": True},
            "table2": {"serialname": "tc_t2", "kend": namelist.npz, "shield": True},
            "wqs": {"serialname": "tab_wq", "kend": namelist.npz, "shield": True},
            "dwdt": {"serialname": "tab_dwq", "kend": namelist.npz, "shield": True},
            "iqs": {"serialname": "tab_iq", "kend": namelist.npz, "shield": True},
            "didt": {"serialname": "tab_diq", "kend": namelist.npz, "shield": True},
        }
        self.max_error = 2.0e-14

        self.stencil_factory = stencil_factory
        self.config = GFDLCloudMPConfig.from_namelist(namelist)

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)

        compute_func = LookupPython(2621)

        table0, table2, wqs, dwdt, iqs, didt = compute_func(**inputs)
        inputs["table0"] = table0
        inputs["table2"] = table2
        inputs["wqs"] = wqs
        inputs["iqs"] = iqs
        inputs["dwdt"] = dwdt
        inputs["didt"] = didt

        return self.slice_output(inputs)


class TranslateTableComputation(TranslatePhysicsFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)

        self.in_vars["data_vars"] = {
            "temp": {"serialname": "tc_temp", "shield": True},
            "table0": {"serialname": "tc_t0", "shield": True},
            "table2": {"serialname": "tc_t2", "shield": True},
            "wqs": {"serialname": "tab_wq", "shield": True},
            "dwdt": {"serialname": "tab_dwq", "shield": True},
            "iqs": {"serialname": "tab_iq", "shield": True},
            "didt": {"serialname": "tab_diq", "shield": True},
            "temp2": {"serialname": "tab_pt", "shield": True},
            "den": {"serialname": "tab_den", "shield": True},
            "ap1": {"serialname": "tc_ap1", "shield": True},
            "it1": {"serialname": "tc_it1", "shield": True},
            "it2": {"serialname": "tc_it2", "shield": True},
        }

        self.out_vars = {
            "table0": {"serialname": "tc_t0", "kend": namelist.npz, "shield": True},
            "table2": {"serialname": "tc_t2", "kend": namelist.npz, "shield": True},
            "wqs": {"serialname": "tab_wq", "kend": namelist.npz, "shield": True},
            "dwdt": {"serialname": "tab_dwq", "kend": namelist.npz, "shield": True},
            "iqs": {"serialname": "tab_iq", "kend": namelist.npz, "shield": True},
            "didt": {"serialname": "tab_diq", "kend": namelist.npz, "shield": True},
            "ap1": {"serialname": "tc_ap1", "shield": True},
            "it1": {"serialname": "tc_it1", "shield": True},
            "it2": {"serialname": "tc_it2", "shield": True},
        }

        self.max_error = 1e-13  # 10^-25 absolute errors at the top of the tables

        self.stencil_factory = stencil_factory
        self.config = GFDLCloudMPConfig.from_namelist(namelist)
        self.config.do_mp_table_emulation = True

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)

        compute_func = CalcTables(
            self.stencil_factory,
            self.config,
        )

        compute_func(**inputs)

        return self.slice_output(inputs)
