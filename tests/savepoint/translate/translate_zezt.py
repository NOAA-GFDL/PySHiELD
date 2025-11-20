from ndsl import StencilFactory
from ndsl.dsl.typing import FloatField, FloatFieldIJ
from pyshield.stencils.gfdl_cld_microphysics import GFDLCloudMPConfig
from pyshield.stencils.gfdl_cld_microphysics.sedimentation import (
    calc_edge_and_terminal_height,
)
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py


class ZeZt:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        config,
        timestep,
    ):
        self._idx = stencil_factory.grid_indexing
        self.config = config
        self._calc_edge_and_terminal_height = stencil_factory.from_origin_domain(
            func=calc_edge_and_terminal_height,
            externals={"timestep": timestep},
            origin=self._idx.origin_compute(),
            domain=self._idx.domain_compute(add=(0, 0, 1)),
        )

    def __call__(
        self,
        z_surface: FloatFieldIJ,
        z_edge: FloatField,
        z_terminal: FloatField,
        delz: FloatField,
        v_terminal: FloatField,
    ):
        self._calc_edge_and_terminal_height(
            z_surface,
            z_edge,
            z_terminal,
            delz,
            v_terminal,
        )


class TranslateZeZt(TranslatePhysicsFortranData2Py):
    def __init__(
        self,
        grid,
        namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "z_surface": {"serialname": "zz_zs", "shield": True},
            "z_edge": {"serialname": "zz_ze", "shield": True},
            "z_terminal": {"serialname": "zz_zt", "shield": True},
            "delz": {"serialname": "zz_dz", "shield": True},
            "v_terminal": {"serialname": "zz_vt", "shield": True},
        }

        self.in_vars["parameters"] = ["dt"]

        self.out_vars = {
            "z_edge": {"serialname": "zz_ze", "kend": namelist.npz + 1, "shield": True},
            "z_terminal": {
                "serialname": "zz_zt",
                "kend": namelist.npz + 1,
                "shield": True,
            },
            "z_surface": {"serialname": "zz_zs", "shield": True},
        }

        self.stencil_factory = stencil_factory
        self.grid_indexing = self.stencil_factory.grid_indexing
        self.config = GFDLCloudMPConfig.from_namelist(namelist)

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)

        compute_func = ZeZt(
            self.stencil_factory,
            self.config,
            timestep=inputs.pop("dt"),
        )

        compute_func(**inputs)

        return self.slice_output(inputs)
