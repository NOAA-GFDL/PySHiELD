from ndsl.dsl.stencil import StencilFactory
from ndsl.namelist import Namelist
from pyshield import PhysicsConfig
from pyshield.stencils.shield_microphysics.subgrid_z_proc import (
    VerticalSubgridProcesses,
)
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py


class TranslateSubgridZProc(TranslatePhysicsFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "qvapor": {"serialname": "sz_qv", "mp3": True},
            "qliquid": {"serialname": "sz_ql", "mp3": True},
            "qrain": {"serialname": "sz_qr", "mp3": True},
            "qice": {"serialname": "sz_qi", "mp3": True},
            "qsnow": {"serialname": "sz_qs", "mp3": True},
            "qgraupel": {"serialname": "sz_qg", "mp3": True},
            "temperature": {"serialname": "sz_pt", "mp3": True},
            "density": {"serialname": "sz_den", "mp3": True},
            "density_factor": {"serialname": "sz_denfac", "mp3": True},
            "delp": {"serialname": "sz_delp", "mp3": True},
            "rh_adj": {"serialname": "sz_rh_adj", "mp3": True},
            "cloud_condensation_nuclei": {"serialname": "sz_ccn", "mp3": True},
            "cloud_ice_nuclei": {"serialname": "sz_cin", "mp3": True},
            "cond": {"serialname": "sz_cond", "mp3": True},
            "dep": {"serialname": "sz_dep", "mp3": True},
            "reevap": {"serialname": "sz_reevap", "mp3": True},
            "sub": {"serialname": "sz_sub", "mp3": True},
        }

        self.in_vars["parameters"] = [
            "dt",
        ]

        self.out_vars = {
            "qvapor": {"serialname": "sz_qv", "kend": namelist.npz, "mp3": True},
            "qliquid": {"serialname": "sz_ql", "kend": namelist.npz, "mp3": True},
            "qrain": {"serialname": "sz_qr", "kend": namelist.npz, "mp3": True},
            "qice": {"serialname": "sz_qi", "kend": namelist.npz, "mp3": True},
            "qsnow": {"serialname": "sz_qs", "kend": namelist.npz, "mp3": True},
            "qgraupel": {"serialname": "sz_qg", "kend": namelist.npz, "mp3": True},
            "temperature": {"serialname": "sz_pt", "kend": namelist.npz, "mp3": True},
            "cloud_condensation_nuclei": {
                "serialname": "sz_ccn",
                "kend": namelist.npz,
                "mp3": True,
            },
            "cloud_ice_nuclei": {
                "serialname": "sz_cin",
                "kend": namelist.npz,
                "mp3": True,
            },
            "cond": {"serialname": "sz_cond", "kend": namelist.npz, "mp3": True},
            "dep": {"serialname": "sz_dep", "kend": namelist.npz, "mp3": True},
            "reevap": {"serialname": "sz_reevap", "kend": namelist.npz, "mp3": True},
            "sub": {"serialname": "sz_sub", "kend": namelist.npz, "mp3": True},
        }

        self.stencil_factory = stencil_factory
        self.grid_indexing = self.stencil_factory.grid_indexing
        pconf = PhysicsConfig.from_namelist(namelist)
        self.config = pconf.microphysics
        self.config.do_mp_table_emulation = True

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)

        compute_func = VerticalSubgridProcesses(
            self.stencil_factory,
            self.config,
            timestep=inputs.pop("dt"),
        )

        compute_func(**inputs)

        return self.slice_output(inputs)
