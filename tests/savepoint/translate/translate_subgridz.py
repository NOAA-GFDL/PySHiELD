from ndsl import StencilFactory
from pyshield.stencils.gfdl_cld_microphysics import GFDLCloudMPConfig
from pyshield.stencils.gfdl_cld_microphysics.subgrid_z_proc import (
    VerticalSubgridProcesses,
)
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py


class TranslateSubgridZProc(TranslatePhysicsFortranData2Py):
    def __init__(
        self,
        grid,
        config,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, config, stencil_factory)
        self.in_vars["data_vars"] = {
            "qvapor": {"serialname": "sz_qv", "shield": True},
            "qliquid": {"serialname": "sz_ql", "shield": True},
            "qrain": {"serialname": "sz_qr", "shield": True},
            "qice": {"serialname": "sz_qi", "shield": True},
            "qsnow": {"serialname": "sz_qs", "shield": True},
            "qgraupel": {"serialname": "sz_qg", "shield": True},
            "temperature": {"serialname": "sz_pt", "shield": True},
            "density": {"serialname": "sz_den", "shield": True},
            "density_factor": {"serialname": "sz_denfac", "shield": True},
            "delp": {"serialname": "sz_delp", "shield": True},
            "rh_adj": {"serialname": "sz_rh_adj", "shield": True},
            "cloud_condensation_nuclei": {"serialname": "sz_ccn", "shield": True},
            "cloud_ice_nuclei": {"serialname": "sz_cin", "shield": True},
            "cond": {"serialname": "sz_cond", "shield": True},
            "dep": {"serialname": "sz_dep", "shield": True},
            "reevap": {"serialname": "sz_reevap", "shield": True},
            "sub": {"serialname": "sz_sub", "shield": True},
        }

        self.in_vars["parameters"] = [
            "dt",
        ]

        self.out_vars = {
            "qvapor": {"serialname": "sz_qv", "kend": config.npz, "shield": True},
            "qliquid": {"serialname": "sz_ql", "kend": config.npz, "shield": True},
            "qrain": {"serialname": "sz_qr", "kend": config.npz, "shield": True},
            "qice": {"serialname": "sz_qi", "kend": config.npz, "shield": True},
            "qsnow": {"serialname": "sz_qs", "kend": config.npz, "shield": True},
            "qgraupel": {"serialname": "sz_qg", "kend": config.npz, "shield": True},
            "temperature": {
                "serialname": "sz_pt",
                "kend": config.npz,
                "shield": True,
            },
            "cloud_condensation_nuclei": {
                "serialname": "sz_ccn",
                "kend": config.npz,
                "shield": True,
            },
            "cloud_ice_nuclei": {
                "serialname": "sz_cin",
                "kend": config.npz,
                "shield": True,
            },
            "cond": {"serialname": "sz_cond", "kend": config.npz, "shield": True},
            "dep": {"serialname": "sz_dep", "kend": config.npz, "shield": True},
            "reevap": {"serialname": "sz_reevap", "kend": config.npz, "shield": True},
            "sub": {"serialname": "sz_sub", "kend": config.npz, "shield": True},
        }

        self.stencil_factory = stencil_factory
        self.grid_indexing = self.stencil_factory.grid_indexing
        self.config = GFDLCloudMPConfig.from_config(config)
        self.config.do_mp_table_emulation = True

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)

        compute_func = VerticalSubgridProcesses(
            self.stencil_factory,
            self.config,
            timestep=inputs.pop("dt"),
        )
        inputs["last_step"] = True

        compute_func(**inputs)

        return self.slice_output(inputs)
