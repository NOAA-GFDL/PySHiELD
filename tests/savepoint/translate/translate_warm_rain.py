from ndsl import StencilFactory
from pyshield.stencils.gfdl_cld_microphysics import GFDLCloudMPConfig
from pyshield.stencils.gfdl_cld_microphysics.warm_rain import WarmRain
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py


class TranslateWarmRain(TranslatePhysicsFortranData2Py):
    def __init__(
        self,
        grid,
        config,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, config, stencil_factory)
        self.in_vars["data_vars"] = {
            "qvapor": {"serialname": "wr_qv", "shield": True},
            "qliquid": {"serialname": "wr_ql", "shield": True},
            "qrain": {"serialname": "wr_qr", "shield": True},
            "qice": {"serialname": "wr_qi", "shield": True},
            "qsnow": {"serialname": "wr_qs", "shield": True},
            "qgraupel": {"serialname": "wr_qg", "shield": True},
            "temperature": {"serialname": "wr_pt", "shield": True},
            "delp": {"serialname": "wr_delp", "shield": True},
            "density": {"serialname": "wr_den", "shield": True},
            "density_factor": {"serialname": "wr_denfac", "shield": True},
            "vterminal_water": {"serialname": "wr_vtw", "shield": True},
            "vterminal_rain": {"serialname": "wr_vtr", "shield": True},
            "cloud_condensation_nuclei": {"serialname": "wr_ccn", "shield": True},
            "reevap": {"serialname": "wr_reevap", "shield": True},
            "h_var": {"serialname": "wr_h_var", "shield": True},
        }

        self.in_vars["parameters"] = [
            "dt",
        ]

        self.out_vars = {
            "qvapor": {"serialname": "wr_qv", "kend": config.npz, "shield": True},
            "qliquid": {"serialname": "wr_ql", "kend": config.npz, "shield": True},
            "qrain": {"serialname": "wr_qr", "kend": config.npz, "shield": True},
            "qice": {"serialname": "wr_qi", "kend": config.npz, "shield": True},
            "qsnow": {"serialname": "wr_qs", "kend": config.npz, "shield": True},
            "qgraupel": {"serialname": "wr_qg", "kend": config.npz, "shield": True},
            "temperature": {
                "serialname": "wr_pt",
                "kend": config.npz,
                "shield": True,
            },
            "cloud_condensation_nuclei": {
                "serialname": "wr_ccn",
                "kend": config.npz,
                "shield": True,
            },
            "reevap": {"serialname": "wr_reevap", "kend": config.npz, "shield": True},
        }

        self.max_error = 5.0e-13  # only qrain in evaporate_rain

        self.stencil_factory = stencil_factory
        self.grid_indexing = self.stencil_factory.grid_indexing
        self.config = GFDLCloudMPConfig.from_config(config)
        self.config.do_mp_table_emulation = True

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)

        compute_func = WarmRain(
            self.stencil_factory,
            self.config,
            timestep=inputs.pop("dt"),
        )

        compute_func(**inputs)

        return self.slice_output(inputs)
