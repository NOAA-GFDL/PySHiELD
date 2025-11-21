from ndsl import StencilFactory
from pyshield.stencils.gfdl_cld_microphysics import GFDLCloudMPConfig
from pyshield.stencils.gfdl_cld_microphysics.ice_cloud import IceCloud
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py


class TranslateIceCloud(TranslatePhysicsFortranData2Py):
    def __init__(
        self,
        grid,
        config,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, config, stencil_factory)
        self.in_vars["data_vars"] = {
            "qvapor": {"serialname": "ic_qv", "shield": True},
            "qliquid": {"serialname": "ic_ql", "shield": True},
            "qrain": {"serialname": "ic_qr", "shield": True},
            "qice": {"serialname": "ic_qi", "shield": True},
            "qsnow": {"serialname": "ic_qs", "shield": True},
            "qgraupel": {"serialname": "ic_qg", "shield": True},
            "temperature": {"serialname": "ic_pt", "shield": True},
            "density": {"serialname": "ic_den", "shield": True},
            "density_factor": {"serialname": "ic_denfac", "shield": True},
            "vterminal_water": {"serialname": "ic_vtw", "shield": True},
            "vterminal_rain": {"serialname": "ic_vtr", "shield": True},
            "vterminal_ice": {"serialname": "ic_vti", "shield": True},
            "vterminal_snow": {"serialname": "ic_vts", "shield": True},
            "vterminal_graupel": {"serialname": "ic_vtg", "shield": True},
            "h_var": {"serialname": "ic_h_var", "shield": True},
        }

        self.in_vars["parameters"] = [
            "dt",
        ]

        self.out_vars = {
            "qvapor": {"serialname": "ic_qv", "kend": config.npz, "shield": True},
            "qliquid": {"serialname": "ic_ql", "kend": config.npz, "shield": True},
            "qrain": {"serialname": "ic_qr", "kend": config.npz, "shield": True},
            "qice": {"serialname": "ic_qi", "kend": config.npz, "shield": True},
            "qsnow": {"serialname": "ic_qs", "kend": config.npz, "shield": True},
            "qgraupel": {"serialname": "ic_qg", "kend": config.npz, "shield": True},
            "temperature": {
                "serialname": "ic_pt",
                "kend": config.npz,
                "shield": True,
            },
        }

        self.stencil_factory = stencil_factory
        self.grid_indexing = self.stencil_factory.grid_indexing
        self.config = GFDLCloudMPConfig.from_config(config)

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)

        compute_func = IceCloud(
            self.stencil_factory,
            self.config,
            timestep=inputs.pop("dt"),
        )

        compute_func(**inputs)

        return self.slice_output(inputs)
