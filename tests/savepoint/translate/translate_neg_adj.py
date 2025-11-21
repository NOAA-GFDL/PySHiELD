from ndsl import StencilFactory
from pyshield.stencils.gfdl_cld_microphysics import GFDLCloudMPConfig
from pyshield.stencils.gfdl_cld_microphysics.neg_adj import AdjustNegativeTracers
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py


class TranslateNegAdjP(TranslatePhysicsFortranData2Py):
    def __init__(
        self,
        grid,
        config,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, config, stencil_factory)
        self.in_vars["data_vars"] = {
            "qvapor": {"serialname": "ne_qv", "shield": True},
            "qliquid": {"serialname": "ne_ql", "shield": True},
            "qrain": {"serialname": "ne_qr", "shield": True},
            "qice": {"serialname": "ne_qi", "shield": True},
            "qsnow": {"serialname": "ne_qs", "shield": True},
            "qgraupel": {"serialname": "ne_qg", "shield": True},
            "temperature": {"serialname": "ne_pt", "shield": True},
            "delp": {"serialname": "ne_delp", "shield": True},
            "condensation": {"serialname": "ne_cond", "shield": True},
        }

        self.in_vars["parameters"] = ["convt"]

        self.out_vars = {
            "qvapor": {"serialname": "ne_qv", "kend": config.npz, "shield": True},
            "qliquid": {"serialname": "ne_ql", "kend": config.npz, "shield": True},
            "qrain": {"serialname": "ne_qr", "kend": config.npz, "shield": True},
            "qice": {"serialname": "ne_qi", "kend": config.npz, "shield": True},
            "qsnow": {"serialname": "ne_qs", "kend": config.npz, "shield": True},
            "qgraupel": {"serialname": "ne_qg", "kend": config.npz, "shield": True},
            "temperature": {
                "serialname": "ne_pt",
                "kend": config.npz,
                "shield": True,
            },
            "delp": {"serialname": "ne_delp", "kend": config.npz, "shield": True},
            "condensation": {"serialname": "ne_cond", "shield": True},
        }

        self.stencil_factory = stencil_factory
        self.grid_indexing = self.stencil_factory.grid_indexing
        mpconfig = GFDLCloudMPConfig.from_config(config)
        self.config = mpconfig.adjustnegative

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)

        compute_func = AdjustNegativeTracers(
            self.stencil_factory,
            self.config,
            convert_mm_day=inputs.pop("convt"),
        )

        compute_func(**inputs)

        return self.slice_output(inputs)
