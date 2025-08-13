from ndsl.dsl.stencil import StencilFactory
from ndsl.namelist import Namelist
from pyshield import PhysicsConfig
from pyshield.stencils.shield_microphysics.neg_adj import AdjustNegativeTracers
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py


class TranslateNegAdjP(TranslatePhysicsFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
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
            "qvapor": {"serialname": "ne_qv", "kend": namelist.npz, "shield": True},
            "qliquid": {"serialname": "ne_ql", "kend": namelist.npz, "shield": True},
            "qrain": {"serialname": "ne_qr", "kend": namelist.npz, "shield": True},
            "qice": {"serialname": "ne_qi", "kend": namelist.npz, "shield": True},
            "qsnow": {"serialname": "ne_qs", "kend": namelist.npz, "shield": True},
            "qgraupel": {"serialname": "ne_qg", "kend": namelist.npz, "shield": True},
            "temperature": {"serialname": "ne_pt", "kend": namelist.npz, "shield": True},
            "delp": {"serialname": "ne_delp", "kend": namelist.npz, "shield": True},
            "condensation": {"serialname": "ne_cond", "shield": True},
        }

        self.stencil_factory = stencil_factory
        self.grid_indexing = self.stencil_factory.grid_indexing
        pconf = PhysicsConfig.from_namelist(namelist)
        mpconfig = pconf.microphysics
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
