from ndsl.dsl.stencil import StencilFactory
from ndsl.namelist import Namelist
from pyshield import PhysicsConfig
from pyshield.stencils.shield_microphysics.warm_rain import WarmRain
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py


class TranslateWarmRain(TranslatePhysicsFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "qvapor": {"serialname": "wr_qv", "mp3": True},
            "qliquid": {"serialname": "wr_ql", "mp3": True},
            "qrain": {"serialname": "wr_qr", "mp3": True},
            "qice": {"serialname": "wr_qi", "mp3": True},
            "qsnow": {"serialname": "wr_qs", "mp3": True},
            "qgraupel": {"serialname": "wr_qg", "mp3": True},
            "temperature": {"serialname": "wr_pt", "mp3": True},
            "delp": {"serialname": "wr_delp", "mp3": True},
            "density": {"serialname": "wr_den", "mp3": True},
            "density_factor": {"serialname": "wr_denfac", "mp3": True},
            "vterminal_water": {"serialname": "wr_vtw", "mp3": True},
            "vterminal_rain": {"serialname": "wr_vtr", "mp3": True},
            "cloud_condensation_nuclei": {"serialname": "wr_ccn", "mp3": True},
            "reevap": {"serialname": "wr_reevap", "mp3": True},
            "h_var": {"serialname": "wr_h_var", "mp3": True},
        }

        self.in_vars["parameters"] = [
            "dt",
        ]

        self.out_vars = {
            "qvapor": {"serialname": "wr_qv", "kend": namelist.npz, "mp3": True},
            "qliquid": {"serialname": "wr_ql", "kend": namelist.npz, "mp3": True},
            "qrain": {"serialname": "wr_qr", "kend": namelist.npz, "mp3": True},
            "qice": {"serialname": "wr_qi", "kend": namelist.npz, "mp3": True},
            "qsnow": {"serialname": "wr_qs", "kend": namelist.npz, "mp3": True},
            "qgraupel": {"serialname": "wr_qg", "kend": namelist.npz, "mp3": True},
            "temperature": {"serialname": "wr_pt", "kend": namelist.npz, "mp3": True},
            "cloud_condensation_nuclei": {
                "serialname": "wr_ccn",
                "kend": namelist.npz,
                "mp3": True,
            },
            "reevap": {"serialname": "wr_reevap", "kend": namelist.npz, "mp3": True},
        }

        self.max_error = 5.0e-13  # only qrain in evaporate_rain

        self.stencil_factory = stencil_factory
        self.grid_indexing = self.stencil_factory.grid_indexing
        pconf = PhysicsConfig.from_namelist(namelist)
        self.config = pconf.microphysics
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
