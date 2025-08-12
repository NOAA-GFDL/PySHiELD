from ndsl.dsl.stencil import StencilFactory
from ndsl.namelist import Namelist
from pyshield import PhysicsConfig
from pyshield.stencils.shield_microphysics.ice_cloud import IceCloud
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py


class TranslateIceCloud(TranslatePhysicsFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "qvapor": {"serialname": "ic_qv", "mp3": True},
            "qliquid": {"serialname": "ic_ql", "mp3": True},
            "qrain": {"serialname": "ic_qr", "mp3": True},
            "qice": {"serialname": "ic_qi", "mp3": True},
            "qsnow": {"serialname": "ic_qs", "mp3": True},
            "qgraupel": {"serialname": "ic_qg", "mp3": True},
            "temperature": {"serialname": "ic_pt", "mp3": True},
            "density": {"serialname": "ic_den", "mp3": True},
            "density_factor": {"serialname": "ic_denfac", "mp3": True},
            "vterminal_water": {"serialname": "ic_vtw", "mp3": True},
            "vterminal_rain": {"serialname": "ic_vtr", "mp3": True},
            "vterminal_ice": {"serialname": "ic_vti", "mp3": True},
            "vterminal_snow": {"serialname": "ic_vts", "mp3": True},
            "vterminal_graupel": {"serialname": "ic_vtg", "mp3": True},
            "h_var": {"serialname": "ic_h_var", "mp3": True},
        }

        self.in_vars["parameters"] = [
            "dt",
        ]

        self.out_vars = {
            "qvapor": {"serialname": "ic_qv", "kend": namelist.npz, "mp3": True},
            "qliquid": {"serialname": "ic_ql", "kend": namelist.npz, "mp3": True},
            "qrain": {"serialname": "ic_qr", "kend": namelist.npz, "mp3": True},
            "qice": {"serialname": "ic_qi", "kend": namelist.npz, "mp3": True},
            "qsnow": {"serialname": "ic_qs", "kend": namelist.npz, "mp3": True},
            "qgraupel": {"serialname": "ic_qg", "kend": namelist.npz, "mp3": True},
            "temperature": {"serialname": "ic_pt", "kend": namelist.npz, "mp3": True},
        }

        self.stencil_factory = stencil_factory
        self.grid_indexing = self.stencil_factory.grid_indexing
        pconf = PhysicsConfig.from_namelist(namelist)
        self.config = pconf.microphysics

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)

        compute_func = IceCloud(
            self.stencil_factory,
            self.config,
            timestep=inputs.pop("dt"),
        )

        compute_func(**inputs)

        return self.slice_output(inputs)
