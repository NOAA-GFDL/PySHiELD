from ndsl.dsl.stencil import StencilFactory
from ndsl.initialization.sizer import SubtileGridSizer
from ndsl.initialization.allocator import QuantityFactory
from ndsl.namelist import Namelist
from pyshield import PhysicsConfig
from pyshield.stencils.shield_microphysics.terminal_fall import TerminalFall
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py


class TranslateTerminalFall(TranslatePhysicsFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "qvapor": {"serialname": "tf_qv", "mp3": True},
            "qliquid": {"serialname": "tf_ql", "mp3": True},
            "qrain": {"serialname": "tf_qr", "mp3": True},
            "qice": {"serialname": "tf_qi", "mp3": True},
            "qsnow": {"serialname": "tf_qs", "mp3": True},
            "qgraupel": {"serialname": "tf_qg", "mp3": True},
            "temperature": {"serialname": "tf_pt", "mp3": True},
            "delp": {"serialname": "tf_dp", "mp3": True},
            "delz": {"serialname": "tf_dz", "mp3": True},
            "ua": {"serialname": "tf_ua", "mp3": True},
            "va": {"serialname": "tf_va", "mp3": True},
            "wa": {"serialname": "tf_wa", "mp3": True},
            "z_edge": {"serialname": "tf_ze", "mp3": True},
            "z_terminal": {"serialname": "tf_zt", "mp3": True},
            "column_energy_change": {"serialname": "tf_dte", "mp3": True},
            "flux": {"serialname": "tf_pfi", "mp3": True},
            "v_terminal": {"serialname": "tf_vt", "mp3": True},
            "precipitation": {"serialname": "tf_i1", "mp3": True},
        }

        self.in_vars["parameters"] = ["dt"]

        self.out_vars = {
            "qvapor": {"serialname": "tf_qv", "kend": namelist.npz, "mp3": True},
            "qliquid": {"serialname": "tf_ql", "kend": namelist.npz, "mp3": True},
            "qrain": {"serialname": "tf_qr", "kend": namelist.npz, "mp3": True},
            "qice": {"serialname": "tf_qi", "kend": namelist.npz, "mp3": True},
            "qsnow": {"serialname": "tf_qs", "kend": namelist.npz, "mp3": True},
            "qgraupel": {"serialname": "tf_qg", "kend": namelist.npz, "mp3": True},
            "temperature": {"serialname": "tf_pt", "kend": namelist.npz, "mp3": True},
            "ua": {"serialname": "tf_ua", "kend": namelist.npz, "mp3": True},
            "va": {"serialname": "tf_va", "kend": namelist.npz, "mp3": True},
            "wa": {"serialname": "tf_wa", "kend": namelist.npz, "mp3": True},
            "flux": {"serialname": "tf_pfi", "kend": namelist.npz, "mp3": True},
            "precipitation": {"serialname": "tf_i1", "mp3": True},
            "column_energy_change": {"serialname": "tf_dte", "mp3": True},
        }

        self.stencil_factory = stencil_factory
        self.grid_indexing = self.stencil_factory.grid_indexing
        pconf = PhysicsConfig.from_namelist(namelist)
        self.config = pconf.microphysics

        sizer = SubtileGridSizer.from_tile_params(
            nx_tile=self.namelist.npx - 1,
            ny_tile=self.namelist.npy - 1,
            nz=self.namelist.npz,
            n_halo=3,
            extra_dim_lengths={},
            layout=self.namelist.layout,
        )

        self.quantity_factory = QuantityFactory.from_backend(
            sizer, self.stencil_factory.backend
        )

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)

        inputs["tracer"] = "ice"

        compute_func = TerminalFall(
            self.stencil_factory,
            self.quantity_factory,
            self.config,
            timestep=inputs.pop("dt"),
        )

        compute_func(**inputs)

        return self.slice_output(inputs)
