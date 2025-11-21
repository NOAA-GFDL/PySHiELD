from ndsl import QuantityFactory, StencilFactory, SubtileGridSizer
from pyshield.stencils.gfdl_cld_microphysics import GFDLCloudMPConfig
from pyshield.stencils.gfdl_cld_microphysics.terminal_fall import TerminalFall
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py


class TranslateTerminalFall(TranslatePhysicsFortranData2Py):
    def __init__(
        self,
        grid,
        config,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, config, stencil_factory)
        self.in_vars["data_vars"] = {
            "qvapor": {"serialname": "tf_qv", "shield": True},
            "qliquid": {"serialname": "tf_ql", "shield": True},
            "qrain": {"serialname": "tf_qr", "shield": True},
            "qice": {"serialname": "tf_qi", "shield": True},
            "qsnow": {"serialname": "tf_qs", "shield": True},
            "qgraupel": {"serialname": "tf_qg", "shield": True},
            "temperature": {"serialname": "tf_pt", "shield": True},
            "delp": {"serialname": "tf_dp", "shield": True},
            "delz": {"serialname": "tf_dz", "shield": True},
            "ua": {"serialname": "tf_ua", "shield": True},
            "va": {"serialname": "tf_va", "shield": True},
            "wa": {"serialname": "tf_wa", "shield": True},
            "z_edge": {"serialname": "tf_ze", "shield": True},
            "z_terminal": {"serialname": "tf_zt", "shield": True},
            "column_energy_change": {"serialname": "tf_dte", "shield": True},
            "flux": {"serialname": "tf_pfi", "shield": True},
            "v_terminal": {"serialname": "tf_vt", "shield": True},
            "precipitation": {"serialname": "tf_i1", "shield": True},
        }

        self.in_vars["parameters"] = ["dt"]

        self.out_vars = {
            "qvapor": {"serialname": "tf_qv", "kend": config.npz, "shield": True},
            "qliquid": {"serialname": "tf_ql", "kend": config.npz, "shield": True},
            "qrain": {"serialname": "tf_qr", "kend": config.npz, "shield": True},
            "qice": {"serialname": "tf_qi", "kend": config.npz, "shield": True},
            "qsnow": {"serialname": "tf_qs", "kend": config.npz, "shield": True},
            "qgraupel": {"serialname": "tf_qg", "kend": config.npz, "shield": True},
            "temperature": {
                "serialname": "tf_pt",
                "kend": config.npz,
                "shield": True,
            },
            "ua": {"serialname": "tf_ua", "kend": config.npz, "shield": True},
            "va": {"serialname": "tf_va", "kend": config.npz, "shield": True},
            "wa": {"serialname": "tf_wa", "kend": config.npz, "shield": True},
            "flux": {"serialname": "tf_pfi", "kend": config.npz, "shield": True},
            "precipitation": {"serialname": "tf_i1", "shield": True},
            "column_energy_change": {"serialname": "tf_dte", "shield": True},
        }

        self.stencil_factory = stencil_factory
        self.grid_indexing = self.stencil_factory.grid_indexing
        self.config = GFDLCloudMPConfig.from_config(config)

        sizer = SubtileGridSizer.from_tile_params(
            nx_tile=self.config.npx - 1,
            ny_tile=self.config.npy - 1,
            nz=self.config.npz,
            n_halo=3,
            data_dimensions={},
            layout=self.config.layout,
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
