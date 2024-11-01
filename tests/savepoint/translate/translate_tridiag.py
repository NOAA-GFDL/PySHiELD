from ndsl.dsl.stencil import StencilFactory
from ndsl.initialization.allocator import QuantityFactory
from ndsl.dsl.typing import Float
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.initialization.sizer import SubtileGridSizer
from pySHiELD.stencils.pbl.tridiag import tridit, tridi2, tridin
from ndsl.stencils.basic_operations import copy_defn
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py

class TridiT:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
    ):
        idx = stencil_factory.grid_indexing
        self._cu = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM],
            units="unknown",
            dtype=Float,
        )
        self._rt = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM],
            units="unknown",
            dtype=Float,
        )
        self._tridit = stencil_factory.from_origin_domain(
            func=tridit,
            origin=idx.origin_compute(),
            domain=idx.domain_compute(),
        )
        self._copy_stencil = stencil_factory.from_origin_domain(
            func=copy_defn,
            origin=idx.origin_compute(),
            domain=idx.domain_compute(),
        )

    def __call__(
        self,
        au,
        ad,
        al,
        f1,
    ):
        self._copy_stencil(au, self._cu)
        self._copy_stencil(f1, self._rt)
        self._tridit(
            self._cu,
            ad,
            al,
            self._rt,
            au,
            f1,
        )

class Tridi2:
    def __init__(
        self,
        stencil_factory: StencilFactory,
    ):
        idx = stencil_factory.grid_indexing
        self._tridi2 = stencil_factory.from_origin_domain(
            func=tridi2,
            origin=idx.origin_compute(),
            domain=idx.domain_compute(),
        )

    def __call__(
        self,
        au,
        ad,
        al,
        f1,
        f2,
    ):
        self._tridi2(
            f1,
            f2,
            au,
            al,
            ad,
            au,
            f1,
            f2,
        )

class TridiN:
    def __init__(
        self,
        stencil_factory: StencilFactory,
    ):
        idx = stencil_factory.grid_indexing
        self._tridin = stencil_factory.from_origin_domain(
            func=tridin,
            origin=idx.origin_compute(),
            domain=idx.domain_compute(),
        )

    def __call__(
        self,
        au,
        ad,
        al,
        f1,
        f2,
        nt,
    ):
        self._tridin(
            al,
            ad,
            au,
            f1,
            f2,
            au,
            f1,
            f2,
            nt,
        )


class TranslateTridit(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "al": {"shield": True},
            "au": {"shield": True, "kend": namelist.npz - 1},
            "ad": {"shield": True},
            "f1": {"shield": True},
        }
        self.out_vars = {
            "al": {"shield": True},
            "au": {"shield": True, "kend": namelist.npz - 1},
            "ad": {"shield": True},
            "f1": {"shield": True},
        }
        self.stencil_factory = stencil_factory
        self.grid_indexing = self.stencil_factory.grid_indexing

    def compute(self, inputs):
        sizer = SubtileGridSizer.from_tile_params(
            nx_tile=self.namelist.npx - 1,
            ny_tile=self.namelist.npx - 1,
            nz=self.namelist.npz,
            n_halo=3,
            extra_dim_lengths={},
            layout=self.namelist.layout,
        )

        quantity_factory = QuantityFactory.from_backend(
            sizer, self.stencil_factory.backend
        )
        
        self.make_storage_data_input_vars(inputs)
        compute_func = TridiT(self.stencil_factory, quantity_factory)

        compute_func(**inputs)

        return self.slice_output(inputs)

class TranslateTridi2(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "al": {"shield": True, "kstart": 1},
            "au": {"shield": True, "kend": namelist.npz - 1},
            "ad": {"shield": True},
            "f1": {"shield": True},
            "f2": {"shield": True},
        }
        self.out_vars = {
            "al": {"shield": True, "kstart": 1},
            "au": {"shield": True, "kend": namelist.npz - 1},
            "ad": {"shield": True},
            "f1": {"shield": True},
            "f2": {"shield": True},
        }
        self.stencil_factory = stencil_factory
        self.grid_indexing = self.stencil_factory.grid_indexing

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        compute_func = Tridi2(self.stencil_factory,)

        compute_func(**inputs)

        return self.slice_output(inputs)

class TranslateTridin(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "al": {"shield": True, "kstart": 1},
            "au": {"shield": True, "kend": namelist.npz - 1},
            "ad": {"shield": True},
            "f1": {"shield": True},
            "f2": {"shield": True},
        }
        self.in_vars.parameters = [
            "nt"
        ]
        self.out_vars = {
            "al": {"shield": True, "kstart": 1},
            "au": {"shield": True, "kend": namelist.npz - 1},
            "ad": {"shield": True},
            "f1": {"shield": True},
            "f2": {"shield": True},
        }
        self.stencil_factory = stencil_factory
        self.grid_indexing = self.stencil_factory.grid_indexing

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        compute_func = Tridi2(self.stencil_factory,)

        compute_func(**inputs)

        return self.slice_output(inputs)
