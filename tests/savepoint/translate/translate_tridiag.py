from gt4py.cartesian.gtscript import PARALLEL, computation, interval

from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.dsl.stencil import StencilFactory
from ndsl.dsl.typing import Float
from ndsl.initialization.allocator import QuantityFactory
from ndsl.initialization.sizer import SubtileGridSizer
from ndsl.stencils.basic_operations import copy_defn
from pyshield._config import TRACER_DIM, FloatFieldTracer
from pyshield.stencils.pbl.tridiag import tridi2, tridin, tridit
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py


def copy_4d(
    q_in: FloatFieldTracer,
    q_out: FloatFieldTracer,
):
    with computation(PARALLEL), interval(...):
        q_out = q_in


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
        quantity_factory: QuantityFactory,
    ):
        idx = stencil_factory.grid_indexing
        quantity_factory.set_extra_dim_lengths(
            **{
                TRACER_DIM: 8,
            }
        )
        self._tridi2 = stencil_factory.from_origin_domain(
            func=tridi2,
            origin=idx.origin_compute(),
            domain=idx.domain_compute(),
        )

        self._copy_stencil = stencil_factory.from_origin_domain(
            func=copy_defn,
            origin=idx.origin_compute(),
            domain=idx.domain_compute(),
        )

        self._copy_4d = stencil_factory.from_origin_domain(
            func=copy_4d,
            origin=idx.origin_compute(),
            domain=idx.domain_compute(),
        )

        self._cu = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM],
            units="unknown",
            dtype=Float,
        )
        self._r1 = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM],
            units="unknown",
            dtype=Float,
        )
        self._r2 = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM, TRACER_DIM],
            units="unknown",
            dtype=Float,
        )

    def __call__(
        self,
        au,
        ad,
        al,
        f1,
        f2,
    ):

        self._copy_stencil(au, self._cu)
        self._copy_stencil(f1, self._r1)
        self._copy_4d(f2, self._r2)

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
        quantity_factory: QuantityFactory,
        ntke,
    ):
        idx = stencil_factory.grid_indexing
        quantity_factory.set_extra_dim_lengths(
            **{
                TRACER_DIM: 8,
            }
        )
        self._ntke = ntke
        self._tridin = stencil_factory.from_origin_domain(
            func=tridin,
            origin=idx.origin_compute(),
            domain=idx.domain_compute(),
        )

        self._copy_stencil = stencil_factory.from_origin_domain(
            func=copy_defn,
            origin=idx.origin_compute(),
            domain=idx.domain_compute(),
        )

        self._copy_4d = stencil_factory.from_origin_domain(
            func=copy_4d,
            origin=idx.origin_compute(),
            domain=idx.domain_compute(),
        )

        self._cu = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM],
            units="unknown",
            dtype=Float,
        )
        self._r1 = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM],
            units="unknown",
            dtype=Float,
        )
        self._r2 = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM, TRACER_DIM],
            units="unknown",
            dtype=Float,
        )

    def __call__(
        self,
        au,
        ad,
        al,
        f1,
        f2,
        nt1,
    ):
        self._copy_stencil(au, self._cu)
        self._copy_stencil(f1, self._r1)
        self._copy_4d(f2, self._r2)

        for n in range(0, int(nt1)):
            dim_n = n if n < self._ntke else n + 1
            breakpoint()
            self._tridin(al, ad, self._cu, self._r1, self._r2, au, f1, f2, int(dim_n))


class TranslateTridit(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "al": {"shield": True, "kend": namelist.npz - 1},
            "au": {"shield": True, "kend": namelist.npz - 1},
            "ad": {"shield": True},
            "f1": {"shield": True},
        }
        self.out_vars = {
            "al": {"shield": True, "kend": namelist.npz - 1},
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
            "al": {"shield": True, "kend": namelist.npz - 1},
            "au": {"shield": True, "kend": namelist.npz - 1},
            "ad": {"shield": True},
            "f1": {"shield": True},
            "f2": {"serialname": "f2_ser", "shield": True},
        }
        self.out_vars = {
            "al": {"shield": True, "kend": namelist.npz - 1},
            "au": {"shield": True, "kend": namelist.npz - 1},
            "ad": {"shield": True},
            "f1": {"shield": True},
            "f2": {"serialname": "f2_ser", "shield": True},
        }
        self.stencil_factory = stencil_factory
        self.grid_indexing = self.stencil_factory.grid_indexing

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
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
        compute_func = Tridi2(self.stencil_factory, quantity_factory)

        compute_func(**inputs)

        return self.slice_output(inputs)


class TranslateTridin(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "al": {"shield": True, "kend": namelist.npz - 1},
            "au": {"shield": True, "kend": namelist.npz - 1},
            "ad": {"shield": True},
            "f1": {"shield": True},
            "f2": {"serialname": "f2_ser", "shield": True},
        }
        self.in_vars["parameters"] = ["nt1"]
        self.out_vars = {
            "al": {"shield": True, "kend": namelist.npz - 1},
            "au": {"shield": True, "kend": namelist.npz - 1},
            "ad": {"shield": True},
            "f1": {"shield": True},
            "f2": {"serialname": "f2_ser", "shield": True},
        }
        self.stencil_factory = stencil_factory
        self.grid_indexing = self.stencil_factory.grid_indexing

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
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
        config = self.namelist.pbl
        compute_func = TridiN(self.stencil_factory, quantity_factory, 8)

        compute_func(**inputs)

        return self.slice_output(inputs)
