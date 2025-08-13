from dataclasses import InitVar, dataclass, field, fields
from typing import Any, Dict, Mapping

import xarray as xr

import ndsl.dsl.gt4py_utils as gt_utils
from ndsl import GridSizer, Quantity, QuantityFactory
from ndsl.constants import X_DIM, Y_DIM
from ndsl.dsl.typing import Float


@dataclass()
class SurfaceState:
    tsfc: Quantity = field(
        metadata={
            "name": "surface_temperature",
            "dims": [X_DIM, Y_DIM],
            "units": "K",
            "intent": "inout",
        }
    )

    snowdepth: Quantity = field(
        metadata={
            "name": "snow_depth_water_equivalent",
            "dims": [X_DIM, Y_DIM],
            "units": "mm",
            "intent": "inout",
        }
    )

    z0rl: Quantity = field(
        metadata={
            "name": "composite_surface_roughness",
            "dims": [X_DIM, Y_DIM],
            "units": "cm",
            "intent": "in",
        }
    )

    ztrl: Quantity = field(
        metadata={
            "name": "t_and_q_surface_roughness",
            "dims": [X_DIM, Y_DIM],
            "units": "cm",
            "intent": "in",
        }
    )

    wind: Quantity = field(
        metadata={
            "name": "wind_speed",
            "dims": [X_DIM, Y_DIM],
            "units": "m/s",
            "intent": "out",
        }
    )

    quantity_factory: InitVar[QuantityFactory]

    @classmethod
    def init_zeros(
        cls,
        quantity_factory,
    ) -> "SurfaceState":
        initial_arrays = {}
        for _field in fields(cls):
            if "dims" in _field.metadata.keys():
                initial_arrays[_field.name] = quantity_factory.zeros(
                    _field.metadata["dims"],
                    _field.metadata["units"],
                    dtype=Float,
                ).data
        return cls(
            **initial_arrays,
            quantity_factory=quantity_factory,
        )

    @classmethod
    def init_from_storages(
        cls,
        storages: Mapping[str, Any],
        sizer: GridSizer,
        quantity_factory: QuantityFactory,
    ) -> "SurfaceState":
        inputs: Dict[str, Quantity] = {}
        for _field in fields(cls):
            if "dims" in _field.metadata.keys():
                dims = _field.metadata["dims"]
                quantity = Quantity(
                    storages[_field.name],
                    dims,
                    _field.metadata["units"],
                    origin=sizer.get_origin(dims),
                    extent=sizer.get_extent(dims),
                )
                inputs[_field.name] = quantity
        return cls(**inputs, quantity_factory=quantity_factory)

    @property
    def xr_dataset(self):
        data_vars = {}
        for name, field_info in self.__dataclass_fields__.items():
            if name not in ["quantity_factory"]:
                if issubclass(field_info.type, Quantity):
                    dims = [
                        f"{dim_name}_{name}" for dim_name in field_info.metadata["dims"]
                    ]
                    data_vars[name] = xr.DataArray(
                        gt_utils.asarray(getattr(self, name).data),
                        dims=dims,
                        attrs={
                            "long_name": field_info.metadata["name"],
                            "units": field_info.metadata.get("units", "unknown"),
                        },
                    )
        return xr.Dataset(data_vars=data_vars)
