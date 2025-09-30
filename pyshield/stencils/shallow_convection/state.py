from dataclasses import dataclass, field, fields
from typing import Any, Dict, Mapping

import xarray as xr

import ndsl.dsl.gt4py_utils as gt_utils
from ndsl import GridSizer, Quantity, QuantityFactory
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.dsl.typing import Float, Int
from pyshield.stencils.shallow_convection._config import SC_TRACER_DIM


@dataclass()
class SAMFShalConvState:
    q1: Quantity = field(
        metadata={
            "name": "specific_humidity",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    t1: Quantity = field(
        metadata={
            "name": "air_temperature",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "degK",
            "intent": "inout",
        }
    )
    u1: Quantity = field(
        metadata={
            "name": "eastward_wind",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "m/s",
            "intent": "inout",
        }
    )
    v1: Quantity = field(
        metadata={
            "name": "northward_wind",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "m/s",
            "intent": "inout",
        }
    )
    qtr: Quantity = field(
        metadata={
            "name": "convective_tracers",
            "dims": [X_DIM, Y_DIM, Z_DIM, SC_TRACER_DIM],
            "units": "kg/kg",
            "intent": "in",
        }
    )
    dot: Quantity = field(
        metadata={
            "name": "layer_mean_vertical_velocity",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "Pa/s",
            "intent": "in",
        }
    )
    hpbl: Quantity = field(
        metadata={
            "name": "pbl_height",
            "dims": [X_DIM, Y_DIM],
            "units": "m",
            "intent": "in",
        }
    )
    prslp: Quantity = field(
        metadata={
            "name": "mean_layer_pressure",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "Pa",
            "intent": "in",
        }
    )
    phil: Quantity = field(
        metadata={
            "name": "layer_geopotential",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "m**2/s**2",
            "intent": "in",
        }
    )
    delp: Quantity = field(
        metadata={
            "name": "pressure_thickness_of_atmospheric_layer",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "Pa",
            "intent": "in",
        }
    )
    cnvw: Quantity = field(
        metadata={
            "name": "convective_cloud_water",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "kg/kg",
            "intent": "out",
        }
    )
    cnvc: Quantity = field(
        metadata={
            "name": "convective_cloud_cover",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "",
            "intent": "out",
        }
    )
    ud_mf: Quantity = field(
        metadata={
            "name": "updraft_mass_flux_times_timestep",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "kg/m**2",
            "intent": "out",
        }
    )
    dt_mf: Quantity = field(
        metadata={
            "name": "ud_mf_at_cloud_top",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "kg/m**2",
            "intent": "out",
        }
    )
    psp: Quantity = field(
        metadata={
            "name": "surface_pressure",
            "dims": [X_DIM, Y_DIM],
            "units": "Pa",
            "intent": "in",
        }
    )
    rn: Quantity = field(
        metadata={
            "name": "convective_rain",
            "dims": [X_DIM, Y_DIM],
            "units": "m",
            "intent": "out",
        }
    )
    kcnv: Quantity = field(
        metadata={
            "name": "flag_for_deep_convection",
            "dims": [X_DIM, Y_DIM],
            "units": "",
            "intent": "inout",
            "dtype": Int,
        }
    )
    kbot: Quantity = (
        field(
            metadata={
                "name": "index_for_cloud_base",
                "dims": [X_DIM, Y_DIM],
                "units": "",
                "intent": "out",
                "dtype": Int,
            }
        ),
    )
    ktop: Quantity = (
        field(
            metadata={
                "name": "index_for_cloud_top",
                "dims": [X_DIM, Y_DIM],
                "units": "",
                "intent": "out",
                "dtype": Int,
            }
        ),
    )
    garea: Quantity = field(
        metadata={
            "name": "grid_area",
            "dims": [X_DIM, Y_DIM],
            "units": "m**2",
            "intent": "in",
        }
    )
    islimsk: Quantity = field(
        metadata={
            "name": "land_mask",
            "dims": [X_DIM, Y_DIM],
            "units": "",
            "intent": "in",
        }
    )

    @classmethod
    def init_zeros(cls, quantity_factory) -> "SAMFShalConvState":
        initial_arrays = {}
        for _field in fields(cls):
            if "dims" in _field.metadata.keys():
                if "dtype" in _field.metadata.keys():
                    dtype = _field.metadata["dtype"]
                else:
                    dtype = Float
                initial_arrays[_field.name] = quantity_factory.zeros(
                    _field.metadata["dims"],
                    _field.metadata["units"],
                    dtype=dtype,
                )
        return cls(**initial_arrays)

    @classmethod
    def init_from_storages(
        cls,
        storages: Mapping[str, Any],
        sizer: GridSizer,
        quantity_factory: QuantityFactory,
    ) -> "SAMFShalConvState":
        inputs: Dict[str, Quantity] = {}
        for _field in fields(cls):
            if "dims" in _field.metadata.keys():
                dims = _field.metadata["dims"]
                if _field.name in storages.keys():
                    quantity = Quantity(
                        storages[_field.name],
                        dims,
                        _field.metadata["units"],
                        origin=sizer.get_origin(dims),
                        extent=sizer.get_extent(dims),
                    )
                else:
                    quantity = quantity_factory.zeros(dims, _field.metadata["units"])
                inputs[_field.name] = quantity
        return cls(**inputs)

    @property
    def xr_dataset(self):
        data_vars = {}
        for name, field_info in self.__dataclass_fields__.items():
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
