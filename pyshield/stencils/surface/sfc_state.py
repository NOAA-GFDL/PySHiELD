from dataclasses import InitVar, dataclass, field, fields
from typing import Any, Dict, Mapping

import xarray as xr

import ndsl.dsl.gt4py_utils as gt_utils
from ndsl import GridSizer, Quantity, QuantityFactory
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.dsl.typing import Bool, Float, Int


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

    stc: Quantity = field(
        metadata={
            "name": "soil_temperature_content",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "K",
            "intent": "inout",
        }
    )

    qsfc: Quantity = field(
        metadata={
            "name": "surface_specific_humidity",
            "dims": [X_DIM, Y_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )

    snowd: Quantity = field(
        metadata={
            "name": "snow_depth_water_equivalent",
            "dims": [X_DIM, Y_DIM],
            "units": "mm",
            "intent": "inout",
        }
    )

    zorl: Quantity = field(
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

    uustar: Quantity = field(
        metadata={
            "name": "boundary_layer_param",
            "dims": [X_DIM, Y_DIM],
            "units": "",
            "intent": "out",
        }
    )

    slmsk: Quantity = field(
        metadata={
            "name": "sea_land_ice_mask",
            "dims": [X_DIM, Y_DIM],
            "units": "",
            "intent": "in",
            "type": Int,
        }
    )

    vegtype: Quantity = field(
        metadata={
            "name": "vegetation_type",
            "dims": [X_DIM, Y_DIM],
            "units": "",
            "intent": "in",
            "type": Int,
        }
    )

    vfrac: Quantity = field(
        metadata={
            "name": "vegetation_fraction",
            "dims": [X_DIM, Y_DIM],
            "units": "",
            "intent": "in",
        }
    )

    shdmax: Quantity = field(
        metadata={
            "name": "max_fractional_green_vegetation_cover",
            "dims": [X_DIM, Y_DIM],
            "units": "",
            "intent": "in",
        }
    )

    ffmm: Quantity = field(
        metadata={
            "name": "fm_PBL_parameter",
            "dims": [X_DIM, Y_DIM],
            "units": "",
            "intent": "inout",
        }
    )

    ffhh: Quantity = field(
        metadata={
            "name": "fh_PBL_parameter",
            "dims": [X_DIM, Y_DIM],
            "units": "",
            "intent": "inout",
        }
    )

    f10m: Quantity = field(
        metadata={
            "name": "sigma1_10m_wind_ratio",
            "dims": [X_DIM, Y_DIM],
            "units": "",
            "intent": "inout",
        }
    )

    sfcemis: Quantity = field(
        metadata={
            "name": "sfc_lw_emissivity_fraction",
            "dims": [X_DIM, Y_DIM],
            "units": "",
            "intent": "inout",
        }
    )

    srflag: Quantity = field(
        metadata={
            "name": "rain_snow_precipitation_flag",
            "dims": [X_DIM, Y_DIM],
            "units": "",
            "intent": "inout",
            "type": Bool,
        }
    )

    hice: Quantity = field(
        metadata={
            "name": "sea_ice_thickness",
            "dims": [X_DIM, Y_DIM],
            "units": "unknown",
            "intent": "inout",
        }
    )

    fice: Quantity = field(
        metadata={
            "name": "ice_fraction_over_open_water_grid",
            "dims": [X_DIM, Y_DIM],
            "units": "",
            "intent": "inout",
        }
    )

    tisfc: Quantity = field(
        metadata={
            "name": "surface_temperature_over_ice_fraction",
            "dims": [X_DIM, Y_DIM],
            "units": "K",
            "intent": "inout",
        }
    )

    weasd: Quantity = field(
        metadata={
            "name": "water_equiv_accumulated_snow_depth",
            "dims": [X_DIM, Y_DIM],
            "units": "kg/m**2",
            "intent": "inout",
        }
    )

    tprcp: Quantity = field(
        metadata={
            "name": "total_precip",
            "dims": [X_DIM, Y_DIM],
            "units": "unknown",
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
                dtype = (
                    _field.metadata["type"]
                    if "type" in _field.metadata.keys()
                    else Float
                )
                initial_arrays[_field.name] = quantity_factory.zeros(
                    _field.metadata["dims"],
                    _field.metadata["units"],
                    dtype=dtype,
                )
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
