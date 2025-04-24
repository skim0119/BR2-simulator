from typing import TYPE_CHECKING
from .protocol import ConfigProtocol

from pydantic import BaseModel, ConfigDict, Field, computed_field


class RodConfig(BaseModel):
    model_config = ConfigDict(strict=False)

    Info: str

    n_elements: int
    direction: list[float]
    normal: list[float]
    base_length: float
    outer_radius: float
    inner_radius: float
    density: float
    damping_constant: float
    youngs_modulus: float
    shear_modulus: float
    alpha: float
    beta: float


class RodLibrary(BaseModel):
    model_config = ConfigDict(strict=False)

    Info: str

    DefaultParams: RodConfig

    # Custom Parameters
    Rods: dict[str, dict[str, str | float | int | list[float]]]

    def get_rod(self, rod_name: str) -> RodConfig:
        assert rod_name in self.Rods, f"Rod {rod_name} not found in RodLibrary"

        custom_fields = self.Rods[rod_name]
        return self.DefaultParams.model_copy(update=custom_fields, deep=True)


if TYPE_CHECKING:
    _: ConfigProtocol = RodLibrary()
