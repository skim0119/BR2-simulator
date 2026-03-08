from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


class SegmentConfig(BaseModel):
    model_config = ConfigDict(strict=False, extra="allow")

    rod_order: list[str]
    base_position: list[list[float]]
    y_rotation: list[float] = Field(
        validation_alias=AliasChoices("y-rotation", "y_rotation"),
        serialization_alias="y-rotation",
    )

    @model_validator(mode="after")
    def check_lengths(self):
        count = len(self.rod_order)
        if len(self.base_position) != count or len(self.y_rotation) != count:
            raise ValueError(
                "Each segment must provide matching lengths for "
                "rod_order, base_position, and y-rotation."
            )
        return self


class AssemblyConfig(BaseModel):
    model_config = ConfigDict(strict=False, extra="allow")

    CaseID: int | None = None
    Date: str | None = None
    Info: str | None = None
    Segments: dict[str, SegmentConfig]
    Activations: dict[str, list[tuple[str, int]]]

    @model_validator(mode="after")
    def check_activation_targets(self):
        for action_name, targets in self.Activations.items():
            for segment_name, rod_idx in targets:
                if segment_name not in self.Segments:
                    raise ValueError(
                        f"Activation '{action_name}' references unknown segment "
                        f"'{segment_name}'."
                    )
                if rod_idx < 0 or rod_idx >= len(self.Segments[segment_name].rod_order):
                    raise ValueError(
                        f"Activation '{action_name}' references invalid rod index "
                        f"{rod_idx} in segment '{segment_name}'."
                    )
        return self
