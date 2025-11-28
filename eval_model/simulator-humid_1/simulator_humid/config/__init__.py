"""Configuration utilities for duct and piping geometry."""

from .duct_geometry import (
    DuctSegment,
    BranchGeometry,
    ZONE_BRANCH_GEOMETRY,
    SUPPLY_TRUNK_GEOMETRY,
    RETURN_TRUNK_GEOMETRY,
)

__all__ = [
    "DuctSegment",
    "BranchGeometry",
    "ZONE_BRANCH_GEOMETRY",
    "SUPPLY_TRUNK_GEOMETRY",
    "RETURN_TRUNK_GEOMETRY",
]
