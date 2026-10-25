"""Runtime settings read from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    api_key: str = "dev-key"
    db_path: str = "experiments/mesh.sqlite"
    model_path: str = "models/link_gnn_v1.pt"
    rate_limit_per_min: int = 60

    def __post_init__(self) -> None:
        if isinstance(self.rate_limit_per_min, bool) or not isinstance(self.rate_limit_per_min, int) \
                or self.rate_limit_per_min < 1:
            raise ValueError("rate_limit_per_min must be a positive integer")

    @staticmethod
    def from_env() -> "Settings":
        return Settings(
            api_key=os.getenv("MESH_API_KEY", "dev-key"),
            db_path=os.getenv("MESH_DB_PATH", "experiments/mesh.sqlite"),
            model_path=os.getenv("MESH_MODEL_PATH", "models/link_gnn_v1.pt"),
            rate_limit_per_min=int(os.getenv("MESH_RATE_LIMIT_PER_MIN", "60")),
        )
