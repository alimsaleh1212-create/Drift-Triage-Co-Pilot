"""Demo helper: inject shifted predictions into the rolling-window table.

Usage:
    uv run python scripts/inject_drift.py --feature euribor3m --shift 2.0 --n 500
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def inject(feature: str, shift: float, n: int) -> None:
    import asyncpg

    from core.settings import get_settings

    settings = get_settings()
    conn = await asyncpg.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        user=settings.postgres_user,
        password=settings.postgres_password,
        database=settings.postgres_db,
    )

    # Build a representative feature vector; shift the target feature
    base_features = {
        "age": 40,
        "job": "admin.",
        "marital": "married",
        "education": "university.degree",
        "default": "no",
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "month": "may",
        "day_of_week": "mon",
        "campaign": 1,
        "pdays": 999,
        "previous": 0,
        "poutcome": "nonexistent",
        "emp.var.rate": -1.8,
        "cons.price.idx": 92.9,
        "cons.conf.idx": -46.2,
        "euribor3m": 1.3,
        "nr.employed": 5099.1,
        "was_previously_contacted": 0,
    }

    # Apply shift to the target feature
    if feature in base_features and isinstance(base_features[feature], (int, float)):
        base_features[feature] = float(base_features[feature]) + shift  # type: ignore[assignment]

    print(f"Injecting {n} rows with {feature} shifted by +{shift}...")
    for i in range(n):
        features = {**base_features}
        if i % 10 == 0:
            features["job"] = "retired"  # also shift categorical
        await conn.execute(
            "INSERT INTO predictions (id, features, label, probability, created_at) "
            "VALUES ($1, $2::jsonb, $3, $4, now())",
            str(uuid4()),
            json.dumps(features),
            1,
            0.85,
        )

    await conn.close()
    print(f"Done. Injected {n} shifted predictions.")
    print("Wait up to 60s for the drift cache TTL to expire, then check GET /api/v1/drift/report")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", default="euribor3m")
    parser.add_argument("--shift", type=float, default=2.0)
    parser.add_argument("--n", type=int, default=500)
    args = parser.parse_args()
    asyncio.run(inject(args.feature, args.shift, args.n))
