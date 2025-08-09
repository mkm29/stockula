#!/usr/bin/env python3
"""Seed AutoTS models into the database directly from the AutoTS library."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlmodel import Session, SQLModel, create_engine

from stockula.data.autots_repository import AutoTSRepository


def main():
    """Main function to seed AutoTS models."""
    # Database path
    db_path = Path(__file__).parent.parent / "stockula.db"
    engine = create_engine(f"sqlite:///{db_path}")

    # Create tables if they don't exist
    SQLModel.metadata.create_all(engine)

    # Seed the database directly from AutoTS
    with Session(engine) as session:
        repo = AutoTSRepository(session)
        models_count, presets_count = repo.seed_from_autots()
        print(f"Successfully seeded {models_count} models and {presets_count} presets from AutoTS")

        # Show some statistics
        print("\nDatabase contents:")
        print(f"- Total models: {len(repo.get_all_models())}")
        print(f"- Total presets: {len(repo.get_all_presets())}")

        # Show categories
        categories = ["univariate", "multivariate", "slow", "gpu", "probabilistic"]
        for category in categories:
            count = len(repo.get_models_by_category(category))
            print(f"- {category.capitalize()} models: {count}")


if __name__ == "__main__":
    main()
