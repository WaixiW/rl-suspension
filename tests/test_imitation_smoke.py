from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from rl_suspension.imitation.collect import collect_dataset
from rl_suspension.imitation.train_bc import train_behavior_cloning
from rl_suspension.imitation.train_dagger import train_dagger


@pytest.mark.integration
def test_collect_bc_and_one_dagger_round(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    bc_dir = tmp_path / "bc"
    dagger_dir = tmp_path / "dagger"

    summary = collect_dataset(
        output_dir=dataset_dir,
        episodes=2,
        teacher="preview",
        seed_start=0,
        curriculum_stage=1,
    )
    assert summary["transitions"] > 0
    assert summary["phase_counts"]["rear_contact"] > 0
    assert summary["phase_counts"]["recovery"] > 0

    metrics = train_behavior_cloning(
        dataset_dir=dataset_dir,
        output_dir=bc_dir,
        loss_name="huber",
        epochs=1,
        batches_per_epoch=1,
        batch_size=8,
        seed=0,
        device="cpu",
        curriculum_stage=1,
    )
    model_path = Path(metrics["model_path"])
    assert model_path.exists()

    dagger = train_dagger(
        dataset_dir=dataset_dir,
        initial_model_path=model_path,
        output_dir=dagger_dir,
        rounds=1,
        episodes_per_round=1,
        betas=[0.0],
        teacher="preview",
        allow_unqualified=True,
        bc_loss="huber",
        bc_epochs=1,
        bc_batches_per_epoch=1,
        seed=0,
        curriculum_stage=1,
        device="cpu",
    )
    assert Path(dagger["final_model_path"]).exists()
