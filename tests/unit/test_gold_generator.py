import json

import pytest

from benchmark.gold.generator import GoldGenerator, GoldGeneratorConfig, write_dataset_to_file
from benchmark.gold.schema import GoldDataset
from benchmark.time.simulated_clock import SimulatedClock


@pytest.mark.unit
def test_gold_generator_deterministic_replay():
    clock = SimulatedClock()
    cfg = GoldGeneratorConfig(seed=12345, users=5, days=3, events_per_day=2)
    gen1 = GoldGenerator(cfg, clock)
    ds1 = gen1.generate()
    # regenerate with same seed and config
    clock2 = SimulatedClock()
    gen2 = GoldGenerator(cfg, clock2)
    ds2 = gen2.generate()
    # serialized dumps must match exactly for deterministic replay
    j1 = json.dumps(ds1.model_dump(), sort_keys=True)
    j2 = json.dumps(ds2.model_dump(), sort_keys=True)
    assert j1 == j2


@pytest.mark.unit
def test_write_and_validate_tmpfile(tmp_path):
    clock = SimulatedClock()
    cfg = GoldGeneratorConfig(seed=1, users=2, days=1, events_per_day=1)
    gen = GoldGenerator(cfg, clock)
    ds = gen.generate()
    out = tmp_path / "generated.json"
    write_dataset_to_file(ds, str(out))
    # read back and validate via pydantic
    loaded = json.loads(out.read_text(encoding="utf-8"))
    GoldDataset.model_validate(loaded)
    assert loaded["scenario"].startswith("generated")
