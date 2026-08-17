"""Gold Oracle — read-only repository for gold truth data.

Loads gold datasets and provides query-time access to expected results.
The oracle never modifies data and has no dependency on memory modules.

Supports loading:
- Our native GoldDataset JSON format
- LoCoMo dataset format (locomo10.json) — detected automatically
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark.exceptions.evaluation_errors import GoldDatasetError
from benchmark.gold.normalizer import normalize_timestamps
from benchmark.gold.schema import GoldDataset, GoldExpectedResult, GoldQuery


class GoldOracle:
    """Read-only repository for gold truth datasets.

    Loads gold datasets from JSON files and provides access to:
    - Memory events to inject per day
    - Expected results per query
    - Evaluation criteria

    Automatically detects whether the input is LoCoMo format or our
    native GoldDataset format and handles both transparently.

    This class has NO dependency on memory modules or the orchestrator.
    """

    def __init__(self) -> None:
        """Initialize an empty gold oracle."""
        self._datasets: dict[str, GoldDataset] = {}
        self._normalization_metadata: dict[str, dict] = {}

    def load_dataset(
        self,
        dataset_path: Path,
        scenario_name: str = "locomo",
    ) -> GoldDataset:
        """Load a gold dataset from a JSON file.

        Automatically detects LoCoMo format (array of conversations with
        'conversation' and 'qa' fields) vs our native GoldDataset format.

        Timestamp normalization is applied unconditionally after loading:
        event days are shifted forward so that the newest memory is within
        ``TARGET_GAP_DAYS`` of the first query.  This ensures decay functions
        produce meaningful differentiation between old and new memories instead
        of collapsing to the archival floor for all events.

        Args:
            dataset_path: Path to the dataset JSON file.
            scenario_name: Scenario name (used when loading LoCoMo format).

        Returns:
            The loaded, validated, and timestamp-normalized GoldDataset.

        Raises:
            GoldDatasetError: If the file cannot be loaded or fails validation.
        """
        raw_data = self._read_json(dataset_path)

        if self._is_locomo_format(raw_data):
            dataset = self._load_locomo(raw_data, scenario_name, dataset_path)
        else:
            dataset = self._validate_dataset(raw_data, source=str(dataset_path))

        dataset, norm_meta = normalize_timestamps(dataset)
        self._normalization_metadata[dataset.scenario] = norm_meta

        self._datasets[dataset.scenario] = dataset
        return dataset

    def get_normalization_metadata(self, scenario_name: str) -> dict:
        """Return timestamp-normalization metadata for a loaded scenario.

        Args:
            scenario_name: The scenario name passed to ``load_dataset``.

        Returns:
            Dict with keys ``applied``, ``delta_days``, etc.  Returns
            ``{"applied": False}`` if the scenario has not been loaded.
        """
        return self._normalization_metadata.get(scenario_name, {"applied": False})

    def get_dataset(self, scenario_name: str) -> GoldDataset:
        """Retrieve a previously loaded gold dataset by scenario name.

        Args:
            scenario_name: The name of the scenario.

        Returns:
            The GoldDataset for the given scenario.

        Raises:
            GoldDatasetError: If the scenario has not been loaded.
        """
        if scenario_name not in self._datasets:
            raise GoldDatasetError(f"Gold dataset not loaded for scenario: {scenario_name}")
        return self._datasets[scenario_name]

    def get_expected_result(
        self, scenario_name: str, query_text: str, day: int
    ) -> GoldExpectedResult:
        """Get the expected result for a specific query on a specific day.

        Args:
            scenario_name: The scenario name.
            query_text: The query string.
            day: The simulated day.

        Returns:
            The expected result for the query.

        Raises:
            GoldDatasetError: If no matching query is found.
        """
        dataset = self.get_dataset(scenario_name)
        for gold_query in dataset.queries:
            if gold_query.day == day and gold_query.query == query_text:
                return gold_query.expected
        raise GoldDatasetError(
            f"No gold query found for scenario={scenario_name}, query='{query_text}', day={day}"
        )

    def get_queries_for_day(self, scenario_name: str, day: int) -> list[GoldQuery]:
        """Get all queries scheduled for a specific simulated day.

        Args:
            scenario_name: The scenario name.
            day: The simulated day.

        Returns:
            List of GoldQuery objects for the given day.
        """
        dataset = self.get_dataset(scenario_name)
        return [query for query in dataset.queries if query.day == day]

    def list_loaded_scenarios(self) -> list[str]:
        """List all loaded scenario names.

        Returns:
            Sorted list of scenario names.
        """
        return sorted(self._datasets.keys())

    def _is_locomo_format(self, data: Any) -> bool:
        """Detect whether data is in LoCoMo format.

        LoCoMo format is either:
        - A JSON array where items have 'conversation' and 'qa' fields
        - A dict with a 'data' key containing such an array
        """
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                return "conversation" in data[0] or "qa" in data[0]
            return False

        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                return self._is_locomo_format(data["data"])
            # Could be a single LoCoMo sample
            return "conversation" in data and "qa" in data

        return False

    def _load_locomo(
        self,
        raw_data: Any,
        scenario_name: str,
        source_path: Path,
    ) -> GoldDataset:
        """Load LoCoMo format data directly into GoldDataset.

        Imports the LoCoMo loader on demand (keeps oracle independent of
        loader internals while still supporting native format).
        """
        from benchmark.gold.locomo_loader import LoCoMoLoader

        loader = LoCoMoLoader()
        try:
            return loader.convert_raw(raw_data, scenario_name)
        except Exception as load_error:
            raise GoldDatasetError(
                f"Failed to load LoCoMo dataset ({source_path}): {load_error}"
            ) from load_error

    def _read_json(self, file_path: Path) -> Any:
        """Read and parse a JSON file.

        Args:
            file_path: Path to the JSON file.

        Returns:
            Parsed JSON data (dict or list).

        Raises:
            GoldDatasetError: If the file cannot be read or parsed.
        """
        if not file_path.exists():
            raise GoldDatasetError(f"Gold dataset file not found: {file_path}")

        try:
            with file_path.open("r", encoding="utf-8") as file_handle:
                return json.load(file_handle)
        except json.JSONDecodeError as json_error:
            raise GoldDatasetError(
                f"Failed to parse gold dataset JSON: {file_path}"
            ) from json_error
        except OSError as io_error:
            raise GoldDatasetError(f"Failed to read gold dataset file: {file_path}") from io_error

    def _validate_dataset(self, data: dict[str, Any], source: str) -> GoldDataset:
        """Validate raw data against the GoldDataset schema.

        Args:
            data: Raw JSON data.
            source: Description of the data source (for error messages).

        Returns:
            Validated GoldDataset instance.

        Raises:
            GoldDatasetError: If validation fails.
        """
        try:
            return GoldDataset.model_validate(data)
        except Exception as validation_error:
            raise GoldDatasetError(
                f"Gold dataset validation failed ({source}): {validation_error}"
            ) from validation_error
