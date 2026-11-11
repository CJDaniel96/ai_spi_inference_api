"""Native OBB appearance followed by measurements against each CSV row's limits."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from dataclasses import replace
from pathlib import Path

from app.application.ingest.io import inspect_source, ping
from app.application.shared.fallback_result import FallbackResultBuilder
from app.application.shared.result_artifacts import ResultArtifactWriter
from app.core.config import AppConfig
from app.core.errors import CsvSchemaError
from app.domain.entities.obb_result import ObbBatchResult, ObbDetection
from app.domain.entities.pipeline_result import PipelineOutcome, PipelineResultManifest
from app.domain.services.component_judge import RULE_ORDER, ComponentJudge
from app.domain.services.csv_merger import CsvMerger
from app.infrastructure.async_io import call_io
from app.infrastructure.deadline_io import DeadlineIO
from app.infrastructure.input.sinic_folder_input import SinicFolderInput
from app.infrastructure.repositories.csv_repository import CsvRepository


class ObbInspectionService:
    def __init__(
        self,
        *,
        config: AppConfig,
        result_root: Path,
        runtime,
        clock=time.time,
        io_factory=DeadlineIO,
    ) -> None:
        self._config = config
        self._runtime = runtime
        self._clock = clock
        self._io_factory = io_factory
        self._result_root = result_root
        self._site_id = config.identity.site
        self._thresholds = dict(config.inference.obb.class_thresholds)
        self._judge = ComponentJudge(self._thresholds)
        self._policy = self._judge.policy_definition()
        definition = {
            "site_id": self._site_id,
            "policy": self._policy,
            "rule_order": list(RULE_ORDER),
            "class_thresholds": self._thresholds,
            "model_conf": config.inference.obb.conf,
        }
        self._version = hashlib.sha256(
            json.dumps(definition, sort_keys=True).encode("utf-8")
        ).hexdigest()
        self._input = SinicFolderInput(config.processing.image_extensions)
        self._csv = CsvRepository()
        self._merger = CsvMerger()
        self._artifacts = ResultArtifactWriter(result_root)
        self._fallback = FallbackResultBuilder(
            config=config, result_root=result_root, clock=clock
        )

    def _metadata(self, result: ObbBatchResult | None = None) -> dict:
        return {
            "site_id": self._site_id,
            "rule_version": self._version,
            "policy": self._policy,
            "limit_source": "csv",
            "rule_order": list(RULE_ORDER),
            "class_thresholds": self._thresholds,
            "model_conf": self._config.inference.obb.conf,
            "model_version": result.model_version if result else None,
            "device": result.device if result else self._config.inference.obb.device,
        }

    def _inputs(self, folder: Path):
        files = self._input.load(folder)
        if not files.csv_files:
            raise CsvSchemaError(f"No CSV files found in folder: {folder}")
        frames = []
        images = {}
        for source in files.csv_files:
            frame = self._merger.add_image_name_column(
                self._csv.read_for_output(source),
                template=self._config.processing.image_name_template,
                csv_stem=source.stem,
            )
            if frame.empty:
                raise CsvSchemaError(f"CSV has no data rows: {source.name}")
            for key in frame["img_name"]:
                image = folder / key
                if (
                    not key
                    or key in images
                    or Path(key).name != key
                    or "\\" in key
                    or image.resolve().parent != folder.resolve()
                    or not image.is_file()
                ):
                    raise CsvSchemaError(
                        f"Missing, duplicate or invalid image key: {key!r}"
                    )
                images[key] = str(image)
            frames.append((source, frame))
        return frames, images

    @staticmethod
    def _validate_result(result: ObbBatchResult, images: dict) -> None:
        if set(result.results) != set(images):
            raise ValueError(
                "Required OBB results do not cover exactly the expected images"
            )
        if not math.isfinite(result.inference_ms) or result.inference_ms < 0:
            raise ValueError("Invalid OBB inference time")
        for boxes in result.results.values():
            if not isinstance(boxes, tuple) or not all(
                isinstance(b, ObbDetection) for b in boxes
            ):
                raise ValueError("Invalid OBB detection result")

    async def prepare(self, *, job_id: str, staged_folder: Path):
        lane = self._io_factory()

        async def operation(function, *args):
            return await call_io(
                lane,
                function,
                self._config,
                self._result_root,
                *args,
                deadline_at=time.time() + self._config.runtime.io_timeout_seconds,
            )

        try:
            await call_io(
                lane,
                ping,
                deadline_at=time.time()
                + self._config.runtime.io_startup_timeout_seconds,
            )
            # Also enforce the policy on an already verified backup restored
            # after a restart, before loading inputs or invoking the model.
            files, _, _ = await call_io(
                lane,
                inspect_source,
                self._config.processing.image_extensions,
                staged_folder,
                deadline_at=time.time() + self._config.runtime.io_timeout_seconds,
            )
            limit = self._config.processing.folder_images_num_threshold
            if files.image_count > limit:
                return await operation(
                    build_fallback,
                    job_id,
                    staged_folder,
                    "images_exceed_threshold",
                    f"Image count {files.image_count} exceeds configured limit {limit}",
                )
            try:
                frames, images = await operation(load_inputs, staged_folder)
            except (ValueError, CsvSchemaError) as exc:
                return await operation(
                    build_fallback, job_id, staged_folder, "invalid_input", str(exc)
                )
            try:
                timeout = self._config.inference.obb.inference_timeout_seconds
                result = await asyncio.wait_for(
                    self._runtime.infer(images, deadline_at=time.time() + timeout),
                    timeout,
                )
                self._validate_result(result, images)
            except TimeoutError as exc:
                self._runtime.close()
                return await operation(
                    build_fallback,
                    job_id,
                    staged_folder,
                    "required_model_timeout",
                    str(exc),
                )
            except Exception as exc:
                return await operation(
                    build_fallback,
                    job_id,
                    staged_folder,
                    "required_model_failure",
                    str(exc),
                )
            try:
                return await operation(evaluate_result, job_id, frames, result)
            except ValueError as exc:
                return await operation(
                    build_fallback,
                    job_id,
                    staged_folder,
                    "invalid_measurement",
                    str(exc),
                )
        finally:
            lane.close()

    def _evaluate(self, job_id, frames, result):
        artifact_dir = self._artifacts.new_artifact_dir(job_id)
        csv_results = []
        counts = {"pass_count": 0, "fail_count": 0}
        for source, frame in frames:
            audits = []
            for row in frame.to_dict("records"):
                boxes = result.results[row["img_name"]]
                decision = self._judge.judge(row, boxes)
                audits.append(self._judge.audit_row(decision, boxes))
                counts["pass_count" if decision.passed else "fail_count"] += 1
                if decision.defect:
                    counts[decision.defect] = counts.get(decision.defect, 0) + 1
            out = frame.copy()
            for column in audits[0]:
                out[column] = [audit[column] for audit in audits]
            out["ai_site_id"] = self._site_id
            out["ai_rule_version"] = self._version
            out["ai_model_version"] = result.model_version
            csv_results.append(
                self._artifacts.persist_csv_result(artifact_dir, source, out)
            )
        manifest = PipelineResultManifest(
            job_id=job_id,
            outcome=PipelineOutcome.NORMAL,
            created_at=self._clock(),
            csv_results=tuple(csv_results),
            counts=counts,
            model_timings_ms={"yolo_obb": result.inference_ms},
            inspection_metadata=self._metadata(result),
        )
        return manifest, manifest.write_atomic(artifact_dir / "manifest.json")


def load_inputs(config, result_root, folder):
    return ObbInspectionService(
        config=config, result_root=result_root, runtime=None
    )._inputs(folder)


def evaluate_result(config, result_root, job_id, frames, result):
    return ObbInspectionService(
        config=config, result_root=result_root, runtime=None
    )._evaluate(job_id, frames, result)


def build_fallback(config, result_root, job_id, folder, reason, error):
    service = ObbInspectionService(config=config, result_root=result_root, runtime=None)
    manifest, path = service._fallback.build_csv_fallback(
        job_id=job_id,
        csv_folder=folder,
        reason=reason,
        errors=(error,) if error else (),
    )
    manifest = replace(manifest, inspection_metadata=service._metadata())
    manifest.write_atomic(path)
    return manifest, path
