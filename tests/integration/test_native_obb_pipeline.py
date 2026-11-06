"""Three-stage handoff using the real resident process and native SX service."""

import asyncio
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from app.application.inference.obb_service import ObbInspectionService
from app.application.inference.worker import InferenceWorker
from app.application.ingest.worker import IngestWorker
from app.application.publisher.service import PipelinePublisherService
from app.application.publisher.worker import PublisherWorker
from app.application.shared.fallback_result import FallbackResultBuilder
from app.core.config import AppConfig
from app.domain.entities.obb_result import ObbBatchResult, ObbDetection
from app.domain.entities.pipeline_job import PipelineJobStatus
from app.infrastructure.inference.resident_process import ResidentObbRuntime
from app.infrastructure.repositories.pipeline_job_repository import (
    PipelineJobRepository,
)


class IntegrationObbModel:
    def __init__(self, settings):
        self.model_version = "integration-model"
        self.device = "cpu"

    def infer(self, images, deadline_at):
        polygon = ((0.0, 0.0), (3.0, 0.0), (3.0, 3.0), (0.0, 3.0))
        return ObbBatchResult(
            {
                key: tuple(
                    ObbDetection("Solder_Paste", conf, polygon)
                    for conf in (0.3, 0.4, 0.8)
                )
                for key in images
            },
            1.0,
            self.model_version,
            self.device,
        )


def test_native_workers_publish_two_jobs_with_one_resident_model(tmp_path):
    config = AppConfig(
        paths={
            "external_output_root": str(tmp_path / "return"),
            "backup_output_root": str(tmp_path / "backup"),
        },
        processing={"image_name_source_column": "image"},
        inference={"mode": "yolo_obb", "obb": {"device": "cpu"}},
        defect_rules={
            "anomaly_threshold": 0.9,
            "high_cover_threshold": 180,
            "short_distance_threshold": 6.8,
            "low_vol_offset": -10,
            "high_vol_offset": 20,
            "high_paste_height_threshold": 200,
        },
        pipeline={
            "enabled": True,
            "watch_root": str(tmp_path / "share"),
            "database_path": str(tmp_path / "jobs.sqlite3"),
            "result_root": str(tmp_path / "results"),
            "source_settle_seconds": 0,
            "publisher_lease_seconds": 0.5,
            "publisher_heartbeat_interval_seconds": 0.1,
        },
        reliability={"primary_publish_reserve_seconds": 1},
        output={"primary_path_layout": "machine_return"},
    )
    repo = PipelineJobRepository(config.pipeline.database_path)
    ingest = IngestWorker(
        config=config,
        repository=repo,
        worker_id="ingest",
        watch_root=Path(config.pipeline.watch_root),
        backup_root=Path(config.paths.backup_output_root),
        staging_root=None,
    )
    publisher = PublisherWorker(
        config=config,
        repository=repo,
        worker_id="publisher",
        publisher=PipelinePublisherService(config=config),
        fallback_builder=FallbackResultBuilder(
            config=config, result_root=tmp_path / "results"
        ),
    )

    async def run():
        async with ResidentObbRuntime(
            config.inference.obb, model_factory=IntegrationObbModel
        ) as runtime:
            service = ObbInspectionService(
                config=config,
                result_root=tmp_path / "results",
                runtime=runtime,
            )
            inference = InferenceWorker(
                config=config,
                repository=repo,
                service=service,
                worker_id="infer",
                prepare_runtime=runtime.start,
            )
            pid = runtime.pid
            for job_id in ("20261106120000", "20261106120001"):
                source = tmp_path / "share" / job_id
                source.mkdir(parents=True)
                csv_name = f"{job_id}.csv"
                pd.DataFrame(
                    [
                        {
                            "component_name": "C1174",
                            "image": "a.jpg",
                            "is_pass": 1,
                            "insp_vol": 60,
                            "insp_area": 160,
                            "insp_hei": 100,
                            "insp_x": -0.04,
                            "insp_y": 0.04,
                            "vol_l_ng": 160,
                            "vol_h_ng": 60,
                            "area_l_ng": 60,
                            "area_h_ng": 160,
                            "hei_l_ng": 160,
                            "hei_h_ng": 60,
                            "shift_x_ng": 0.04,
                            "shift_y_ng": 0.04,
                        },
                        {
                            "component_name": "U0500",
                            "image": "b.jpg",
                            "is_pass": 1,
                            "insp_vol": 161,
                            "insp_area": 100,
                            "insp_hei": 100,
                            "insp_x": 0,
                            "insp_y": 0,
                            "vol_l_ng": 40,
                            "vol_h_ng": 160,
                            "area_l_ng": 40,
                            "area_h_ng": 160,
                            "hei_l_ng": 40,
                            "hei_h_ng": 160,
                            "shift_x_ng": 0.03,
                            "shift_y_ng": 0.03,
                        },
                    ]
                ).to_csv(source / csv_name, index=False)
                for name in ("a.jpg", "b.jpg"):
                    assert cv2.imwrite(
                        str(source / name), np.zeros((8, 8, 3), dtype=np.uint8)
                    )
                for _ in range(4):
                    ingest.run_once(target_date=date(2026, 11, 6))
                    if repo.get(job_id).status is PipelineJobStatus.READY:
                        break
                assert repo.get(job_id).status is PipelineJobStatus.READY
                assert await inference.run_once()
                assert runtime.pid == pid
                assert repo.get(job_id).status is PipelineJobStatus.RESULT_READY
                assert publisher.run_once()
                assert publisher.run_once()
                assert repo.get(job_id).status is PipelineJobStatus.DONE
                returned = pd.read_csv(tmp_path / "return" / csv_name)
                assert returned["is_pass"].tolist() == [22, 23]
                assert returned["vol_l_ng"].tolist() == [160, 40]
                assert returned["vol_h_ng"].tolist() == [60, 160]

    try:
        asyncio.run(run())
    finally:
        ingest.close()
        publisher.close()
