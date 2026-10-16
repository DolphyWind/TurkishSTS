import csv
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Union

import mlflow
from mlflow_logger import MLFlowLogger, NullLogger


class MetricLogger:
    def __init__(
        self,
        is_main_process: bool,
        experiment_name: str,
        run_name: str,
        run_id: Optional[str],
        out_dir: Path,
    ) -> None:
        self.is_main_process: bool = is_main_process
        self.mlflow_logger: Union[MLFlowLogger, NullLogger] = NullLogger()
        self.out_path: Path = out_dir / "metrics.csv"

        self.experiment_name: str = experiment_name
        self.run_name: str = run_name
        self.run_id: Optional[str] = run_id
        self.ctx = nullcontext()

        if self.is_main_process:
            mlflow.set_experiment(experiment_name)
            self.ctx = mlflow.start_run(
                run_name=run_name,
                run_id=self.run_id,
            )
            self.mlflow_logger = MLFlowLogger(run_id=self.run_id)

        self.metrics: DefaultDict[int, Dict[str, float]] = defaultdict(dict)
        self.load_metrics()

    def set_run_id(self, run_id: Optional[str]) -> None:
        if not self.is_main_process:
            return

        self.run_id = run_id
        self.mlflow_logger.run_id = run_id

    def load_metrics(self) -> None:
        if not self.is_main_process or not self.out_path.exists():
            return

        with open(
            self.out_path,
            "r",
            newline="",
        ) as f:
            reader: csv.DictReader = csv.DictReader(f)
            for row in reader:
                step: int = int(row.pop("step"))
                for name, value in row.items():
                    if value == "":
                        continue
                    self.metrics[step][name] = float(value)

    def log_cfg(self, cfg) -> None:
        if not self.is_main_process or self.run_id is not None:
            return

        for k, v in asdict(cfg).items():
            self.mlflow_logger.log_param(k, v)

    def log_metric(
        self,
        name: str,
        value: float,
        step: int,
        **kwargs,
    ) -> None:
        if not self.is_main_process:
            return

        self.mlflow_logger.log_metric(
            name,
            value,
            step=step,
            **kwargs,
        )
        self.metrics[step][name] = value

    def write_to_disk(self) -> None:
        if not self.is_main_process:
            return

        names: List[str] = sorted({name for row in self.metrics.values() for name in row})
        with open(
            self.out_path,
            "w",
            newline="",
        ) as f:
            writer: csv.DictWriter = csv.DictWriter(
                f,
                fieldnames=["step", *names],
                restval="",
            )
            writer.writeheader()
            for step in sorted(self.metrics):
                writer.writerow({
                    "step": step,
                    **self.metrics[step]
                })

    def stop(self) -> None:
        self.mlflow_logger.stop()
        self.write_to_disk()
