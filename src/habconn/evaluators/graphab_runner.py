# src/habconn/evaluators/graphab_runner.py

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import shutil
import subprocess
import tempfile
import uuid


@dataclass(slots=True)
class GraphabRuntimeConfig:
    """
    Runtime configuration for invoking Graphab through Java.
    """

    graphab_jar_path: Path
    work_root: Path

    java_executable: str = "java"
    keep_workdirs: bool = True
    headless: bool = True
    jvm_memory: Optional[str] = None
    extra_java_options: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.graphab_jar_path = Path(self.graphab_jar_path).expanduser().resolve()
        self.work_root = Path(self.work_root).expanduser().resolve()

        if not self.graphab_jar_path.exists():
            raise FileNotFoundError(
                f"Graphab jar not found: {self.graphab_jar_path}"
            )

        self.work_root.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class GraphabProjectConfig:
    """
    Graphab project and metric settings for the first exact evaluator.
    """

    habitat_value: int = 1
    nodata_value: int = -32768
    max_patch_size: int = 10
    con8: bool = True

    linkset_name: str = "linkset_main"
    distance_type: str = "cost"
    max_cost: float = 300.0
    topo: str = "planar"
    save_paths: bool = False

    graph_name: str = "graph_main"
    graph_threshold: float = 130.0
    cost_converted_threshold: bool = True

    metric_name: str = "PC"
    metric_d: float = 130.0
    metric_p: float = 0.01
    metric_beta: float = 1.0
    cost_converted_metric_d: bool = True


@dataclass(slots=True)
class GraphabRunResult:
    """
    Structured output from one Graphab pipeline execution.
    """

    project_name: str
    project_dir: Path
    project_xml_path: Path
    metric_file_path: Path

    create_result: subprocess.CompletedProcess
    link_result: subprocess.CompletedProcess
    graph_result: subprocess.CompletedProcess
    metric_result: subprocess.CompletedProcess


class GraphabRunner:
    """
    Low-level Graphab subprocess wrapper.

    This intentionally mirrors the working approach from the original codebase,
    but moves paths and command construction into one isolated component.
    """

    def __init__(
        self,
        runtime_config: GraphabRuntimeConfig,
        project_config: Optional[GraphabProjectConfig] = None,
    ) -> None:
        self.runtime_config = runtime_config
        self.project_config = project_config or GraphabProjectConfig()

    def make_run_directory(self, prefix: str = "graphab_run") -> Path:
        run_dir = Path(
            tempfile.mkdtemp(
                prefix=f"{prefix}_",
                dir=str(self.runtime_config.work_root),
            )
        )
        return run_dir

    def cleanup_run_directory(self, run_dir: Path) -> None:
        if self.runtime_config.keep_workdirs:
            return
        shutil.rmtree(run_dir, ignore_errors=True)

    def _java_base_command(self) -> list[str]:
        cmd = [self.runtime_config.java_executable]

        if self.runtime_config.jvm_memory:
            cmd.append(f"-Xmx{self.runtime_config.jvm_memory}")

        if self.runtime_config.headless:
            cmd.append("-Djava.awt.headless=true")

        cmd.extend(self.runtime_config.extra_java_options)
        cmd.extend(["-jar", str(self.runtime_config.graphab_jar_path)])
        return cmd

    def _run(self, args: list[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
        cmd = self._java_base_command() + args
        return subprocess.run(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            check=True,
        )

    def create_project(
        self,
        *,
        project_name: str,
        habitat_raster_path: Path,
        run_dir: Path,
    ) -> subprocess.CompletedProcess:
        cfg = self.project_config
        args = [
            "--create",
            project_name,
            str(habitat_raster_path),
            f"habitat={cfg.habitat_value}",
            f"nodata={cfg.nodata_value}",
            f"maxsize={cfg.max_patch_size}",
            f"con8={str(cfg.con8).lower()}",
            f"dir={run_dir.as_posix()}",
        ]
        return self._run(args)

    def link(
        self,
        *,
        project_xml_path: Path,
        resistance_raster_path: Path,
        run_dir: Path,
    ) -> subprocess.CompletedProcess:
        cfg = self.project_config
        args = [
            "--project",
            str(project_xml_path),
            "--linkset",
            f"distance={cfg.distance_type}",
            f"name={cfg.linkset_name}",
            f"maxcost={cfg.max_cost}",
            *(["nopathsaved"] if not cfg.save_paths else []),
            f"extcost={str(resistance_raster_path)}",
            f"topo={cfg.topo}",
            f"dir={run_dir.as_posix()}",
        ]
        return self._run(args)

    def graph(
        self,
        *,
        project_xml_path: Path,
    ) -> subprocess.CompletedProcess:
        cfg = self.project_config
        threshold_value = (
            f"threshold={{{cfg.graph_threshold}}}"
            if cfg.cost_converted_threshold
            else f"threshold={cfg.graph_threshold}"
        )

        args = [
            "--project",
            str(project_xml_path),
            "--uselinkset",
            cfg.linkset_name,
            "--graph",
            f"name={cfg.graph_name}",
            threshold_value,
        ]
        return self._run(args)

    def metric(
        self,
        *,
        project_xml_path: Path,
    ) -> subprocess.CompletedProcess:
        cfg = self.project_config
        d_value = (
            f"d={{{cfg.metric_d}}}"
            if cfg.cost_converted_metric_d
            else f"d={cfg.metric_d}"
        )

        args = [
            "--project",
            str(project_xml_path),
            "--usegraph",
            cfg.graph_name,
            "--gmetric",
            cfg.metric_name,
            d_value,
            f"p={cfg.metric_p}",
            f"beta={cfg.metric_beta}",
        ]
        return self._run(args)

    def run_full_pipeline(
        self,
        *,
        habitat_raster_path: Path,
        resistance_raster_path: Path,
        run_label: Optional[str] = None,
    ) -> GraphabRunResult:
        run_dir = self.make_run_directory(prefix=run_label or "graphab")
        project_name = f"project_{uuid.uuid4().hex[:12]}"
        project_dir = run_dir / project_name
        project_xml_path = project_dir / f"{project_name}.xml"
        metric_file_path = project_dir / f"{self.project_config.metric_name}.txt"

        try:
            create_result = self.create_project(
                project_name=project_name,
                habitat_raster_path=habitat_raster_path,
                run_dir=run_dir,
            )
            link_result = self.link(
                project_xml_path=project_xml_path,
                resistance_raster_path=resistance_raster_path,
                run_dir=run_dir,
            )
            graph_result = self.graph(project_xml_path=project_xml_path)
            metric_result = self.metric(project_xml_path=project_xml_path)

            return GraphabRunResult(
                project_name=project_name,
                project_dir=project_dir,
                project_xml_path=project_xml_path,
                metric_file_path=metric_file_path,
                create_result=create_result,
                link_result=link_result,
                graph_result=graph_result,
                metric_result=metric_result,
            )
        except Exception:
            if not self.runtime_config.keep_workdirs:
                self.cleanup_run_directory(run_dir)
            raise
