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
    graphab_jar_path: Path
    work_root: Path

    java_executable: str = "java"
    keep_workdirs: bool = True
    headless: bool = True
    jvm_memory: Optional[str] = None
    extra_java_options: list[str] = field(default_factory=list)
    n_proc: Optional[int] = None

    # Logging / diagnostics
    capture_output: bool = True
    max_log_lines_on_error: int = 80

    def __post_init__(self) -> None:
        self.graphab_jar_path = Path(self.graphab_jar_path).expanduser().resolve()
        self.work_root = Path(self.work_root).expanduser().resolve()

        if not self.graphab_jar_path.exists():
            raise FileNotFoundError(f"Graphab jar not found: {self.graphab_jar_path}")

        self.work_root.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class GraphabProjectConfig:
    """
    Graphab 2.8-style config.

    Current evaluator = exact recreate-from-raster reference evaluator.
    """

    habitat_codes: tuple[int, ...] = (1,)
    nodata_value: int = -32768
    minarea: Optional[float] = None
    maxsize: Optional[float] = 10.0
    con8: bool = True

    linkset_name: str = "linkset_main"
    distance_type: str = "cost"
    complete: bool = False
    max_cost: float = 300.0
    save_paths: bool = False

    graph_name: str = "graph_main"
    graph_threshold: float = 130.0
    cost_converted_threshold: bool = True
    nointra: bool = False

    metric_name: str = "PC"
    metric_d: float = 130.0
    metric_p: float = 0.01
    metric_beta: Optional[float] = None
    metric_output_filename: str = "PC.txt"


@dataclass(slots=True)
class GraphabRunResult:
    project_name: str
    run_dir: Path
    project_dir: Path
    project_xml_path: Path
    metric_file_path: Path
    full_command: list[str]
    pipeline_result: subprocess.CompletedProcess

    @property
    def stdout(self) -> str:
        return self.pipeline_result.stdout or ""

    @property
    def stderr(self) -> str:
        return self.pipeline_result.stderr or ""


class GraphabRunner:
    def __init__(
        self,
        runtime_config: GraphabRuntimeConfig,
        project_config: Optional[GraphabProjectConfig] = None,
    ) -> None:
        self.runtime_config = runtime_config
        self.project_config = project_config or GraphabProjectConfig()

    def make_run_directory(self, prefix: str = "graphab_run") -> Path:
        return Path(
            tempfile.mkdtemp(prefix=f"{prefix}_", dir=str(self.runtime_config.work_root))
        )

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

    def _truncate_log(self, text: str) -> str:
        if not text:
            return ""
        lines = text.splitlines()
        max_lines = self.runtime_config.max_log_lines_on_error
        if len(lines) <= max_lines:
            return text
        tail = "\n".join(lines[-max_lines:])
        return f"[... trimmed {len(lines) - max_lines} earlier lines ...]\n{tail}"

    def _run(self, args: list[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
        cmd = self._java_base_command()

        if self.runtime_config.n_proc is not None:
            cmd.extend(["-proc", str(self.runtime_config.n_proc)])

        cmd.extend(args)

        print("GRAPHAB CMD:", " ".join(cmd))

        try:
            return subprocess.run(
                cmd,
                cwd=str(cwd) if cwd is not None else None,
                capture_output=self.runtime_config.capture_output,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            stdout = self._truncate_log(e.stdout or "")
            stderr = self._truncate_log(e.stderr or "")
            raise RuntimeError(
                "Graphab command failed.\n\n"
                f"Command:\n{' '.join(e.cmd)}\n\n"
                f"STDOUT:\n{stdout}\n\n"
                f"STDERR:\n{stderr}"
            ) from e

    def run_full_pipeline(
        self,
        *,
        habitat_raster_path: Path,
        resistance_raster_path: Path,
        run_dir: Optional[Path] = None,
        run_label: Optional[str] = None,
    ) -> GraphabRunResult:
        cfg = self.project_config
        local_run_dir = run_dir if run_dir is not None else self.make_run_directory(prefix=run_label or "graphab")

        project_name = f"project_{uuid.uuid4().hex[:12]}"
        project_dir = local_run_dir / project_name
        project_xml_path = project_dir / f"{project_name}.xml"
        metric_file_path = project_dir / cfg.metric_output_filename

        habitat_codes_str = ",".join(str(x) for x in cfg.habitat_codes)

        threshold_value = (
            f"threshold={{{cfg.graph_threshold}}}"
            if cfg.cost_converted_threshold
            else f"threshold={cfg.graph_threshold}"
        )
        d_value = (
            f"d={{{cfg.metric_d}}}"
            if cfg.cost_converted_threshold
            else f"d={cfg.metric_d}"
        )

        args = [
            "--create",
            project_name,
            str(habitat_raster_path),
            f"habitat={habitat_codes_str}",
            f"nodata={cfg.nodata_value}",
        ]

        if cfg.minarea is not None:
            args.append(f"minarea={cfg.minarea}")
        if cfg.maxsize is not None:
            args.append(f"maxsize={cfg.maxsize}")
        if cfg.con8:
            args.append("con8")

        args.append(f"dir={local_run_dir.as_posix()}")

        args.extend(
            [
                "--linkset",
                f"distance={cfg.distance_type}",
                f"name={cfg.linkset_name}",
            ]
        )

        if cfg.complete:
            args.append("complete")

        args.extend(
            [
                f"maxcost={cfg.max_cost}",
                *(["nopathsaved"] if not cfg.save_paths else []),
                f"extcost={str(resistance_raster_path)}",
                "--graph",
                f"name={cfg.graph_name}",
            ]
        )

        if cfg.nointra:
            args.append("nointra")

        args.append(threshold_value)

        args.extend(
            [
                "--gmetric",
                cfg.metric_name,
                f"resfile={cfg.metric_output_filename}",
                d_value,
                f"p={cfg.metric_p}",
            ]
        )

        if cfg.metric_beta is not None:
            args.append(f"beta={cfg.metric_beta}")

        pipeline_result = self._run(args)

        return GraphabRunResult(
            project_name=project_name,
            run_dir=local_run_dir,
            project_dir=project_dir,
            project_xml_path=project_xml_path,
            metric_file_path=metric_file_path,
            full_command=(self._java_base_command()
                          + ([ "-proc", str(self.runtime_config.n_proc) ] if self.runtime_config.n_proc is not None else [])
                          + args
            ),
            pipeline_result=pipeline_result,
        )