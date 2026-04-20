"""Java service backend for exact incremental patch addition.

Manages a persistent Java subprocess running GraphabService, communicating
via JSON-over-stdin/stdout. Supports only additive patch operations in v1.
Resistance-changing cases must be routed to CliExactBackend.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from habconn.evaluators.base import (
    ActionType,
    BackendResult,
    BackendType,
    GraphabBackend,
    UnsupportedActionError,
    classify_action,
)
from habconn.evaluators.graphab_evaluator import GraphabEvaluator
from habconn.evaluators.graphab_runner import (
    GraphabProjectConfig,
    GraphabRunner,
    GraphabRuntimeConfig,
)
from habconn.problems.vector_problem import VectorConnectivityProblem
from habconn.state.landscape_state import LandscapeState

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class JavaServiceConfig:
    """Configuration for the Java service process."""

    java_executable: str
    graphab_jar_path: Path
    service_class_dir: Path
    jvm_memory: str = "4G"
    headless: bool = True
    startup_timeout_seconds: float = 30.0
    command_timeout_seconds: float = 120.0


class JavaServiceBackend(GraphabBackend):
    """Backend using a persistent Java service over Graphab internals.

    v1 scope: supports BASELINE and ADDITIVE_PATCH only.
    Raises UnsupportedActionError for RESISTANCE_CHANGE.

    The service must be initialized with a base project created by the
    CLI exact path. The service loads that project and applies incremental
    patch additions in-memory.
    """

    def __init__(
        self,
        service_config: JavaServiceConfig,
        runner: GraphabRunner,
        project_config: GraphabProjectConfig,
    ) -> None:
        self._config = service_config
        self._runner = runner
        self._project_config = project_config
        self._process: Optional[subprocess.Popen] = None
        self._base_project_dir: Optional[Path] = None
        self._session_active = False
        self._session_problem_key: Optional[str] = None
        self._patches_added = 0

    def evaluate(
        self,
        problem: VectorConnectivityProblem,
        state: LandscapeState,
        *,
        run_label: Optional[str] = None,
    ) -> BackendResult:
        action_type = classify_action(problem, state)

        if not self.supports_action_type(action_type):
            raise UnsupportedActionError(BackendType.JAVA_SERVICE, action_type)

        # Ensure service is running and session is loaded
        self._ensure_session(problem)

        # Reset to base state, then apply all selected patches in order
        self._reset_to_base()

        for pu_id in state.selected_pu_ids:
            self._add_patch(problem, pu_id)

        # Compute PC
        pc_result = self._compute_pc()

        return BackendResult(
            pc_value=pc_result["pc_value"],
            backend_type=BackendType.JAVA_SERVICE,
            action_type=action_type,
            selected_pu_ids=list(state.selected_pu_ids),
            metadata={
                "patches_added": pc_result.get("patches_added", 0),
                "step_count": state.step_count,
                "base_project_dir": str(self._base_project_dir),
                "run_label": run_label,
            },
        )

    def supports_action_type(self, action_type: ActionType) -> bool:
        return action_type in (ActionType.BASELINE, ActionType.ADDITIVE_PATCH)

    def reset_session(self) -> None:
        if self._session_active:
            self._reset_to_base()
        self._session_active = False
        self._session_problem_key = None
        self._base_project_dir = None

    @property
    def backend_type(self) -> BackendType:
        return BackendType.JAVA_SERVICE

    def start_service(self) -> None:
        """Start the Java service subprocess.

        If the process is dead (exited or killed), session state is
        invalidated before restarting so that _ensure_session() will
        create a fresh session on the new process.
        """
        if self._process is not None and self._process.poll() is None:
            return  # Already running

        # Process is dead or never started — clear stale session state
        if self._process is not None:
            logger.info("Java service process is dead (rc=%s), restarting",
                        self._process.returncode)
        self._session_active = False
        self._session_problem_key = None

        cfg = self._config
        cmd = [cfg.java_executable]

        if cfg.jvm_memory:
            cmd.append(f"-Xmx{cfg.jvm_memory}")
        if cfg.headless:
            cmd.append("-Djava.awt.headless=true")

        cmd.extend([
            "-cp",
            f"{cfg.service_class_dir}{_classpath_sep()}{cfg.graphab_jar_path}",
            "GraphabService",
        ])

        logger.info("Starting Java service: %s", " ".join(cmd))

        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line-buffered
        )

        # Wait for ready signal
        response = self._read_response(timeout=cfg.startup_timeout_seconds)
        if response.get("status") != "ready":
            self.stop_service()
            raise RuntimeError(
                f"Java service failed to start. Response: {response}"
            )

        logger.info("Java service started (pid=%d)", self._process.pid)

    def stop_service(self) -> None:
        """Stop the Java service subprocess."""
        if self._process is None:
            return

        try:
            if self._process.poll() is None:
                self._send_command({"cmd": "shutdown"})
                self._process.wait(timeout=5)
        except Exception:
            pass

        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()

        self._process = None
        self._session_active = False
        logger.info("Java service stopped")

    def __del__(self) -> None:
        self.stop_service()

    # --- Internal methods ---

    @staticmethod
    def _problem_key(problem: VectorConnectivityProblem) -> str:
        """Stable identity key for a problem instance (name + data paths)."""
        return (
            f"{problem.name}|{problem.habitat_raster_path}"
            f"|{problem.resistance_raster_path}|{problem.vector_path}"
        )

    def _ensure_session(self, problem: VectorConnectivityProblem) -> None:
        """Ensure the service is running and a session is loaded for this problem."""
        self.start_service()

        key = self._problem_key(problem)
        if self._session_active and self._session_problem_key == key:
            return

        # Different problem or no session: tear down and rebuild
        if self._session_active:
            logger.info(
                "Session problem changed (%s -> %s), reloading",
                self._session_problem_key, key,
            )
            self._session_active = False

        # Clean up old base-project temp directory if present
        old_base = self._base_project_dir
        if old_base is not None and old_base.exists():
            import shutil
            try:
                shutil.rmtree(old_base, ignore_errors=True)
                logger.info("Cleaned up old base project: %s", old_base)
            except Exception:
                pass
            self._base_project_dir = None

        # Create base project using CLI (once per problem)
        base_dir = self._create_base_project(problem)
        self._base_project_dir = base_dir

        # Read the actual converted metric parameters from the CLI output.
        # When cost_converted_threshold is True, the CLI converts d from
        # cost-distance to Euclidean distance. The converted value is
        # written to the metric output file (e.g. PC.txt).
        cfg = self._project_config
        project_subdir = self._find_project_subdir(base_dir)
        effective_d = self._read_effective_d(project_subdir, cfg)

        # Load session in Java service
        response = self._send_command({
            "cmd": "create_session",
            "project_dir": str(project_subdir),
            "linkset_name": cfg.linkset_name,
            "graph_name": cfg.graph_name,
            "metric_name": cfg.metric_name,
            "metric_d": effective_d,
            "metric_p": cfg.metric_p,
            "cost_converted": 0.0,  # Already converted
        })

        if response.get("status") != "ok":
            raise RuntimeError(f"Failed to create session: {response}")

        self._session_active = True
        self._session_problem_key = key
        self._patches_added = 0
        logger.info(
            "Session created for %s: %d base patches",
            problem.name, response.get("n_patches", -1),
        )

    def _create_base_project(self, problem: VectorConnectivityProblem) -> Path:
        """Create a base Graphab project via CLI for the baseline (no patches selected)."""
        run_dir = self._runner.make_run_directory(prefix="java_service_base")
        run_result = self._runner.run_full_pipeline(
            habitat_raster_path=problem.habitat_raster_path,
            resistance_raster_path=problem.resistance_raster_path,
            run_dir=run_dir,
            run_label="java_service_base",
        )
        return run_result.run_dir

    def _read_effective_d(
        self, project_dir: Path, cfg: GraphabProjectConfig,
    ) -> float:
        """Read the actual d parameter from the CLI metric output file.

        When cost_converted_threshold is True, the Graphab CLI converts
        d from cost-distance to Euclidean distance via linkset regression.
        The converted value is written to the metric output file.
        If the file is not found or can't be parsed, returns the raw d.
        """
        import pandas as pd

        metric_file = project_dir / cfg.metric_output_filename
        if not metric_file.exists():
            logger.warning(
                "Metric file not found at %s, using raw d=%f",
                metric_file, cfg.metric_d,
            )
            return cfg.metric_d

        try:
            df = pd.read_csv(metric_file, sep="\t", header=0)
            # Column names may have whitespace
            cols = {str(c).strip().lower(): str(c).strip() for c in df.columns}
            if "d" in cols:
                effective_d = float(df[cols["d"]].iloc[0])
                logger.info(
                    "Read effective d=%f from %s (input d=%f)",
                    effective_d, metric_file, cfg.metric_d,
                )
                return effective_d
        except Exception as e:
            logger.warning(
                "Failed to read d from metric file %s: %s", metric_file, e,
            )

        return cfg.metric_d

    def _find_project_subdir(self, run_dir: Path) -> Path:
        """Find the Graphab project subdirectory inside a run directory."""
        for child in run_dir.iterdir():
            if child.is_dir() and (child / f"{child.name}.xml").exists():
                return child
        raise FileNotFoundError(
            f"No Graphab project directory found in {run_dir}"
        )

    def _reset_to_base(self) -> None:
        """Reset the Java service session to the base project state."""
        if not self._session_active:
            return

        response = self._send_command({"cmd": "reset_session"})
        if response.get("status") != "ok":
            raise RuntimeError(f"Failed to reset session: {response}")
        self._patches_added = 0

    def _add_patch(
        self,
        problem: VectorConnectivityProblem,
        pu_id: int,
    ) -> dict:
        """Add a single patch to the Java service session.

        Capacity semantics: the CLI path with maxsize fragmentation creates
        one patch per raster pixel, each with capacity = pixel_area. The
        service adds one polygon patch. To minimize the self-term distortion
        (capacity^2 in PC), we use pixel_area as capacity — matching the
        scale of existing patches in the project.

        This is a documented v1 approximation. The fundamental mismatch
        between one service patch and N CLI pixel-patches remains.
        """
        row = problem.get_planning_unit_row(pu_id)
        geom = row.geometry
        wkt = geom.wkt
        # Use pixel area as capacity to match the scale of raster-detected
        # patches. Each existing patch has capacity = pixel_area because
        # maxsize fragmentation creates one patch per pixel.
        pixel_area = abs(problem.raster_transform.a * problem.raster_transform.e)
        capacity = float(pixel_area)

        restored_resistance = problem.restored_resistance_value or 1.0

        response = self._send_command({
            "cmd": "add_patch",
            "wkt": wkt,
            "capacity": capacity,
            "restored_resistance": float(restored_resistance),
        })

        if response.get("status") != "ok":
            raise RuntimeError(
                f"Failed to add patch pu_id={pu_id}: {response}"
            )

        self._patches_added += 1
        return response

    def _compute_pc(self) -> dict:
        """Request PC computation from the Java service."""
        response = self._send_command({"cmd": "compute_pc"})
        if response.get("status") != "ok":
            raise RuntimeError(f"Failed to compute PC: {response}")
        return response

    def _send_command(self, command: dict) -> dict:
        """Send a JSON command and read the JSON response."""
        if self._process is None or self._process.poll() is not None:
            raise RuntimeError("Java service is not running")

        line = json.dumps(command, separators=(",", ":"))
        self._process.stdin.write(line + "\n")
        self._process.stdin.flush()

        return self._read_response(timeout=self._config.command_timeout_seconds)

    def _read_response(self, timeout: float = 30.0) -> dict:
        """Read one JSON response line from the service stdout.

        Uses a background thread for readline() so the timeout is
        enforced even if the Java process hangs without emitting output.
        """
        if self._process is None:
            raise RuntimeError("Java service is not running")

        import queue
        import threading

        result_queue: queue.Queue[str | Exception] = queue.Queue()

        def _reader() -> None:
            try:
                line = self._process.stdout.readline()
                result_queue.put(line)
            except Exception as exc:
                result_queue.put(exc)

        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            # readline is blocked — service is hanging
            # Kill the process so the thread eventually unblocks
            try:
                self._process.kill()
            except Exception:
                pass
            # Invalidate session state so the next evaluate() call does not
            # assume the dead session is still loaded.
            self._process = None
            self._session_active = False
            self._session_problem_key = None
            raise TimeoutError(
                f"Java service did not respond within {timeout}s"
            )

        # Thread finished — check if process exited
        if self._process.poll() is not None and result_queue.empty():
            stderr = ""
            try:
                stderr = self._process.stderr.read()
            except Exception:
                pass
            rc = self._process.returncode
            # Invalidate session state — process is dead
            self._process = None
            self._session_active = False
            self._session_problem_key = None
            raise RuntimeError(
                f"Java service exited unexpectedly (rc={rc}). "
                f"stderr: {stderr[:2000]}"
            )

        try:
            item = result_queue.get_nowait()
        except queue.Empty:
            raise RuntimeError("No response from Java service reader thread")

        if isinstance(item, Exception):
            raise RuntimeError(f"Error reading from Java service: {item}") from item

        line = item.strip() if item else ""
        if not line:
            # Empty line — check for process exit
            if self._process.poll() is not None:
                stderr = ""
                try:
                    stderr = self._process.stderr.read()
                except Exception:
                    pass
                rc = self._process.returncode
                # Invalidate session state — process is dead
                self._process = None
                self._session_active = False
                self._session_problem_key = None
                raise RuntimeError(
                    f"Java service exited (rc={rc}). "
                    f"stderr: {stderr[:2000]}"
                )
            raise RuntimeError("Empty response from Java service")

        try:
            return json.loads(line)
        except json.JSONDecodeError as e:
            logger.warning("Invalid JSON from service: %s", line)
            raise RuntimeError(
                f"Invalid JSON from Java service: {line}"
            ) from e


def _classpath_sep() -> str:
    """Return the classpath separator for the current platform."""
    import sys
    return ";" if sys.platform == "win32" else ":"
