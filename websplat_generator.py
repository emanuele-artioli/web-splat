#!/usr/bin/env python3
"""Python wrapper for fast repeated 6DoF renders with web-splat.

This module launches the Rust `stream_render` binary once and keeps it alive,
so rendering multiple camera poses avoids reloading the point cloud.
"""

from __future__ import annotations

import argparse
import collections
import json
import struct
import subprocess
import threading
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence, Tuple

import numpy as np


Pose6D = Tuple[float, float, float, float, float, float]


def _infer_intrinsics_from_scene(scene_path: Path) -> Tuple[int, int, float, float]:
    with scene_path.open("r", encoding="utf-8") as f:
        cameras = json.load(f)

    if not cameras:
        raise ValueError(f"Scene file is empty: {scene_path}")

    cam0 = cameras[0]
    required = ("width", "height", "fx", "fy")
    missing = [key for key in required if key not in cam0]
    if missing:
        raise ValueError(f"Scene file {scene_path} is missing keys in first camera: {missing}")

    return int(cam0["width"]), int(cam0["height"]), float(cam0["fx"]), float(cam0["fy"])


class WebSplatGenerator:
    """Long-lived web-splat renderer with generator helpers.

    Pose convention for `render_pose` and `render_many`:
        - 6 values: `(x, y, z, roll_deg, pitch_deg, yaw_deg)`
        - 12 values: `(x, y, z, r00, r01, r02, r10, r11, r12, r20, r21, r22)`
            with row-major 3x3 rotation matrix entries from scene JSON

        Euler order in renderer is: pitch (x), yaw (y), roll (z).

    Output frame:
    - `numpy.ndarray` with shape `(height, width, 4)`
    - dtype `uint8`
    - channel order RGBA
    """

    def __init__(
        self,
        point_cloud_path: str | Path,
        scene_path: Optional[str | Path] = None,
        *,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        background_rgb: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        project_root: Optional[str | Path] = None,
        release: bool = True,
    ) -> None:
        self.point_cloud_path = Path(point_cloud_path).expanduser().resolve()
        if not self.point_cloud_path.exists():
            raise FileNotFoundError(f"Point cloud file not found: {self.point_cloud_path}")

        self.scene_path = Path(scene_path).expanduser().resolve() if scene_path else None
        self.project_root = (
            Path(project_root).expanduser().resolve()
            if project_root is not None
            else Path(__file__).resolve().parent
        )

        if any(v is None for v in (width, height, fx, fy)):
            if self.scene_path is None:
                raise ValueError(
                    "Either provide width/height/fx/fy explicitly or pass scene_path to infer them"
                )
            width, height, fx, fy = _infer_intrinsics_from_scene(self.scene_path)

        self.width = int(width)
        self.height = int(height)
        self.fx = float(fx)
        self.fy = float(fy)
        self.background_rgb = tuple(float(v) for v in background_rgb)

        self._stderr_lines: collections.deque[str] = collections.deque(maxlen=200)
        self._stderr_thread: Optional[threading.Thread] = None

        cmd = ["cargo", "run"]
        if release:
            cmd.append("--release")
        if self.point_cloud_path.suffix.lower() == ".npz":
            cmd.extend(["--features", "npz"])
        cmd.extend(
            [
                "--quiet",
                "--bin",
                "stream_render",
                "--",
                str(self.point_cloud_path),
                "--width",
                str(self.width),
                "--height",
                str(self.height),
                "--fx",
                str(self.fx),
                "--fy",
                str(self.fy),
                "--bg-r",
                str(self.background_rgb[0]),
                "--bg-g",
                str(self.background_rgb[1]),
                "--bg-b",
                str(self.background_rgb[2]),
            ]
        )

        try:
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(self.project_root),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
                bufsize=0,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "cargo executable not found. Install Rust toolchain (rustup/cargo) "
                "or adapt this wrapper to use a prebuilt stream_render binary."
            ) from exc
        self._start_stderr_drain()
        self._wait_for_ready()

    def _start_stderr_drain(self) -> None:
        assert self._proc.stderr is not None

        def _drain() -> None:
            while True:
                raw = self._proc.stderr.readline()
                if not raw:
                    break
                try:
                    line = raw.decode("utf-8", errors="replace").rstrip("\n")
                except Exception:
                    line = repr(raw)
                self._stderr_lines.append(line)

        self._stderr_thread = threading.Thread(target=_drain, daemon=True)
        self._stderr_thread.start()

    def _wait_for_ready(self) -> None:
        assert self._proc.stdout is not None
        ready_line = self._proc.stdout.readline()
        if not ready_line.startswith(b"READY "):
            stderr_tail = "\n".join(self._stderr_lines)
            raise RuntimeError(
                "stream_render failed to start correctly. "
                f"Expected READY line, got: {ready_line!r}\n"
                f"stderr:\n{stderr_tail}"
            )

    def _read_exact(self, n: int) -> bytes:
        assert self._proc.stdout is not None
        buf = bytearray(n)
        view = memoryview(buf)
        offset = 0
        while offset < n:
            chunk = self._proc.stdout.read(n - offset)
            if not chunk:
                stderr_tail = "\n".join(self._stderr_lines)
                raise RuntimeError(
                    f"Unexpected EOF while reading {n} bytes (got {offset}).\n"
                    f"stderr:\n{stderr_tail}"
                )
            view[offset : offset + len(chunk)] = chunk
            offset += len(chunk)
        return bytes(buf)

    def render_pose(self, pose: Sequence[float]) -> np.ndarray:
        """Render one pose and return an RGBA uint8 frame as a numpy array."""
        if len(pose) not in (6, 12):
            raise ValueError(
                "Expected 6 values (Euler pose) or 12 values "
                f"(position + 3x3 rotation matrix), got {len(pose)}"
            )

        if self._proc.poll() is not None:
            stderr_tail = "\n".join(self._stderr_lines)
            raise RuntimeError(
                f"stream_render process already exited with code {self._proc.returncode}.\n"
                f"stderr:\n{stderr_tail}"
            )

        pose_line = " ".join(f"{float(v):.8f}" for v in pose) + "\n"
        assert self._proc.stdin is not None
        self._proc.stdin.write(pose_line.encode("ascii"))
        self._proc.stdin.flush()

        frame_size = struct.unpack("<I", self._read_exact(4))[0]
        if frame_size == 0:
            stderr_tail = "\n".join(self._stderr_lines)
            raise RuntimeError(
                "Renderer reported an error while processing the pose. "
                "Check the stderr tail below for details.\n"
                f"stderr:\n{stderr_tail}"
            )
        payload = self._read_exact(frame_size)
        expected = self.width * self.height * 4
        if frame_size != expected:
            raise RuntimeError(
                f"Unexpected frame size {frame_size}, expected {expected} "
                f"for {self.width}x{self.height} RGBA"
            )

        return np.frombuffer(payload, dtype=np.uint8).reshape(self.height, self.width, 4)

    def render_pose_matrix(
        self,
        position_xyz: Sequence[float],
        rotation_rows: Sequence[Sequence[float]],
    ) -> np.ndarray:
        """Render using explicit world-space position and row-major 3x3 rotation matrix."""
        if len(position_xyz) != 3:
            raise ValueError(f"position_xyz must contain 3 values, got {len(position_xyz)}")
        if len(rotation_rows) != 3 or any(len(row) != 3 for row in rotation_rows):
            raise ValueError("rotation_rows must be a 3x3 matrix (three rows with three values)")

        pose12 = [
            float(position_xyz[0]),
            float(position_xyz[1]),
            float(position_xyz[2]),
            float(rotation_rows[0][0]),
            float(rotation_rows[0][1]),
            float(rotation_rows[0][2]),
            float(rotation_rows[1][0]),
            float(rotation_rows[1][1]),
            float(rotation_rows[1][2]),
            float(rotation_rows[2][0]),
            float(rotation_rows[2][1]),
            float(rotation_rows[2][2]),
        ]
        return self.render_pose(pose12)

    def render_many(self, poses: Iterable[Sequence[float]]) -> Iterator[np.ndarray]:
        """Yield frames for a sequence of poses."""
        for pose in poses:
            yield self.render_pose(pose)

    def interactive(self) -> Iterator[np.ndarray]:
        """Coroutine-style generator: send a pose, receive a frame.

        Usage:
            gen = renderer.interactive()
            next(gen)
            frame = gen.send((x, y, z, roll, pitch, yaw))
        """
        pose = yield
        while True:
            frame = self.render_pose(pose)
            pose = yield frame

    def close(self) -> None:
        if self._proc.poll() is None:
            try:
                assert self._proc.stdin is not None
                self._proc.stdin.write(b"quit\n")
                self._proc.stdin.flush()
            except Exception:
                pass

            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()

        if self._proc.stdin:
            self._proc.stdin.close()
        if self._proc.stdout:
            self._proc.stdout.close()
        if self._proc.stderr:
            self._proc.stderr.close()

    def __enter__(self) -> "WebSplatGenerator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _parse_pose(raw: str) -> Pose6D:
    vals = [float(v.strip()) for v in raw.split(",") if v.strip()]
    if len(vals) != 6:
        raise argparse.ArgumentTypeError("pose must contain 6 comma-separated numbers")
    return vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]


def _main() -> None:
    parser = argparse.ArgumentParser(description="Render one pose with persistent web-splat process")
    parser.add_argument("point_cloud", type=Path, help="Path to point cloud (.ply or .npz)")
    parser.add_argument(
        "--scene",
        type=Path,
        default=None,
        help="Optional cameras.json used to infer width/height/fx/fy",
    )
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument(
        "--pose",
        type=_parse_pose,
        required=True,
        help="6DoF pose as x,y,z,roll,pitch,yaw (degrees)",
    )
    parser.add_argument(
        "--np-out",
        type=Path,
        default=Path("frame.npy"),
        help="Output .npy path for the RGBA frame",
    )
    args = parser.parse_args()

    with WebSplatGenerator(
        point_cloud_path=args.point_cloud,
        scene_path=args.scene,
        width=args.width,
        height=args.height,
        fx=args.fx,
        fy=args.fy,
    ) as renderer:
        frame = renderer.render_pose(args.pose)
        np.save(args.np_out, frame)


if __name__ == "__main__":
    _main()