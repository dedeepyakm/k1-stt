"""Output sinks: where each STTResult goes.

Each sink has send(result) and close(). Adding a new destination means adding
a new class here; the rest of the pipeline does not change.
"""
from __future__ import annotations

from pathlib import Path

from stt.config import Config
from stt.schema import STTResult


class ConsoleSink:
    """One readable line per command."""

    def send(self, r: STTResult) -> None:
        status = "ACCEPTED" if r.accepted else f"REJECTED ({r.reject_reason})"
        print(f'  [{status}] "{r.transcript}"  '
              f"confidence={r.confidence:.2f}  time={r.processing_time_s:.2f}s")
        if r.discarded_text:
            print(f'  (ignored speech before Start: "{r.discarded_text}")')

    def close(self) -> None:
        pass


class JsonlSink:
    """Append each result as one JSON line: a complete log for debugging and reports."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file = open(path, "a", encoding="utf-8")

    def send(self, r: STTResult) -> None:
        self._file.write(r.to_json() + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class Ros2Sink:
    """Publish results to a ROS 2 topic as std_msgs/String containing the JSON."""

    def __init__(self, topic: str, node_name: str, publish_rejected: bool = False) -> None:
        try:
            import rclpy
            from std_msgs.msg import String
        except ImportError as e:
            raise RuntimeError(
                "ROS 2 is not available in this environment. On the lab PC, run "
                "'source /opt/ros/<distro>/setup.bash' first, or remove 'ros2' "
                "from output.sinks in config.yaml."
            ) from e
        self._rclpy = rclpy
        self._String = String
        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node(node_name)
        self._publisher = self._node.create_publisher(String, topic, 10)
        self.publish_rejected = publish_rejected

    def send(self, r: STTResult) -> None:
        if not r.accepted and not self.publish_rejected:
            return
        msg = self._String()
        msg.data = r.to_json()
        self._publisher.publish(msg)

    def close(self) -> None:
        self._node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()


def build_sinks(cfg: Config) -> list:
    sinks = []
    for name in cfg.output.sinks:
        if name == "console":
            sinks.append(ConsoleSink())
        elif name == "jsonl":
            sinks.append(JsonlSink(cfg.output.jsonl_path))
        elif name == "ros2":
            sinks.append(Ros2Sink(cfg.ros2.topic, cfg.ros2.node_name,
                                  cfg.ros2.publish_rejected))
        else:
            raise ValueError(f"Unknown sink '{name}' in output.sinks")
    return sinks