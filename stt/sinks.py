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
        if r.matched_command:
            print(f'  (approved command: "{r.matched_command}", {r.match_distance} word(s) different)')
    """
    def send(self, r: STTResult) -> None:
        status = "ACCEPTED" if r.accepted else f"REJECTED ({r.reject_reason})"
        print(f'  [{status}] "{r.transcript}"  '
              f"confidence={r.confidence:.2f}  time={r.processing_time_s:.2f}s")
        if r.discarded_text:
            print(f'  (ignored speech before Start: "{r.discarded_text}")')
    """
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


class HttpSink:
    """POST each result as JSON to a web address, e.g. the VLN module's endpoint.

    Uses only the Python standard library. If the receiver is down or slow,
    send() raises after timeout_s; the pipeline turns that into a warning, so
    STT keeps working.
    """

    def __init__(self, url: str, timeout_s: float = 1.0, send_rejected: bool = False) -> None:
        self.url = url
        self.timeout_s = timeout_s
        self.send_rejected = send_rejected

    def send(self, r: STTResult) -> None:
        if not r.accepted and not self.send_rejected:
            return
        import urllib.request
        request = urllib.request.Request(
            self.url,
            data=r.to_json().encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            response.read()

    def close(self) -> None:
        pass


class Ros2Sink:
    """Publish results to a ROS 2 topic as std_msgs/String containing the JSON.

    Included for future use with ROS 2 based systems; not yet tested on hardware.
    ROS 2 is only imported when this sink is enabled, so machines without ROS
    are unaffected.
    """

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
        elif name == "http":
            sinks.append(HttpSink(cfg.http.url, cfg.http.timeout_s, cfg.http.send_rejected))
        elif name == "ros2":
            sinks.append(Ros2Sink(cfg.ros2.topic, cfg.ros2.node_name,
                                  cfg.ros2.publish_rejected))
        else:
            raise ValueError(f"Unknown sink '{name}' in output.sinks")
    return sinks