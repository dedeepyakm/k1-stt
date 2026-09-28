"""ROS 2 test listener: receives STT commands, measures delivery delay and packet loss.

Run on the lab PC in a second terminal while app.py is running with the ros2 sink enabled:

    source /opt/ros/<distro>/setup.bash
    python scripts/ros2_listener.py                  # default topic from config.yaml
    python scripts/ros2_listener.py --topic /other   # a different topic

Speak some commands, then press Ctrl+C for the summary.

This is also a minimal example of how VLN can subscribe: parse msg.data as JSON
and act only when "accepted" is true.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # allow "import stt"
from stt.config import load_config  # noqa: E402


def expected_ids(log_path: Path, since: float) -> set:
    """Accepted commands the STT app logged after the listener started."""
    if not log_path.exists():
        return set()
    ids = set()
    for line in log_path.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("unix_time", 0) >= since and d.get("accepted"):
            ids.add(d["command_id"])
    return ids


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Measure STT -> VLN delivery over ROS 2")
    parser.add_argument("--topic", default=cfg.ros2.topic)
    args = parser.parse_args()

    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    class Listener(Node):
        def __init__(self) -> None:
            super().__init__("stt_test_listener")
            self.delays_ms = []
            self.received_ids = set()
            self.create_subscription(String, args.topic, self.on_message, 10)
            self.get_logger().info(f"Listening on {args.topic}. Ctrl+C for summary.")

        def on_message(self, msg) -> None:
            arrived = time.time()
            data = json.loads(msg.data)                       # the STT JSON message
            delay_ms = (arrived - data["unix_time"]) * 1000
            self.delays_ms.append(delay_ms)
            self.received_ids.add(data["command_id"])
            status = "ACCEPTED" if data["accepted"] else "REJECTED"
            print(f'[{status}] "{data["transcript"]}"  confidence={data["confidence"]:.2f}  '
                  f"delay={delay_ms:.1f} ms")

    started = time.time()
    rclpy.init()
    node = Listener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    # ---- summary ----
    expected = expected_ids(cfg.output.jsonl_path, started)
    missing = expected - node.received_ids
    d = node.delays_ms
    print("\n--- Summary ---")
    print(f"Received          : {len(node.received_ids)}")
    print(f"Sent (from log)   : {len(expected)}")
    if expected:
        print(f"Missing           : {len(missing)}  (packet loss {len(missing) / len(expected):.1%})")
    if d:
        print(f"Delay mean/median : {statistics.mean(d):.1f} / {statistics.median(d):.1f} ms")
        print(f"Delay min/max     : {min(d):.1f} / {max(d):.1f} ms")

    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()