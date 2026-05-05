from __future__ import annotations

import asyncio
import socket
import sys
from collections import deque
from pathlib import Path
from typing import Optional, Tuple, Deque, Callable, Any

import numpy as np

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from datetime import datetime

from scipy.signal import butter, sosfiltfilt

import json
import paho.mqtt.client as mqtt

MQTT_BROKER_HOST = "127.0.0.1"  
MQTT_BROKER_PORT = 1883 # 1883和9001对应同一个broker，但不同协议
MQTT_BASE_TOPIC = "das"
MQTT_CLIENT_ID = "das_server"

def mqtt_topic(*parts: str) -> str:
    items = [MQTT_BASE_TOPIC.strip("/")]
    for p in parts:
        p = str(p).strip("/")
        if p:
            items.append(p)
    return "/".join(items)

def bandpass(x, fs, low=5.0, high=300.0, order=4):
    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def bandpass_2d(arr, fs, low=5.0, high=300.0, order=4):
    """
    arr: [T, C]
    沿时间轴对每个通道做带通滤波
    """
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={arr.shape}")

    t, c = arr.shape
    out = np.empty_like(arr, dtype=np.float64)

    for ch in range(c):
        out[:, ch] = bandpass(
            arr[:, ch].astype(np.float64),
            fs=fs,
            low=low,
            high=high,
            order=order,
        )

    return out.astype(np.float32)


PROJECT_SRC = Path("/home/sente/das_ai_project/src")
if str(PROJECT_SRC) not in sys.path:
    sys.path.append(str(PROJECT_SRC))

from infer_tft import TFTInferencer


START_FLAG = b"\x33\x55"
DEVICE_TYPE = b"\x0C\x00\x00\x00"
END_FLAG = b"\x33\xAA"

# 上位机绑定端口（接收 DAS 上行 UDP）
DEFAULT_BIND_IP = "192.168.1.100"
DEFAULT_BIND_PORT = 8009

# DAS设备默认地址（发送开始/停止指令时用）
DEFAULT_DAS_IP = "192.168.1.240"
DEFAULT_DAS_CMD_PORT = 8007

DEFAULT_CKPT_PATH = Path("/home/sente/das_ai_project/checkpoints/best.pt")


def build_start_stream_cmd() -> bytes:
    """
    高速数据开始发送指令：
    CC 55 + 0C 00 00 00 + 10 01 00 + 00 + CC AA
    """
    return bytes.fromhex("CC550C00000010010000CCAA")


def build_stop_stream_cmd() -> bytes:
    """
    高速数据停止发送指令：
    CC 55 + 0C 00 00 00 + 10 01 FF + 00 + CC AA
    """
    return bytes.fromhex("CC550C0000001001FF00CCAA")


def send_cmd(
    cmd: bytes,
    target_ip: str = DEFAULT_DAS_IP,
    target_port: int = DEFAULT_DAS_CMD_PORT,
) -> None:
    """
    发送 UDP 指令到 DAS 设备。
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.sendto(cmd, (target_ip, target_port))


class DASReceiver:
    """
    异步 UDP 接收器，拆成三层：
    1) recv_loop   : 只收包，放进 raw queue
    2) parse_loop  : 只拼帧、解析、更新矩阵
    3) infer_loop  : 只消费最新矩阵做推理

    这样高频 UDP 场景下更稳，不容易因为推理太慢导致收包卡住。
    """

    def __init__(
        self,
        bind_ip: str = DEFAULT_BIND_IP,
        bind_port: int = DEFAULT_BIND_PORT,
        history_len: int = 128,
        target_head: Tuple[int, int] = (0x80, 0x11),
        scale_to_pi: bool = True,
        raw_queue_size: int = 2048,
        actual_sample_rate: int = 2000,
        target_sample_rate: int = 1000,
        low_cut: float = 5.0,
        high_cut: float = 300.0,
        batch_size: int = 256,
        enable_infer: bool = True,
    ):
        self.low_cut = low_cut
        self.high_cut = high_cut
        self.batch_size = batch_size
        self.enable_infer = enable_infer
        self.bind_ip = bind_ip
        self.bind_port = bind_port
        self.history_len = history_len
        self.target_head = target_head
        self.scale_to_pi = scale_to_pi
        self.raw_queue_size = raw_queue_size

        self.actual_sample_rate = actual_sample_rate
        self.target_sample_rate = target_sample_rate
        self.sample_stride = max(1, int(round(actual_sample_rate / target_sample_rate)))
        self._frame_count = 0

        self._sock: Optional[socket.socket] = None
        self._closed = False

        # 收到的原始 UDP datagram 先进这个队列
        self._raw_q: asyncio.Queue[Optional[bytes]] = asyncio.Queue(maxsize=raw_queue_size)

        # 解析用字节缓冲
        self._buf = bytearray()

        # 滚动窗口，保存最近 history_len 行
        self._rows: Deque[np.ndarray] = deque(maxlen=history_len)

        # 本地递增序号
        self._seq = 0

        # 新数据到达事件，供 infer 任务等待
        self._update_event = asyncio.Event()

    @property
    def seq(self) -> int:
        return self._seq

    def latest_matrix(self) -> Optional[np.ndarray]:
        if not self._rows:
            return None
        return np.stack(self._rows, axis=0)

    def latest_row(self) -> Optional[np.ndarray]:
        if not self._rows:
            return None
        return self._rows[-1]
    
    def get_config(self) -> dict:
        return {
            "actual_sample_rate": self.actual_sample_rate,
            "target_sample_rate": self.target_sample_rate,
            "low_cut": self.low_cut,
            "high_cut": self.high_cut,
            "history_len": self.history_len,
            "batch_size": self.batch_size,
            "scale_to_pi": self.scale_to_pi,
            "enable_infer": self.enable_infer,
        }

    def get_infer_config(self) -> dict:
        return {
            "target_sample_rate": self.target_sample_rate,
            "low_cut": self.low_cut,
            "high_cut": self.high_cut,
            "batch_size": self.batch_size,
        }

    def apply_config(self, cfg: dict) -> None:
        if "actual_sample_rate" in cfg:
            self.actual_sample_rate = int(cfg["actual_sample_rate"])

        if "target_sample_rate" in cfg:
            self.target_sample_rate = int(cfg["target_sample_rate"])

        if "low_cut" in cfg:
            self.low_cut = float(cfg["low_cut"])

        if "high_cut" in cfg:
            self.high_cut = float(cfg["high_cut"])

        if "batch_size" in cfg:
            self.batch_size = int(cfg["batch_size"])

        if "scale_to_pi" in cfg:
            self.scale_to_pi = bool(cfg["scale_to_pi"])

        if "enable_infer" in cfg:
            self.enable_infer = bool(cfg["enable_infer"])

        if "history_len" in cfg:
            new_len = max(1, int(cfg["history_len"]))
            if new_len != self.history_len:
                old_rows = list(self._rows)
                self.history_len = new_len
                self._rows = deque(old_rows[-new_len:], maxlen=new_len)

        self.sample_stride = max(
            1,
            int(round(self.actual_sample_rate / max(1, self.target_sample_rate)))
        )

    def start(self) -> None:
        """
        创建 UDP socket，绑定本地地址。
        """
        if self._sock is not None:
            return

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # 尽量增大内核接收缓冲，降低高频 UDP 丢包概率
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        except OSError:
            pass

        sock.bind((self.bind_ip, self.bind_port))
        sock.setblocking(False)  # 非阻塞，配合 asyncio 使用
        self._sock = sock

    def close(self) -> None:
        self._closed = True

        # 关闭 socket
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

        # 唤醒可能阻塞在 queue.get() 的 parse_loop
        try:
            self._raw_q.put_nowait(None)
        except asyncio.QueueFull:
            pass

        # 唤醒可能阻塞在 wait_next_update() 的 infer_loop
        self._update_event.set()

    async def recv_loop(self) -> None:
        """
        只负责收 UDP 包，然后放进 raw queue。
        这个任务尽量短，不做任何解析。
        """
        if self._sock is None:
            self.start()

        loop = asyncio.get_running_loop()

        while not self._closed:
            try:
                data, _ = await loop.sock_recvfrom(self._sock, 65535)
                # print("收到UDP包:", len(data))
            except (OSError, asyncio.CancelledError):
                break

            if not data:
                continue

            # 队列满了就丢掉最旧的包，保留最新数据，适合实时场景
            try:
                self._raw_q.put_nowait(data)
            except asyncio.QueueFull:
                try:
                    _ = self._raw_q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    self._raw_q.put_nowait(data)
                except asyncio.QueueFull:
                    pass

    async def parse_loop(self) -> None:
        """
        只负责：
        - 从 raw queue 取原始 UDP 包
        - 拼到 bytearray 缓冲
        - 尽可能多地解析完整帧
        - 更新 rows / seq / event
        """
        while not self._closed:
            try:
                item = await self._raw_q.get()
            except asyncio.CancelledError:
                break

            if item is None:
                break

            self._buf.extend(item)
            self._parse_available_frames()

    async def wait_next_update(self) -> None:
        """
        等待下一帧更新。给 infer_loop 用。
        """
        await self._update_event.wait()
        self._update_event.clear()

    def _parse_available_frames(self) -> None:
        """
        从 bytearray 缓冲区中尽可能多地解析完整帧。
        只处理：
        - 上行帧起始：33 55
        - 设备类型：0C 00 00 00
        - 数据体标志：DA
        - 目标振动头：0x80 0x11
        """
        while True:
            # 找起始标志
            # print(self._buf[:20])
            start_idx = self._buf.find(START_FLAG)
            if start_idx < 0:
                # 保留最后 1 个字节，防止 START_FLAG 被拆成两段
                if len(self._buf) > 1:
                    del self._buf[:-1]
                return

            if start_idx > 0:
                del self._buf[:start_idx]

            # 最小有体帧长度：2 + 4 + 3 + 1 + 4 + 2 = 16
            if len(self._buf) < 16:
                return

            if self._buf[2:6] != DEVICE_TYPE:
                # 不是合法设备帧，丢掉一个起始字节后继续找
                del self._buf[:2]
                continue

            head0 = self._buf[6]
            head1 = self._buf[7]
            head2 = self._buf[8]
            body_included = self._buf[9]

            # 只保留目标高速数据帧
            if (head0, head1) != self.target_head:
                del self._buf[:2]
                continue

            # 只处理带 body 的帧
            if body_included != 0xDA:
                del self._buf[:2]
                continue

            body_len = int.from_bytes(self._buf[10:14], byteorder="little", signed=False)
            total_len = 16 + body_len  # 2+4+3+1+4+body+2

            if len(self._buf) < total_len:
                return

            if self._buf[14 + body_len : 16 + body_len] != END_FLAG:
                # 帧尾不对，说明当前对齐错了，继续重新找
                del self._buf[:2]
                continue

            body = bytes(self._buf[14 : 14 + body_len])
            del self._buf[:total_len]

            row = self._body_to_row(body)
            print("成功解析一帧！")

            self._frame_count += 1
            if self._frame_count % self.sample_stride != 0:
                continue

            self._rows.append(row)
            self._seq += 1
            self._update_event.set()

    def _body_to_row(self, body: bytes) -> np.ndarray:
        """
        将 Body 转成一行 numpy。

        振动解调：
        - little-endian int16
        - 换算：value / 256 * pi
        """
        row = np.frombuffer(body, dtype="<i2")

        if self.scale_to_pi:
            row = row.astype(np.float32) / 256.0 * np.pi
        else:
            row = row.astype(np.int16)

        return row

    async def run(
        self,
        infer_fn: Callable[[np.ndarray], Any],
        on_result: Callable[[str, int], None] | None = None,
    ) -> None:
        recv_task = asyncio.create_task(self.recv_loop())
        parse_task = asyncio.create_task(self.parse_loop())
        infer_task = asyncio.create_task(self.infer_loop(infer_fn, on_result=on_result))

        try:
            await asyncio.gather(recv_task, parse_task, infer_task)
        finally:
            recv_task.cancel()
            parse_task.cancel()
            infer_task.cancel()

    async def infer_loop(
        self,
        infer_fn: Callable[[np.ndarray], Any],
        on_result: Callable[[str, int], None] | None = None,
    ) -> None:
        """
        只要有新帧就触发推理。
        推理放到线程里，避免阻塞事件循环。
        """
        last_seq = -1

        while not self._closed:
            try:
                await self.wait_next_update()
            except asyncio.CancelledError:
                break

            if self._closed:
                break

            if self.seq == last_seq:
                continue

            matrix = self.latest_matrix()
            if matrix is None:
                continue

            if matrix.shape[0] < self.history_len:
                continue

            last_seq = self.seq

            # 重型推理放线程池，避免阻塞收包和解析，因为 task 本质是单线程
            if not self.enable_infer:
                continue

            result = await asyncio.to_thread(infer_fn, matrix)
            print(f"infer result: {result} | seq: {self.seq}")

            if on_result is not None:
                on_result(str(result), self.seq)


def waterfall_infer(matrix: np.ndarray) -> str:
    """
    将最新窗口画成瀑布图，并保存到本地文件。
    """
    data = bandpass_2d(matrix, fs=1000, low=5.0, high=300.0, order=4)
    if data.ndim != 2 or data.size == 0:
        return "invalid matrix"

    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)

    im = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )

    ax.set_xlabel("Channel")
    ax.set_ylabel("Time step")
    ax.set_title("DAS Waterfall")

    fig.colorbar(im, ax=ax, label="Phase")

    fig.tight_layout()

    now = datetime.now()
    filename = now.strftime("%Y%m%d_%H%M%S")

    out_path = f"waterfall_{filename}.png"
    fig.savefig(out_path)

    plt.close(fig)
    return out_path


def build_realtime_infer_fn(
    ckpt_path: str | Path,
    device: str = "cuda",
    config_getter: Callable[[], dict] | None = None,
) -> Callable[[np.ndarray], str]:
    inferencer = TFTInferencer(ckpt_path, device=device)

    def _infer(matrix: np.ndarray) -> str:
        cfg = config_getter() if config_getter is not None else {}

        result = inferencer.predict(
            matrix,
            batch_size=int(cfg.get("batch_size", 256)),
            fs=int(cfg.get("target_sample_rate", 1000)),
            low_cut=float(cfg.get("low_cut", 5.0)),
            high_cut=float(cfg.get("high_cut", 300.0)),
            order=4,
        )

        if not result["has_positive_windows"]:
            return f"pred=0(nature)"

        segs = []
        for w in result["window_results"]:
            segs.append(
                f"[{w['channel_start']}:{w['channel_end']}]={w['pred_id']}({w['pred_name']})"
            )
        return " | ".join(segs)

    return _infer


def build_infer_with_waterfall(ckpt_path: str | Path, device: str = "cuda") -> Callable[[np.ndarray], str]:
    inferencer = TFTInferencer(ckpt_path, device=device)

    def _infer(matrix: np.ndarray) -> str:
        result = inferencer.predict(matrix)

        if not result["has_positive_windows"]:
            pred_text = f"pred=0({result['pred_name']})"
        else:
            segs = []
            for w in result["window_results"]:
                segs.append(
                    f"[{w['channel_start']}:{w['channel_end']}]={w['pred_id']}({w['pred_name']})"
                )
            pred_text = " | ".join(segs)

        waterfall_path = waterfall_infer(matrix)
        return f"{pred_text} | waterfall={waterfall_path}"

    return _infer


async def main():
    receiver = DASReceiver(
        bind_ip="192.168.1.100",
        bind_port=8009,
        history_len=1024,
        target_head=(0x80, 0x11),
        scale_to_pi=True,
        raw_queue_size=2048,
        actual_sample_rate=2000,
        target_sample_rate=1000,
        low_cut=5.0,
        high_cut=300.0,
        batch_size=256,
        enable_infer=True,
    )

    receiver.start()

    client = mqtt.Client(client_id=MQTT_CLIENT_ID, protocol=mqtt.MQTTv311)

    def publish_state(kind: str, payload: dict) -> None:
        client.publish(
            mqtt_topic("state", kind),
            json.dumps(payload, ensure_ascii=False),
            qos=0,
            retain=False,
        )

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("MQTT connected")
            client.subscribe(mqtt_topic("cmd", "#"))
            publish_state("status", {"status": "mqtt_connected"})
            publish_state("config", receiver.get_config())
        else:
            print(f"MQTT connect failed: rc={rc}")

    def on_message(client, userdata, msg):
        text = msg.payload.decode("utf-8", errors="ignore").strip()

        try:
            data = json.loads(text) if text else {}
        except json.JSONDecodeError:
            data = {"raw": text}

        if msg.topic.endswith("/cmd/request_state"):
            publish_state("config", receiver.get_config())
            publish_state("status", {"status": "state_sent"})

        elif msg.topic.endswith("/cmd/set_config"):
            receiver.apply_config(data)
            publish_state("config", receiver.get_config())
            publish_state("status", {"status": "config_updated"})

        elif msg.topic.endswith("/cmd/control"):
            action = str(data.get("action", "")).lower()

            if action == "start":
                print("收到了")
                send_cmd(build_start_stream_cmd())
                publish_state("status", {"status": "start_sent"})

            elif action == "stop":
                send_cmd(build_stop_stream_cmd())
                publish_state("status", {"status": "stop_sent"})

            else:
                publish_state("status", {"status": f"unknown_action: {action}"})

    def publish_result(text: str, seq: int) -> None:
        publish_state(
            "result",
            {
                "seq": seq,
                "text": text,
                "time": datetime.now().isoformat(timespec="seconds"),
            },
        )

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
    client.loop_start()

    # 这里不要自动 start，交给网页按钮控制
    infer_fn = build_realtime_infer_fn(
        ckpt_path=DEFAULT_CKPT_PATH,
        device="cuda",
        config_getter=receiver.get_infer_config,
    )

    try:
        await receiver.run(infer_fn, on_result=publish_result)
    finally:
        send_cmd(build_stop_stream_cmd())
        client.loop_stop()
        client.disconnect()
        receiver.close()


if __name__ == "__main__":
    asyncio.run(main())