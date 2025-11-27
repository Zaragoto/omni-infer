#!/usr/bin/python3
import zmq
import msgpack
import argparse
import time


class ServerCmdClient:
    """
    对接 C++ ox_server 的 ZMQ_PULL 端。
    这里用 PUSH，只发不收。
    """
    def __init__(self, address="tcp://localhost:6000"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)

        print(f"[server_single] Connecting to ZMQ PULL at: {address}")
        self.socket.connect(address)

    def send_cmd(self, table_id, layer_id, block_ids, rank, seq=None):
        """
        构造 ZmqCmd:
        struct ZmqCmd {
            table_id_t table_id = 0;
            size_t layer_id = 0;
            block_list_t block_ids;
            int rank = 0;
        };
        """
        try:
            cmd_dict = {
                "table_id": table_id,
                "layer_id": layer_id,
                "block_ids": block_ids,
                "rank": rank
            }

            packed = msgpack.packb(cmd_dict)
            self.socket.send(packed)

            if seq is not None:
                print(f"[server_single] Sent #{seq} ZmqCmd: {cmd_dict}")
            else:
                print(f"[server_single] Sent ZmqCmd: {cmd_dict}")

        except Exception as e:
            print(f"[server_single] Error sending command: {e}")

    def close(self):
        self.socket.close()
        self.context.term()
        print("[server_single] Closed connection.")


def interactive_loop(address):
    client = ServerCmdClient(address)

    print("\n=== server_single.py interactive mode ===")
    print("手动输入 ZmqCmd 的内容并发送给 C++ ox_server.")
    print("输入 'exit' 结束.\n")

    try:
        while True:
            line = input("输入: table layer block_ids rank > ").strip()
            if line.lower() in ["exit", "quit"]:
                break

            # 解析输入，例如：0 3 1,2,3 0
            parts = line.split()
            if len(parts) != 4:
                print("格式错误，应为: table layer block_ids rank")
                continue

            table_id = int(parts[0])
            layer_id = int(parts[1])
            block_ids = [int(x) for x in parts[2].split(",") if x]
            rank = int(parts[3])

            client.send_cmd(table_id, layer_id, block_ids, rank)

    except KeyboardInterrupt:
        pass
    finally:
        client.close()


def batch_send_layers(address, num_layers: int = 61, sleep_ms: int = 0):
    """
    批量模拟：同一个 uid（同一批 block_ids），连续发送 layer_id=0..num_layers-1 的 ZmqCmd。
    这样可以观察 C++ listen_zmq_global 是否只处理了前 N 条。
    """
    client = ServerCmdClient(address)

    # 这里固定一个 block_ids，用来生成同一个 uid
    # 你可以改成你实际场景里的 block_list（例如 [1,2,3,...]）
    block_ids = [1, 2, 3, 4]

    table_id = 0
    rank = 0

    print(f"\n=== server_single.py batch mode ===")
    print(f"将连续发送 {num_layers} 条 ZmqCmd，layer_id = 0..{num_layers-1}")
    print(f"table_id={table_id}, block_ids={block_ids}, rank={rank}")
    if sleep_ms > 0:
        print(f"每条之间 sleep {sleep_ms} ms")
    print()

    try:
        for i in range(num_layers):
            client.send_cmd(table_id=table_id,
                            layer_id=i,
                            block_ids=block_ids,
                            rank=rank,
                            seq=i + 1)

            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000.0)

    except KeyboardInterrupt:
        pass
    finally:
        client.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="server_single.py for testing ox_server's ZMQ commands"
    )
    parser.add_argument("--host", type=str, default="localhost",
                        help="ZMQ host (default: localhost)")
    parser.add_argument("--port", type=int, default=6000,
                        help="ZMQ port (default: 6000)")
    parser.add_argument("--mode", type=str, choices=["interactive", "batch"],
                        default="interactive",
                        help="Mode: interactive or batch (default: interactive)")
    parser.add_argument("--num-layers", type=int, default=61,
                        help="Number of layers to send in batch mode (default: 61)")
    parser.add_argument("--sleep-ms", type=int, default=0,
                        help="Sleep milliseconds between sends in batch mode (default: 0)")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    address = f"tcp://{args.host}:{args.port}"

    if args.mode == "interactive":
        interactive_loop(address)
    else:
        batch_send_layers(address, num_layers=args.num_layers, sleep_ms=args.sleep_ms)