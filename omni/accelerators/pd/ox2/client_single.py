#!/usr/bin/python
import zmq
import msgpack
import uuid
import argparse
from typing import TYPE_CHECKING, Any, Optional, Tuple, List, Dict, Union


class RouterDealerClient:
    def __init__(self, server_address="tcp://localhost:5555"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)

        client_id = f"client_{uuid.uuid4().hex[:8]}".encode('utf-8')
        self.socket.setsockopt(zmq.IDENTITY, client_id)
        self.socket.connect(server_address)

        print(f"[OK] Connected to ox ZMQ at {server_address}  (client_id={client_id.decode()})")

    def send_request(self, request_data):
        try:
            packed_data = msgpack.packb(request_data)
            self.socket.send(packed_data)
            print(f"[SEND] {request_data}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to send: {e}")
            return False

    # def receive_response(self):
    #     try:
    #         frames = self.socket.recv_multipart()
    #         responses = [msgpack.unpackb(f) for f in frames]
    #         return responses
    #     except Exception as e:
    #         print(f"[ERROR] Failed to receive response: {e}")
    #         return None

    def receive_response(self):
        try:
            frames = self.socket.recv_multipart()
            payload = frames[-1]
            response = msgpack.unpackb(payload)
            return response
        except Exception as e:
            print(f"[ERROR] Failed to send: {e}")
            return None

    def close(self):
        self.socket.close()
        self.context.term()
        print("Client closed.")


def parse_user_input(line: str):
    """解析:
       request_id  table_id  src_ids  dst_ids  cluster_id
       示例:
       0 0 1,2,3 4,5,6 0
    """
    parts = line.strip().split()
    if len(parts) != 5:
        print("[ERROR] 输入格式不对，需要 5 个字段")
        return None

    request_id = parts[0]
    table_id = int(parts[1])
    src_ids = [int(x) for x in parts[2].split(",")] if parts[2] else []
    dst_ids = [int(x) for x in parts[3].split(",")] if parts[3] else []
    cluster_id = int(parts[4])

    return {
        "request_id": request_id,
        "table_id": table_id,
        "src_block_ids": list(src_ids),
        "dst_block_ids": list(dst_ids),
        "cluster_id": cluster_id,
    }


def main():
    # ----------- 解析命令行参数 -----------
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=5555, help="ZMQ 端口号")
    args = parser.parse_args()

    address = f"tcp://7.150.12.133:{args.port}"
    print(f"[INFO] 使用 ZMQ 地址: {address}")

    # ----------- 启动客户端 -----------
    client = RouterDealerClient(address)

    print("\n=== ZMQ Interactive Request Sender ===")
    print("输入格式：")
    print("    request_id  table_id  src_ids  dst_ids  cluster_id")
    print("示例：")
    print("    0 0 1,2,3 4,5,6 0")
    print("输入 exit 退出\n")

    while True:
        line = input("请输入 request_data > ").strip()
        if line.lower() in ("exit", "quit"):
            break

        req = parse_user_input(line)
        if req is None:
            continue

        # 发送一次请求
        client.send_request(req)

        # 连续收这个 request 对应的所有消息（每层 + 整体完成）
        print("[WAIT] 等待 ox 返回多条消息...")

        while True:
            resp = client.receive_response()
            print("[RECV]", resp)

            if resp is None:
                # 收不到就继续等，不要跳回输入
                continue

            if not isinstance(resp, dict):
                continue

            req_id = resp.get("request_id")
            if not isinstance(req_id, str):
                continue

            # 每一层完成的通知：形如 "0#L0", "0#L1" ...
            if "#L" in req_id:
                print(f"[LAYER DONE] {req_id}")
                continue

            # 整个 request 完成：形如 "0"
            if req_id == req["request_id"]:
                print("[DONE] request 完整接收")
                break

    client.close()



if __name__ == "__main__":
    main()
