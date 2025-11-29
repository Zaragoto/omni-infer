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

    def receive_response(self, timeout_ms=10000):
        """接收响应，支持超时"""
        try:
            self.socket.RCVTIMEO = timeout_ms
            frames = self.socket.recv_multipart()
            responses = []
            for f in frames:
                try:
                    resp = msgpack.unpackb(f)
                    responses.append(resp)
                except:
                    # 如果不是msgpack数据，可能是身份帧
                    pass
            return responses
        except zmq.Again:
            print(f"[TIMEOUT] No response received within {timeout_ms}ms")
            return None
        except Exception as e:
            print(f"[ERROR] Failed to receive response: {e}")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=5555, help="ZMQ 端口号")
    args = parser.parse_args()

    address = f"tcp://7.150.12.133:{args.port}"
    print(f"[INFO] 使用 ZMQ 地址: {address}")

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

        if client.send_request(req):
            print("[WAIT] 等待 ox 返回结果...")
            
            # 持续接收响应，直到收到完成信号
            layers_received = set()
            while True:
                resp = client.receive_response(timeout_ms=5000)
                if resp is None:
                    print("[TIMEOUT] 等待响应超时")
                    break
                
                for response in resp:
                    if isinstance(response, dict):
                        request_id = response.get('request_id', '')
                        layer_id = response.get('layer_id', -1)
                        success = response.get('success', False)
                        
                        if layer_id == -1:
                            print(f"[COMPLETE] 请求 {request_id} 全部完成!")
                            break
                        elif layer_id >= 0:
                            if layer_id not in layers_received:
                                layers_received.add(layer_id)
                                print(f"[LAYER] 请求 {request_id} 层 {layer_id} 接收成功: {success}")
                        else:
                            print(f"[RESPONSE] {response}")
                    
                # 检查是否应该退出接收循环
                if any(isinstance(r, dict) and r.get('layer_id') == -1 for r in resp if isinstance(r, dict)):
                    break

    client.close()


if __name__ == "__main__":
    main()