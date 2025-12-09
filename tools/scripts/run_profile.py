#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#仅在新profile设置下使用
import argparse
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Dict, List
import yaml

def to_ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: to_ns(v) for k, v in obj.items()})
    return obj

def eval_jinja_expr(val: Any, ctx: Dict[str, Any]) -> Any:
    # 只处理类似 "{{ base_api_port + port_offset.D + node_rank }}" 的简单表达式
    if isinstance(val, str) and "{{" in val and "}}" in val:
        expr = val[val.find("{{") + 2 : val.rfind("}}")].strip()
        ns = to_ns(ctx)
        return eval(expr, {}, ns.__dict__)
    return val

def load_targets(inv: Dict[str, Any], groups: List[str]):
    all_vars = inv.get("all", {}).get("vars", {}) or {}
    children = inv.get("all", {}).get("children", {}) or {}
    targets = []
    for g in groups:
        hosts = children.get(g, {}).get("hosts", {}) or {}
        for name, hv in hosts.items():
            ctx = dict(all_vars)
            ctx.update(hv or {})
            host_ip = hv.get("host_ip") or hv.get("ansible_host")
            host_ip = eval_jinja_expr(host_ip, ctx)
            api_port = eval_jinja_expr(hv.get("api_port"), ctx)
            if host_ip is None or api_port is None:
                raise ValueError(f"{g}/{name} 缺少 host_ip 或 api_port")
            targets.append({"group": g, "name": name, "host_ip": str(host_ip), "api_port": int(api_port)})
    return targets

def post_curl(url: str) -> int:
    # 返回 HTTP 状态码，失败返回 0
    try:
        res = subprocess.run(
            ["curl", "-sS", "-X", "POST", url, "-o", "/dev/null", "-w", "%{http_code}"],
            check=False,
            capture_output=True,
            text=True,
        )
        code_str = (res.stdout or "").strip()
        return int(code_str) if code_str.isdigit() else 0
    except Exception:
        return 0

def main():
    ap = argparse.ArgumentParser(description="批量向 P/D 组 API 发送 start/stop profile 请求（简化版）")
    ap.add_argument("-i", "--inventory", default="ansible_inventory.yml")
    ap.add_argument("-g", "--groups", nargs="+", choices=["P", "D"], default=["P", "D"])
    ap.add_argument("-a", "--actions", nargs="+", choices=["start", "stop"], default=["start"])
    ap.add_argument("--scheme", default="http", choices=["http", "https"])
    args = ap.parse_args()

    with open(args.inventory, "r", encoding="utf-8") as f:
        inv = yaml.safe_load(f)

    targets = load_targets(inv, args.groups)
    if not targets:
        print("未找到目标主机")
        return 1

    total, ok = 0, 0
    for t in targets:
        for act in args.actions:
            endpoint = "start_profile" if act == "start" else "stop_profile"
            url = f"{args.scheme}://{t['host_ip']}:{t['api_port']}/{endpoint}"
            code = post_curl(url)
            success = 200 <= code < 300
            total += 1
            ok += 1 if success else 0
            print(f"[{'OK' if success else 'ERR'}] {t['group']}/{t['name']} {act}: {url} -> {code}")

    print(f"完成: 成功 {ok} / 总计 {total}")
    return 0 if ok == total else 1

if __name__ == "__main__":
    sys.exit(main())