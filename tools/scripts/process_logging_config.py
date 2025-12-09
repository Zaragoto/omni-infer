import os
import sys
import json
import re
import argparse
from typing import Any, Union

def replace_env(config_str):
    def replacer(match):
        var, default = match.groups()
        return os.getenv(var, default or "")
    return re.sub(r'\$\{([^:}]+)(?::-(.*?))?\}', replacer, config_str)


def set_env_logger_file(config_json):
    if isinstance(config_json, dict):
        return {k: set_env_logger_file(v) for k, v in config_json.items()}
    elif isinstance(config_json, list):
        return [set_env_logger_file(v) for v in config_json]
    elif isinstance(config_json, str):
        return replace_env(config_json)
    else:
        return config_json


def main():
    parser = argparse.ArgumentParser(description="Interpolate env vars in JSON config")
    parser.add_argument("input", help="Input config file (JSON)")
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    parser.add_argument("--inplace", action="store_true", help="Modify input file in-place")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        config_json = json.load(f)

    config_json = set_env_logger_file(config_json)

    output_target = args.output or (args.input if args.inplace else None)
    if output_target:
        with open(output_target, "w", encoding="utf-8") as f:
            json.dump(config_json, f)
    else:
        print(json.dumps(config_json))


if __name__ == "__main__":
    main()