import os
import sys
import fcntl
import socket
import struct
import argparse
import warnings
import subprocess

def get_path_before_omniinfer():
    """Get the base path before the 'omniinfer' directory in the current script's path.

    Returns:
        str: The path segment before 'omniinfer' directory.
    Raises:
        ValueError: If 'omniinfer' directory is not found in the path.
    """
    # Get absolute path of the currently executing script
    script_path = os.path.abspath(sys.argv[0])

    # Split path into components using OS-specific separator
    path_parts = script_path.split(os.sep)

    # Find the index of 'omniinfer' in the path components
    try:
        omni_index = path_parts.index('omniinfer')
    except ValueError:
        raise ValueError("'omniinfer' directory not found in path")

    # Reconstruct path up to (but not including) 'omniinfer'
    before_omni = os.sep.join(path_parts[:omni_index])

    return before_omni

def get_network_interfaces():
    """
    Retrieves primary network interface information excluding loopback.
    Returns a dictionary with interface name and its IP address.
    Falls back to 'eth0' if no interfaces found.
    """
    # List all network interfaces except loopback (lo)
    if_names = [name for name in os.listdir('/sys/class/net') if name != 'lo']

    # Select first available interface or default to 'eth0'
    if_name = if_names[0] if if_names else 'eth0'

    try:
        # Get IP address for selected interface
        ip = get_ip_address(if_name)

        # Compose result dictionary
        interfaces = {
            'if_name': if_name,  # Network interface name
            'ip': ip             # IPv4 address of the interface
        }
    except Exception as e:
        print(f"Error getting network interfaces: {if_name}:{e}")
        interfaces = {}  # Return empty dict on error

    return interfaces

def get_ip_address(if_name):
    """
    Retrieves the IPv4 address of a network interface using ioctl.
    Args:
        if_name: Name of the network interface (e.g., 'eth0')
    Returns:
        IPv4 address as string
    Raises:
        RuntimeError on failure
    """
    # Create UDP socket for ioctl operations
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # SIOCGIFADDR = 0x8915 (get interface address)
        # Pack interface name into byte structure (max 15 chars)
        packed_ifname = struct.pack('256s', if_name[:15].encode('utf-8'))

        # Perform ioctl call to get interface info
        # [20:24] slices the IP address from the returned structure
        ip_bytes = fcntl.ioctl(
            s.fileno(),
            0x8915,  # SIOCGIFADDR constant
            packed_ifname
        )[20:24]

        # Convert packed binary IP to dotted-quad string
        return socket.inet_ntoa(ip_bytes)
    except Exception as e:
        raise RuntimeError(f"Failed to get IP address for interface {if_name}: {e}")


def run_default_mode(args):
    """Run in mixed deployment mode"""

    if (args.network_interface is not None and args.host_ip is None) or \
        (args.network_interface is None and args.host_ip is not None):
        warnings.warn(
            "For best results, please specify both --network-interface AND --host-ip "
            "together. Falling back to auto-detection for missing values.",
            RuntimeWarning
        )
    # Get network interface
    if args.network_interface:
        intf = {'if_name': args.network_interface, 'ip': get_ip_address(args.network_interface)}
    else:
        intf = get_network_interfaces()
        if not intf:
            raise RuntimeError("No network interface found and none specified")

    # Override IP if host-ip was specified
    if args.host_ip:
        intf['ip'] = args.host_ip


    env = os.environ.copy()
    # Network config for distributed training
    env['GLOO_SOCKET_IFNAME'] = intf['if_name']
    env['TP_SOCKET_IFNAME'] = intf['if_name']

    # Hardware and framework settings
    env['ASCEND_RT_VISIBLE_DEVICES'] = args.server_list  # Use first 8 NPUs
    env['VLLM_WORKER_MULTIPROC_METHOD'] = 'fork'   # Process spawning method
    env['VLLM_USE_V1'] = '1'
    env['ASCEND_GLOBAL_LOG_LEVEL'] = '3'
    env['VLLM_LOGGING_LEVEL'] = 'INFO'

    env['HCCL_OP_EXPANSION_MODE'] = 'AIV'
    env['TNG_HOST_COPY'] = '1'
    env['TASK_QUEUE_ENABLE'] = '2'
    env['CPU_AFFINITY_CONF'] = '2'

    if args.graph_true.lower() == 'false':
        extra_args = '--enforce-eager '   # Disable graph execution
        if args.local_media_path is not None:
            extra_args += f'--allowed-local-media-path {args.local_media_path} '

        # Base command for API server
        cmd = [
            'python',  os.path.join(args.code_path, 'omniinfer/tools/scripts/start_api_servers.py'),
            '--num-servers', '1',
            '--model-path', args.model_path,
            '--master-ip', intf['ip'],       # Coordinator IP
            '--master-port', args.master_port,         # Coordinator port
            '--tp', '1',
            '--served-model-name', args.model_name,
            '--base-api-port', args.https_port,        # HTTP service port
            '--log-dir', args.log_path,  # Log directory
            '--gpu-util', '0.9',  # Target NPU utilization
            '--max-model-len', args.max_model_len,
            '--extra-args', extra_args
        ]

        if hasattr(args, 'additional_config') and args.additional_config:
            cmd.extend(['--additional-config', args.additional_config])

    # Graph mode specific optimizations
    elif args.graph_true.lower() == 'true':
        additional_config = args.additional_config if args.additional_config else \
                '{"graph_model_compile_config": {"level":1, "use_ge_graph_cached":false}, "enable_hybrid_graph_mode": true}'

        extra_args = f'--max-num-batched-tokens {args.max_num_batched_tokens} --max-num-seqs {args.max_num_seqs} '
        if args.local_media_path is not None:
            extra_args += f'--allowed-local-media-path {args.local_media_path} '

        cmd = [
            'python',  os.path.join(args.code_path,'omniinfer/tools/scripts/start_api_servers.py'),
            '--num-servers', '1',
            '--model-path', args.model_path,
            '--master-ip', intf['ip'],       # Coordinator IP
            '--master-port', args.master_port,    # Coordinator port
            '--tp', '1',
            '--served-model-name', args.model_name,
            '--base-api-port', args.https_port,        # HTTP service port
            '--log-dir', args.log_path,  # Log directory
            '--gpu-util', '0.9',  # Target NPU utilization
            '--max-model-len', args.max_model_len,
            '--extra-args', extra_args,
            '--additional-config', additional_config
        ]

    print(f'Starting with NIC: {intf["if_name"]}, IP: {intf["ip"]}')

    subprocess.run(cmd, env=env)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description="OmniInfer Deployment Script - Launches model servers in different deployment modes"
        )
    parser.add_argument('--model-path', type=str, required=True,
                        help="Absolute path to the model checkpoint directory (required)")
    parser.add_argument('--graph-true', type=str, default='false',
                        help="Enable graph optimization mode: 'true' for optimized execution, 'false' for standard mode (default: false)")

    parser.add_argument('--network-interface', type=str, default=None,
                        help="Network interface name for distributed communication (default: auto-detect)")
    parser.add_argument('--host-ip', type=str, default=None,
                        help="Local machine's IP address for service binding (default: auto-detect from network interface)")
    parser.add_argument('--model-name', type=str, default='default_model',
                        help="Model identifier used for API endpoints (default: default_model)")
    parser.add_argument('--max-model-len', type=str, default='8192',
                        help="Maximum context length supported by the model in tokens (default: 8192)")
    parser.add_argument('--max-num-batched-tokens', type=str, default='32768',
                        help="Maximum context length supported by the model in tokens (default: 32768)")
    parser.add_argument('--max-num-seqs', type=str, default='64',
                        help="Maximum number of sequences supported by the model in tokens (default: 64)")
    parser.add_argument('--log-path', type=str, default='./apiserverlog',
                        help="Directory path for storing service logs (default: ./apiserverlog)")

    parser.add_argument('--local-media-path', type=str, default=None,
                        help="allowed local media path (default: None)")

    parser.add_argument('--server-list', type=str, default='0',
                        help="NPU device ID (default: 0)")

    parser.add_argument('--master-port', type=str, default='8888',
                        help="The --master-port parameter in your command specifies the central coordination port used" \
                             " for distributed communication between different components of the inference system.")
    parser.add_argument('--https-port', type=str, default='8001',
                        help="Port for accepting HTTPS requests (default: 8001)")

    parser.add_argument('--additional-config', type=str, default=None,
                        help="JSON format advanced config, e.g. '{\"enable_graph_mode\":true}'")


    args = parser.parse_args()
    args.code_path = get_path_before_omniinfer()  # Get base path before 'omniinfer'

    # Validate critical paths
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path does not exist: {args.model_path}")

    run_default_mode(args)
