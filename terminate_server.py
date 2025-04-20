#!/usr/bin/env python3
"""
Script to terminate running MCP Memory server instances.

This script helps to cleanly terminate any running MCP Memory server
processes, which is useful during development or when you need to
restart the server.
"""

import os
import signal
import subprocess
import sys

def find_mcp_mem_processes():
    """Find running MCP Memory server processes."""
    try:
        # Use ps to find python processes
        result = subprocess.run(
            ["ps", "-ef"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        mcp_processes = []
        for line in result.stdout.splitlines():
            if "mcp-mem" in line and "python" in line and "terminate_server.py" not in line:
                parts = line.split()
                if len(parts) >= 2:
                    pid = int(parts[1])
                    mcp_processes.append((pid, line))
        
        return mcp_processes
    
    except subprocess.CalledProcessError:
        print("Error: Failed to execute process search command")
        return []
    except Exception as e:
        print(f"Error: {str(e)}")
        return []

def terminate_processes(processes):
    """Terminate the specified processes."""
    if not processes:
        print("No MCP Memory server processes found.")
        return
    
    print(f"Found {len(processes)} MCP Memory server processes:")
    for i, (pid, cmd) in enumerate(processes):
        print(f"{i+1}. PID {pid}: {cmd}")
    
    if len(processes) == 1:
        choice = 0
    else:
        try:
            choice = int(input("\nEnter number to terminate (0 for all, -1 to cancel): ")) - 1
        except ValueError:
            print("Invalid input. Aborting.")
            return
    
    if choice == -2:  # -1 after the -1 adjustment
        print("Operation cancelled.")
        return
    
    if choice == -1:  # 0 after the -1 adjustment
        # Terminate all processes
        for pid, _ in processes:
            try:
                os.kill(pid, signal.SIGTERM)
                print(f"Sent SIGTERM to process {pid}")
            except OSError as e:
                print(f"Failed to terminate process {pid}: {e}")
    elif 0 <= choice < len(processes):
        # Terminate selected process
        pid = processes[choice][0]
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"Sent SIGTERM to process {pid}")
        except OSError as e:
            print(f"Failed to terminate process {pid}: {e}")
    else:
        print("Invalid selection. Aborting.")

if __name__ == "__main__":
    processes = find_mcp_mem_processes()
    terminate_processes(processes)