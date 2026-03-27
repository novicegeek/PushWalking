import subprocess
import os
import sys
import numpy as np
import pickle
import time
import struct


class C3DReader():
    """Read c3d file via subprocess to circumvent the conflict between ezc3d and opensim modules."""

    def __init__(self):
        self.script_path = r"D:\Overseas\German Sport University Cologne\20.Course Materials\TSM11-Project&Applied Research Methods\24.Scripts\readc3d_subprocess.py"
        # Start the persistent worker
        self.worker = subprocess.Popen(
            [sys.executable, self.script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0  # Unbuffered for immediate communication
        )

    def get_data_via_subprocess(self, filepath, moment_point=0):
        poll = self.worker.poll()
        if poll is not None:
            # Process died! Let's see why.
            stderr_data = self.worker.stderr.read().decode()
            raise RuntimeError(f"Subprocess died with code {poll}. Error: {stderr_data}")
        try:
            # 1. Send the filepath to the worker via stdin
            self.worker.stdin.write((filepath + "\n").encode())
            self.worker.stdin.flush()

            # 2. Read the size of the incoming pickle data (4 bytes)
            raw_size = self.worker.stdout.read(4)
            if not raw_size:
                return None, None
            size = struct.unpack('>I', raw_size)[0]

            # 3. Read the actual pickled data
            data_payload = self.worker.stdout.read(size)
            result = pickle.loads(data_payload)

            if "error" in result:
                print(f"Subprocess error: {result['error']}")
                return None, None
            
            print(f"Data reading successful: {os.path.basename(filepath)}")
            return result['data'], result['metadata']
        except Exception as e:
            print(f"Parent process error: {e}")
            return None, None

    def close(self):
        """Properly shut down the worker."""
        if self.worker:
            self.worker.stdin.close()
            self.worker.terminate()
            self.worker.wait()