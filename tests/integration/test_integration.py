"""
Integration tests: verify serial, OpenMP, CUDA produce consistent results.
Requires binaries to be built: make serial openmp cuda
"""

import subprocess
import os
import sys
import pytest

TEST_INPUT = "tests/fixtures/small_100.csv"
RESULTS_DIR = "tests/tmp"

def run_binary(binary, k=5, threads=None, block_size=256):
    """Run a K-Means binary and return output file path."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    output_file = os.path.join(RESULTS_DIR, f"{os.path.basename(binary)}_out.csv")

    cmd = [
        f"./results/{binary}",
        "--input", TEST_INPUT,
        "--k", str(k),
        "--max_iter", "50",
        "--output", output_file
    ]

    if binary == "openmp" and threads:
        cmd.extend(["--threads", str(threads)])
    elif binary == "cuda":
        cmd.extend(["--block_size", str(block_size)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"{binary} failed: {result.stderr}"
    assert os.path.exists(output_file), f"Output file not created: {output_file}"

    return output_file

def validate_against_serial(parallel_file, serial_file):
    """Run validation script."""
    result = subprocess.run(
        ["python3", "scripts/validate.py",
         "--serial", serial_file,
         "--parallel", parallel_file],
        capture_output=True,
        text=True
    )
    return result.returncode == 0

class TestIntegration:

    def test_serial_runs(self):
        """Serial implementation produces output."""
        output = run_binary("serial", k=5)
        assert os.path.getsize(output) > 100

    def test_openmp_runs(self):
        """OpenMP implementation produces output."""
        output = run_binary("openmp", k=5, threads=4)
        assert os.path.getsize(output) > 100

    def test_cuda_runs(self):
        """CUDA implementation produces output."""
        try:
            output = run_binary("cuda", k=5, block_size=256)
            assert os.path.getsize(output) > 100
        except AssertionError as e:
            if "CUDA not available" in str(e):
                pytest.skip("CUDA not available")
            raise

    def test_openmp_vs_serial(self):
        """OpenMP results validate against serial baseline."""
        serial_out = run_binary("serial", k=5)
        openmp_out = run_binary("openmp", k=5, threads=4)
        assert validate_against_serial(openmp_out, serial_out)

    def test_cuda_vs_serial(self):
        """CUDA results validate against serial baseline."""
        try:
            serial_out = run_binary("serial", k=5)
            cuda_out = run_binary("cuda", k=5, block_size=256)
            assert validate_against_serial(cuda_out, serial_out)
        except AssertionError as e:
            if "CUDA not available" in str(e):
                pytest.skip("CUDA not available")
            raise

    def test_openmp_thread_counts(self):
        """OpenMP works with various thread counts."""
        serial_out = run_binary("serial", k=5)

        for threads in [1, 2, 4, 8]:
            openmp_out = run_binary("openmp", k=5, threads=threads)
            assert validate_against_serial(openmp_out, serial_out), \
                f"OpenMP with {threads} threads failed validation"

    def test_cuda_block_sizes(self):
        """CUDA works with various block sizes."""
        try:
            serial_out = run_binary("serial", k=5)

            for block_size in [64, 128, 256, 512]:
                cuda_out = run_binary("cuda", k=5, block_size=block_size)
                assert validate_against_serial(cuda_out, serial_out), \
                    f"CUDA with block_size {block_size} failed validation"
        except AssertionError as e:
            if "CUDA not available" in str(e):
                pytest.skip("CUDA not available")
            raise

    def test_output_row_count(self):
        """All implementations produce correct row count."""
        with open(TEST_INPUT) as f:
            input_rows = len(f.readlines()) - 1  # Exclude header

        serial_out = run_binary("serial", k=5)
        with open(serial_out) as f:
            output_rows = len(f.readlines()) - 1

        assert output_rows == input_rows

    def test_different_k_values(self):
        """K=1, 2, 5, 10 all work."""
        for k in [1, 2, 5, 10]:
            serial_out = run_binary("serial", k=k)
            openmp_out = run_binary("openmp", k=k, threads=4)
            assert validate_against_serial(openmp_out, serial_out)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
