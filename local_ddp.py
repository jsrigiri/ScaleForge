import os
import subprocess
import sys


def main():

    world_size = 2

    master_addr = "127.0.0.1"
    master_port = "29501"

    extra_args = sys.argv[1:]

    processes = []

    for rank in range(world_size):

        env = os.environ.copy()

        env["MASTER_ADDR"] = master_addr
        env["MASTER_PORT"] = master_port
        env["WORLD_SIZE"] = str(world_size)
        env["RANK"] = str(rank)
        env["LOCAL_RANK"] = str(rank)

        env["USE_LIBUV"] = "0"

        cmd = [
            sys.executable,
            "train.py",
        ] + extra_args

        p = subprocess.Popen(
            cmd,
            env=env,
        )

        processes.append(p)

    exit_codes = [
        p.wait()
        for p in processes
    ]

    if any(code != 0 for code in exit_codes):
        raise SystemExit(
            f"DDP failed with exit codes: {exit_codes}"
        )


if __name__ == "__main__":
    main()