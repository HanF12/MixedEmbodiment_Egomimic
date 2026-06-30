import argparse
import submitit
from egomimic.scripts.pl_train import main, train_argparse
import os
import datetime
import time
import sys

if __name__ == "__main__":
    # os.environ["TQDM_DISABLE"]="1"

    args = train_argparse()

    # the AutoExecutor class is your interface for submitting function to a cluster or run them locally.
    # The specified folder is used to dump job information, logs and result when finished
    # %j is replaced by the job id at runtime
    assert args.name is not None, "Must provide a name for the experiment"
    assert args.description is not None, "Must provide a description for the experiment"
    base_dir = f"/coc/flash7/rpunamiya6/EgoPlay/trained_models_highlevel/{args.name}/"
    if "DT" not in args.description:
        time_str = f"{args.description}_DT_{datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')}"
        args.description = time_str
    log_dir = os.path.join(base_dir, args.description, "slurm")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "command_log.txt")
    with open(log_file, "w") as f:
        f.write(" ".join(sys.argv))

    executor = submitit.AutoExecutor(folder=log_dir)
    # The AutoExecutor provides a simple abstraction over SLURM to simplify switching between local and slurm jobs (or other clusters if plugins are available).
    # specify sbatch parameters (here it will timeout after 4min, and run on dev)
    # This is where you would specify `gpus_per_node=1` for instance
    # Cluster specific options must be appended by the cluster name:
    # Eg.: slurm partition can be specified using `slurm_partition` argument. It
    # will be ignored on other clusters:
    if args.overcap:
        slurm_partition, slurm_account, slurm_qos = "overcap", "hoffman-lab", None
    else:
        slurm_partition, slurm_account, slurm_qos = (
            args.partition,
            args.partition,
            "short",
        )

    executor.update_parameters(
        slurm_job_name=args.description,
        slurm_partition=slurm_partition,
        slurm_account=slurm_account,
        cpus_per_task=13,
        nodes=args.num_nodes,
        slurm_ntasks_per_node=args.gpus_per_node,
        gpus_per_node=f"a40:{args.gpus_per_node}",
        slurm_qos=slurm_qos,
        slurm_mem_per_gpu="40G",
        timeout_min=60 * 24 * 2,
        slurm_exclude="starrysky",
    )
    # The submission interface is identical to concurrent.futures.Executor

    job = executor.submit(main, args)
