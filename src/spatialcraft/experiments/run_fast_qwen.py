"""Opt-in entry point: unchanged experiment protocol, explicit FLA/two-GPU backend."""

from . import run, runtime
from .fast_qwen import FastAuditedProvider, FastExperimentRuntime


def main():
    # run.main imports ExperimentRuntime lazily after all normal preflight checks.
    # This process-local replacement never affects another job or default entry.
    runtime.ExperimentRuntime = FastExperimentRuntime
    runtime.AuditedProvider = FastAuditedProvider
    run.main()


if __name__ == "__main__":
    main()
