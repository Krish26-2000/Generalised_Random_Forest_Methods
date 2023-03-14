"""This module contains the general configuration of the project."""
from pathlib import Path


SRC = Path(__file__).parent.resolve()
BLD = SRC.joinpath("..", "..", "bld").resolve()

TEST_DIR = SRC.joinpath("..", "..", "tests").resolve()
PAPER_DIR = SRC.joinpath("..", "..", "paper").resolve()

TASK_1 = ["outcome_train", "features_train", "treatment_train", "instrument_train", "features_test"]

TASK_2 = ["outcome2_train", "features2_train", "treatment2_train", "instrument2_train", "features2_test"]

GROUPS = ["marital_status", "qualification"]

__all__ = ["BLD", "SRC", "TEST_DIR"]

NO_LONG_TASKS = True
