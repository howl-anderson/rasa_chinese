import os
import shutil
import subprocess

import pytest


@pytest.fixture
def run_rasa_train():
    return _run_rasa_train


def _run_rasa_train(file__):
    # def _run_rasa_train():
    cwd = os.path.dirname(file__)
    # cwd = "."

    # remove cached model
    shutil.rmtree(os.path.join(cwd, "models"), ignore_errors=True)
    # no news is a good news
    return_code = subprocess.call("rasa train", shell=True, cwd=cwd)

    assert return_code == 0
