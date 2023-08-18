from click.testing import CliRunner
from main import main
import numpy as np

def test_main():
    runner = CliRunner()
    runner.invoke(
        main, ["image","mask","out"]
    )
    assert None != main