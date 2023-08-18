from click.testing import CliRunner
from Treepoint_Tiler.main import main


def test_main():
    runner = CliRunner()
    result = runner.invoke(main, ["git_img.jpg", "outpath", -9999, 0.2, 64, 32])
    assert result != None
