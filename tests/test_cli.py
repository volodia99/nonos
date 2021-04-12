from nonos.main import main, AnalysisNonos
import os
import toml

def test_no_inifile(capsys, tmp_path):
    os.chdir(tmp_path)
    ret = main([])
    assert ret != 0
    out, err = capsys.readouterr()
    assert out == ""
    assert err == "Error | idefix.ini, pluto.ini or variables.par not found.\n"


def test_default_conf(capsys, tmp_path):
    os.chdir(tmp_path)
    ret = main(["-config"])
    assert ret == 0
    out, err = capsys.readouterr()
    assert err == ""

    # validate output is reusable
    with open("config.toml", "w") as fh:
        fh.write(out)
    AnalysisNonos(directory_of_script=".")
