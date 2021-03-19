from nonos.main import main
import os

def test_no_inifile(capsys, tmp_path):
    os.chdir(tmp_path)
    ret = main([])
    assert ret != 0
    out, err = capsys.readouterr()
    assert out == ""
    assert err == "Error | idefix.ini, pluto.ini or variables.par not found.\n"