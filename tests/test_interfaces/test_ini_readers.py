from math import isnan

import inifix
import pytest

from nonos._readers.ini import Fargo3DReader, FargoADSGReader
from nonos._types import FrameType


class TestFargoReaders:

    ini_template = [
        "ninterm 10",
        "DT 0.2",
        "frame <FRAME>",
        "omegaframe 3.0",
    ]

    @pytest.mark.parametrize(
        "reader, template_lines",
        [
            (
                Fargo3DReader,
                [
                    "NINTERM 10",
                    "DT 0.2",
                    "FRAME %s",
                    "OMEGAFRAME 3.0",
                ],
            ),
            (
                FargoADSGReader,
                [
                    "Ninterm 10",
                    "DT 0.2",
                    "Frame %s",
                    "OmegaFrame 3.0",
                ],
            ),
        ],
    )
    @pytest.mark.parametrize(
        "FRAME, expected_frame, expected_rotational_rate",
        [
            ("F", FrameType.FIXED, 3.0),
            ("C", FrameType.COROT, float("nan")),
            ("test", FrameType.UNSET, 0.0),
        ],
    )
    def test_read_frame(
        self,
        tmp_path,
        reader,
        template_lines,
        FRAME,
        expected_frame,
        expected_rotational_rate,
    ):
        body = "\n".join(template_lines) % FRAME
        inifile = tmp_path / "variables.par"
        inifile.write_text(body)

        expected_meta = inifix.loads(body)

        ini = reader.read(inifile)
        assert ini.frame is expected_frame
        if isnan(expected_rotational_rate):
            assert isnan(ini.rotational_rate)
        else:
            assert ini.rotational_rate == expected_rotational_rate
        assert ini.output_time_interval == 2.0
        assert ini.meta == expected_meta
