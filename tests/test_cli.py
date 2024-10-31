import os
import textwrap
from importlib.metadata import version

import inifix

from nonos.main import main


def test_no_inifile(capsys, tmp_path):
    os.chdir(tmp_path)
    ret = main([])
    assert ret != 0
    out, err = capsys.readouterr()
    assert out == ""
    assert "Could not find a parameter file in" in err


def test_default_conf(capsys):
    ret = main(["-config"])
    assert ret == 0
    out, err = capsys.readouterr()
    assert err == ""

    # validate output is reusable
    inifix.loads(out)


def test_version(capsys):
    ret = main(["-version"])
    assert ret == 0

    out, err = capsys.readouterr()
    assert err == ""
    assert out == version("nonos") + "\n"


def test_logo(capsys):
    ret = main(["-logo"])

    assert ret == 0
    out, err = capsys.readouterr()
    expected = textwrap.dedent(
        r"""
                                                                     `!)}$$$$})!`
                 `,!>|))|!,                                        :}&#&$}{{}$&#&};`
              ~)$&&$$}}$$&#$).                                   '}#&}|~'.```.'!($#$+
           `=$&$(^,..``.'~=$&&=                                `|&#}!'`         `'?$#}.
          !$&}:.`         `.!$#$'                             :$#$^'`       ``     .$#$`
         ^&&+.              `'(&$!                          `)&&),`                 !##!
        `$#^      `.   `     `={$&{`                       ^$&$(!'  ``     `,.  ``  !##!
        ,#$ ``                 .>}&${?!:'`   ```..'',:!+|{$&${:`   `'`             `}&}`
        `$$`                       ,;|{}$$$&&&&&$$$$}}()<!,`   '`                 `}&}`
         +&}`   `   |:|.\    |:|            `.```                                  :$$!
          !$$'`!}|  |:|\.\   |:|      __                      __       ___       .{#$
           '$&})$:  |:| \.\  |:|   /./  \.\   |:|.\  |:|   /./  \.\   |:|  \.\  '$$#}
            `}#&;   |:|  \.\ |:|  |:|    |:|  |:|\.\ |:|  |:|    |:|  |:|___     :$)&$`
            `{&!    |:|   \.\|:|  |:|    |:|  |:| \.\|:|  |:|    |:|       |:|    :!{&}`
           :$$,     |:|    \.|:|   \.\__/./   |:|  \.|:|   \.\__/./   \.\__|:|     `:}#}`
          ^&$.                                                                       .$#^
          +&$.                 ``'^)}$$$$$({}}}$$$$$$$$$$}}(|>!.`~:,.                 }#)
         '&#|                 ,|$##$>'`                `'~!)$##$$$)?^,`           ` :&&:
         ,&#}`  ``       .` `:{$&}:                          ~}&$)^^+=^`  `` ..  .|&#)
          |&#$:```   `` '::!}$&},                              !$&$|++^^:,:~',!!($#&^
           ,}&#${^~,,,:!|}$&&(.                                  ^$#$}{)|?|)(}$&#$?`
             :{&##$$$$$&##$).                                      ~($&#&&##&$}=,
               `:|}$$$$}):`

        Analysis tool for idefix/pluto/fargo3d simulations (in polar coordinates).
        """.lstrip("\n")
    )
    expected += f"Version {version('nonos')}\n"
    assert out == expected
    assert err == ""
