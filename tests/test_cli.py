import os
import textwrap

import inifix

from nonos import __version__
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
    assert out == str(__version__) + "\n"


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
        """.lstrip(
            "\n"
        )
    )
    expected += f"Version {__version__}\n"
    assert out == expected
    assert err == ""
