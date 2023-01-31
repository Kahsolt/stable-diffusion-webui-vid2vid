@ECHO OFF

SET SD_PATH=%~dp0\..\..\..
PUSHD %SD_PATH%
SET SD_PATH=%CD%
POPD

SET VENV_PATH=%SD_PATH%\venv
ECHO VENV_PATH = %VENV_PATH%
ECHO.

SET PATH=%VENV_PATH%\Scripts;%PATH%
DOSKEY py=python.exe $*
DOSKEY ml=python.exe show_mask_lowcut.py $*
DOSKEY ss=python.exe show_sigma_schedule.py $*

ECHO Cammand shortcuts:
ECHO.   py           start python shell
ECHO.   ss           show_sigma_schedule.py
ECHO.   ml ^<file^>    show_mask_lowcut.py
ECHO.
ECHO Python executables:

CMD /K "activate.bat & where python.exe"
