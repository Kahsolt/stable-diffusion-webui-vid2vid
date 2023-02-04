@ECHO OFF

SET SD_PATH=%~dp0\..\..\..
PUSHD %SD_PATH%
SET SD_PATH=%CD%
POPD

REM SET VENV_PATH=C:\Miniconda3
SET VENV_PATH=%SD_PATH%\venv
ECHO VENV_PATH = %VENV_PATH%
ECHO.

SET PATH=%VENV_PATH%\Scripts;%PATH%
SET PY_BIN=python.exe

%PY_BIN% --version > NUL
IF ERRORLEVEL 1 GOTO die


DOSKEY py=%PY_BIN% $*
DOSKEY ss=%PY_BIN% sigma_schedule.py $*
DOSKEY sr=%PY_BIN% size_recommend.py $*
DOSKEY im=%PY_BIN% img_utils.py $*
DOSKEY md=%PY_BIN% mask_depth.py $*
DOSKEY mm=%PY_BIN% mask_motion.py $*

ECHO Command shortcuts:
ECHO   py     start python shell
ECHO   ss     run sigma_schedule.py, view schedulers and sigma curves
ECHO   sr     run size_recommend.py, giving advices on img2img canvas size
ECHO   im     run img_utils.py, inspect basic image info
ECHO   md     run mask_depth.py, draw grid
ECHO   mm     run mask_motion.py, draw grid

CMD /K "activate.bat"


GOTO EOF

:die
ECHO ERRORLEVEL: %ERRORLEVEL%
ECHO PATH: %PATH%
ECHO Python executables:
WHERE python.exe

PAUSE

:EOF
