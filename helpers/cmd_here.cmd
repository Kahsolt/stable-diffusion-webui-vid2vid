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
DOSKEY im=%PY_BIN% img_utils.py $*
DOSKEY sr=%PY_BIN% size_recommend.py $*
DOSKEY ss=%PY_BIN% sigma_schedule.py $*
DOSKEY md=%PY_BIN% mask_depth_grid.py $*
DOSKEY mm=%PY_BIN% mask_motion_grid.py $*
DOSKEY fdc=%PY_BIN% debug_fdc.py $*

ECHO Command shortcuts:
ECHO   py     start python shell
ECHO   im     img_utils, inspect basic image info
ECHO   sr     size_recommend, giving advices on img2img canvas size
ECHO   ss     sigma_schedule, view schedulers and sigma curves
ECHO   md     mask_depth_grid, draw grid view of low-cut
ECHO   mm     mask_motion_grid, draw grid view of low-cut ^& high-ext
ECHO   fdc    debug_fdc, inspect into the FDC trick frame by frame

CMD /K "activate.bat"


GOTO EOF

:die
ECHO ERRORLEVEL: %ERRORLEVEL%
ECHO PATH: %PATH%
ECHO Python executables:
WHERE python.exe

PAUSE

:EOF
