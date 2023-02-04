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
DOSKEY ss=python.exe sigma_schedule.py $*
DOSKEY sr=python.exe size_recommend.py $*
DOSKEY im=python.exe image_info.py $*
DOSKEY md=python.exe mask_depth.py $*
DOSKEY mm=python.exe mask_motion.py $*
DOSKEY fd=python.exe frame_delta.py $*

ECHO Cammand shortcuts:
ECHO   py                    start python shell
ECHO   ss                    run sigma_schedule.py
ECHO   sr                    run size_recommend.py
ECHO   im ^<file^>             run image_info.py
ECHO   md ^<file^>             run mask_depth.py
ECHO   mm ^<file^>             run mask_motion.py
ECHO   fd ^<folder^>           run frame_delta.py ^(make^)
ECHO   fd ^<folder^> ^<folder^>  run frame_delta.py ^(compare^)
REM ECHO Python executables:

REM CMD /K "activate.bat & where python.exe"
CMD /K "activate.bat"
