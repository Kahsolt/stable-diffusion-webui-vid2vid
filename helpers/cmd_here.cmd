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
DOSKEY ml=python.exe mask_lowcut.py $*
DOSKEY im=python.exe image_mode.py $*
DOSKEY fd=python.exe frame_delta.py $*
DOSKEY fdd=python.exe frame_delta_denoise.py $*

ECHO Cammand shortcuts:
ECHO   py                    start python shell
ECHO   ss                    run sigma_schedule.py
ECHO   sr                    run size_recommend.py
ECHO   ml ^<file^>             run mask_lowcut.py
ECHO   im ^<file^>             run image_mode.py
ECHO   fd ^<folder^>           run frame_delta.py ^(make^)
ECHO   fd ^<folder^> ^<folder^>  run frame_delta.py ^(compare^)
ECHO   fdd ^<file^>             run frame_delta_denoise.py
ECHO.
ECHO Python executables:

CMD /K "activate.bat & where python.exe"
