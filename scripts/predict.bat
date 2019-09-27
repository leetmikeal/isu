@echo off
setlocal

if "%*" equ "" (
    echo "not found argument" 1>&2
    echo "  sh predict.sh [dataset name]" 1>&2
    echo "  ex) sh predict.sh alcon01" 1>&2
	pause
    exit /b
)
set L=%1
#exit /b

echo "dataset :"%L%
isu predict-2d ^
    --in-settings setting.ini ^
    --dataset %L% ^
    --verbose
if errorlevel 1 call :stop
isu predict-3d ^
    --in-settings setting.ini ^
    --dataset %L% ^
    --verbose
if errorlevel 1 call :stop
isu analyze ensemble ^
    --in-settings setting.ini ^
    --dataset %L% ^
    --verbose
if errorlevel 1 call :stop
isu analyze connection ^
    --in-settings setting.ini ^
    --dataset %L% ^
    --verbose
if errorlevel 1 call :stop
exit /b    
	
:stop
echo;
echo error
pause
exit /b