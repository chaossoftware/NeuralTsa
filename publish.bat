@echo off

set VERSION=%1
set OUT_DIR=%2
set PROJ_PATH=%cd%\src\NeuralNetTsa\NeuralNetTsa.csproj

dotnet publish %PROJ_PATH% --configuration Release --framework net6.0 --output %OUT_DIR%\neural-tsa\net6.0-windows -p:VersionPrefix=%VERSION%