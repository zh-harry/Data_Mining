@echo off
set PROGRAM=main.exe
set OUTPUT_FILE=results.txt

type nul > %OUTPUT_FILE%

%PROGRAM% "..\..\classification\KNN\knn_prediction.csv" >> %OUTPUT_FILE%

pause