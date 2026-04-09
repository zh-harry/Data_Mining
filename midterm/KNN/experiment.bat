@echo off
set PROGRAM=main.exe
set OUTPUT_FILE=results.txt

type nul > %OUTPUT_FILE%

echo Accuracy of KNN (w/o standardize) at k = 1 to 100:
echo ======================================

FOR /L %%k IN (1, 1, 100) DO (
    echo running for k = %%k ...
    %PROGRAM% false %%k >> %OUTPUT_FILE%
)

echo ======================================

echo Accuracy of KNN (w/ standardize) at k = 1 to 100:
echo ======================================

FOR /L %%k IN (1, 1, 100) DO (
    echo running for k = %%k ...
    %PROGRAM% true %%k >> %OUTPUT_FILE%
)

echo ======================================
echo Test completed
pause