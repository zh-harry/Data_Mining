@echo off

cd ..\..\classification\KNN

python knn.py 21 0.7

cd ..\..\clustering\DBSCAN

set PROGRAM=main.exe
set OUTPUT_FILE=results.txt

type nul > %OUTPUT_FILE%

%PROGRAM% "..\..\classification\KNN\knn_prediction.csv" "..\..\data\dry_bean_test_std.csv" 1 17 >> %OUTPUT_FILE%

pause