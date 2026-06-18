@echo off

cd ..\..\classification\KNN

set THRESHOLD_PROB=0.9744188571144557
set K=34
set TRAIN="..\..\data\dry_bean_train_std.csv"
set TEST="..\..\data\dry_bean_test_std.csv"

echo ==============================
echo KNN+KMEANS
echo.
echo KNN classification:
echo (k = %K%, threshold probability = %THRESHOLD_PROB%)

python knn.py %TRAIN% %TEST% %K% %THRESHOLD_PROB%

cd ..\..\clustering\kmeans

set PROGRAM=main.exe
set OUTPUT_FILE=knn_kmeans.txt
set K=17
set CLASSIFY="..\..\classification\KNN\knn_prediction.csv"
set SOURCE="..\..\data\dry_bean_test_std.csv"

echo.
echo KMEANS clustering:
echo (k = %K%)

type nul > %OUTPUT_FILE%

%PROGRAM% %CLASSIFY% %SOURCE% %K% >> %OUTPUT_FILE%

echo ==============================
echo.

cd ..\..\classification\SVM

set KERNEL=poly
set THRESHOLD_PROB=0.9749742232127162
set TRAIN="..\..\data\dry_bean_train_std.csv"
set TEST="..\..\data\dry_bean_test_std.csv"

echo ==============================
echo SVM+KMEANS
echo SVM classification:
echo (kernel = %KERNEL%, threshold probability = %THRESHOLD_PROB%)

python svm.py %TRAIN% %TEST% %KERNEL% %THRESHOLD_PROB%

cd ..\..\clustering\kmeans

set PROGRAM=main.exe
set OUTPUT_FILE=svm_kmeans.txt
set K=18
set CLASSIFY="..\..\classification\SVM\svm_prediction.csv"
set SOURCE="..\..\data\dry_bean_test_std.csv"

echo.
echo KMEANS clustering:
echo (k = %K%)

type nul > %OUTPUT_FILE%

%PROGRAM% %CLASSIFY% %SOURCE% %K% >> %OUTPUT_FILE%

echo ==============================
echo.

pause