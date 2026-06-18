@echo off


cd ..\..\classification\KNN

set THRESHOLD_PROB=0.9511737363335939
set K=40
set TRAIN="..\..\data\dry_bean_train_std.csv"
set TEST="..\..\data\dry_bean_test_std.csv"

echo ==============================
echo KNN+DBSCAN:
echo.
echo KNN classification:
echo (k = %K%, threshold probability = %THRESHOLD_PROB%)

python knn.py %TRAIN% %TEST% %K% %THRESHOLD_PROB%

cd ..\..\clustering\DBSCAN

set PROGRAM=main.exe
set OUTPUT_FILE=knn_dbscan.txt
set EPS=2.1911499961737353 
set MIN_SMP=27
set CLASSIFY="..\..\classification\KNN\knn_prediction.csv"
set SOURCE="..\..\data\dry_bean_test_std.csv"

echo.
echo DBSCAN clustering:
echo (epsilon = %EPS%, minimum samples = %MIN_SMP%)

type nul > %OUTPUT_FILE%

%PROGRAM% %CLASSIFY% %SOURCE% %EPS% %MIN_SMP% >> %OUTPUT_FILE%

echo ==============================
echo.

cd ..\..\classification\SVM

set KERNEL=poly  
set THRESHOLD_PROB=0.892845379940649
set TRAIN="..\..\data\dry_bean_train_std.csv"
set TEST="..\..\data\dry_bean_test_std.csv"

echo ==============================
echo SVM+DBSCAN
echo.
echo SVM classification:
echo (kernel = %KERNEL%, threshold probability = %THRESHOLD_PROB%)

python svm.py %TRAIN% %TEST% %KERNEL% %THRESHOLD_PROB%

cd ..\..\clustering\DBSCAN

set PROGRAM=main.exe
set OUTPUT_FILE=svm_dbscan.txt
set EPS=1.2920396403865566
set MIN_SMP=42
set CLASSIFY="..\..\classification\SVM\svm_prediction.csv"
set SOURCE="..\..\data\dry_bean_test_std.csv"

echo epsilon = %EPS%, minimum samples = %MIN_SMP%

echo.
echo DBSCAN clustering:
echo (epsilon = %EPS%, minimum samples = %MIN_SMP%)

type nul > %OUTPUT_FILE%

%PROGRAM% %CLASSIFY% %SOURCE% %EPS% %MIN_SMP% >> %OUTPUT_FILE%

echo ==============================
echo.

pause