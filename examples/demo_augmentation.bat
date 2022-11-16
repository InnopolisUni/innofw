for %%I in (.) do set current_dir=%%~nxI

if "%current_dir%"=="examples" (
cd ..
)

streamlit run ui\pages\augmentation.py -- IM_190722_vwer3f23_oneshotlearning
