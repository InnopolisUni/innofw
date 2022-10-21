PWD=$(pwd)
if [ "${PWD##*/}"="examples" ]; then
  cd ..
fi
experiment_name=IM_190722_vwer3f23_oneshotlearning
streamlit run ui/pages/Аугментация.py -- $experiment_name
