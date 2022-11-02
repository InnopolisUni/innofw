PWD=$(pwd)
current_dir=${PWD##*/}
if [ $current_dir="examples" ]; then
  cd ..
fi
experiment_name=IM_190722_vwer3f23_oneshotlearning
streamlit run ui/pages/Аугментация.py -- $experiment_name
