PWD=$(pwd)
current_dir=${PWD##*/}

var2="examples"

if [ "$current_dir" = "$var2" ]; then
	cd ..
fi

experiment_name=one-shot-learning/IM_190722_vwer3f23_oneshotlearning
streamlit run ui/pages/augmentation.py -- $experiment_name
