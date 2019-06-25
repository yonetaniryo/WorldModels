for i in `seq 1 3`
do
ENVNAME=car_racing_2$i
xvfb-run -a -s "-screen 0 1400x900x24" python 01_generate_data.py $ENVNAME --total_episodes 2000 --time_steps 300 --root_dir=./$ENVNAME
python 02_train_vae.py --new_model --root_dir=./$ENVNAME
python 03_generate_rnn_data.py --root_dir=./$ENVNAME
python 04_train_rnn.py --new_model --root_dir=./$ENVNAME
# CUDA_VISIBLE_DEVICES= xvfb-run -s "-screen 0 1400x900x24" python 05_train_controller.py $ENVNAME --num_worker 16 --num_worker_trial 2 --num_episode 4 --max_length 1000 --eval_steps 25  --root_dir=./$ENVNAMExvfb-run -s "-screen 0 1400x900x24" python model.py $ENVNAME --filename ./$ENVNAME/controller/car_racing.cma.4.32.best.json --record_video --root_dir=./$ENVNAME
# xvfb-run -s "-screen 0 1400x900x24" python model.py $ENVNAME --filename ./$ENVNAME/controller/$ENVNAME.cma.4.32.best.json --record_video --root_dir=./$ENVNAME
done