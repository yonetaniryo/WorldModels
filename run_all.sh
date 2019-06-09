xvfb-run -a -s "-screen 0 1400x900x24" python 01_generate_data.py car_racing --total_episodes 2000 --time_steps 300 --root_dir=./data
python 02_train_vae.py --new_model --root_dir=./data
python 03_generate_rnn_data.py --root_dir=./data
python 04_train_rnn.py --new_model --root_dir=./data
CUDA_VISIBLE_DEVICES= xvfb-run -s "-screen 0 1400x900x24" python 05_train_controller.py car_racing --num_worker 16 --num_worker_trial 2 --num_episode 4 --max_length 1000 --eval_steps 25  --root_dir=./data
xvfb-run -s "-screen 0 1400x900x24" python model.py car_racing --filename ./data/controller/car_racing.cma.4.32.best.json --record_video --root_dir=./data