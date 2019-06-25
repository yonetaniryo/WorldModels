for i in `seq 1 3`
do
for j in `seq 1 3`
do
ENVNAME=car_racing_$i$j
CUDA_VISIBLE_DEVICES= xvfb-run -s "-screen 0 1400x900x24" python 05_train_controller.py $ENVNAME --num_worker 16 --num_worker_trial 2 --num_episode 4 --max_length 1000 --eval_steps 25  --root_dir=./$ENVNAME
done
done

for i in `seq 1 3`
do
for j in `seq 1 3`
do
ENVNAME=car_racing_$i$j
xvfb-run -s "-screen 0 1400x900x24" python model.py $ENVNAME --filename ./$ENVNAME/controller/car_racing.cma.4.32.best.json --record_video --root_dir=./$ENVNAME
done
done