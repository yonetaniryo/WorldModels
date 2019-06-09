#python 03_generate_rnn_data.py

from vae.arch import VAE
import argparse
import config
import numpy as np
import os
import subprocess


def get_filelist(N, ROLLOUT_DIR_NAME):
    filelist = os.listdir(ROLLOUT_DIR_NAME)
    filelist = [x for x in filelist if x != '.DS_Store']
    filelist.sort()
    length_filelist = len(filelist)

    if length_filelist > N:
        filelist = filelist[:N]

    if length_filelist < N:
        N = length_filelist

    return filelist, N

def encode_episode(vae, episode):

    obs = episode['obs']
    action = episode['action']
    reward = episode['reward']
    done = episode['done']

    done = done.astype(int)  
    reward = np.where(reward>0, 1, 0) * np.where(done==0, 1, 0)

    mu, log_var = vae.encoder_mu_log_var.predict(obs)
    
    initial_mu = mu[0, :]
    initial_log_var = log_var[0, :]

    return (mu, log_var, action, reward, done, initial_mu, initial_log_var)


def main(args):
    ROOT_DIR_NAME = args.root_dir
    ROLLOUT_DIR_NAME = ROOT_DIR_NAME + "/rollout/"
    SERIES_DIR_NAME = ROOT_DIR_NAME + "/series/"
    VAE_DIR_NAME = ROOT_DIR_NAME + "/vae/"
    subprocess.run(['mkdir', '-p', SERIES_DIR_NAME])

    N = args.N

    vae = VAE()

    try:
        vae.set_weights(VAE_DIR_NAME + 'weights.h5')
    except:
        print("./vae/weights.h5 does not exist - ensure you have run 02_train_vae.py first")
        raise


    filelist, N = get_filelist(N, ROLLOUT_DIR_NAME)

    file_count = 0

    initial_mus = []
    initial_log_vars = []

    for file in filelist:
        try:
      
            rollout_data = np.load(ROLLOUT_DIR_NAME + file)

            mu, log_var, action, reward, done, initial_mu, initial_log_var = encode_episode(vae, rollout_data)

            np.savez_compressed(SERIES_DIR_NAME + file, mu=mu, log_var=log_var, action = action, reward = reward, done = done)
            initial_mus.append(initial_mu)
            initial_log_vars.append(initial_log_var)

            file_count += 1

            if file_count%50==0:
                print('Encoded {} / {} episodes'.format(file_count, N))

        except:
            print('Skipped {}...'.format(file))

    print('Encoded {} / {} episodes'.format(file_count, N))

    initial_mus = np.array(initial_mus)
    initial_log_vars = np.array(initial_log_vars)

    print('ONE MU SHAPE = {}'.format(mu.shape))
    print('INITIAL MU SHAPE = {}'.format(initial_mus.shape))

    np.savez_compressed(ROOT_DIR_NAME + '/initial_z.npz', initial_mu=initial_mus, initial_log_var=initial_log_vars)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Generate RNN data'))
    parser.add_argument('--N', default = 10000, type=int, help='number of episodes to use to train')
    parser.add_argument('--root_dir', default='./data', type=str, help='save directory')
    args = parser.parse_args()

    main(args)