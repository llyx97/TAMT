import torch, logging, os, json
import numpy as np


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
sparsity = 0.8

root_dir_imp = 'imp_and_fine_tune/pretrain_prun/imp_pretrain/wikitext-103'  # root directory for IMP subnetwork masks
root_dir = 'mask_training/models/prun_bert/unstructured'  #  root directory for TAMT-KD and TAMT-MLM subnetwork masks
output_dir = ''    # output director
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

steps = {}
steps['imp'] = '200 300 500 900 1396 2792'.split()
steps['mlm'] = '1000 2000 3000 5000 10000 20000 25000'.split()
steps['kd'] = [str(int(step)-1) for step in steps['mlm']]

mask_dirs = {
        'mag': 'imp_and_fine_tune/pretrain_prun/oneshot/%s/mask.pt'%str(sparsity),
        }

mask_dirs['imp'] = {
        step: {
            seed: os.path.join(root_dir_imp, 'prun_step%s'%step, 'seed%d/%s/mask.pt'%(seed, str(sparsity)))
            for seed in range(1, 4)
            }
        for step in steps['imp']
        }
mask_dirs['kd'] = {
        step: {
            seed: os.path.join(root_dir, 'train_kd/wikitext-103/length512/full_cosine/mag_init/%s/seed%d/step_%s/mask.pt'%(str(sparsity), seed, step))
            for seed in range(1, 4)
            }
        for step in steps['kd']
        }
mask_dirs['mlm'] = {
        step: {
            seed: os.path.join(root_dir, 'train_mlm/wikitext-103/length512/%s/seed%d/checkpoint-%s/mask.pt'%(str(sparsity), seed, step))
            for seed in range(1, 4)
            }
        for step in steps['mlm']
        }
sim_mat = {key: {seed: [] for seed in range(1, 4)} for key in ['imp', 'mlm', 'kd']}


def compare(mask1_dir, mask2_dir, show_every_matrix=True):
    mask1 = torch.load(mask1_dir)
    mask2 = torch.load(mask2_dir)
    avg_sim = 0.

    assert len(mask1.keys())==len(mask2.keys())
    for key1, key2 in zip(mask1.keys(), mask2.keys()):
        sim = (mask1[key1] & mask2[key2]).sum().float().div((mask1[key1] | mask2[key2]).sum().float())
        avg_sim += sim
        if show_every_matrix:
            print(key1, sim)
    avg_sim = avg_sim / len(mask1.keys())
    return avg_sim


def main():
    key1 = 'mag'
    for key2 in ['imp', 'mlm', 'kd']:
        for seed in range(1, 4):
            logger.info("Compute mask distance for %s seed%d"%(key2, seed))
            for step in steps[key2]:
                sim = compare(mask_dirs[key1], mask_dirs[key2][step][seed], False)
                logger.info(sim)
                sim_mat[key2][seed].append(sim.numpy())

    logger.info(sim_mat)
    for key in sim_mat:
        sim_mat[key] = np.array([sim_mat[key][seed] for seed in range(1, 4)])
    np.savez(os.path.join(output_dir, 'mask_dist'), imp=sim_mat['imp'], mlm=sim_mat['mlm'], kd=sim_mat['kd'])

if __name__ == "__main__":
    main()
