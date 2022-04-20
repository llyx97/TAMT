import torch, logging, os, json

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
sparsity = '0.8'
steps = {'0.6': 14000, '0.7': 17000, '0.8': 20000}

root_dir = 'mask_training/models/prun_bert/unstructured'      # root directory for TAMT-KD and TAMT-MLM subnetwork masks
root_dir_imp = 'imp_and_fine_tune/pretrain_prun/imp_pretrain/wikitext-103'  # root directory for IMP subnetwork masks
root_dir_kd = '%s/train_kd/wikitext-103/length512/full_cosine/mag_init/%s'%(root_dir, sparsity)
root_dir_mlm = '%s/train_mlm/wikitext-103/length512/%s/'%(root_dir, sparsity)
output_dir = ''    # output directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

mask_dirs = {
        'mag': 'imp_and_fine_tune/pretrain_prun/oneshot/%s/mask.pt'%str(sparsity),
        }

for seed in range(1, 4):
    mask_dirs['kd_seed%d'%seed] = '%s/seed%d/step_%d/mask.pt'%(root_dir_kd, seed, steps[sparsity]-1)
for seed in range(1, 4):
    mask_dirs['mlm_seed%d'%seed] = '%s/seed%d/checkpoint-%d/mask.pt'%(root_dir_mlm, seed, steps[sparsity])
for seed in range(1, 4):
    mask_dirs['imp_seed%d'%seed] = os.path.join(root_dir_imp, 'prun_step2792', 'seed%d/%s/mask.pt'%(seed, str(sparsity)))

sim_mat = {key:[] for key in mask_dirs}


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

def print_2d_tensor(tensor):
        """ Print a 2D tensor """
        logger.info("lv, h >\t" + "\t".join(f"{key}" for key in tensor))
        for key in tensor:
            logger.info(f"{key}:\t" + "\t".join(f"{x:.5f}" for x in tensor[key]))

def main():
    for key1 in mask_dirs:
        logger.info("Compute mask sims for %s"%key1)
        for key2 in mask_dirs:
            sim = compare(mask_dirs[key1], mask_dirs[key2], False)
            logger.info(sim)
            sim_mat[key1].append(sim.item())

    print_2d_tensor(sim_mat)
    output_file = open(output_dir+'/mask_sim.json', 'w')
    json.dump(sim_mat, output_file)
    output_file.close()

if __name__ == "__main__":
    main()
