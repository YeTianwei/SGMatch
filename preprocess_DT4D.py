import os
import scipy.io as sio
import numpy as np
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm

import torch

from utils.geometry_util import laplacian_decomposition, get_operators, get_elas_operators
from utils.shape_util import read_shape, compute_geodesic_distmat, write_off
from utils.dino_util import get_shape_dino_features


if __name__ == '__main__':
    # parse arguments
    parser = ArgumentParser('Preprocess .off files')
    parser.add_argument('--data_root', required=True, help='data root contains /off sub-folder.')
    parser.add_argument('--n_eig', type=int, default=200, help='number of eigenvectors/values to compute.')
    parser.add_argument('--no_eig', action='store_true', help='no laplacian eigen-decomposition')
    parser.add_argument('--no_elas_eig', action='store_true', help='no elastic eigen-decomposition')
    parser.add_argument('--bending_weight', type=float, default=1e-2, help='bending weight for elastic energy') 
    parser.add_argument('--no_dist', action='store_true', help='no geodesic matrix.')
    parser.add_argument('--no_normalize', action='store_true', help='no normalization of face area.')
    parser.add_argument('--no_dino', action='store_true', help='no dino feature precomputation')
    parser.add_argument('--dino_out', default='dino', help='output folder name for dino features (under data_root)')
    args = parser.parse_args()

    # sanity check
    data_root = args.data_root
    n_eig = args.n_eig
    no_eig = args.no_eig
    no_elas_eig = args.no_elas_eig
    bending_weight = args.bending_weight
    no_dist = args.no_dist
    no_normalize = args.no_normalize
    no_dino = args.no_dino
    dino_dir = os.path.join(data_root, args.dino_out)
    if not no_dino:
        os.makedirs(dino_dir, exist_ok=True)
    assert n_eig > 0, f'Invalid n_eig: {n_eig}'
    assert os.path.isdir(data_root), f'Invalid data root: {data_root}'
    for folder in ["crypto", "drake", "mannequin", "mousey", "ninja", "ortiz", "prisoner", "pumpkinhulk", "skeletonzombie", "zlorp"]:
        if not no_eig:
            spectral_dir = os.path.join(data_root, 'diffusion')
            os.makedirs(spectral_dir, exist_ok=True)

        if not no_dist:
            dist_dir = os.path.join(data_root, 'dist', folder)
            os.makedirs(dist_dir, exist_ok=True)

        if not no_elas_eig:
            elas_spectral_dir = os.path.join(data_root,'elastic')
            os.makedirs(elas_spectral_dir, exist_ok=True)
    
        # read .off files
        off_files = sorted(glob(os.path.join(data_root, 'off',folder, '*.off')))
        assert len(off_files) != 0

        for off_file in tqdm(off_files):
            verts, faces = read_shape(off_file)
            filename = os.path.basename(off_file)

            if not no_normalize:
                # center shape
                verts -= np.mean(verts, axis=0)

                # normalize verts by sqrt face area
                old_sqrt_area = laplacian_decomposition(verts=verts, faces=faces, k=1)[-1]
                print(f'Old face sqrt area: {old_sqrt_area:.3f}')
                verts /= old_sqrt_area

                # save new verts and faces
                write_off(off_file, verts, faces)

            if not no_dino:
                dino_pt_dir = os.path.join(dino_dir, folder)
                dino_cache_dir = os.path.join(data_root, "dino_cache")
                os.makedirs(dino_pt_dir, exist_ok=True)
                os.makedirs(dino_cache_dir, exist_ok=True)

                out_path = os.path.join(dino_pt_dir, filename.replace('.off', '.pt'))

                if not os.path.exists(out_path):
                    # compute dino features
                    v = torch.from_numpy(verts).float()          # [V,3]
                    f = torch.from_numpy(faces).long()           # [F,3]

                    dino_feat = get_shape_dino_features(v, f, cache_dir=dino_cache_dir) * 0.5 # [V,768]
                    torch.save(dino_feat.cpu(), out_path)

            if not no_eig:
                # recompute laplacian decomposition
                get_operators(torch.from_numpy(verts).float(), torch.from_numpy(faces).long(),
                            k=n_eig, cache_dir=spectral_dir)

            if not no_dist:
                # compute distance matrix
                dist_mat = compute_geodesic_distmat(verts, faces)
                # save results
                sio.savemat(os.path.join(dist_dir, filename.replace('.off', '.mat')), {'dist': dist_mat})
            
            if not no_elas_eig:
                # recompute elastic decomposition
                # important not convert to float() 32, it's better to keep higher precision 64 for elastic
                get_elas_operators(torch.from_numpy(verts), torch.from_numpy(faces).long(),
                                k=n_eig, bending_weight=bending_weight, cache_dir=elas_spectral_dir)
