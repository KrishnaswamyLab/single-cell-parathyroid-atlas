import os, scipy
from scipy import io
import scanpy as sc
import numpy as np
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO)

# Load raw data files
samples = ['sample1', 'sample2', 'sample3', 'sample4']
PATH = '../data'
filenames = [os.path.join(PATH, sample+'_filtered_feature_bc_matrix.h5') for sample in samples]

adatas = []
for filename in filenames:
    tmp = sc.read_10x_h5(filename)
    tmp.var_names_make_unique()
    adatas.append(tmp)

sdata = adatas[0].concatenate(adatas[1:])
del(adatas)

# Get sample labels
id_to_batch = dict(zip([str(x) for x in range(4)], samples))
sdata.obs['sample'] = sdata.obs.batch.map(id_to_batch)

logging.info(f'Pre-filtering')
logging.info(f'...{sdata.n_obs}, {sdata.n_vars}')
logging.info(f'{Counter(sdata.obs["sample"])}')

# Get mito information
mito_gene_names = ["ND3", "ND5", "ND2", "COX1", "ND6", "COX3", "ND1", "ATP6", "ND4L", "COX2", "CYTB", "ATP8", "ND4"]
sdata.obs['mito_percent'] = np.sum(sdata[:, mito_gene_names].X, axis=1).A1 / np.sum(sdata.X, axis=1).A1

# Cell filter
sc.pp.filter_cells(sdata, min_genes=500)
sc.pp.filter_cells(sdata, min_counts=1000)

logging.info(f'After cell filtering')
logging.info(f'...{sdata.n_obs}, {sdata.n_vars}')

# Mito filter
sdata_mito_mask = sdata.obs['mito_percent'] < 0.4
sdata = sdata[sdata_mito_mask]

logging.info(f'After mito filtering')
logging.info(f'...{sdata.n_obs}, {sdata.n_vars}')

# Gene filter
sc.pp.filter_genes(sdata, min_cells=5)

logging.info(f'After gene filtering')
logging.info(f'...{sdata.n_obs}, {sdata.n_vars}')

# Save
sdata.write('../data/intermediate_files/1_preprocessed_primate_data.h5ad')
sdata.obs['sample'].to_csv('../data/intermediate_files/1_preprocessed_primate_data_sample_ids.csv')
io.mmwrite('../data/intermediate_files/1_preprocessed_primate_data.mtx', sdata.X)
