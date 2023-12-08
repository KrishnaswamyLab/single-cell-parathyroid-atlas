import scipy, scprep, graphtools, phate
import scanpy as sc
import numpy as np
from scipy import io, sparse
from pygsp import graphs, filters

# Load data
sdata = sc.read_h5ad('../data/intermediate_files/4_human_parathyroid_cells_all_genes.h5ad')
mito_gene_names = ['MT-ND1', 'MT-ND2', 'MT-CO1', 'MT-CO2', 'MT-ATP8', 'MT-ATP6', 'MT-CO3','MT-ND3', 'MT-ND4L', 'MT-ND4', 'MT-ND5', 'MT-ND6', 'MT-CYB']

# Recompute features for PTH-only cells
sdata.obs['mito_percent'] = np.sum(sdata[:, mito_gene_names].X, axis=1).A1 / np.sum(sdata.X, axis=1).A1
sdata.obs['n_genes'] = (sdata.to_df() > 0).sum(axis=1)
sdata.var['n_cells'] = (sdata.to_df() > 0).sum(axis=0)
del(sdata.obs['n_counts']) # don't have access to pre-normalized counts

# Batch mean center
sdata.raw = sdata
sdata.X = sparse.csr_matrix(scprep.normalize.batch_mean_center(sdata.X.toarray(), sample_idx=sdata.obs['batch']))

# Select highly variable genes
normalized, HVG_vars = scprep.select.highly_variable_genes(sdata.X.toarray(), sdata.var.index, cutoff=0.05, percentile=None)
sdata.var['highly_variable'] = sdata.var.index.isin(HVG_vars)

# Get normalized counts back instead of mean centered values as pca will mean center
sdata.layers["X_batch_mean"] = sdata.X
sdata.X = sdata.raw.X

# Subset to HVGs for PCs
data_hvg = sdata[:, sdata.var.highly_variable].X.toarray()

# Compute PCs for initial cell graph
data_pca, SVs = scprep.reduce.pca(data_hvg, n_components=100, return_singular_values=True)
sdata.obsm["X_pca"] = data_pca[0:,0:]

# Make initial cellwise graph with mnn
G = graphtools.Graph(sdata.obsm["X_pca"], use_pygsp=True, n_pca=None, random_state=42, sample_idx=sdata.obs.batch, bandwidth_scale=0.5)

phate_op = phate.PHATE(n_components=2, random_state=0, use_pygsp=True)
data_phate = phate_op.fit_transform(G)
sdata.obsm["X_phate_mnn"] = data_phate[0:,0:]

sdata.write('../data/processed_files/4_human_parathyroid_cells_all_genes.h5ad')