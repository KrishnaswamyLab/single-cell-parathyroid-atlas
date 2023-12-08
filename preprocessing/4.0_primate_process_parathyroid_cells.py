import scipy, scprep, graphtools, phate
import scanpy as sc
import numpy as np
from scipy import io, sparse
from pygsp import graphs, filters

# Load data
sdata = sc.read_h5ad('../data/intermediate_files/4_primate_parathyroid_cells_all_genes.h5ad')
mito_gene_names = ["ND3", "ND5", "ND2", "COX1", "ND6", "COX3", "ND1", "ATP6", "ND4L", "COX2", "CYTB", "ATP8", "ND4"]

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

# Subset to HVGs for PCs
data_hvg = sdata[:, sdata.var.highly_variable].X.toarray()

# Compute PCs for initial cell graph
data_pca, SVs = scprep.reduce.pca(data_hvg, n_components=100, return_singular_values=True)
sdata.obsm["X_pca"] = data_pca[0:,0:]

# Get normalized counts back instead of mean centered values
sdata.layers["X_batch_mean"] = sdata.X
sdata.X = sdata.raw.X

# Make initial cellwise graph with mnn
G = graphtools.Graph(sdata.obsm["X_pca"], use_pygsp=True, n_pca=None, random_state=42, sample_idx=sdata.obs.batch, bandwidth_scale=0.5)

phate_op = phate.PHATE(n_components=2, random_state=0, use_pygsp=True, t=7)
data_phate = phate_op.fit_transform(G)
sdata.obsm["X_phate_mnn"] = data_phate[0:,0:]

# Flip axis so "early" in time
anndata_primate.obsm['X_phate_mnn'][:, 0] = -1 * anndata_primate.obsm['X_phate_mnn'][:, 0]
anndata_primate.obs['pseudotime'] = G.U[:, 1]

sdata.write('../data/processed_files/4_primate_parathyroid_cells_all_genes.h5ad')