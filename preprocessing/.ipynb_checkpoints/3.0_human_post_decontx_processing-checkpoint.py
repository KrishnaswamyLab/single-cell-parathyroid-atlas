import scipy, scprep, graphtools, phate, magic
import scanpy as sc
import numpy as np
from scipy import io, sparse
from pygsp import graphs, filters

# Load data
data = io.mmread('intermediate_files/2_decontx_human_data.mtx')
orig_data = sc.read_h5ad('intermediate_files/1_preprocessed_human_data.h5ad')
data = data.T.toarray() # decontx uses transposed matrix

# Normalize & Transform
data = scprep.normalize.library_size_normalize(data)
data = scprep.transform.sqrt(data)

# Make Anndata object
sdata = sc.AnnData(X=sparse.csr_matrix(data), obs=orig_data.obs, var=orig_data.var)

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

sdata.write('processed_files/3_human_all_cell_types_all_genes.h5ad')
