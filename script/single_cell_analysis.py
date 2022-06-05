import argparse, os, sys
from os import listdir
from os.path import isfile, join

import math
import collections

import numpy as np
import pandas as pd
import scanpy as sc
import warnings
warnings.filterwarnings('ignore')

import kaplanmeier as km
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['font.size'] = 10

def runGSEAPY(adata, group_by='louvain', gene_sets=['GO_Biological_Process_2021'], organism='Human', cutoff=0.05, logfc_threshold=2, outdir='.'):
    import gseapy as gp
    from gseapy.plot import barplot

    df_list = []
    cluster_list = []
    celltypes = sorted(adata.obs[group_by].unique())

    for celltype in celltypes:
        indlist_logfc = adata.uns['rank_genes_groups']['logfoldchanges'][celltype] >= logfc_threshold
        indlist_adjp = adata.uns['rank_genes_groups']['pvals_adj'][celltype] <= 1e-2
        indlist_p = adata.uns['rank_genes_groups']['pvals'][celltype] <= 1e-2
        #indlist_pts = adata.uns['rank_genes_groups']['pts'][celltype] >= 0.1
        
        indlist = indlist_logfc * indlist_adjp * indlist_p 

        ind = [x for x in range(0, len(indlist)) if indlist[x] ]
        degs = adata.uns['rank_genes_groups']['names'][celltype][ind].tolist()
        
        if not degs:
            continue

        enr = gp.enrichr(gene_list=degs,
                gene_sets=gene_sets,
                organism=organism, 
                description=celltype,
                no_plot=True
                )
        barplot(enr.res2d,title='{gene_sets}: C{celltype}', color='grey')
        if ax.lines:
            pp.savefig()
            plt.close()
        df_list.append(enr.res2d)
        cluster_list.append(celltype)

    columns = ['Cluster', 'Gene_set', 'Term', 'Overlap', 'P-value', 'Adjusted P-value', 'Genes']

    df = pd.DataFrame(columns = columns)
    for cluster_ind, df_ in zip(cluster_list, df_list):
        df_ = df_[df_['Adjusted P-value'] <= cutoff]
        df_ = df_.assign(Cluster = cluster_ind)
        if(df_.shape[0] > 0):
            df = pd.concat([df, df_[columns]], sort=False)
        else:
            print('No pathway with an adjusted P-value less than the cutoff (={}) for cluster {}'.format(cutoff, cluster_ind))
    
    return df

def getBulkProfile(bulkpath, gencode_table):
    bk_gep = pd.read_csv(bulkpath, sep=',', compression='gzip')
    # covert gene ID to gene name
    bk_gep_name = bk_gep.merge(gencode_table, how='left', on='id')
    bk_gep_name.drop_duplicates(subset='name', inplace=True)
    bk_gep_name = bk_gep_name.set_index('name')
    bk_gep_name = bk_gep_name.drop(columns=['id'])
    return bk_gep_name

def getSpecCellDict(bk_gep_name, dict_deg):
    score_table = pd.DataFrame(index = bk_gep_name.columns)
    for gene in bk_gep_name.index:
        median = bk_gep_name.loc[gene, :].median()
        score_table[gene] = list(bk_gep_name.loc[gene, :] >= median)
    score_table = score_table.T

    spec_score_table = pd.DataFrame(index = dict_deg.keys())
    for sample in score_table.columns:
        values = []
        for _, genes in dict_deg.items():
            values.append(score_table.loc[genes, sample].sum())
        spec_score_table[sample] = values
    spec_score_table = spec_score_table.T
    
    dict_low  = {}
    dict_high = {}
    for col in spec_score_table.columns:
        dict_low[col] = spec_score_table[col].quantile(q=0.25)
        dict_high[col] = spec_score_table[col].quantile(q=0.75)
        
    dict_celltype = collections.defaultdict(lambda : collections.defaultdict(dict))
    for c in spec_score_table.columns.tolist():
        dict_celltype[c]['high'] = [x for x in spec_score_table.index if spec_score_table.loc[x,c] >= dict_high[c]]
        dict_celltype[c]['low'] = [x for x in spec_score_table.index if spec_score_table.loc[x,c] <= dict_low[c]]
    return dict_celltype

def drawSurvivalPlot(dict_celltype, clinical_df, project_id):
    n_types = len(dict_celltype.keys())
    n_cols = 3
    n_rows = math.ceil(n_types/n_cols)
    dict_group = {'high':1, 'low':0}
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(2.5*n_cols, 2.5*n_rows))
    i_col = 0
    i_row = 0
    alpha_val = 0.05
    cph_selected_cols = ['project', 'cell type', 'coef', 'exp(coef)', 'se(coef)', 'coef lower 95%', 'coef upper 95%','exp(coef) lower 95%', 'exp(coef) upper 95%', 'z', 'p', '-log2(p)']
    df_hazard = pd.DataFrame(columns = cph_selected_cols)

    for c in sorted(dict_celltype.keys(), key=int):
        ax = axs[i_row][i_col]
        dict_km = {'time':[], 'died':[], 'group':[], 'abundance':[]}
        for g in ['high', 'low']:
            death_time = [int(x) for x in clinical_df.loc[dict_celltype[c][g], '_event_time_']]
            dict_km['time'].extend(death_time)
            dict_km['died'].extend([not x.startswith('\'') for x in clinical_df.loc[dict_celltype[c][g], 'days_to_death']])
            dict_km['group'].extend([g]*len(death_time))
            dict_km['abundance'].extend([dict_group[g]]*len(death_time))

        df = pd.DataFrame(dict_km)
        T = df['time']
        E = df['died']
        dem = (df['group'] == 'high')

        # Compute survival
        kmf = KaplanMeierFitter(alpha=alpha_val)
        kmf.fit(T[dem], event_observed=E[dem], label='High')
        kmf.plot_survival_function(ax=ax, ci_show=False)
        kmf.fit(T[~dem], event_observed=E[~dem], label='Low')
        kmf.plot_survival_function(ax=ax, ci_show=False)
        out_lr = logrank_test(T[dem], T[~dem], E[dem], E[~dem], alpha=1-alpha_val)
        p_value = out_lr.p_value

        # Hazard Rate
        df.drop(['group'], axis=1, inplace=True)
        cph = CoxPHFitter()
        cph.fit(df, duration_col='time', event_col='died')
        hr = cph.hazard_ratios_.values[0]
        dict_tmp = cph.summary.to_dict(orient='records')[0]
        dict_tmp['project'] = project_id
        dict_tmp['cell type'] = c
        df_hazard = df_hazard.append(dict_tmp, ignore_index=True)

        ax.set_title('C{} (Log-rank P={:.2f}, HR={:.2f})'.format(c, p_value, hr), fontsize=8)

        # update i_col, i_row
        i_col += 1
        if i_col >= n_cols: 
            i_col = 0
            i_row += 1

    # remove empty axes
    while(i_col < n_cols):
        fig.delaxes(axs[i_row][i_col])
        i_col += 1

    fig.suptitle('Survival Analysis: '+project_id.rsplit('-',1)[1], fontsize=12, y=1)
    fig.tight_layout()
    plt.tight_layout()
    pp.savefig(fig, bbox_inches='tight')
    plt.close('all')
    
    return df_hazard



## Parse command-line arguments
# process arguments
parser = argparse.ArgumentParser(description="scRNA-seq data analysis")

parser.add_argument("-i", "--input", required=True, help="path to input 10x directory or CSV file")
parser.add_argument("-f", "--format", default='10x', help="input format, 10x (default) | csv | h5ad (Anndata object for subclustering with --clusters CLUSTERS)")
parser.add_argument("-o", "--output", default='./', help="path to output directory, default='./'")
parser.add_argument("-r", "--resolution", type=float, default=0.6, help="resolution for clustering, default=0.6")
parser.add_argument("--auto-resolution", action="store_true", help="automatically determine resolution for clustering")
parser.add_argument("-m", "--metadata", default=None, help="path to metadata CSV file for batch correction (index as input in first column)")
parser.add_argument("-b", "--batch", default=None, help="column in metadata (or adata.obs) for batch correction, e.g. 'PatientID'")
parser.add_argument("-c", "--clusters", default=None, help="perform single cell analysis only on specified clusters, e.g. '1,3,8,9'")
parser.add_argument("--GEP", action="store_true", help=' generate Gene Expression Profile file.')
parser.add_argument("--annotation", action="store_true", help="perform cell type annotation")
parser.add_argument("--gsea", action="store_true", help="perform gene set enrichment analysis (GSEA)")
parser.add_argument("--cpus", default=1, type=int, help="number of CPU used for auto-resolution and annotation, default=1")
parser.add_argument("--survival", action="store_true", help="perform survival analysis")
parser.add_argument("--tcga", default='/scDrug/data/TCGA/', help="path to TCGA data")
parser.add_argument("--id", default=None, help='Specify TCGA project id in the format "TCGA-xxxx", e.g., "TCGA-LIHC"')
parser.add_argument("--not_treated", action="store_true", help='only consider untreated samples from TCGA for survival analysis.')

args = parser.parse_args()

# check format, input and clusters
if not os.path.exists(args.input):
    sys.exit("The input path does not exist.")
if args.format == 'csv':
    if args.input[-4:] != '.csv':
        sys.exit("The input file is not a CSV file.")
elif args.format == '10x':
    if not os.path.exists(os.path.join(args.input, 'matrix.mtx')) and not os.path.exists(os.path.join(args.input, 'matrix.mtx.gz')):
        sys.exit("Cannot find 'matrix.mtx' file in the input directory.")
    if not os.path.exists(os.path.join(args.input, 'genes.tsv')) and not os.path.exists(os.path.join(args.input, 'genes.tsv.gz')):
        if not os.path.exists(os.path.join(args.input, 'features.tsv')) and not os.path.exists(os.path.join(args.input, 'features.tsv.gz')):
            sys.exit("Cannot find 'genes.tsv' or 'features.tsv' file in the input directory.")
    if not os.path.exists(os.path.join(args.input, 'barcodes.tsv')) and not os.path.exists(os.path.join(args.input, 'barcodes.tsv.gz')):
        sys.exit("Cannot find 'barcodes.tsv' file in the input directory.")
elif args.format == 'h5ad':
    if args.input[-5:] != '.h5ad':
        sys.exit("The input file is not a h5ad file.")
else:
     sys.exit("The format can only be '10x' or 'csv'.")

# check output
if not os.path.isdir(args.output):
    sys.exit("The output directory does not exist.")

# check metadata
if not args.metadata is None:
    if not os.path.exists(args.metadata):
        sys.exit("The metadata file does not exist.")
    if args.metadata[-4:] != '.csv':
        sys.exit("The metadata file is not a CSV file.")

# write metadata
if not args.metadata is None:
    metadata_df = pd.read_csv(args.metadata, index_col=0)

# check batch in metadata
if not args.batch is None:
    if args.metadata is None and args.format != 'h5ad':
        sys.exit("Please provide the metadata file for batch correction with --metadata METADATA.")

    if not args.batch in metadata_df.columns:
        sys.exit("The batch column is not in the metadata file.")
        
# check arguments for survival analysis
if args.survival:
    if not os.path.isdir(args.tcga):
        sys.exit("The path to TCGA files does not exist.")
    if not os.path.isfile(f'{args.tcga}/clinical.tsv'):
        sys.exit("The TCGA clinical file does not exist.")
    else:
        clinicalpath = f'{args.tcga}/clinical.tsv'
    if not os.path.isfile(f'{args.tcga}/gencode.v22.annotation.id.name.gtf'):
        sys.exit("The gencode file for id conversion does not exist.")
    else:
        gencode = f'{args.tcga}/gencode.v22.annotation.id.name.gtf'
    if args.not_treated:
        treatment_status = 'no'
    else:
        treatment_status = ''
    project_ids = [f.rsplit('.csv',1)[0] for f in listdir(args.tcga) if isfile(join(args.tcga, f)) and f.startswith('TCGA')]
    if args.id:
        if not args.id in project_ids:
            sys.exit(f"Cannot find the TCGA file for the specified project_id: {args.id}\nCandidates:{project_ids}")

## Preprocessing
print('Preprocessing...')
results_file = os.path.join(args.output, 'scanpyobj.h5ad')
pdfname = f'{args.output}/results.pdf'
pp = PdfPages(pdfname)


useraw = False
if args.format == 'h5ad':
    adata = sc.read(args.input)
    if args.clusters:
        clusters = [x.strip() for x in args.clusters.split(',')]
        adata = adata[adata.obs['louvain'].isin(clusters)]
        # get back to raw data if raw data existed
        if adata.raw:
            useraw = True
            adata = adata.raw.to_adata()
    if not args.metadata is None:
        adata.obs = adata.obs.merge(metadata_df, left_index=True, right_index=True)
    adata_GEP = adata.raw.to_adata()

else:
    if args.format == 'csv':
        adata = sc.read_csv(args.input)
    elif args.format == '10x':
        adata = sc.read_10x_mtx(args.input, var_names='gene_symbols', cache=True)
    adata.var_names_make_unique()

    if not args.metadata is None:
        adata.obs = metadata_df

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    
    adata = adata[adata.obs.pct_counts_mt < 30, :]

    adata_GEP = adata.copy()

if args.format != 'h5ad' or useraw:
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata)
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]
    sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])

    sc.pp.scale(adata)

    ## Principal component analysis
    sc.tl.pca(adata, svd_solver='arpack')

# adata.write(results_file)
    
    
## Batch Correction with Harmony
if not args.batch is None:
    if not args.metadata is None:
        adata.obs[args.batch] = [str(x) for x in metadata_df.loc[adata.obs.index][args.batch]]
    elif args.format == 'h5ad' and not args.batch in adata.obs.columns:
        sys.exit("The batch column is not in the Anndata object.")
    print('Batch correction...')
    sc.external.pp.harmony_integrate(adata, args.batch, adjusted_basis='X_pca')


## Clustering
sc.pp.neighbors(adata, n_pcs=20)
sc.tl.umap(adata)
if args.auto_resolution:
    print("Automatically determine clustering resolution...")
    
    from sklearn.metrics import silhouette_score
    import multiprocess as mp
    from functools import partial
    
    def subsample_clustering(adata, sample_n, subsample_n, resolution, subsample):
        subadata = adata[subsample]
        sc.tl.louvain(subadata, resolution=resolution)
        cluster = subadata.obs['louvain'].tolist()
        
        subsampling_n = np.zeros((sample_n, sample_n), dtype=bool)
        coclustering_n = np.zeros((sample_n, sample_n), dtype=bool)
        
        for i in range(subsample_n):
            for j in range(subsample_n):
                x = subsample[i]
                y = subsample[j]
                subsampling_n[x][y] = True
                if cluster[i] == cluster[j]:
                    coclustering_n[x][y] = True
        return (subsampling_n, coclustering_n)
    
    rep_n = 5
    subset = 0.8
    sample_n = len(adata.obs)
    subsample_n = int(sample_n * subset)
    resolutions = np.linspace(0.4, 1.4, 6)
    silhouette_avg = {}
    np.random.seed(1)
    best_resolution = 0
    highest_sil = 0
    for r in resolutions:
        r = np.round(r, 1)
        print("Clustering test: resolution = ", r)
        subsamples = [np.random.choice(sample_n, subsample_n, replace=False) for t in range(rep_n)]
        p = mp.Pool(args.cpus)
        func = partial(subsample_clustering, adata, sample_n, subsample_n, r)
        resultList = p.map(func, subsamples)
        p.close()
        p.join()
        
        subsampling_n = sum([result[0] for result in resultList])
        coclustering_n = sum([result[1] for result in resultList])
        
        subsampling_n[np.where(subsampling_n == 0)] = 1e6
        distance = 1.0 - coclustering_n / subsampling_n
        np.fill_diagonal(distance, 0.0)
        
        sc.tl.louvain(adata, resolution=r, key_added = 'louvain_r' + str(r))
        silhouette_avg[r] = silhouette_score(distance, adata.obs['louvain_r' + str(r)], metric="precomputed")
        if silhouette_avg[r] > highest_sil:
            highest_sil = silhouette_avg[r]
            best_resolution = r
        print("robustness score = ", silhouette_avg[r])
        print()
    
    
    adata.obs['louvain'] = adata.obs['louvain_r' + str(best_resolution)]
    print("resolution with highest score: ", best_resolution)
    res = best_resolution
    # write silhouette record to uns and remove the clustering results except for the one with the best resolution
    adata.uns['sihouette score'] = silhouette_avg
    # draw lineplot
    df_sil = pd.DataFrame(silhouette_avg.values(), columns=['silhouette score'], index=silhouette_avg.keys())
    df_sil.plot.line(style='.-', color='green', title='Auto Resolution', xticks=resolutions, xlabel='resolution', ylabel='silhouette score', legend=False)
    pp.savefig()
    plt.close()
    # for r in [x for x in resolutions if x != best_resolution]:
    #     del adata.obs['louvain_r' + str(r)]
    
else:
    print("Clustering with resolution = ", args.resolution)
    sc.tl.louvain(adata, resolution=args.resolution)
    res = args.resolution


print('Exporting UMAP...')
sc.settings.autosave = True
sc.settings.figdir = args.output
fig = sc.pl.umap(adata, color=['louvain'], use_raw=False, show=False, return_fig=True,
           title='louvain, resolution='+str(res))
pp.savefig(fig, bbox_inches='tight')
plt.close()
if not args.batch is None:
    fig = sc.pl.umap(adata, color=[args.batch], use_raw=False, show=False, return_fig=True, title=args.batch)
    pp.savefig(fig, bbox_inches='tight')
    plt.close()


groups = sorted(adata.obs['louvain'].unique(), key=int)
if args.annotation:
    print('Cell type annotation...')
    
    # Export csv used by scMatch
    mat = np.zeros((len(adata.raw.var.index), len(groups)), dtype=float)
    for group in groups:
        mat[: , int(group)] = adata.raw.X[adata.obs['louvain']==group].mean(axis=0)
    dat = pd.DataFrame(mat, index = adata.raw.var.index, columns = groups)
    dat.to_csv(os.path.join(args.output, 'cluster_mean_exp.csv'))
    
    os.system('python /opt/scMatch/scMatch.py --refDS /opt/scMatch/refDB/FANTOM5 \
              --dFormat csv --testDS {} --coreNum {}'.format(
              os.path.join(args.output, 'cluster_mean_exp.csv'), args.cpus))
    
    # Cell annotation result
    scMatch_cluster_df = pd.read_csv(os.path.join(args.output, 'cluster_mean_exp') + '/annotation_result_keep_all_genes/human_Spearman_top_ann.csv')
    scMatch_cluster_names = [group + " " + scMatch_cluster_df.loc[scMatch_cluster_df['cell']==int(group)]\
                              ['cell type'].tolist()[0] for group in groups]
    adata.obs['cell_type'] = adata.obs['louvain'].cat.rename_categories(scMatch_cluster_names)
    scMatch_candidate_df = pd.read_excel(os.path.join(args.output, 'cluster_mean_exp') + '/annotation_result_keep_all_genes/human_Spearman.xlsx', skiprows=4, header=None, index_col=0)
    for i in range(len(scMatch_candidate_df.columns)):
        if i%2 == 0:
            scMatch_candidate_df.iloc[:, i] = [x.split(',',1)[0].split(':',1)[0] for x in scMatch_candidate_df.iloc[:, i]]
    dict_candidates = {}
    for i in range(int(len(scMatch_candidate_df.columns)/2)):
        candidates = list(set(scMatch_candidate_df.iloc[:5, i*2]))
        idx = 5
        while len(candidates) < 5:
            cell = scMatch_candidate_df.iloc[idx, i*2]
            if not cell in candidates:
                candidates.append(cell)
            idx += 1
        dict_candidates[str(i)] = candidates
    df_candidate = pd.DataFrame(dict_candidates).T.reset_index().rename(columns={'index':'cluster'})
    del scMatch_candidate_df

    fig = sc.pl.umap(adata, color=['cell_type'], use_raw=False, show=False, return_fig=True, title='cell type')
    pp.savefig(fig, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots()
    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    table = ax.table(cellText=df_candidate.values, colLabels=df_candidate.columns, loc='center')
    # table.auto_set_font_size(False)
    # table.set_fontsize(8)
    table.auto_set_column_width(col=list(range(len(df_candidate.columns))))

    for cell in table._cells:
        if cell[0] == 0:
            table._cells[cell].set_color('lightblue')
            table._cells[cell].set_height(.05)
    
    ax.set_title('Top 5 annotation')
    pp.savefig(fig, bbox_inches='tight')
    plt.close('all')


    


## Finding differentially expressed genes
print('Finding Differentially Expressed Genes...')
method = "t-test"
sc.tl.rank_genes_groups(adata, 'louvain', method=method, pts=True)

# cluster DEGs
result = adata.uns['rank_genes_groups']
dat = pd.DataFrame({group + '_' + key: result[key][group] for group in groups for key in ['names', 'logfoldchanges','scores','pvals']})
dat.to_csv(os.path.join(args.output, 'cluster_DEGs.csv'))

# perform GSEA
if args.gsea:
    print('Gene Set Enrichment Analysis...')
    df_gsea = runGSEAPY(adata)
    df_gsea.to_csv(os.path.join(args.output, 'GSEA_results.csv'))

if args.survival:
    print('Survival Analysis...')
    gencode_table = pd.read_csv(gencode, names=['id','name'])
    clinical_df_all = pd.read_csv(clinicalpath, sep='\t', index_col=0)
    cph_selected_cols = ['project', 'cell type', 'coef', 'exp(coef)', 'se(coef)', 'coef lower 95%', 'coef upper 95%','exp(coef) lower 95%', 'exp(coef) upper 95%', 'z', 'p', '-log2(p)']
    df_hazard = pd.DataFrame(columns = cph_selected_cols)
    if not args.id: # user didn't specify tumor type
        run_project_ids = project_ids
    else:
        run_project_ids = [args.id]

    for project_id in run_project_ids:
        print(project_id)
        
        # Read bulk GEP
        bk_gep_name = getBulkProfile(f'{args.tcga}/{project_id}.csv.gz', gencode_table)
        
        # Extract DEGs (masked: mitochondrial genes and genes that not exist in bulk GEPs)
        dict_deg = {}
        all_degs = []
        for celltype in adata.obs['louvain'].unique().tolist():
            degs = [x for x in adata.uns['rank_genes_groups']['names'][celltype] if not (x.startswith('MT-') or not x in bk_gep_name.index)][:20]
            dict_deg[celltype] = degs
            all_degs.extend(degs)
        all_degs = list(set(all_degs))

        # Keep expression of mutual genes in bulk GEP
        bk_gep_name = bk_gep_name.loc[all_degs,:]
        
        # Add clinical info
        if treatment_status == 'no':
            clinical_df = clinical_df_all[(clinical_df_all['project_id'] == project_id) & (clinical_df_all['isTreated'] == treatment_status)]
        else:
            clinical_df = clinical_df_all[clinical_df_all['project_id'] == project_id]
        
        # Remove treated/untreated samples
        bk_gep_name = bk_gep_name.loc[:, [x for x in clinical_df['case_submitter_id'].tolist() if x in bk_gep_name.columns]]
        if bk_gep_name.shape[1] < 10:
            print(f'Skipped: the number of {project_id} patients (treated: {treatment_status}) is less than 10.')
            continue
        
        clinical_df['_event_time_'] = clinical_df['days_to_death']
        for ind in clinical_df.index:
            if not clinical_df.loc[ind, '_event_time_'].isnumeric():
                clinical_df.loc[ind, '_event_time_'] = clinical_df.loc[ind, 'days_to_last_follow_up']
        
        # Build the score table
        # Find patients that show high/low expression for each cell type
        dict_celltype = getSpecCellDict(bk_gep_name, dict_deg)
            # Compute and draw plot
        df_new_hazard = pd.DataFrame()
        try:
            df_new_hazard = drawSurvivalPlot(dict_celltype, clinical_df, project_id)
            df_new_hazard.iloc[:,2:] = df_new_hazard.iloc[:,2:].round(2)
            #df_new_hazard = df_new_hazard.iloc[:, [0, 1, 2, 3, 7, 8, 9, 10]]
            fig, ax = plt.subplots()
            # hide axes
            fig.patch.set_visible(False)
            ax.axis('off')
            ax.axis('tight')
            table = ax.table(cellText=df_new_hazard.iloc[:,1:].values, colLabels=df_new_hazard.columns[1:], loc='center')
            table.auto_set_font_size(False)
            table.auto_set_column_width(col=list(range(len(df_new_hazard.columns[1:]))))
            table.set_fontsize(8)
            for cell in table._cells:
                if cell[0] == 0:
                    table._cells[cell].set_fontsize(6)
                    table._cells[cell].set_color('lightblue')
                    table._cells[cell].set_height(.05)
            
            ax.set_title(project_id)
            pp.savefig(fig, bbox_inches='tight')
            plt.close('all')
        except Exception as e:
            print(f'Survival analysis failed for {project_id}. The reason might be that the abundances, when conditioned on the presence or absence of death events, have very low variance and thus fail to converge.')
            continue
        else:
            df_hazard = pd.concat([df_hazard, df_new_hazard], ignore_index=True)


    adata.uns['survival_analysis'] = df_hazard
    
    df_hazard.to_csv(f'{args.output}/HR_{treatment_status}.csv')

pp.close()
adata.write(results_file)

# GEP format
if args.id: # select the tumor clusters with hr value 
    harm_clusters = df_hazard[(df_hazard['exp(coef)']>0) & (df_hazard['p']<=0.05)]['cell type'].tolist()
    if len(harm_clusters) < 2:
        sys.exit('The number of the cell types with a Hazard Ratio greater than 0 and passed the threshold (0.05) is less than 2, which is below the requirement for cell type abundance prediction.')
    else:
        adata_GEP = adata_GEP[adata.obs['louvain'].isin(harm_clusters)]
        adata.write('{}_filtered.h5ad'.format(results_file.rsplit('.',1)[0]))
if args.GEP:
    print('Exporting GEP...')
    sc.pp.normalize_total(adata_GEP, target_sum=1e6)
    mat = adata_GEP.X.transpose()
    if type(mat) is not np.ndarray:
        mat = mat.toarray()
    GEP_df = pd.DataFrame(mat, index=adata_GEP.var.index)
    GEP_df.columns = adata.obs['louvain'].tolist()
    # GEP_df = GEP_df.loc[adata.var.index[adata.var.highly_variable==True]]
    GEP_df.dropna(axis=1, inplace=True)
    GEP_df.to_csv(os.path.join(args.output, 'GEP.txt'), sep='\t')