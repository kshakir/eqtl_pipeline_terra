import argparse
import logging
import math

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gtfparse import read_gtf

logger = logging.getLogger(__name__)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--donors', dest='donor_list', type=str,
                        help="donor IDs to keep (default: %(default)s)", default="ALL")
    parser.add_argument('--genes', dest='gene_list', type=str,
                        help="genes to keep (default: %(default)s)", default="ALL")
    parser.add_argument('--thresh-umis', dest='thresh_umis', type=int,
                        help="minimum # UMIs to keep a cell (default: %(default)s)", default=0)
    parser.add_argument('--thresh-cells', dest='thresh_cells', type=int,
                        help="minimum # cells to keep a donor (default: %(default)s)", default=0)
    parser.add_argument('--remove-pct-exp', dest='remove_pct_exp', type=float,
                        help="remove the bottom percent of expressed genes (default: %(default)s)", default=0.0)
    parser.add_argument('--downscale-median-factor', dest='downscale_median_factor', type=float,
                        help="factor times median to downscale high-UMI cells (default: %(default)s)", default=2.0)
    parser.add_argument('--ignore-chr', dest='ignore_chrs', type=str,
                        help="ignore genes on chromosome", action='append')
    parser.add_argument(dest="counts", type=str,
                        help="H5AD file of counts")
    parser.add_argument(dest="donormap", type=str,
                        help="cell to donor table")
    parser.add_argument(dest="output_prefix", type=str,
                        help="prefix for output files")
    parser.add_argument(dest="gtf", type=str,
                        help="GTF file of gene info")
    parser.add_argument('-d', '--debug', dest="log_level",
                        help="Print debugging statements", action="store_const",
                        const=logging.DEBUG, default=logging.WARNING)
    parser.add_argument('-v', '--verbose', dest="log_level",
                        help="Print verbose status", action="store_const",
                        const=logging.INFO)
    args = parser.parse_args()

    logging.basicConfig(
        format='%(levelname)s - %(name)s - %(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=args.log_level,
        # Other libraries are configuring logging during their import. Force logging to use our config.
        force=True,
    )

    # load counts
    logger.info(f'load counts: {args.counts}')
    counts = anndata.read_h5ad(args.counts)
    logger.debug(f'counts original shape {counts}')
    logger.debug('make names unique')
    counts.var_names_make_unique()

    logger.debug('calculate the number of reads per cell')
    reads_all = counts.X.sum(axis=1).A.ravel()

    if args.downscale_median_factor > 0:
        # downscale high-UMI cells
        logger.debug('calculate median')
        median_count = np.median(reads_all)
        logger.debug('calculate scale_factor')
        scale_factor = np.minimum(1, args.downscale_median_factor * median_count / reads_all)  # per cell scale factor
        scale_factor = np.expand_dims(scale_factor, axis=1)
        logger.debug('calculate counts')
        logger.info(f'downscale high-UMI cells above {args.downscale_median_factor} times the median {median_count}')
        counts = anndata.AnnData(counts.to_df() * scale_factor)
    else:
        logger.info('skip downscale of high-UMI cells')
        counts = anndata.AnnData(counts.to_df())

    # remove low-UMI cells
    logger.info(f'remove cells with <= {args.thresh_umis} UMIs')
    counts = anndata.AnnData(counts[reads_all > args.thresh_umis, :].to_df())

    # plot UMIs per cell
    logger.debug('sum UMIs per cell')
    reads_all = counts.X.sum(axis=1)
    fig, ax = plt.subplots(facecolor='w')
    ax.hist(reads_all, bins=100)
    ax.set_xlabel('# UMIs per cell')
    ax.set_ylabel('# cells')
    ax.set_title(f'after downscale high-UMI cells above {args.downscale_median_factor:.2f} x median')
    fig.patch.set_facecolor('w')
    logger.info(f'plot UMIs per cell: {args.output_prefix}.umis_per_cell.postfilter.png')
    plt.savefig(f'{args.output_prefix}.umis_per_cell.postfilter.png', dpi=300)

    # load cell to donor map
    logger.info(f'load cell to donor map: {args.donormap}')
    cell_to_donor = pd.read_table(args.donormap)
    logger.debug('reassign cell_to_donor columns')
    cell_to_donor.columns = "cell donor".split()

    # filter out cells not assigned to donor
    logger.debug('filter out cells not assigned to donor')
    cell_to_donor = cell_to_donor[cell_to_donor.cell.isin(counts.obs_names)]

    # filter out donors with not enough cells
    logger.debug(f'create filter for donors with <= {args.thresh_cells} cells')
    keep_donors = cell_to_donor['donor'].value_counts()[cell_to_donor['donor'].value_counts() > args.thresh_cells].index
    logger.info(f'filter out donors with <= {args.thresh_cells} cells')
    cell_to_donor = cell_to_donor[cell_to_donor.donor.isin(keep_donors)]

    # filter to donor list
    if args.donor_list != 'ALL':
        logger.info(f'keep donors from {args.donor_list}')
        keep_donors = pd.read_csv(args.donor_list, sep='\t', header=None)[0].values
        cell_to_donor = cell_to_donor[cell_to_donor.donor.isin(keep_donors)]
    else:
        logger.info('keep all remaining donors')

    # filter to gene list
    if args.gene_list != 'ALL':
        logger.info(f'keep genes from {args.gene_list}')
        keep_genes = pd.read_csv(args.gene_list, sep='\t', header=None)[0].values
    elif 0 < args.remove_pct_exp < 100:
        logger.debug('calculate number of genes to keep')
        keep_genes_factor = (100 - args.remove_pct_exp) / 100.0
        keep_genes_count = int(math.ceil(counts.n_vars * keep_genes_factor))
        logger.debug('subset counts to cells in the donor list')
        cell_counts = counts[cell_to_donor.cell, :]
        logger.debug('calculate total expression per gene')
        expression_per_gene = cell_counts.X.sum(axis=0).ravel()
        # Get the top keep_genes_count by index
        # Via: https://stackoverflow.com/questions/6910641#answer-23734295
        logger.debug(f'generate indices for the top {keep_genes_count} of {counts.n_vars} genes')
        keep_genes_ind = np.argpartition(expression_per_gene, -keep_genes_count)[-keep_genes_count:]
        logger.info(f'keep top {keep_genes_count} of {counts.n_vars} genes, remove {args.remove_pct_exp}%')
        keep_genes = cell_counts.var_names[keep_genes_ind]
    else:
        logger.info('keep all genes')
        keep_genes = counts.var_names  # all the genes in the count matrix

    # sum counts to donors
    logger.info('sum counts to donors')
    donor_counts = pd.DataFrame(columns=keep_genes)
    for donor, cells in cell_to_donor.groupby("donor"):
        # group by donor
        logger.debug(f'group by donor {donor}')
        donor_counts.loc[donor] = counts[cells.cell, keep_genes].X.sum(axis=0).ravel()

    # transpose to a Genes x Donors table
    logger.debug('Transpose to a Genes x Donors table')
    gene_counts = donor_counts.T
    gene_counts.index.name = 'gene'

    # get gene info
    logger.info(f'get gene info: {args.gtf}')
    gene_info = read_gtf(args.gtf)
    logger.debug('query genes')
    gene_info = gene_info.query("feature == 'gene'")
    logger.debug('group by gene_name')
    gene_info = gene_info.groupby("gene_name").first().copy()
    logger.debug('locate TSS')
    gene_info['TSS'] = gene_info.start.where(gene_info.strand == '+', gene_info.end)

    # drop unknown genes
    if not args.ignore_chrs:
        logger.info('drop unknown genes')
        gene_counts = gene_counts[gene_counts.index.isin(gene_info.index)]
    else:
        logger.info(f'drop unknown genes and genes from chromosomes {args.ignore_chrs}')
        keep_gene_info = ~gene_info.seqname.isin(args.ignore_chrs)
        gene_counts = gene_counts[gene_counts.index.isin(gene_info[keep_gene_info].index)]

    # add other columns
    logger.debug('add chr column')
    gene_counts["chr"] = gene_counts.index.map(gene_info.seqname).astype(str)
    logger.debug('add start column')
    gene_counts["start"] = gene_counts.index.map(gene_info.TSS)
    logger.debug('add end column')
    gene_counts["end"] = gene_counts.index.map(gene_info.TSS) + 1
    logger.debug('add end column')
    gene_counts["strand"] = gene_counts.index.map(gene_info.strand)

    # write out filtered count matrix
    logger.debug('reset gene_counts index')
    gene_counts = gene_counts.reset_index()["chr start end gene gene strand".split() +
                                            donor_counts.index.tolist()]
    logger.debug('set gene_counts columns')
    gene_counts.columns = "#chr start end gid pid strand".split() + donor_counts.index.tolist()
    logger.info(f'write out filtered count matrix {gene_counts.shape}: {args.output_prefix}.counts.filtered.txt')
    gene_counts.to_csv(f'{args.output_prefix}.counts.filtered.txt', sep="\t", index=None)
