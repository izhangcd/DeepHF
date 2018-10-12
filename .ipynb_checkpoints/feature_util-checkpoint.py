'''
- The nucleotide sequence feature construction code was mainly modified from Azimuth project(https://github.com/MicrosoftResearch/Azimuth).
- The structure features were inspired from WU-CRISPR(http://crispr.wustl.edu/)
- RNAfold binary should be configured in PATH environment. Vienna RNA package from ViennaRNA(2.4.5) was recommened to be installed.
- Concurrent processing of structure features was automatically opened.
'''
import pandas
import time
import sklearn
import numpy as np
import Bio.SeqUtils as SeqUtil
import Bio.Seq as Seq
import math
import sys
import Bio.SeqUtils.MeltingTemp as Tm
import pickle
import itertools
from multiprocessing import Pool

######feature options########
feature_options = {
                 "testing_non_binary_target_name": 'ranks',
                 'include_pi_nuc_feat': True,
                 "gc_features": True,
                 "nuc_features": True,
                 "include_Tm": True,
                 "include_structure_features": True,
                 "order": 3,
                 "num_proc": 20,
                 "normalize_features":None
                 }

######parallelize processing########
def parallelize_dataframe(df, func, num_partitions, num_processes=1):
    df_split = np.array_split( df, num_partitions )
    pool = Pool( num_processes )
    a = pool.map( func, df_split )

    dataList = []
    aList = []
    bList = []
    for x in a:
        dataList.append( x[0] )
        aList.append( x[1] )
        bList.append( x[2] )
    pool.close()
    pool.join()

    return pandas.concat( dataList ), pandas.concat( aList ), pandas.concat( bList )


def featurize_data(data, feature_options, length_audit=True, quiet=True):
    '''
    assumes that data contains the 21mer column
    returns set of features from which one can make a kernel for each one
    '''
    all_lens = data['21mer'].apply( len ).values
    unique_lengths = np.unique( all_lens )

    num_lengths = len( unique_lengths )

    assert num_lengths == 1, "should only have sequences of a single length, but found %s: %s" % (
        num_lengths, str( unique_lengths ))

    if not quiet:
        print( "Constructing features..." )
    t0 = time.time()

    feature_sets = {}
    data_rows = data['21mer']

    if feature_options["include_structure_features"]:
        num_partitions = math.ceil( len( data['21mer'] ) / 1000 )  # number of partitions to split dataframe
        num_processes = feature_options['num_proc'] * 5  # number of cores on your machine
        data_rows, feature_sets['dG_features'], ba_rows = parallelize_dataframe( data['21mer'], get_structural_feat,
                                                                                 num_partitions, num_processes )

        get_all_order_ba_features( ba_rows, feature_sets, feature_options, feature_options["order"], max_index_to_use=99,
                                   quiet=False )
        check_feature_set( feature_sets, 3 )

    if feature_options["nuc_features"]:
        # spectrum kernels (position-independent) and weighted degree kernels (position-dependent)
        get_all_order_nuc_features( data_rows, feature_sets, feature_options, feature_options["order"],
                                    max_index_to_use=21, quiet=quiet )

    check_feature_set( feature_sets )

    if feature_options["gc_features"]:
        gc_above_10, gc_below_10, gc_count = gc_features( data_rows, length_audit )
        feature_sets['gc_above_10'] = pandas.DataFrame( gc_above_10 )
        feature_sets['gc_below_10'] = pandas.DataFrame( gc_below_10 )
        feature_sets['gc_count'] = pandas.DataFrame( gc_count )

    if feature_options["include_Tm"]:
        feature_sets["Tm"] = Tm_feature( data_rows, feature_options=None )

    t1 = time.time()
    if not quiet:
        print( "\t\tElapsed time for constructing features is %.2f seconds" % (t1 - t0) )

    if feature_options['normalize_features']:
        assert (
            "should not be here as doesn't make sense when we make one-off predictions, but could make sense for internal model comparisons when using regularized models")
        feature_sets = normalize_feature_sets( feature_sets )
        check_feature_set( feature_sets )

    return feature_sets


def check_feature_set(feature_sets, debug=0):
    '''
    Ensure the # of people is the same in each feature set
    TypeError: ufunc 'isnan' not supported for the input types, and the inputs could not be safely coerced to
    any supported types according to the casting rule ''safe'' 
    '''
    assert feature_sets != {}, "no feature sets present"

    N = None
    for ft in feature_sets.keys():
        N2 = feature_sets[ft].shape[0]
        if N is None:
            print( "N-{},N2-{}".format( N, N2 ) )
            N = N2
        else:
            assert N >= 1, "should be at least one individual"
            assert N == N2, "# of individuals do not match up across feature sets"

    for set in feature_sets.keys():
        if np.any( np.isnan( feature_sets[set] ) ):
            raise Exception( "found Nan in set %s" % set )


def countGC(s, length_audit=True):
    '''
    GC content for only the 20mer, as per the Doench paper/code
    '''
    if length_audit:
        assert len( s ) == 21, "seems to assume 21mer"
    return len( s[0:20].replace( 'A', '' ).replace( 'T', '' ) )


def gc_cont(seq):
    return (seq.count( 'G' ) + seq.count( 'C' )) / float( len( seq ) )


def Tm_feature(data, feature_options=None):

    if feature_options is None or 'Tm segments' not in feature_options.keys():
        segments = [(15, 21), (4, 13), (0, 4)]
    else:
        segments = feature_options['Tm segments']

    sequence = data.values
    featarray = np.ones( (sequence.shape[0], 4) )

    for i, seq in enumerate( sequence ):
        rna = False
        featarray[i, 0] = Tm.Tm_staluc( seq, rna=rna )  # 21mer Tm
        featarray[i, 1] = Tm.Tm_staluc( seq[segments[0][0]:segments[0][1]],
                                        rna=rna )  # 5nts immediately proximal of the NGG PAM
        featarray[i, 2] = Tm.Tm_staluc( seq[segments[1][0]:segments[1][1]], rna=rna )  # 8-mer
        featarray[i, 3] = Tm.Tm_staluc( seq[segments[2][0]:segments[2][1]], rna=rna )  # 4-mer

    feat = pandas.DataFrame( featarray, index=data.index,
                             columns=["Tm global_%s" % rna, "5mer_end_%s" % rna, "8mer_middle_%s" % rna,
                                      "4mer_start_%s" % rna] )
    return feat


def gc_features(data, audit=True):
    gc_count = data.apply( lambda seq: countGC( seq, audit ) )
    gc_count.name = 'GC count'
    gc_above_10 = (gc_count > 10) * 1
    gc_above_10.name = 'GC > 10'
    gc_below_10 = (gc_count < 10) * 1
    gc_below_10.name = 'GC < 10'
    return gc_above_10, gc_below_10, gc_count


def normalize_features(data, axis):
    '''
    input: Pandas.DataFrame of dtype=np.float64 array, of dimensions
    mean-center, and unit variance each feature
    '''
    data -= data.mean( axis )
    data /= data.std( axis )
    # remove rows with NaNs
    data = data.dropna( 1 )
    if np.any( np.isnan( data.values ) ): raise Exception( "found NaN in normalized features" )
    return data


def get_all_order_nuc_features(data, feature_sets, feature_options, maxorder, max_index_to_use, prefix="", quiet=True):
    print('Constructing nucleotide features.')
    for order in range( 1, maxorder + 1 ):
        nuc_features_pd, nuc_features_pi = apply_sparse_seq_features( data, order, feature_options["num_proc"],
                                                                      include_pos_independent=True,
                                                                      max_index_to_use=max_index_to_use,
                                                                      raw_alphabet=['A', 'T', 'C', 'G'], prefix=prefix )
        feature_sets['%s_nuc_pd_Order%i' % (prefix, order)] = nuc_features_pd
        if feature_options['include_pi_nuc_feat']:
            feature_sets['%s_nuc_pi_Order%i' % (prefix, order)] = nuc_features_pi
        check_feature_set( feature_sets )


def get_all_order_ba_features(data, feature_sets, feature_options, maxorder, max_index_to_use, prefix="", quiet=True):
    print('Constructing sencondary structure features.')
    for order in range( 1, maxorder + 1 ):
        nuc_features_pd, nuc_features_pi = apply_sparse_seq_features( data, order, feature_options["num_proc"],
                                                                      include_pos_independent=True,
                                                                      max_index_to_use=max_index_to_use,
                                                                      raw_alphabet=['D', 'B'], prefix=prefix )
        feature_sets['%s_ba_pd_Order%i' % (prefix, order)] = nuc_features_pd
        if feature_options['include_pi_nuc_feat']:
            feature_sets['%s_ba_pi_Order%i' % (prefix, order)] = nuc_features_pi
        check_feature_set( feature_sets, 3 )


def apply_sparse_seq_features(seq_data, order, num_proc, include_pos_independent, max_index_to_use, raw_alphabet,
                              prefix=""):
    fast = True
    if include_pos_independent:
        feat_pd = seq_data.apply( sparse_features,
                                  args=(order, max_index_to_use, prefix, raw_alphabet, 'pos_dependent') )
        feat_pi = seq_data.apply( sparse_features,
                                  args=(order, max_index_to_use, prefix, raw_alphabet, 'pos_independent') )
        assert not np.any( np.isnan( feat_pd ) ), "nans here can arise from sequences of different lengths"
        assert not np.any( np.isnan( feat_pi ) ), "nans here can arise from sequences of different lengths"
        return feat_pd, feat_pi
    else:
        feat_pd = seq_data_frame.apply( nucleotide_features,
                                        args=(order, max_index_to_use, prefix, raw_alphabet, 'pos_dependent') )
        assert not np.any( np.isnan( feat_pd ) ), "found nan in feat_pd"
        return feat_pd


def get_alphabet(order, raw_alphabet=['A', 'T', 'C', 'G']):
    alphabet = ["".join( i ) for i in itertools.product( raw_alphabet, repeat=order )]
    return alphabet


def sparse_features(s, order, max_index_to_use, prefix="", raw_alphabet=['A', 'T', 'C', 'G'], feature_type='all'):
    assert feature_type in ['all', 'pos_independent', 'pos_dependent']
    if max_index_to_use <= len( s ):
        # print "WARNING: trimming max_index_to use down to length of string=%s" % len(s)
        max_index_to_use = len( s )

    if max_index_to_use is not None:
        s = s[:max_index_to_use]
 
    alphabet = get_alphabet( order, raw_alphabet=raw_alphabet )
    features_pos_dependent = np.zeros( len( alphabet ) * (len( s ) - (order - 1)) )
    features_pos_independent = np.zeros( np.power( len( raw_alphabet ), order ) )

    index_dependent = []
    index_independent = []

    for position in range( 0, len( s ) - order + 1, 1 ):
        for l in alphabet:
            index_dependent.append( '%s%s_%d' % (prefix, l, position) )

    for l in alphabet:
        index_independent.append( '%s%s' % (prefix, l) )

    for position in range( 0, len( s ) - order + 1, 1 ):
        nucl = s[position:position + order]
        features_pos_dependent[alphabet.index( nucl ) + (position * len( alphabet ))] = 1.0
        features_pos_independent[alphabet.index( nucl )] += 1.0

        # this is to check that the labels in the pd df actually match the nucl and position
        assert index_dependent[alphabet.index( nucl ) + (position * len( alphabet ))] == '%s%s_%d' % (
            prefix, nucl, position)
        assert index_independent[alphabet.index( nucl )] == '%s%s' % (prefix, nucl)

    if np.any( np.isnan( features_pos_dependent ) ):
        raise Exception( "found nan features in features_pos_dependent" )
    if np.any( np.isnan( features_pos_independent ) ):
        raise Exception( "found nan features in features_pos_independent" )

    if feature_type == 'all' or feature_type == 'pos_independent':
        if feature_type == 'all':
            res = pandas.Series( features_pos_dependent, index=index_dependent ), pandas.Series(
                features_pos_independent, index=index_independent )
            assert not np.any( np.isnan( res.values ) )
            return res
        else:
            res = pandas.Series( features_pos_independent, index=index_independent )
            assert not np.any( np.isnan( res.values ) )
            return res

    res = pandas.Series( features_pos_dependent, index=index_dependent )
    assert not np.any( np.isnan( res.values ) )
    return res


def concatenate_feature_sets(feature_sets, keys=None):
    import numpy as np
    assert feature_sets != {}, "no feature sets present"
    if keys is None:
        keys = feature_sets.keys()

    F = feature_sets[list( keys )[0]].shape[0]
    for set in feature_sets.keys():
        F2 = feature_sets[set].shape[0]
        assert F == F2, "not same # individuals for features %s and %s" % (keys[0], set)

    N = feature_sets[list( keys )[0]].shape[0]
    inputs = np.zeros( (N, 0) )
    feature_names = []
    dim = {}
    dimsum = 0
    for set in keys:
        inputs_set = feature_sets[set].values
        dim[set] = inputs_set.shape[1]
        dimsum += dim[set]
        inputs = np.hstack( (inputs, inputs_set) )
        feature_names.extend( feature_sets[set].columns.tolist() )

    if False:
        inputs.shape
        for j in keys: print( j + str( feature_sets[j].shape ) )
        import ipdb;
        ipdb.set_trace()

    # print "final size of inputs matrix is (%d, %d)" % inputs.shape
    return inputs, dim, dimsum, feature_names


############################################################
################***Structure Featuer***#####################
############################################################


def generate_bytes_file(data):
    scaffold_seq = "GTTTTAGAGCTAGAAATAGCAAGTTAAAATAAGGCTAGTCCGTTATCAACTTGAAAAAGTGGCACCGAGTCGGTGCTTT";
    gRNA_bytes = '\n'.join( list( data.apply( lambda x: '>{0}\n{1}'.format( x[:21], x[:20] ) ) ) ).encode()
    gRNA_plus_tracr_bytes = '\n'.join(
        list( data.apply( lambda x: '>{0}\n{1}'.format( x[:21], x[:20] + scaffold_seq ) ) ) ).encode()
    return gRNA_bytes, gRNA_plus_tracr_bytes


def get_structural_feat(rows):
    bytes_list = generate_bytes_file( rows )
    # print('Get bytes_list 0,20bp')
    # 20bp input for free energy feaure
    lst_0 = get_dG( bytes_list )[0].split( '\n' )
    # 99bp input（20bp + 79bp tracr）
    # print('Get bytes_list 1,99bp')
    lst_1 = get_dG( bytes_list )[1].split( '\n' )
    r = []
    '''
    a--('>AAAAAACACAAGCAAGACCG', '>AAAAAACACAAGCAAGACCG'),fasta header
    b--('AAAAAACACAAGCAAGACCG', 'AAAAAACACAAGCAAGACCGGUUUUAG...AGCUAGAAAUA'), RNAFold transfromed gRNA
    c--('.................... (  0.00)', '........((....(((((((((((((.(((。。。。 (-27.60)')，secondary structure
    '''
    base_pair_List = []

    for a, b, c in grouped( zip( lst_0, lst_1 ), 3 ):
        align_seq = c[1][:99]
        base_pair_List.append( align_seq.replace( '.', 'D' ).replace( '(', 'B' ).replace( ')', 'B' ) )

        # whether exists stem-loop
        ext_stem = "(((((((((.((((....))))...)))))))"
        aligned_stem = align_seq[18:18 + len( ext_stem )]
        stem = 1 if ext_stem == aligned_stem else 0
        dG = c[0].split( ' (' )[1][:-2].strip()
        dG_binding_20 = dG_binding( a[0][1:21] )
        dg_binding_7to20 = dG_binding( a[0][8:21] )
        simple_feature_group = [stem, dG, dG_binding_20, dg_binding_7to20]
        r.append( simple_feature_group )

    ba_rows = pandas.Series( base_pair_List )
    df_feat = pandas.DataFrame( r ).astype( 'float64' )
    colums_name_list = ['stem', 'dG', 'dG_binding_20', 'dg_binding_7to20']
    df_feat.columns = colums_name_list

    return rows, df_feat, ba_rows


def grouped(iterable, n):
    "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
    return zip( *[iter( iterable )] * n )


def base_accessibility(align_seq):
    alignment = align_seq.replace( '.', 'D' ).replace( '(', 'B' ).replace( ')', 'B' )
    r = []
    for i, v in enumerate( alignment ):
        if i in feature_options['secondary_structure_list']:
            r.append( v )
    ext_stem = "(((((((((.((((....))))...)))))))"
    aligned_stem = align_seq[18:18 + len( ext_stem )]
    if ext_stem == aligned_stem:
        r.append( 1 )
    else:
        r.append( 0 )
    return r


def get_dG(bytes_list, RNAfold_BIN='RNAfold'):
    import subprocess
    CMD = [RNAfold_BIN, '--noPS']
    CMD = ' '.join( str( v ) for v in CMD )
    r = []
    for data in bytes_list:
        p = subprocess.Popen( CMD,
                              shell=True,
                              stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              executable='/bin/bash' )
        stdout, stderr = p.communicate( input=data )
        stdout = stdout.decode( 'utf-8' )
        stderr = stderr.decode( 'utf-8' )
        r.append( stdout )
    return r


def dG_binding(seq):
    seq = seq.lower()
    dG = {'aa': -0.2, 'tt': -1, 'at': -0.9,
          'ta': -0.6, 'ca': -1.6, 'tg': -0.9,
          'ct': -1.8, 'ag': -0.9, 'ga': -1.5,
          'tc': -1.3, 'gt': -2.1, 'ac': -1.1,
          'cg': -1.7, 'gc': -2.7, 'gg': -2.1, 'cc': -2.9}

    seq = seq.replace( 'u', 't' )
    binding_dG = 0
    dGi = 3.1
    for i in range( 0, len( seq ) - 1 ):
        key = seq[i:i + 2]
        binding_dG += dG[key]
    binding_dG += dGi
    return binding_dG