import os
import pandas as pd
import keras
from keras.preprocessing import text
from keras.preprocessing import sequence
from keras.models import *
from feature_util import *

#load models for eSpCas9(1.1) and SpCas9-HF1
dir_path = os.path.dirname( os.path.realpath( __file__ ) )

wt_u6_model_file_path = os.path.join( dir_path, 'models/DeepWt_U6.hd5' )
wt_t7_model_file_path = os.path.join( dir_path, 'models/DeepWt_T7.hd5' )

esp_model_file_path = os.path.join( dir_path, 'models/esp_rnn_model.hd5' )
hf_model_file_path = os.path.join( dir_path, 'models/hf_rnn_model.hd5' )

model_wt_u6 = load_model( wt_u6_model_file_path )
model_wt_t7 = load_model( wt_t7_model_file_path )

model_hf = load_model( hf_model_file_path )
model_esp = load_model( esp_model_file_path )



#get embedding data
def make_data(X):
    vectorizer = text.Tokenizer( lower=False, split=" ", num_words=None, char_level=True )
    vectorizer.fit_on_texts( X )
    # construct a new vocabulary
    alphabet = "ATCG"
    char_dict = {}
    for i, char in enumerate(alphabet):
        char_dict[char] = i + 1
    word_index = {k:(v+1) for k,v in char_dict.items()}
    word_index["PAD"] = 0
    word_index["START"] = 1
    vectorizer.word_index = word_index.copy()
    index_word = {v:k for k,v in word_index.items()}
    X = vectorizer.texts_to_sequences(X)
    X = [[word_index["START"]] + [w for w in x] for x in X]
    X = sequence.pad_sequences(X)
    return X

def my_feature(df_model, feature_options):
    feature_options['order'] = 1
    feature_sets = featurize_data( df_model, feature_options )
    inputs, dim, dimsum, feature_names = concatenate_feature_sets( feature_sets )
    return inputs, dim, dimsum, feature_names, feature_sets

def get_embedding_data(data, feature_options):
    feature_options['order'] = 1
    #generating biofeatures
    r = my_feature( data, feature_options )
    lst_features = [0, 1, 2, 3, -7, -6, -5, -4, -3, -2, -1]
    feat_names = list( r[3][i] for i in lst_features )
    biofeat = r[0][:, lst_features]
    df_biofeat = pd.DataFrame( data=biofeat, columns=feat_names )
    # sequence embedding representation
    X_1 = make_data( data['21mer'] )
    # biological feature reperesentation
    X_biofeat = np.array( df_biofeat )
    return X_1, X_biofeat
def output_prediction_old(inputs, df, model_type='esp'):
    import os
    from sklearn.externals import joblib
    from sklearn.linear_model import LinearRegression
    model = load_model(model_file_path) 
    Efficiency = model.predict( inputs )
    df['gRNA_Seq'] = df['21mer'].apply( lambda x: x[:-1] )
    df['Efficiency'] = np.clip( Efficiency, 0, 1 )
    df = df.drop( ['21mer'], axis=1 )
    df.reset_index( inplace=True )
    return df.sort_values( by='Efficiency', ascending=False ).to_dict( orient='records' )

def output_prediction(inputs, df, model_type='esp'):
    import os
    from sklearn.externals import joblib
    from sklearn.linear_model import LinearRegression
    #dir_path = os.path.dirname( os.path.realpath( __file__ ) )
    #model_file = model_type + '_rnn.hd5'
    #model_file_path = os.path.join( dir_path, model_file )
    if model_type == 'esp':
        model = model_esp
    elif model_type == 'wt_u6':
        model = model_wt_u6
    elif model_type == 'wt_t7':
        model = model_wt_t7
    elif model_type == 'hf':
        model = model_hf
    Efficiency = model.predict( inputs )
    df['gRNA_Seq'] = df['21mer'].apply( lambda x: x[:-1] )
    df['Efficiency'] = np.clip( Efficiency, 0, 1 )
    r = model.predict([np.zeros((1, 22)),np.zeros((1,11))])
    df = df.drop( ['21mer'], axis=1 )
    df.reset_index( inplace=True )
    return df.sort_values( by='Efficiency', ascending=False )


def effciency_predict(sequence, model_type='esp'):
    sequence = sequence.strip()
    import re
    # 找出sequence后面20位之后含GG的index
    indexs = [m.start() for m in re.finditer( '(?=GG)', sequence ) if m.start() > 20]
    gRNA = []
    Cut_Pos = []
    Strand = []
    PAM = []
    for i in indexs:
        Strand.append( '+' )
        gRNA.append( sequence[i - 21:i] )
        Cut_Pos.append( i - 4 )
        PAM.append( sequence[i - 1:i + 2] )

    sequence_complement = str( Seq.Seq( sequence ).reverse_complement() )

    index_reverse = [m.start() for m in re.finditer( '(?=GG)', sequence_complement ) if m.start() > 20]

    for i in index_reverse:
        Strand.append( '-' )
        gRNA.append( sequence_complement[i - 21:i] )
        Cut_Pos.append( i - 4 )
        PAM.append( sequence_complement[i - 1:i + 2] )

    pandas.set_option( 'Precision', 5 )
    df = pandas.DataFrame( {'Cut_Pos': Cut_Pos,
                            'Strand': Strand,
                            '21mer': gRNA,
                            'PAM': PAM}, columns=['Strand', 'Cut_Pos', '21mer', 'PAM'] )
    X,X_biofeat = get_embedding_data(df,feature_options)
    return output_prediction( [X,X_biofeat], df, model_type )