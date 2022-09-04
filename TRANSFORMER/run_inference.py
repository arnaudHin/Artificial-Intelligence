"""
@author : Arnaud HINCELIN
@date : 01/07/2022
"""

#CUDA_VISIBLE_DEVICES=<4>
#import preprocess_dataset as myPreprocessing
from preprocess_dataset import myTokenizer, create_dataset_fr_en, VOCAB_FR, VOCAB_EN
from transformer_model import Translator, Transformer, load_saved_transformer_model

import os
import time
import numpy as np
import tensorflow as tf
import colorama
from colorama import Fore
from colorama import Style


###-----------------------------------------------------------------###
###                             INFERENCE                           ###
###-----------------------------------------------------------------###

def compute_translation(sentence_fr, transfo):
    assert isinstance(sentence_fr, str)
    time_s = time.time()
    text_predicted, text_predicted_tokens, _ , _ = transfo(
        sentence = tf.constant(sentence_fr)
        )
    time_stop = time.time()
    print("__________________________________________________________________")
    print(Style.BRIGHT+f"-- TRANSLATION FR-to-EN | inference time : {(time_stop-time_s):.2f} seconds --")
    print(Fore.GREEN+f"{'FR input':12s} : {sentence_fr} ")
    #print( text_predicted_tokens )
    print(Fore.YELLOW+f"{'EN predicted':12s} : {text_predicted.numpy().decode('utf-8')}" + Style.RESET_ALL  )
    print("__________________________________________________________________")

def write_ALL_attention_into_pickle(input_sentence, translator_mod, attention_name):
    dir_save = "att_saved"
    assert isinstance(input_sentence, str)
    translated_text, translated_tokens, encoder_attention, decoder_attention = translator_mod(
    tf.constant(input_sentence) )

    tokens_ids_input = tokenizer_fr.tokenize(tf.convert_to_tensor([input_sentence])).to_tensor()
    tokens_inputs = tokenizer_fr.lookup(tokens_ids_input)[0]

    print(f'{"Text input:":15s}: { input_sentence }')
    print(f'{"Text predicted":15s}: { translated_text }')
    print(f'{"Tokens input":15s}: { tokens_inputs }')
    print(f'{"Tokens pred":15s}: {translated_tokens}')

    att_array = np.array( 
        [encoder_attention, decoder_attention] ) #shape : (2,)
    # att_array[0] : encoder : dict (len=4)
    # att_array[1] : decoder : dict (len=8)

    pickle_out = open( f"{dir_save}/att_{attention_name}.pickle" , "wb")
    pickle.dump(att_array, pickle_out)
    pickle_out.close()
    


###-----------------------------------------------------------------###
###                             TOKENIZERS                          ###
###-----------------------------------------------------------------###

tokenizer_fr = myTokenizer( VOCAB_FR ) 
tokenizer_en = myTokenizer( VOCAB_EN )

###-----------------------------------------------------------------###
###                             DATASET                             ###
###-----------------------------------------------------------------###
NB_TRAINING_SAMPLES = 10_000
trainset, valset, testset = create_dataset_fr_en(
            nb_samples          = NB_TRAINING_SAMPLES,
            batch_size_train    = 256,
            nb_workers          = 0
        )



########################################################
###                     MAIN                         ###
########################################################
DIR_MODELS_NAME = 'MODELS'
MODEL_NAME = "base_frTOen_noWE_part_25M"
PATH_MODEL = os.path.join( DIR_MODELS_NAME, MODEL_NAME )


model_transfo = load_saved_transformer_model( PATH_MODEL )

translator = Translator(
    tokizer_input=tokenizer_fr, 
    tokizer_out=tokenizer_en, 
    transformer=model_transfo)



sentence = "Cette équipe est absolument géniale et j'apprécie beaucoup travailler avec ces personnes."
compute_translation(sentence, translator)

sentence = "Bonjour Patrick, je te suggère une réunion pour parler du projet ce jeudi à partir de 13h ou bien à partir de 17h."
compute_translation(sentence, translator)

sentence = "Nous devons régler rapidement ce problème qui fait perdre des millions de dollars à notre entreprise."
compute_translation(sentence, translator)



if 0 :
    sentence = "Bonjour Patrick, je te suggère une réunion pour parler du projet ce jeudi à partir de 13h ou bien à partir de 17h."
    compute_translation(sentence, translator)

    sentence = "Sorin est le meilleur !"
    compute_translation(sentence, translator)

    sentence = "Katy habite à Fort Collins et elle dirige notre équipe technique."
    compute_translation(sentence, translator)
    #write_ALL_attention_into_pickle(sentence, translator)

    sentence = "Hana est ma tutrice de stage, et elle est la plus gentille."
    compute_translation(sentence, translator)

    sentence = "Nous devons régler rapidement ce problème qui fait perdre des millions de dollars à notre entreprise."
    compute_translation(sentence, translator)




