
import os
import pathlib
import re

import psutil
import colorama
from colorama import Style, Fore
def print_ram_used() -> None:
    ram_used_bytes = psutil.Process().memory_info().rss
    ram_used_MB = ram_used_bytes/(1024 * 1024)
    print( "------->> "+Fore.RED + Style.BRIGHT+ f"RAM used by process : {ram_used_MB:.2f} MB" + Style.RESET_ALL )


from numpy import array, pad, linspace, concatenate
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text
 

#tf.config.run_functions_eagerly(True)
"""
gpus = tf.config.list_physical_devices('GPU')
print("Num physical GPUs Available: ", len(gpus) )
for gpu in gpus:
    print( tf.config.experimental.get_device_details(gpu) )
    tf.config.experimental.set_memory_growth(gpu, True)
"""

"""
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,2,3,4"
"""

MAX_TOKENS_PER_SENTENCE = 200
BUFFER_SIZE = 10_000
VOC_FR_SIZE = 0
VOC_EN_SIZE = 0

###-----------------------------------------------------------------###
###                          TOKENIZERS                             ###
###-----------------------------------------------------------------###

def add_start_end(ragged, reserved_tokens):
    START = tf.argmax(tf.constant(reserved_tokens) == "[START]")
    END = tf.argmax(tf.constant(reserved_tokens) == "[END]")
    count = ragged.bounding_shape()[0]
    starts = tf.fill([count,1], START)
    ends = tf.fill([count,1], END)
    return tf.concat([starts, ragged, ends], axis=1)

def cleanup_text(reserved_tokens, token_txt):
    # Drop the reserved tokens, except for "[UNK]".
    bad_tokens = [re.escape(tok) for tok in reserved_tokens if tok != "[UNK]"]
    bad_token_re = "|".join(bad_tokens)

    bad_cells = tf.strings.regex_full_match(token_txt, bad_token_re)
    result = tf.ragged.boolean_mask(token_txt, ~bad_cells)

    # Join them into strings.
    result = tf.strings.reduce_join(result, separator=' ', axis=-1)

    return result


class myTokenizer(tf.Module):
    def __init__(self, vocab_path):

        self.tokenizer = tensorflow_text.BertTokenizer(
            vocab_lookup_table = vocab_path,
            suffix_indicator='##',
            max_bytes_per_word=100,
            max_chars_per_token=None,
            token_out_type=tf.int64,
            unknown_token='[UNK]',
            split_unknown_characters=False,               
            lower_case=True)

        self._reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]
        self._vocab_path = tf.saved_model.Asset(vocab_path)

        vocab = pathlib.Path(vocab_path).read_text().splitlines()
        self.vocab = tf.Variable(vocab)

        ## Create the signatures for export:   

        # Include a tokenize signature for a batch of strings. 
        self.tokenize.get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string))

        # Include `detokenize` and `lookup` signatures for:
        #   * `Tensors` with shapes [tokens] and [batch, tokens]
        #   * `RaggedTensors` with shape [batch, tokens]
        self.detokenize.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.detokenize.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

        self.lookup.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.lookup.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

        # These `get_*` methods take no arguments
        self.get_vocab_size.get_concrete_function()
        self.get_vocab_path.get_concrete_function()
        self.get_reserved_tokens.get_concrete_function()

    @tf.function
    def tokenize(self, strings):
        enc = self.tokenizer.tokenize(strings)
        # Merge the `word` and `word-piece` axes.
        enc = enc.merge_dims(-2,-1)
        enc = add_start_end(
            ragged=enc, 
            reserved_tokens=self._reserved_tokens )
        return enc

    @tf.function
    def detokenize(self, tokenized):
        words = self.tokenizer.detokenize(tokenized)
        return cleanup_text(self._reserved_tokens, words)

    @tf.function
    def lookup(self, token_ids):
        return tf.gather(self.vocab, token_ids)

    @tf.function
    def get_vocab_size(self):
        return tf.shape(self.vocab)[0]

    @tf.function
    def get_vocab_path(self):
        return self._vocab_path

    @tf.function
    def get_reserved_tokens(self):
        return tf.constant(self._reserved_tokens)


USE_DET_AI = False

if USE_DET_AI:
    VOCAB_FR = "CONFIG_DT_AI/fr_vocab_para_crawl.txt"
    VOCAB_EN = "CONFIG_DT_AI/en_vocab_para_crawl.txt"
else:
    VOCAB_FR = "../TOKENIZATION/fr_vocab_para_crawl.txt"
    VOCAB_EN = "../TOKENIZATION/en_vocab_para_crawl.txt"

tokenizer_fr = myTokenizer( VOCAB_FR ) 
tokenizer_en = myTokenizer( VOCAB_EN )

VOC_EN_SIZE = tokenizer_en.get_vocab_size().numpy() #30_000
VOC_FR_SIZE = tokenizer_fr.get_vocab_size().numpy() #30_000

tf.saved_model.save(tokenizer_fr, "module_tokenizer_fr")


###-----------------------------------------------------------------###
###                        DATASET CREATION                         ###
###-----------------------------------------------------------------###


def filter_max_tokens(inp, tar):
    num_tokens = tf.maximum(tf.shape(inp)[1] , tf.shape(tar)[1])
    return num_tokens < MAX_TOKENS_PER_SENTENCE

def map_fct_FR_EN(en, fr):
    fr = tokenizer_fr.tokenize( fr )
    # Convert from ragged to dense, padding with zeros.
    fr = fr.to_tensor()
    en = tokenizer_en.tokenize( en )
    # Convert from ragged to dense, padding with zeros.
    en = en.to_tensor()

    return fr, en

def create_dataset_fr_en(nb_samples, batch_size_train, nb_workers):

    print( "Creating dataset..." )

    if USE_DET_AI:
        name='para_crawl/enfr_plain_text'
    else:
        name = 'para_crawl/enfr'

    samples_frTOen, metadata = tfds.load(
        name=name, 
        split=['train[:90%]', 'train[90%:95%]', 'train[95%:100%]'],
        with_info=True,
        as_supervised=True, 
        data_dir="/cstor/SHARED/datasets/NLP/translation/")
    # samples_frTOen total : 31,374,161 samples
    # samples_frTOen = list of len 3 according to split argument
    
    trainset = samples_frTOen[0]
    valset = samples_frTOen[1]
    testset = samples_frTOen[2]

    #Validation set : (batchsize, int64, int64)
    valset = valset.shuffle(BUFFER_SIZE).batch(16)
    valset = valset.map(map_fct_FR_EN, num_parallel_calls=tf.data.AUTOTUNE)
    valset = valset.filter(filter_max_tokens)

    #Training set : (batchsize, int64, int64)
    trainset = trainset.shuffle(BUFFER_SIZE)
    trainset = trainset.take( nb_samples )
    trainset = trainset.batch(batch_size_train)
    trainset = trainset.map(map_fct_FR_EN, num_parallel_calls=tf.data.AUTOTUNE)
    trainset = trainset.filter(filter_max_tokens)
    trainset = trainset.cache()
    trainset = trainset.prefetch(buffer_size = batch_size_train)

    #Test set : (String, String)
    testset = testset.shuffle(BUFFER_SIZE).batch(batch_size_train)

    """
    CACHE
    -->> dataset.cache(filename='') <<--
    
    -> The first time the dataset is iterated over, its elements will be cached either in the 
    specified file or in memory. Subsequent iterations will use the cached data.
    
    dataset = tf.data.Dataset.range(5)
    dataset = dataset.cache()
    ## The first time reading through the data will generate the data using
    list(dataset.as_numpy_iterator())

    ## Subsequent iterations read from the cache.
    list(dataset.as_numpy_iterator())

    -> if arg filename : dataset = dataset.cache("/path/to/file")
    When caching to a file, the cached data will persist across runs. 
    Even the first iteration through the data will read from the cache file. 
    Changing the input pipeline before the call to .cache() will have no effect until the cache file is removed or the filename is changed.    

    dataset = tf.data.Dataset.range(10)
    dataset = dataset.cache("/path/to/file")
    ## IF the file "/path/to/file" exists, the cached dataset will be loaded ! 
    ## either, this dataset range(10) will be cached into this file
    """

    return ( trainset, valset, testset )


"""
The tf.data.Dataset.cache transformation can cache a dataset, 
either in memory or on local storage. This will save some operations 
(like file opening and data reading) from being executed during each 
epoch. The next epochs will reuse the data cached by the cache 
transformation.

Prefetch overlaps the preprocessing and model execution of a training step.
While the model is executing training step s, the input pipeline is reading the data for step s+1. 
Doing so reduces the step time to the maximum (as opposed to the sum) of the training and the time it takes to extract the data.


Pbs : 
WARNING:tensorflow:Efficient allreduce is not supported for 2 IndexedSlices
WARNING:tensorflow:Efficient allreduce is not supported for 2 IndexedSlices
WARNING:tensorflow:Efficient allreduce is not supported for 2 IndexedSlices
WARNING:tensorflow:Efficient allreduce is not supported for 2 IndexedSlices

Responses : 
Hi @ratovarius, right now MirroredStrategy.reduce will do a concatenation of IndexedSlices on one device, and broadcast the result back to all GPUs. This is not efficient, hence the warning. This is a known limitation and the current suggestion is to use MultiWorkerMirroredStrategy, which has a slightly better implementation for handling IndexedSlices. Despite its name, MultiWorkerMirroredStrategy can be used with a single worker without additional setup.

Thanks for your answer @nikitamaia
Is there a way to configure witch GPUs to use with MultiWorkerMirroredStrategy ? By default it uses all GPUs, right?
I want to make some performance measurements.

I tested your suggestion, using MultiWorkerMirroredStrategy removes the warning, but it is taking twice time for training using
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy() # Default -> 8GPUs
compared to,
strategy = tf.distribute.MirroredStrategy(['/gpu:0'])

I'm not sure about the concept of IndexedSlices. Does it get generated when using multi-input model and not when using a single input one?


"""


###-----------------------------------------------------------------###
###                         DATASET VIEWS                           ###
###-----------------------------------------------------------------###


def print_lines_from_dataset(name_dataset, nb_lines):
    # name_dataset= 'para_crawl/enfr_plain_text'
    trainset, metadata = tfds.load(
        name=name_dataset, 
        split=['train'],
        with_info=True,
        as_supervised=True, 
        data_dir="/cstor/SHARED/datasets/NLP/translation/")
    # samples_frTOen total : 31,374,161 samples
    # samples_frTOen = list of len 3 according to split argument
    for en, fr in trainset.take(nb_lines):
        print( en.numpy().decode('utf-8') )
        print( fr.numpy().decode('utf-8') )
        print()


def visualize_tokens_len(name_dataset):

    # name_dataset= 'para_crawl/enfr_plain_text'

    import matplotlib.pyplot as plt

    lengths = []
    trainset, metadata = tfds.load(
        name=name_dataset, 
        split=['train'],
        with_info=True,
        as_supervised=True, 
        data_dir="/cstor/SHARED/datasets/NLP/translation/")
    # samples_frTOen total : 31,374,161 samples
    # samples_frTOen = list of len 3 according to split argument

    print( trainset)
    for en, fr in trainset[0].batch(1024):
        fr = tokenizer_fr.tokenize( fr )
        lengths.append(fr.row_lengths())
        en = tokenizer_en.tokenize( en )
        lengths.append(en.row_lengths())
        print('.', end='', flush=True)

    all_lengths = concatenate(lengths)

    plt.hist(all_lengths, linspace(0, 500, 101))
    plt.ylim(plt.ylim())
    max_length = max(all_lengths)
    plt.plot([max_length, max_length], plt.ylim())
    plt.title(f'Max tokens per example: {max_length}')
    plt.savefig("tokens_len.png" )



if 0:

    print("NONO : ")
    fr_str = "La trotinette électrique est un bon moyen de transport, en ville ou en campagne, même quand il fait chaud. Dejbox c'est bien."
    print( " ".join(tokenizer_fr.lookup(tokenizer_fr.tokenize( fr_str )).numpy().tolist()) )
    print("\n HUGO : ")
    print("la tro ##tine ##tte electrique est un bon moyen de transport , en ville ou en campagne , meme quand il fait chaud . dej ##box c ' est bien ."
    )

    print("NONO : ")
    en_str = "Sorin brought pain au chocolat this morning, and we ate them while watching some unicorns fly by the window. Champollion then exploded in feux d'artifices all over the place."
    print( " ".join(tokenizer_en.lookup(tokenizer_en.tokenize( en_str )).numpy().tolist())  )
    print("\n HUGO : ")
    print("sor ##in brought pain au chocol ##at this morning , and we ate them while watching some unic ##orn ##s fly by the window . champ ##oll ##ion then exploded in fe ##ux d ' artific ##es all over the place ."
    )

print_ram_used()

ds = create_dataset_fr_en(1_000_000, 256, 4)

print_ram_used()



