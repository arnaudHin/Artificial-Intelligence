"""
@author : Arnaud HINCELIN
@date : 05/07/2022
"""

import sys
sys.path.append("../shared")
#CUDA_VISIBLE_DEVICES=<4>
#import preprocess_dataset as myPreprocessing
from preprocess_dataset import myTokenizer, create_dataset_fr_en, MAX_TOKENS_PER_SENTENCE
from transformer_model import Translator, Transformer

import time
import tensorflow as tf
from utils import print_ram_used


GPATH_EN_FR_TRANSLATION_PARA = "/cstor/SHARED/datasets/NLP/translation/"

TK_FR = "../TOKENIZATION/fr_vocab_para_crawl.txt"
TK_EN = "../TOKENIZATION/en_vocab_para_crawl.txt"

MAX_TOKENS = MAX_TOKENS_PER_SENTENCE



###-----------------------------------------------------------------###
###                             OPTIMIZER                           ###
###-----------------------------------------------------------------###
"""
from origianl Transfo paper : https://arxiv.org/pdf/1706.03762.pdf
Optimizer Adam, with b1=0.9, b2=0.98, and epsilon=1.10^-9 

lr = d_model^(-0.5) * min( step_num^(-0.5), step_num*warmup_steps^(-1.5) )

step_num = curretn strp
warmup_steps=4000 (all steps)

We increase the learning rate linearly for the first warmup_steps training step. 
and decrease it after 

"""
class CustomSchedule( tf.keras.optimizers.schedules.LearningRateSchedule ):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

"""
LOSS : distance between the predicted value by your model and the true value.
commun : Regression=Mean Square Error (MSE), and for classification/NN=Cross-Entrpopy

y_true = integer representing the id of classe, so the token (as many classes as vocab_size )
y_pred = a vector of fp32 representing proba for all classes

note : if y_true is onehot vector instead of integer, use CategoricalCrossentropy()
    else if y_true is an integer (id of class), use SparseCategoricalCrossentropy()

For a sentence of words : 
y_true = vector of seq_len integers (tokens) : (1, seq_len, vocab_size)
y_pred = seq_len vector of vocab_size vectors : (1, seq_len)
"""

def loss_function(loss_object, real, pred):
    #pred : (batch_size, tar_seq_len, target_vocab_size)
    #real : (batch_size, tar_seq_len)
    
    ## 1 -> Cross loss between the label and the predictions
    #Crosse entropy, real=tokens and predict_seq=vectors of proba
    loss_ = loss_object(real, pred)# (batch_size, seq_len)
    
    ## 2 -> Create mask vector for real : (contains True if PAD(0), and False otherwise )
    # equal(x, y,) : Returns the truth value of (x == y) element-wise.
    # logical_not(x) : Returns the truth value of NOT x element-wise
    padding_mask = tf.math.equal(real, 0) # (batch_size, seq_len)
    padding_mask = tf.math.logical_not( padding_mask ) # bool (batch_size, seq_len)
    padding_mask = tf.cast(padding_mask, dtype=loss_.dtype) #fp32 (batch_size, seq_len)

    loss_ *= padding_mask #fp32 (batch_size, seq_len)
    total_loss = tf.reduce_sum(loss_) / tf.reduce_sum(padding_mask) #fp32 (1,1)

    return total_loss

"""
ACC : the number of error you made on the data
It is one of the metrics to measure the performance of your model.
ACC = Nb_good_prediction / Nb_total_prediction
"""
def accuracy_function(real, pred):

    ## 1 -> Build accuracies : Is each token correctly predicted
    #take ids of the tokens of thepredicted sentence
    predicted_seq_tokens = tf.argmax(pred, axis=2) #int (batch_size, tar_seq_len)
    
    #take the bool value of the prediction : True if token predicted is right, False otherwise
    accuracies = tf.math.equal(real, predicted_seq_tokens) #bool (batch_size, tar_seq_len)

    ## 2 -> Create mask vector for target_seq : (contains True if 0(PAD), and False otherwise )
    padding_mask = tf.math.equal(real, 0) # bool (batch_size, tar_seq_len)

    #mask where False is a padding, and True is a word
    padding_mask = tf.math.logical_not( padding_mask ) # bool (batch_size, tar_seq_len)

    #mask where True is a word correctly predicted, and which is not a PADDING, False otherwise
    accuracies = tf.math.logical_and(padding_mask, accuracies) # bool (batch_size, tar_seq_len)

    #cast into fp32 befiore to sum
    accuracies = tf.cast( accuracies, dtype=tf.float32) # fp32 (batch_size, tar_seq_len)
    padding_mask = tf.cast( padding_mask, dtype=tf.float32) # fp32 (batch_size, tar_seq_len)

    #compute the general accuracy : good_prediction / total_prediction
    general_acc = tf.reduce_sum(accuracies)/ tf.reduce_sum(padding_mask) #fp32 (1,1)

    return general_acc







# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.

def train_step(sample):
    inp, tar = sample
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    with tf.GradientTape() as tape:
        predictions, _ ,_ = transformer([inp, tar_inp], training = True)
        loss_val = loss_function(loss_object, tar_real, predictions)

    gradients = tape.gradient(loss_val, transformer.trainable_variables)
    opt.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss_val)
    train_accuracy(accuracy_function(tar_real, predictions))
    return loss_val 


@tf.function
def distributed_train_epoch(trainset_dist):
    total_loss = 0.0
    num_batchs_per_worker = 0
    for (inp, tar) in trainset_dist:
        per_replica_losses = strategy.run(train_step, args=((inp, tar),))
        total_loss += strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
        num_batchs_per_worker += 1
    return [total_loss / tf.cast(num_batchs_per_worker, dtype=tf.float32), num_batchs_per_worker]



def print_translation(sentence, tokens, ground_truth):
    print(f'{"Input:":15s}: {sentence}')
    print(f'{"Prediction":15s}: { tokens.numpy().decode("utf-8") }')
    print(f'{"Ground truth":15s}: {ground_truth}')


###-----------------------------------------------------------------###
###                             TOKENIZERS                          ###
###-----------------------------------------------------------------###

tokenizer_fr = myTokenizer( TK_FR ) 
tokenizer_en = myTokenizer( TK_EN ) 

###-----------------------------------------------------------------###
###                             DATASET                             ###
###-----------------------------------------------------------------###
NB_TRAINING_SAMPLES = 10_000_000
assert (NB_TRAINING_SAMPLES<27_000_000), "Too many data..."

########################################################
###                     MAIN                         ###
########################################################

## ---->>>> Set the Transformer configuration
"""
- Models from origianl Transfo paper : https://arxiv.org/pdf/1706.03762.pdf : 
-- Big Model 
    num_layers = 6 | d_model = 1024 | dff = 4096 | num_heads = 16

-- Base Model 
    num_layers = 6 | d_model = 512 | dff = 2048 | num_heads = 8

- Others Models : 
-- Little Model : 
    num_layers = 4 | d_model = 128 | dff = 512 | num_heads = 8

"""

num_layers = 6
d_model = 1024
dff = 4096
num_heads = 16
dropout_rate = 0.1



## ---->>>> Prepare the distributed training
BATCH_SIZE_per_worker = 256
b_size = BATCH_SIZE_per_worker
VOCAB_EN_SIZE = tokenizer_en.get_vocab_size().numpy() #30_000
VOCAB_FR_SIZE = tokenizer_fr.get_vocab_size().numpy() #30_000
MODEL_NAME = "big_frTOen_noWE_part_20M"
FILE_LOG_NAME = "LOG_training_2.txt"

# Define optimizer parameters
b1, b2, epsilon = 0.9, 0.98, 1e-9


# Define the strategy : MirroredStrategy or MultiWorkerMirroredStrategy
strategy = tf.distribute.MirroredStrategy(  )
# if want a configuration with a special number of gpus
# EX for 2 gpus : strategy = tf.distribute.MirroredStrategy( [ '/gpu:0' , '/gpu:1' ] )
GLOBAL_BATCH_SIZE = BATCH_SIZE_per_worker * strategy.num_replicas_in_sync
tf.print( GLOBAL_BATCH_SIZE )


with strategy.scope():

    ## Dataset distribution
    # create the dataset
    trainset, _, _ = create_dataset_fr_en(
        nb_samples=NB_TRAINING_SAMPLES,
        batch_size_train=GLOBAL_BATCH_SIZE,
        nb_workers=strategy.num_replicas_in_sync
        )
    # conversion into a tf.distribute.DistributedDataset (can only iterate )
    # -> Three key actions happening under the hood of this method are batching, sharding, and prefetching.
    trainset_dist = strategy.experimental_distribute_dataset(trainset)

    ## Variables distribution (Model)
    transformer = Transformer(
        num_layers= num_layers,
        d_model=    d_model,
        num_heads=  num_heads,
        dff=        dff,
        input_vocab_size=VOCAB_FR_SIZE,
        target_vocab_size=VOCAB_EN_SIZE,
        rate=dropout_rate,
        name="fr_to_en_noWE",
        nb_training_sample=NB_TRAINING_SAMPLES
    )

    ## Optimizer distribution
    #create the lr
    #lr = d_model^(-0.5) * min( step_num^(-0.5), step_num*warmup_steps^(-1.5) )
    learning_rate = CustomSchedule( transformer.encoder.d_model )
    opt = tf.keras.optimizers.Adam(learning_rate, beta_1=b1, beta_2=b2, epsilon=epsilon)    

    ## Loss distribution
    #from_logits : True, because y_pred is a logit tensor and not a proba distribution (False)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, 
        reduction=tf.keras.losses.Reduction.NONE)
    train_loss =tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')


print(f"CONFIG : num_layers : {num_layers} | d_model : {d_model} | dff : {dff} | \
num_heads : {num_heads} | \
VOCAB_SIZE : {VOCAB_EN_SIZE} | \
N_SAMPLE : {NB_TRAINING_SAMPLES} | \
BATCH_SIZE : {b_size}")


## ---->>>> Training

EPOCHS = 20
TRAIN = True
LOSS_list, ACC_list = [], []


print("start training...")


if TRAIN : 

    f = open(FILE_LOG_NAME, "a")
    f.write(f"CONFIG : num_layers : {num_layers} | d_model : {d_model} | dff : {dff} | \
num_heads : {num_heads} | \
VOCAB_SIZE : {VOCAB_EN_SIZE} | \
N_SAMPLE : {NB_TRAINING_SAMPLES} | \
BATCH_SIZE : {b_size} \n ")
    f.close()
    
    start_time = time.time()
    epoch_start_time = start_time
    for epoch in range( EPOCHS ):
        train_accuracy.reset_states()
        train_loss.reset_states()

        loos_per_epoch, num_batchs_per_worker = distributed_train_epoch( trainset_dist)
        epoch_time = time.time() - epoch_start_time

        tf.print(f'[Epoch {epoch + 1}]:{num_batchs_per_worker} batchs/gpu  | time(s) per epoch:{epoch_time:.2f} | train_loss:{train_loss.result():.4f} | Accuracy:{train_accuracy.result():.4f}')
        epoch_start_time = time.time()
        
        LOSS_list.append( float( f"{train_loss.result():.4f}" ) )
        ACC_list.append( float( f"{train_accuracy.result():.4f}" ) )

        f = open(FILE_LOG_NAME, "a")
        f.write( f'[Epoch {epoch + 1}]:{num_batchs_per_worker} batchs/gpu | time(s) per epoch:{epoch_time:.2f} | train_loss:{train_loss.result():.4f} | Accuracy:{train_accuracy.result():.4f}\n' )
        f.close()
    
    train_time = time.time() - start_time
    print( f"TOTAL TRAINING TILE : {train_time}s -> {train_time/60}min -> {(train_time/60)/60}h " )

    f = open(FILE_LOG_NAME, "a")
    f.write( f"TOTAL TRAINING TILE : {train_time}s -> {train_time/60}min -> {(train_time/60)/60}h " )
    f.close()

    print( transformer.summary() )
    print("LOSS : ", LOSS_list )
    print("ACC :", ACC_list )
    transformer.save( MODEL_NAME )




"""
-->> Object : tf.distribute.DistributedDataset

tf.distribute.DistributedDataset instances are not of type tf.data.Dataset. 
It only supports two usages we will mention below: iteration and element_spec. 
We don't support any other APIs to transform or inspect the dataset.


-->> Method : strategy.experimental_distribute_dataset(Object Dataset) -> Object DistributedDataset


- Three key actions happening under the hood of this method are batching, sharding, and prefetching.

"""


"""
OOM stands for "out of memory"
Your GPU is running out of memory, so it can't allocate memory for this tensor. There are a few things you can do:

-Decrease the number of filters in your Dense, Conv2D layers
-Use a smaller batch_size (or increase steps_per_epoch and validation_steps)
-Use grayscale images (you can use tf.image.rgb_to_grayscale)
-Reduce the number of layers
-Use MaxPooling2D layers after convolutional layers
-Reduce the size of your images (you can use tf.image.resize for that)
-Use smaller float precision for your input, namely np.float32
-If you're using a pre-trained model, freeze the first layers (like this)


"""