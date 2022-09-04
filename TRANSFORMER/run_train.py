"""
@author : Arnaud HINCELIN
@date : 04/05/2022
"""

import sys
sys.path.append("../shared")

#CUDA_VISIBLE_DEVICES=<4>
#import preprocess_dataset as myPreprocessing
from preprocess_dataset import myTokenizer, create_dataset_fr_en, MAX_TOKENS_PER_SENTENCE
from transformer_model import Translator, Transformer
from utils import print_ram_used
import time
import tensorflow as tf

GPATH_EN_FR_TRANSLATION = "/cstor/SHARED/datasets/NLP/translation/en-fr.csv"
GPATH_EN_FR_TRANSLATION_PARA = "/cstor/SHARED/datasets/NLP/translation/"

TOKENIZERS_PATH = "../../TOKENIZATION/tokenizers_WIKI_fr_en"
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

def loss_function(real, pred):
    #pred : (batch_size, tar_seq_len, target_vocab_size)
    #real : (batch_size, tar_seq_len)
    
    ## 1 -> Cross loss between the label and the predictions
    #from_logits : True, because y_pred is a logit tensor and not a proba distribution (False)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
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
train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]
@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):

    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    with tf.GradientTape() as tape:
        predictions, _ ,_ = transformer([inp, tar_inp],
                                    training = True)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    opt.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy( accuracy_function(tar_real, predictions) )


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
NB_TRAINING_SAMPLES = 200_000
BATCH_SIZE_per_worker = 256
b_size = BATCH_SIZE_per_worker
GLOBAL_BATCH_SIZE = BATCH_SIZE_per_worker

trainset, valset, testset = create_dataset_fr_en(
        nb_samples=NB_TRAINING_SAMPLES,
        batch_size_train=GLOBAL_BATCH_SIZE,
        nb_workers=1)

########################################################
###                     MAIN                         ###
########################################################
#strategy = tf.distribute.MirroredStrategy()

## ---->>>> Set the Transformer configuration
"""
- Models from origianl Transfo paper : https://arxiv.org/pdf/1706.03762.pdf : 
-- Big Model 
    num_layers = 8 | d_model = 1024 | dff = 4096 | num_heads = 16

-- Base Model 
    num_layers = 6 | d_model = 512 | dff = 2048 | num_heads = 8

- Others Models : 
-- Little Model : 
    num_layers = 4 | d_model = 128 | dff = 512 | num_heads = 8

"""
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1



## ---->>>> Create the Transformer

VOCAB_EN_SIZE = tokenizer_en.get_vocab_size().numpy() #30_000
VOCAB_FR_SIZE = tokenizer_fr.get_vocab_size().numpy() #30_000

print_ram_used()

MODEL_NAME = "testmodel_frTOen_noWE"
transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=VOCAB_FR_SIZE,
    target_vocab_size=VOCAB_EN_SIZE,
    rate=dropout_rate,
    name="fr_to_en_noWE",
    nb_training_sample=NB_TRAINING_SAMPLES
    )
print(f"CONFIG : num_layers : {num_layers} | d_model : {d_model} | dff : {dff} | \
num_heads : {num_heads} | \
VOCAB_SIZE : {VOCAB_EN_SIZE} | \
N_SAMPLE : {NB_TRAINING_SAMPLES} | \
BATCH_SIZE : {b_size}")

print_ram_used()

## ---->>>> Prepare the training
EPOCHS = 20
TRAIN = True
LOSS_list, ACC_list = [], []

b1, b2, epsilon = 0.9, 0.98, 1e-9
#create the lr
#lr = d_model^(-0.5) * min( step_num^(-0.5), step_num*warmup_steps^(-1.5) )
learning_rate = CustomSchedule( transformer.encoder.d_model )
opt = tf.keras.optimizers.Adam(learning_rate, beta_1=b1, beta_2=b2, epsilon=epsilon)
train_loss =tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')


"""
for (batch, (inp, tar)) in enumerate( trainset.take(1) ) :
    
    tar_inp = tar[:, :-1]
    
    print( inp.shape ) #(256, len_seq_inp)
    print( tar.shape ) #(256, len_seq_tar)
    print( tar_inp.shape ) #(256, len_seq_tar-1)
    N_S = 10
    inp_one = inp[:N_S]
    tar_inp_one = tar_inp[:N_S]

    print( inp_one.shape ) 

    time_s = time.time()
    predictions, _ ,_ = transformer([inp, tar_inp], training = False)
    print( f"END of {inp.shape[0]} inferences containing {inp.shape[1]}\
    inputs tokens in {time.time() - time_s} s" )


    time_s = time.time()
    predictions, _ ,_ = transformer([ inp_one, tar_inp_one ], training = False)

    print( f"END of {inp_one.shape[0]} inferences containing { inp_one.shape[1]}\
        inputs tokens in {(time.time()-time_s):.4f} s" )

"""

## ---->>>> Training

if TRAIN : 

    start_train = time.time()
    print("start training...")

    f = open("LOG_training.txt", "a")
    f.write(f"CONFIG : num_layers : {num_layers} | d_model : {d_model} | dff : {dff} | \
num_heads : {num_heads} | \
VOCAB_SIZE : {VOCAB_EN_SIZE} | \
N_SAMPLE : {NB_TRAINING_SAMPLES} | \
BATCH_SIZE : {b_size} \n ")
    f.close()

    for epoch in range(EPOCHS):
        start = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()
        # inp -> fr, tar -> en
        for (batch, (inp, tar)) in enumerate( trainset ) :
            
            train_step(inp, tar)

            if batch % 100 == 0:
                current_min = int((time.time() - start)/60)
                loss, acc = train_loss.result(), train_accuracy.result()
                print(f"Time : {current_min}min - [Epoch]: {epoch+1} [batch] : {batch} - [LOSS] : {loss:.4f} [Accuracy] : {acc:.4f} - [DATA] : {batch*b_size}")
                
                f = open("LOG_training.txt", "a")
                f.write( f"Time : {current_min}min - [Epoch]: {epoch+1} [batch] : {batch} - [LOSS] : {loss:.4f} [Accuracy] : {acc:.4f} - [DATA] : {batch*b_size} \n")
                f.close()
        print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
        print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')
        
        LOSS_list.append( float( f"{train_loss.result():.4f}" ) )
        ACC_list.append( float( f"{train_accuracy.result():.4f}" ) )
    train_time = time.time() - start_train

    print( f"TOTAL TRAINING TILE : {train_time}s -> {train_time/60}min -> {(train_time/60)/60}h " )
    print( f"NB batch : {batch} | = {batch*b_size} samples" )
    print( transformer.summary() )

    print("LOSS : ", LOSS_list )
    print("ACC :", ACC_list )
    #transformer.save( MODEL_NAME )







