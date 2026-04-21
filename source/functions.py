# Imports
import os
import numpy as np
import tensorflow as tf
import keras
from keras import layers, optimizers, callbacks
from keras.layers import LeakyReLU



# DATA EXPLORATION FUNCTIONS
# DATA EXPLORATION FUNCTIONS

def summarize(name, data, unit=""):
    """
    Print basic descriptive statistics for a numeric dataset.
    Parameters:
        - name : Label used to identify the data being summarized. It will be displayed in the output header.
        - data : A sequence of numeric values (e.g., list, NumPy array) for which the statistics will be computed.
        - unit : Unit of measurement associated with the values (e.g., "px", "MB", etc.). The unit is appended to the printed statistics.
        Default is an empty string.
    Returns: The function prints the summary statistics to the console.
    """
    data = np.array(data)

    print(f"\n=== {name} ===")
    print(f"Min  : {data.min():.1f}{unit}")
    print(f"Max  : {data.max():.1f}{unit}")
    print(f"Mean : {data.mean():.1f}{unit}")
    print(f"Std  : {data.std():.1f}{unit}")


# MODELING FUNCTIONS

# DEF FUNCTION TO RESIZE IMAGES IN DATASET
def build_resized_ds(train_ds, val_ds, image_model_size, AUTOTUNE):
    """
    Resize images from the training and validation datasets to match the input size required by the model.
    Parameters:
        - train_ds : TensorFlow dataset containing the training images and labels.
        - val_ds : TensorFlow dataset containing the validation images and labels.
        - image_model_size : Target size (height and width) that images should be resized to. 
            The same value is used for both dimensions.
        - AUTOTUNE : TensorFlow constant used to automatically tune the number of parallel calls 
            during dataset mapping for better performance.
    Returns:
        - train_ds : Training dataset with resized images.
        - val_ds : Validation dataset with resized images.
    """
    def resize_fn(image, label):
        image = tf.image.resize(image, [image_model_size, image_model_size])
        return image, label

    train_ds = train_ds.map(resize_fn, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(resize_fn, num_parallel_calls=AUTOTUNE)

    return train_ds, val_ds


# DEF PREPROCESSING FUNCTION FOR DIFFERENT MODELS
def apply_preprocess_ds(train_resized, val_resized, preprocess_fn, AUTOTUNE, batch_size=32):
    """
    Apply model-specific preprocessing to training and validation datasets, and prepare them for training.
    Parameters:
        - train_resized : TensorFlow dataset containing resized training images and labels.
        - val_resized : TensorFlow dataset containing resized validation images and labels.
        - preprocess_fn : Preprocessing function associated with a specific pretrained model 
        (e.g., EfficientNet, ConvNeXt). This function adapts the input images to the format expected by the model.
        - AUTOTUNE : TensorFlow constant used to automatically tune the number of parallel calls and 
        prefetch buffer size for better pipeline performance.
        - batch_size : Number of samples per batch used during training. Default is 32.
    Returns:
        - t_ds : Preprocessed, batched, and prefetched training dataset.
        - v_ds : Preprocessed, batched, and prefetched validation dataset.
    """

    def apply_preprocess_img(image, label):
        image = preprocess_fn(image)
        return image, label

    t_ds = (train_resized
            .map(apply_preprocess_img, num_parallel_calls=AUTOTUNE)
            .batch(batch_size)
            .prefetch(AUTOTUNE))

    v_ds = (val_resized
            .map(apply_preprocess_img, num_parallel_calls=AUTOTUNE)
            .batch(batch_size)
            .prefetch(AUTOTUNE))

    return t_ds, v_ds


class SparseF1Score(keras.metrics.F1Score):
    """ Custom F1 Score metric
    This class extends the Keras F1Score metric to handle sparse integer labels by converting them to one-hot encoding before computing the F1 score. """
    def update_state(self, y_true, y_pred, sample_weight=None):
        """Override the update_state method to convert sparse integer labels to one-hot encoding before computing the F1 score.
        Parameters:
            - y_true: Tensor of true labels, expected to be in sparse integer format (e.g., class indices).
            - y_pred: Tensor of predicted probabilities or logits for each class.
            - sample_weight: Optional tensor of weights for each sample, used to weight the contribution of each sample to the overall metric. Default is None, which means all samples are equally weighted.
        Returns: The method updates the internal state of the metric with the converted one-hot labels and predictions, and does not return a value. The F1 score can be retrieved later using the result() method of the metric instance.
"""
        # converte inteiros → one-hot antes de passar ao F1Score
        y_true_onehot = tf.one_hot(tf.cast(y_true, tf.int32), depth=NUM_CLASSES)
        return super().update_state(y_true_onehot, y_pred, sample_weight)
    

# METRICS - USED IN MODEL EVALUATION AND COMPARISON
def make_metrics():
    """ Create a list of metrics to be used during model compilation and evaluation. The metrics include a custom SparseF1Score for macro-averaged F1 score and a SparseTopKCategoricalAccuracy for top-3 accuracy.
     Returns:
        A list of Keras metrics.
     """
    return [
        SparseF1Score(average='macro', name='macro_f1'),
        keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top3_accuracy'),
    ]




def build_base_model(backbone_name, backbone_configs, num_classes,
                     activation_name='relu', head_config=None):
    """
    Build a model with a pretrained backbone and a configurable classification head.
 
    Parameters:
    backbone_name : str
        Key into backbone_configs identifying the backbone.
    backbone_configs : dict
        Dict mapping backbone names to their configuration (model_fn, image_size, ...).
    num_classes : int
        Number of output classes.
    activation_name : str, default 'relu'
        Activation function used in the Dense layers of the head.
        Supports any Keras activation string, plus 'leaky_relu' (handled explicitly).
    head_config : dict, optional
        Head architecture configuration. Expected keys:
            - 'dense_units': list[int]   neurons per Dense layer, in order
            - 'dropout_rate': float      dropout rate applied after each Dense
        If None, defaults to {'dense_units': [256], 'dropout_rate': 0.3},
        which reproduces the original single-layer head with dropout only before
        the softmax (baseline behavior for backward compatibility).
 
    Returns:
    (model, backbone) : tuple
        model is the full keras.Model (backbone + head).
        backbone is the raw backbone (kept trainable=False by default).

    """
    backbone_config = backbone_configs.get(backbone_name)
    if not backbone_config:
        raise ValueError(f"Backbone '{backbone_name}' not found in configs.")
 
    backbone_fn = backbone_config['model_fn']
    IMG_SIZE    = backbone_config['image_size']
 
    # Default head: replicates original [256] + Dropout(0.3) before softmax
    if head_config is None:
        head_config = {'dense_units': [256], 'dropout_rate': 0.3}
 
    dense_units  = head_config['dense_units']
    dropout_rate = head_config['dropout_rate']
 
    bb = backbone_fn(include_top=False, weights='imagenet', input_shape=(*IMG_SIZE, 3))
    bb.trainable = False
 
    inp = keras.Input(shape=(*IMG_SIZE, 3))
    x   = bb(inp, training=False)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.BatchNormalization()(x)
 
    # Build the head: stack of Dense + (activation) + Dropout
    for units in dense_units:
        if activation_name == 'leaky_relu':
            x = layers.Dense(units)(x)
            x = layers.LeakyReLU(negative_slope=0.01)(x)
        else:
            x = layers.Dense(units, activation=activation_name)(x)
        x = layers.Dropout(dropout_rate)(x)
 
    out = layers.Dense(num_classes, activation='softmax')(x)
 
    return keras.Model(inp, out), bb








def run_phase1(model, backbone, train, val, backbone_name, phase1_config, phase2_config, make_metrics, class_weight_dict, optimizer_fn=None):
    """ Train the model with the backbone frozen.
    Parameters:
        - model: The Keras Model instance to be trained.
        - backbone: The backbone model instance whose layers are frozen during this phase.
        - train: The training dataset, prepared with augmentation and preprocessing.
        - val: The validation dataset, prepared with augmentation and preprocessing.
        - backbone_name: String identifier for the backbone architecture, used for naming callbacks and checkpoints.
        - phase1_config: Dictionary containing configuration parameters specific to phase 1, such as learning rate, number of epochs,
        early stopping patience, learning rate reduction factor, etc.
        - phase2_config: Dictionary containing configuration parameters specific to phase 2, which may be needed for callbacks.
        - make_metrics: A function that returns a list of metrics to be used during model compilation.
        - class_weight_dict: A dictionary mapping class indices to weights, used to address class imbalance during training.
        - optimizer_fn: Optional function that returns an optimizer instance. If None, a default Adam optimizer with a learning rate from phase1_config will be used.
    Returns:
        - hist: The training history object returned by model.fit(), containing details about the training process, including loss and metrics for each epoch.
        - model.get_weights(): The weights of the model after training in phase 1, which can be used to initialize the model for phase 2 fine-tuning.
    """
    if optimizer_fn is None:
        optimizer_fn = lambda: optimizers.Adam(learning_rate=phase1_config['PHASE1_LR'])  # default

    backbone.trainable = False
    model.compile(
        optimizer=optimizer_fn(),
        loss='sparse_categorical_crossentropy',
        metrics=make_metrics()
    )
    hist = model.fit(
        train,
        epochs=phase1_config['PHASE1_EPOCHS'],
        validation_data=val,
        class_weight=class_weight_dict,
        callbacks=make_callbacks(backbone_name, phase1_config=phase1_config, phase2_config=phase2_config, phase=1),
        verbose=1
    )
    return hist, model.get_weights()


def run_phase2(model, backbone, train, val, phase1_weights, backbone_name, n_unfreeze, phase1_config, phase2_config, make_metrics, class_weight_dict):
    """Unfreeze the last n layers and do fine tuning.
    Parameters:
        - model: The Keras Model instance to be fine-tuned.
        - backbone: The backbone model instance whose last n layers will be unfrozen for fine-tuning.
        - train: The training dataset, prepared with augmentation and preprocessing.
        - val: The validation dataset, prepared with augmentation and preprocessing.
        - phase1_weights: The weights of the model obtained after training in phase 1, used to initialize the model before fine-tuning.
        - backbone_name: String identifier for the backbone architecture, used for naming callbacks and checkpoints.
        - n_unfreeze: Integer specifying the number of layers from the end of the backbone to unfreeze for fine-tuning. The last n layers will be set to trainable, while the rest will remain frozen.
        - phase1_config: Dictionary containing configuration parameters specific to phase 1, which may be needed for callbacks.
        - phase2_config: Dictionary containing configuration parameters specific to phase 2, such as learning rate, number of epochs, early stopping patience, learning rate reduction factor, etc.
        - make_metrics: A function that returns a list of metrics to be used during model compilation.
        - class_weight_dict: A dictionary mapping class indices to weights, used to address class imbalance during training.
    Returns:
        - hist: The training history object returned by model.fit(), containing details about the fine-tuning process, including loss and metrics for each epoch.
    """
    model.set_weights(phase1_weights)
    backbone.trainable = True
    for layer in backbone.layers[:-n_unfreeze]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=phase2_config['PHASE2_LR']),
        loss='sparse_categorical_crossentropy',
        metrics=make_metrics()
    )
    hist = model.fit(
        train,
        epochs=phase2_config['PHASE2_EPOCHS'],
        validation_data=val,
        class_weight=class_weight_dict,
        callbacks=make_callbacks(f"{backbone_name}_unfreeze{n_unfreeze}", phase1_config=phase1_config, phase2_config=phase2_config, phase=2),
        verbose=1
    )
    return hist


def make_callbacks(name, phase, phase1_config, phase2_config):
    """Generate a list of Keras callbacks for training, including EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint. The parameters for these callbacks are determined based on the current phase (1 or 2) and the corresponding configuration dictionaries.
    Parameters:
        - name: String used to identify the model and phase in the checkpoint filenames.
        - phase: Integer (1 or 2) indicating the current training phase, which determines which configuration parameters to use for the callbacks.
        - phase1_config: Dictionary containing configuration parameters specific to phase 1, such as early stopping patience, learning rate reduction factor, etc.
        - phase2_config: Dictionary containing configuration parameters specific to phase 2, which may be needed for callbacks during fine-tuning.
    Returns:
        - A list of Keras callback instances configured according to the specified phase and parameters.
    """
    es_patience  = phase1_config['ES_PATIENCE_P1']  if phase == 1 else phase2_config['ES_PATIENCE_P2']
    lr_factor    = phase1_config['LR_FACTOR_P1']    if phase == 1 else phase2_config['LR_FACTOR_P2']
    lr_patience  = phase1_config['LR_PATIENCE_P1']  if phase == 1 else phase2_config['LR_PATIENCE_P2']
    lr_min       = phase1_config['LR_MIN_P1']       if phase == 1 else phase2_config['LR_MIN_P2']

    # the strategy will be to use val loss to trigger early stopping, learn rate reeduction and model checkpointing,
    # so for each backbone the best model will be the one with the lowest val loss but when comparing backbones we will look at val F1 and top-3 accuracy
    # as the main comparison metrics, since they are more informative about the model's performance on imbalanced multi-class classification than val loss alone.

    return [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=es_patience,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=lr_factor,
            patience=lr_patience,
            min_lr=lr_min,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=os.path.join("checkpoints", f"{name}_p{phase}_best.keras"),
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        ),
    ]


def extract_best_metrics(hist, prefix):
    """Extract the best validation metrics from the training history based on the epoch with the lowest validation loss. 
    Parameters:
        - hist: The training history object returned by model.fit(), which contains the loss and metrics for each epoch.
        - prefix: A string prefix used to label the metrics in the returned dictionary (e.g., 'p1' for phase 1 or 'p2' for phase 2).
    Returns:
        - A dictionary containing the best validation F1 score, training F1 score at the best epoch, validation loss 
        at the best epoch, validation top-3 accuracy (if available), and an overfitting measure.
    """
    best_epoch = np.argmin(hist.history['val_loss'])
    return {
        f'{prefix}_val_f1':   hist.history['val_macro_f1'][best_epoch],
        f'{prefix}_train_f1': hist.history['macro_f1'][best_epoch],
        f'{prefix}_val_loss': hist.history['val_loss'][best_epoch],
        f'{prefix}_val_top3': hist.history.get('val_top3_accuracy', [None])[best_epoch],
        f'{prefix}_overfit':  hist.history['macro_f1'][best_epoch] - hist.history['val_macro_f1'][best_epoch],
        f'{prefix}_history':  hist.history,
    }



# FUNCTIONS FOR AUGMENTATION + PREPROCESSING

def apply_augmented_preprocess_ds(train_raw, val_raw, augmentation_model, preprocess_fn, image_size, AUTOTUNE, batch_size=32, seed=42):
    """Apply data augmentation and model-specific preprocessing to the training and validation datasets, and prepare them for training.
    Parameters:
        - train_raw: TensorFlow dataset containing raw training images and labels.
        - val_raw: TensorFlow dataset containing raw validation images and labels.
        - augmentation_model: A Keras model or function that applies data augmentation transformations to the input 
        images. This model should be designed to perform augmentations during training.
        - preprocess_fn: Preprocessing function associated with a specific pretrained model (e.g., EfficientNet, 
        ConvNeXt). This function adapts the input images to the format expected by the model.
        - AUTOTUNE: TensorFlow constant used to automatically tune the number of parallel calls and prefetch buffer 
        size for better pipeline performance.
        - batch_size: Number of samples per batch used during training. Default is 32.
        - seed: Integer seed for random operations to ensure reproducibility of the data augmentation. Default is 42.
    Returns:
        - t_ds: The training dataset with augmented and preprocessed images, batched and prefetched for performance.
        - v_ds: The validation dataset with preprocessed images (without augmentation), batched and prefetched for performance.
    """
    def train_map(image, label):
        image = tf.cast(image, tf.float32)
        image = augmentation_model(image, training=True)
        image = preprocess_fn(image)
        return image, label

    def val_map(image, label):
        image = tf.cast(image, tf.float32)
        image = preprocess_fn(image)
        return image, label

    t_aug = (train_raw
            .shuffle(10000, seed=seed, reshuffle_each_iteration=True)
            .map(train_map, num_parallel_calls=AUTOTUNE)
            .batch(batch_size)
            .prefetch(AUTOTUNE))
    
    v_aug = (val_raw
            .map(val_map, num_parallel_calls=AUTOTUNE)
            .batch(batch_size)
            .prefetch(AUTOTUNE))
    
    t_resized, val_resized = build_resized_ds(t_aug, v_aug, image_size, AUTOTUNE)


    return t_resized, val_resized




def run_augmentation_experiment(train_raw, val_raw, backbone_name, cfg, backbone_configs, aug_name, augmentation_model, n_unfreeze, num_classes, AUTOTUNE, batch_size, seed, class_weight_dict, make_metrics, phase1_config, phase2_config):
    """ Run a complete training experiment using a specified backbone architecture, data augmentation strategy, and number of layers to unfreeze for fine-tuning. 
    Parameters:
        - train_raw: The raw training dataset.
        - val_raw: The raw validation dataset.
        - backbone_name: String identifier for the backbone architecture to use (e.g., 'EfficientNetB0', 'ConvNeXtTiny').
        - cfg: A configuration dictionary containing necessary parameters and functions for building the model, preprocessing the data, 
        and training (e.g., datasets, preprocessing functions, etc.).
        - aug_name: String identifier for the augmentation strategy being applied, used for labeling results and checkpoints.
        - augmentation_model: A Keras model or function that applies data augmentation transformations to the input images during training.
        - n_unfreeze: Integer specifying the number of layers from the end of the backbone to unfreeze for fine-tuning in phase 2. 
        The last n layers will be set to trainable, while the rest will remain frozen.
    Returns:
        - results: A dictionary containing the backbone name, augmentation name, number of layers unfrozen, and the best validation F1 score,
        training F1 score at the best epoch, validation loss at the best epoch, validation top-3 accuracy (if available), and overfitting measure for both phase 1 and phase 2 of training.
    """
    print(f'\n{"="*70}')
    print(f'Backbone     : {backbone_name}')
    print(f'Augmentation : {aug_name}')
    print(f'Unfreeze     : {n_unfreeze}')
    print(f'{"="*70}')

    model, backbone = build_base_model(
        backbone_name=backbone_name,
        backbone_configs=backbone_configs,
        num_classes=num_classes,
        activation_name='swish'
    )

    train, val = apply_augmented_preprocess_ds(
        train_raw=train_raw,
        val_raw=val_raw,
        augmentation_model=augmentation_model,
        preprocess_fn=cfg['preprocess'],
        image_size=cfg['image_size'][0],
        AUTOTUNE=AUTOTUNE,
        batch_size=batch_size,
        seed=seed
    )

    hist1, phase1_weights = run_phase1(
        model=model,
        backbone=backbone,
        train=train,
        val=val,
        backbone_name=f'{backbone_name}_{aug_name}',
        phase1_config=phase1_config,
        phase2_config=phase2_config,
        make_metrics=make_metrics,
        class_weight_dict=class_weight_dict
    )

    hist2 = run_phase2(
        model=model,
        backbone=backbone,
        train=train,
        val=val,
        phase1_weights=phase1_weights,
        backbone_name=f'{backbone_name}_{aug_name}',
        n_unfreeze=n_unfreeze,
        phase1_config=phase1_config,
        phase2_config=phase2_config,
        make_metrics=make_metrics,
        class_weight_dict=class_weight_dict
    )

    results = {
        'backbone': backbone_name,
        'augmentation': aug_name,
        'n_unfreeze': n_unfreeze,
        **extract_best_metrics(hist1, 'p1'),
        **extract_best_metrics(hist2, 'p2')
    }

    return results