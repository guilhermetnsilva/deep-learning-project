# Imports
import os
import numpy as np
import tensorflow as tf
import keras
from keras import layers, optimizers, callbacks
from keras.layers import LeakyReLU
import keras_tuner as kt



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





def build_tuner_model(hp, backbone_name, activation_name):
    """
    Hypermodel builder for Keras Tuner — wraps build_base_model with a tunable head.

    Defines the hyperparameter search space and constructs a compiled model
    for each trial. Called internally by the Hyperband tuner for every trial.

    Parameters
    ----------
    hp : keras_tuner.HyperParameters
        Hyperparameter object provided by the tuner. Defines the following
        search space:
            - dense_units_1 : int in {256, 512, 768}
                Number of units in the first Dense layer.
            - dense_units_2 : int in {128, 256, 512}
                Number of units in the second Dense layer (pyramidal structure).
            - dropout : float in [0.2, 0.5]
                Dropout rate applied after each Dense layer.
            - lr_p1 : float in [1e-4, 1e-2] (log scale)
                Learning rate for Phase 1 (frozen backbone).
    backbone_name : str
        Key into BACKBONE_CONFIGS identifying the backbone to use.
    activation_name : str
        Activation function for the Dense layers (e.g. 'relu', 'swish').

    Returns
    -------
    model : keras.Model
        Compiled model ready for tuner.search().
        Note: backbone is not returned as Keras Tuner expects a single model.
    """
    dense_units_1 = hp.Choice('dense_units_1', values=[256, 512, 768])
    dense_units_2 = hp.Choice('dense_units_2', values=[128, 256, 512])
    dropout       = hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)
    lr_p1         = hp.Float('lr_p1', min_value=1e-4, max_value=1e-2, sampling='log')

    model, _ = build_base_model(
        backbone_name=backbone_name,
        backbone_configs=BACKBONE_CONFIGS,
        num_classes=NUM_CLASSES,
        activation_name=activation_name,
        head_config={
            'dense_units':  [dense_units_1, dense_units_2],
            'dropout_rate': dropout
        }
    )

    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr_p1),
        loss='sparse_categorical_crossentropy',
        metrics=make_metrics()
    )
    return model




# MAIN FUNCTION TO RUN FULL HYPERBAND PIPELINE FOR A GIVEN BACKBONE
def run_hyperband(backbone_name, activation_name, train, val, n_unfreeze,
                  max_epochs=20, factor=3):
    """
    Run a full two-phase training pipeline with Hyperband hyperparameter search.

    Phase 1 uses Hyperband to search over the head architecture and learning
    rate with a frozen backbone. The best hyperparameters are then used to
    retrain Phase 1 in full, followed by Phase 2 fine-tuning with the last
    n_unfreeze backbone layers unfrozen. The Phase 2 learning rate is set
    as lr_p1 / 10 following standard fine-tuning practice.

    Parameters
    ----------
    backbone_name : str
        Key into BACKBONE_CONFIGS identifying the backbone to use.
    activation_name : str
        Activation function for the Dense layers (e.g. 'relu', 'swish').
    train : tf.data.Dataset
        Preprocessed and batched training dataset.
    val : tf.data.Dataset
        Preprocessed and batched validation dataset.
    n_unfreeze : int
        Number of backbone layers to unfreeze during Phase 2 fine-tuning.
    max_epochs : int, default 20
        Maximum number of epochs any trial can receive during Hyperband search.
        Trials eliminated early will receive fewer epochs.
    factor : int, default 3
        Hyperband reduction factor. At each round, the bottom (1 - 1/factor)
        fraction of trials is eliminated. Higher values are more aggressive.

    Returns
    -------
    best_hp : keras_tuner.HyperParameters
        Best hyperparameter configuration found by the tuner.
    results : dict
        Metrics at the best epoch for both phases, as returned by
        extract_best_metrics() with prefixes 'p1' and 'p2'.
    """
    print(f'\n{"="*60}')
    print(f'  Hyperband Phase 1 — {backbone_name}')
    print(f'{"="*60}')

    tuner = kt.Hyperband(
        hypermodel=lambda hp: build_tuner_model(hp, backbone_name, activation_name),
        objective=kt.Objective('val_macro_f1', direction='max'),
        max_epochs=max_epochs,
        factor=factor,
        directory='hyperband',
        project_name=f'{backbone_name}_p1',
        overwrite=True
    )

    tuner.search(
        train,
        epochs=max_epochs,
        validation_data=val,
        class_weight=class_weight_dict,
        callbacks=make_callbacks(f'{backbone_name}_tuner_p1', phase=1),
        verbose=1
    )

    best_hp = tuner.get_best_hyperparameters(1)[0]
    print(f'\n  Best HPs for {backbone_name}:')
    print(f'  dense_units_1 : {best_hp.get("dense_units_1")}')
    print(f'  dense_units_2 : {best_hp.get("dense_units_2")}')
    print(f'  dropout       : {best_hp.get("dropout")}')
    print(f'  lr_p1         : {best_hp.get("lr_p1"):.6f}')

    best_model, best_bb = build_base_model(
        backbone_name=backbone_name,
        backbone_configs=BACKBONE_CONFIGS,
        num_classes=NUM_CLASSES,
        activation_name=activation_name,
        head_config={
            'dense_units':  [best_hp.get('dense_units_1'), best_hp.get('dense_units_2')],
            'dropout_rate': best_hp.get('dropout')
        }
    )

    hist_p1, phase1_weights = run_phase1(
        best_model, best_bb, train, val,
        backbone_name=f'{backbone_name}_best',
        optimizer_fn=lambda: optimizers.Adam(learning_rate=best_hp.get('lr_p1'))
    )

    print(f'\n{"="*60}')
    print(f'  Phase 2 — {backbone_name} — unfreeze {n_unfreeze}')
    print(f'{"="*60}')

    lr_p2   = best_hp.get('lr_p1') / 10
    hist_p2 = run_phase2(
        best_model, best_bb, train, val,
        phase1_weights=phase1_weights,
        backbone_name=f'{backbone_name}_best',
        n_unfreeze=n_unfreeze,
        optimizer_fn=lambda: optimizers.Adam(learning_rate=lr_p2)
    )

    results = {
        **extract_best_metrics(hist_p1, prefix='p1'),
        **extract_best_metrics(hist_p2, prefix='p2'),
    }

    r = results
    print(f'\n  {backbone_name} — unfreeze {n_unfreeze}')
    print(f'  P1 Val F1 : {r["p1_val_f1"]:.4f}')
    print(f'  P2 Val F1 : {r["p2_val_f1"]:.4f}')
    print(f'  P2 Top-3  : {r["p2_val_top3"]:.4f}')
    print(f'  Overfit   : {r["p2_overfit"]:.4f}')

    return best_hp, results