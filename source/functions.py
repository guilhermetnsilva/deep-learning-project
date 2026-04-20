# Imports
import os
import numpy as np
import tensorflow as tf
import keras
from keras import layers, optimizers, callbacks
from keras.layers import LeakyReLU



# DATA EXPLORATION FUNCTIONS

def summarize(name, data, unit=""):
    data = np.array(data)
    print(f"\n=== {name} ===")
    print(f"Min  : {data.min():.1f}{unit}")
    print(f"Max  : {data.max():.1f}{unit}")
    print(f"Mean : {data.mean():.1f}{unit}")
    print(f"Std  : {data.std():.1f}{unit}")


# MODELING FUNCTIONS


# DEF FUNCTION TO RESIZE IMAGES IN DATASET
def build_resized_ds(train_ds, val_ds, image_model_size, AUTOTUNE):
    def resize_fn(image, label):
        image = tf.image.resize(image, [image_model_size, image_model_size])
        return image, label

    train_ds = train_ds.map(resize_fn, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(resize_fn, num_parallel_calls=AUTOTUNE)

    return train_ds, val_ds

# DEF PREPROCESSING FUNCTION FOR DIFFERENT MODELS
def apply_preprocess_ds(train_resized, val_resized, preprocess_fn, AUTOTUNE, batch_size=32):
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









    





def build_base_model(backbone_name, backbone_configs, num_classes, activation_name='relu'):  
    backbone_config = backbone_configs.get(backbone_name)
    if not backbone_config:
        raise ValueError(f"Backbone '{backbone_name}' not found in configs.")

    backbone_fn = backbone_config['model_fn']
    IMG_SIZE    = backbone_config['image_size']

    bb = backbone_fn(include_top=False, weights='imagenet', input_shape=(*IMG_SIZE, 3))
    bb.trainable = False

    inp = keras.Input(shape=(*IMG_SIZE, 3))
    x   = bb(inp, training=False)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.BatchNormalization()(x)
    if activation_name == 'leaky_relu':
        x = layers.Dense(256)(x)
        x = layers.LeakyReLU(negative_slope=0.01)(x)
    else:
        x = layers.Dense(256, activation=activation_name)(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)

    return keras.Model(inp, out), bb








def run_phase1(model, backbone, train, val, backbone_name, phase1_config, phase2_config, make_metrics, class_weight_dict, optimizer_fn=None):
    """Train frozen backbones"""
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
    """Unfreeze the last n layers and do fine tuning"""
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
    """Gera callbacks para uma dada fase e nome de experiência."""
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
    """Extrai métricas na melhor epoch (menor val_loss). Prefixo: 'p1' ou 'p2'."""
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

def apply_augmented_preprocess_ds(train_resized, val_resized, augmentation_model, preprocess_fn, AUTOTUNE, batch_size=32, seed=42):
    def train_map(image, label):
        image = tf.cast(image, tf.float32)
        image = augmentation_model(image, training=True)
        image = preprocess_fn(image)
        return image, label

    def val_map(image, label):
        image = tf.cast(image, tf.float32)
        image = preprocess_fn(image)
        return image, label

    t_ds = (train_resized
            .shuffle(10000, seed=seed, reshuffle_each_iteration=True)
            .map(train_map, num_parallel_calls=AUTOTUNE)
            .batch(batch_size)
            .prefetch(AUTOTUNE))

    v_ds = (val_resized
            .map(val_map, num_parallel_calls=AUTOTUNE)
            .batch(batch_size)
            .prefetch(AUTOTUNE))

    return t_ds, v_ds




def run_augmentation_experiment(backbone_name, cfg, aug_name, augmentation_model, n_unfreeze):
    print(f'\n{"="*70}')
    print(f'Backbone     : {backbone_name}')
    print(f'Augmentation : {aug_name}')
    print(f'Unfreeze     : {n_unfreeze}')
    print(f'{"="*70}')

    model, backbone = build_base_model(
        backbone_name=backbone_name,
        backbone_configs=AUG_BACKBONE_CONFIGS,
        num_classes=NUM_CLASSES,
        activation_name='swish'
    )

    train, val = apply_augmented_preprocess_ds(
        train_resized=cfg['train_ds'],
        val_resized=cfg['val_ds'],
        augmentation_model=augmentation_model,
        preprocess_fn=cfg['preprocess'],
        AUTOTUNE=AUTOTUNE,
        batch_size=BATCH_SIZE,
        seed=SEED
    )

    hist1, phase1_weights = run_phase1(
        model=model,
        backbone=backbone,
        train=train,
        val=val,
        backbone_name=f'{backbone_name}_{aug_name}',
        phase1_config=PHASE1_CONFIG,
        phase2_config=PHASE2_CONFIG,
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
        phase1_config=PHASE1_CONFIG,
        phase2_config=PHASE2_CONFIG,
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