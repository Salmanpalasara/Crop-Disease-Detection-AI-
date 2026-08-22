from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    CSVLogger
)

from config import (
    CHECKPOINT_PATH,
    HISTORY_PATH,
    PATIENCE,
    REDUCE_LR_PATIENCE,
    MIN_LR
)


def get_callbacks():

    callbacks = [

        EarlyStopping(

            monitor="val_loss",

            patience=PATIENCE,

            restore_best_weights=True,

            verbose=1

        ),

        ReduceLROnPlateau(

            monitor="val_loss",

            factor=0.2,

            patience=REDUCE_LR_PATIENCE,

            min_lr=MIN_LR,

            verbose=1

        ),

        ModelCheckpoint(
    filepath=CHECKPOINT_PATH,
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    save_weights_only=False,
    verbose=1
),   

        CSVLogger(

            HISTORY_PATH,

            append=False

        )

    ]

    return callbacks