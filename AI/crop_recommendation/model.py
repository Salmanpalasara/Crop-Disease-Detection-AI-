from xgboost import XGBClassifier


# ==========================================================
# BUILD XGBOOST MODEL
# ==========================================================

def build_model(
    num_classes
):

    model = XGBClassifier(

        n_estimators=300,

        max_depth=6,

        learning_rate=0.05,

        subsample=0.9,

        colsample_bytree=0.9,

        objective="multi:softprob",

        num_class=num_classes,

        eval_metric="mlogloss",

        random_state=42,

        n_jobs=-1

    )


    return model