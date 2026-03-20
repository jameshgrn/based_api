import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from data_format import generate_data

optuna.logging.set_verbosity(optuna.logging.WARNING)

FEATURES = ['log_Q', 'log_w', 'log_S']
MODEL_PATH = 'based_model.ubj'
VALIDATION_IMG = 'img/BASED_validation.png'

# PyPI xgboost wheels don't ship the Metal backend; hist+cpu uses all cores natively
DEVICE = 'cpu'


def add_log_features(df):
    df = df.copy()
    df['log_Q'] = np.log10(df['discharge'])
    df['log_w'] = np.log10(df['width'])
    df['log_S'] = np.log10(df['slope'])
    df['log_h'] = np.log10(df['depth'])
    return df


def smape(y_true, y_pred):
    return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))


def optuna_objective(trial, X_train, y_train):
    params = {
        'verbosity': 0,
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'device': DEVICE,
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'min_child_weight': trial.suggest_int('min_child_weight', 3, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.5, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 10.0, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 0.85),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.85),
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in kf.split(X_train):
        dtrain = xgb.DMatrix(X_train.iloc[train_idx], label=y_train.iloc[train_idx])
        dval = xgb.DMatrix(X_train.iloc[val_idx], label=y_train.iloc[val_idx])
        model = xgb.train(
            params, dtrain,
            num_boost_round=1500,
            evals=[(dval, 'val')],
            early_stopping_rounds=30,
            verbose_eval=False,
        )
        scores.append(smape(y_train.iloc[val_idx].values, model.predict(dval)))
    return np.mean(scores)


def main():
    generate_data()

    df = pd.read_csv('data/based_input_data_clean.csv')
    df = df[~df['source'].str.contains('Trampush', case=False, na=False)]
    df = add_log_features(df)

    X, y = df[FEATURES], df['log_h']

    # Stratify split by depth quantile so all depth ranges are represented
    depth_bins = pd.qcut(df['depth'], q=5, labels=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=depth_bins
    )
    print(f"Device: {DEVICE}  |  Train: {len(X_train)}  Test: {len(X_test)}")

    # Hyperparameter search
    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: optuna_objective(trial, X_train, y_train),
        n_trials=100,
        show_progress_bar=True,
    )
    best_params = {
        **study.best_params,
        'verbosity': 0,
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'device': DEVICE,
    }
    print(f"Best params: {best_params}")

    # Find optimal number of rounds via CV with early stopping
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    cv_results = xgb.cv(
        best_params, dtrain,
        num_boost_round=2000,
        nfold=5,
        metrics=['mae'],
        early_stopping_rounds=30,
        seed=42,
    )
    n_best = int(cv_results['test-mae-mean'].idxmin()) + 1
    print(f"Optimal rounds: {n_best}")

    # Train final model on full training set
    final_model = xgb.train(best_params, dtrain, num_boost_round=n_best)
    final_model.save_model(MODEL_PATH)
    print(f"Saved model → {MODEL_PATH}")

    # Evaluate in original (non-log) space
    log_pred = final_model.predict(dtest)
    pred = 10 ** log_pred
    true = 10 ** y_test.values

    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)
    mape = np.mean(np.abs((true - pred) / true)) * 100

    print(f"\nTest set metrics (original space):")
    print(f"  MAE:  {mae:.3f} m")
    print(f"  RMSE: {rmse:.3f} m")
    print(f"  R2:   {r2:.4f}")
    print(f"  MAPE: {mape:.1f}%")

    # Validation plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(true, pred, color='#FFCCBC', edgecolor='k', s=60, alpha=0.7)
    lim = [min(true.min(), pred.min()) * 0.8, max(true.max(), pred.max()) * 1.2]
    ax.plot(lim, lim, 'k--', lw=2, label='1:1')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Measured Channel Depth (m)')
    ax.set_ylabel('Predicted Channel Depth (m)')
    ax.set_title(f'BASED Validation | n = {len(y_test)}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(VALIDATION_IMG, dpi=250)
    print(f"Saved validation plot → {VALIDATION_IMG}")


if __name__ == '__main__':
    main()
