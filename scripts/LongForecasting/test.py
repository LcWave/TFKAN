import optuna

def objective(trial):
    return trial.suggest_float("x", -5, 5)**2

study = optuna.create_study()
study.optimize(objective, n_trials=100)

print("Best trial:", study.best_trial.params)
