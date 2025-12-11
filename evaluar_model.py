import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from gestor_descriptor import *
from prediccion import *


def evaluar_modelo(model, kmeans, X_test, y_test, clases, nombre_csv="resultado_modelo.csv"):
    """
    Evalúa un modelo concreto y guarda todas las métricas en un CSV.
    """

    print("\n===== EVALUANDO MODELO =====")

    # -------- PREDICCIONES --------
    y_pred = model.predict(X_test)

    # -------- PROBABILIDADES --------
    y_prob = model.predict_proba(X_test)

    # -------- MÉTRICAS GENERALES --------
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf = confusion_matrix(y_test, y_pred, labels=clases)

    fila = {
        "accuracy": acc,
        "precision_macro": report["macro avg"]["precision"],
        "recall_macro": report["macro avg"]["recall"],
        "f1_macro": report["macro avg"]["f1-score"]
    }

    # -------- MÉTRICAS POR CLASE --------
    for cls in clases:
        fila[f"precision_{cls}"] = report[str(cls)]["precision"]
        fila[f"recall_{cls}"] = report[str(cls)]["recall"]
        fila[f"f1_{cls}"] = report[str(cls)]["f1-score"]

    # -------- MATRIZ DE CONFUSIÓN --------
    for i, cls_i in enumerate(clases):
        for j, cls_j in enumerate(clases):
            fila[f"conf_{cls_i}_to_{cls_j}"] = conf[i, j]

    # -------- PROBABILIDADES POR CLASE --------
    for idx, cls in enumerate(clases):
        fila[f"prob_media_{cls}"] = float(np.mean(y_prob[:, idx]))
        fila[f"pred_ratio_{cls}"] = float(np.mean(y_pred == cls))

    # -------- GUARDAR CSV --------
    df = pd.DataFrame([fila])
    df.to_csv(nombre_csv, index=False)

    print(f"✓ CSV guardado como {nombre_csv}")
    print(f"Accuracy: {acc:.4f}")

    return df

if __name__ == "__main__":

    train_root = "../descriptores_train"
    test_root = "../descriptores_test"

    train = GestorDescriptor()
    test = GestorDescriptor()

    train.inicializar(train_root, existe=True)
    test.inicializar(test_root, existe=True)

    train.set_norm("root sift")
    test.set_norm("root sift")

    # Cargas tu modelo y tu kmeans
    kmeans = joblib.load("kmeans_mejor.pkl")
    model = joblib.load("svm_mejor.pkl")

    # Preparar datos de test
    X_test, y_test = procesar_datos(test, kmeans, len(kmeans.cluster_centers_))
    X_test = power_norm(X_test)
    clases = np.unique(y_test)

    # Evaluación
    df_result = evaluar_modelo(model, kmeans, X_test, y_test, clases,
                               nombre_csv="evaluacion_modelo_unico.csv")