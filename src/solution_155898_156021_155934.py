
import os
import random
import re
import time
import warnings

import config
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


warnings.filterwarnings("ignore")

RANDOM_STATE = 42
MODEL_NAME = "microsoft/codebert-base"
MAX_LENGTH = 512

MAX_TRAIN_ROWS = None # None oznacza, że nie ograniczamy liczby wierszy i używamy całego zbioru treningowego. 
MAX_VAL_ROWS = None # None oznacza, że nie ograniczamy liczby wierszy i używamy całego zbioru walidacyjnego.
MAX_TEST_ROWS = None # None oznacza, że nie ograniczamy liczby wierszy i używamy całego zbioru testowego.

NUM_TRAIN_EPOCHS = 1
LEARNING_RATE = 2e-5
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2
OUTPUT_DIR = "better_codebert"


def require_dependencies(): 
    # Funkcja sprawdza, czy wymagane biblioteki są zainstalowane. Jeśli nie, wyświetla komunikat o błędzie i instrukcje instalacji.
    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise SystemExit(
            "Brakuje bibliotek dla CodeBERT.\n"
            "Zainstaluj: pip install torch transformers datasets accelerate\n"
            f"Szczegoly: {exc}"
        )

    return {
        "torch": torch,
        "Dataset": Dataset,
        "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
        "AutoTokenizer": AutoTokenizer,
        "DataCollatorWithPadding": DataCollatorWithPadding,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
        "set_seed": set_seed,
    }


deps = require_dependencies()
torch = deps["torch"]
Dataset = deps["Dataset"]
AutoModelForSequenceClassification = deps["AutoModelForSequenceClassification"]
AutoTokenizer = deps["AutoTokenizer"]
DataCollatorWithPadding = deps["DataCollatorWithPadding"]
Trainer = deps["Trainer"]
TrainingArguments = deps["TrainingArguments"]
set_seed = deps["set_seed"]


def prepare(df):
    # Przygotowuje dane do trenowania modelu. 
    # Uzupełnia brakujące wartości w kolumnie "code" pustym stringiem, 
    # konwertuje kolumnę "label" na typ int, i zwraca tylko te dwie kolumny.
    df = df.copy()
    df["code"] = df["code"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    return df[["code", "label"]]


def stratified_limit(df, max_rows, name):
    # Ogranicza liczbę wierszy w DataFrame do max_rows, zachowując proporcje klas (stratyfikacja).
    if max_rows is None or len(df) <= max_rows:
        return df.reset_index(drop=True)

    sampled, _ = train_test_split(
        df,
        train_size=max_rows,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )
    sampled = sampled.reset_index(drop=True)
    print(f"{name}: ograniczono z {len(df)} do {len(sampled)} wierszy")
    return sampled


def load_data():
    # Ładuje dane z plików Parquet, przygotowuje je i opcjonalnie ogranicza 
    # liczbę wierszy dla każdego zbioru (treningowego, walidacyjnego, testowego).
    train = prepare(pd.read_parquet(config.traindatapath))
    validation = prepare(pd.read_parquet(config.valdatapath))
    test = prepare(pd.read_parquet(config.testdatapath))

    train = stratified_limit(train, MAX_TRAIN_ROWS, "train")
    validation = stratified_limit(validation, MAX_VAL_ROWS, "val")
    test = stratified_limit(test, MAX_TEST_ROWS, "test")

    return train, validation, test


def class_weights(labels):
    # Oblicza wagi klas na podstawie liczby przykładów w każdej klasie. 
    # Wagi są odwrotnie proporcjonalne do liczby przykładów, 
    # co pomaga modelowi lepiej radzić sobie z niezbalansowanymi danymi.
    counts = np.bincount(labels, minlength=2).astype(np.float32)
    weights = counts.sum() / (2.0 * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


class WeightedTrainer(Trainer):
    # Rozszerza klasę Trainer z biblioteki transformers, aby uwzględnić wagi klas podczas obliczania straty.
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_function = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )
        loss = loss_function(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )

        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    # Funkcja oblicza metryki F1 i macro F1 na podstawie przewidywanych logitów i prawdziwych etykiet.
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "f1": f1_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
    }


def tokenize_datasets(tokenizer, train, validation, test):
    # Tokenizuje dane tekstowe w kolumnie "code" dla zbiorów treningowego, walidacyjnego i testowego,
    def tokenize(batch):
        return tokenizer(batch["code"], truncation=True, max_length=MAX_LENGTH)

    train_ds = Dataset.from_pandas(train, preserve_index=False)
    val_ds = Dataset.from_pandas(validation, preserve_index=False)
    test_ds = Dataset.from_pandas(test, preserve_index=False)

    train_ds = train_ds.map(tokenize, batched=True, desc="Tokenizacja train")
    val_ds = val_ds.map(tokenize, batched=True, desc="Tokenizacja val")
    test_ds = test_ds.map(tokenize, batched=True, desc="Tokenizacja test")

    train_ds = train_ds.remove_columns(["code"])
    val_ds = val_ds.remove_columns(["code"])
    test_ds = test_ds.remove_columns(["code"])

    return train_ds, val_ds, test_ds


def build_training_args():
    # Buduje obiekt TrainingArguments z ustawieniami treningu, takimi jak liczba epok, rozmiar batcha, strategia ewaluacji i zapisywania modelu, itp.
    common_args = {
        "output_dir": OUTPUT_DIR,
        "learning_rate": LEARNING_RATE,
        "per_device_train_batch_size": TRAIN_BATCH_SIZE,
        "per_device_eval_batch_size": EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        "weight_decay": 0.01,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "fp16": torch.cuda.is_available(),
        "logging_steps": 100,
        "report_to": "none",
        "dataloader_num_workers": 0,
    }

    try:
        return TrainingArguments(
            **common_args,
            eval_strategy="steps",
            eval_steps=5000,
            save_strategy="steps",
            save_steps=5000,
        )
    except TypeError:
        return TrainingArguments(
            **common_args,
            evaluation_strategy="steps",
            eval_steps=5000,
            save_strategy="steps",
            save_steps=5000,
        )


def print_gpu_status():
    # Sprawdza dostępność GPU i wyświetla informacje o nim, takie jak nazwa, ilość pamięci, wersja CUDA i testowy tensor.
    print("Sprawdzanie GPU...", flush=True)
    print(f"PyTorch: {torch.__version__}", flush=True)
    print(f"CUDA dostepna: {torch.cuda.is_available()}", flush=True)

    if not torch.cuda.is_available():
        print("UWAGA: CodeBERT uruchomi sie na CPU i bedzie bardzo wolny.", flush=True)
        return

    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    print(f"GPU: {properties.name}", flush=True)
    print(f"Pamiec GPU: {properties.total_memory / 1024**3:.1f} GB", flush=True)
    print(f"CUDA PyTorch: {torch.version.cuda}", flush=True)
    test_tensor = torch.tensor([1.0], device="cuda")
    print(f"Test tensora CUDA: {test_tensor.device}", flush=True)


def main():
    # Główna funkcja, która wykonuje cały proces trenowania i ewaluacji modelu CodeBERT.
    started_at = time.perf_counter()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    print(f"Model: {MODEL_NAME}", flush=True)
    print_gpu_status()

    # Używa biblioteki tqdm do wyświetlania paska postępu dla różnych etapów procesu, takich jak ładowanie danych, tokenizacja, trenowanie i ewaluacja.
    with tqdm(total=6, desc="Etapy", unit="etap") as progress:
        print("\nŁadowanie danych...", flush=True)
        df_train, df_val, df_test = load_data()
        print(
            f"Rozmiary: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}",
            flush=True,
        )
        progress.update(1)

        print("\nŁadowanie tokenizera i modelu...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2,
        )
        progress.update(1)

        print("\nTokenizacja...", flush=True)
        train_ds, val_ds, test_ds = tokenize_datasets(
            tokenizer,
            df_train,
            df_val,
            df_test,
        )
        progress.update(1)
        
        # Przygotowuje trenera z ważonymi stratami, aby lepiej radzić sobie z niezbalansowanymi danymi, a następnie trenuje model na zbiorze treningowym.
        trainer = WeightedTrainer(
            model=model,
            args=build_training_args(),
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=compute_metrics,
        )
        trainer.class_weights = class_weights(df_train["label"].to_numpy())

        print("\nTrenowanie CodeBERT...", flush=True)
        trainer.train()
        progress.update(1)

        #### Ewaluacja wewnętrzna####
        print("Ewaluacja na zbiorze treningowym:")
        train_pred = np.argmax(trainer.predict(train_ds).predictions, axis=-1)
        train_acc = np.mean(train_pred == df_train["label"].to_numpy())
        print(f"Dokładność(treningowy): {train_acc:.4f}")
        progress.update(1)

        print("\nEwaluacja na zbiorze walidacyjnym:")
        y_pred = np.argmax(trainer.predict(val_ds).predictions, axis=-1)
        val_f1 = f1_score(df_val["label"], y_pred, average="macro")

        print(f"F1(walidacyjny):  {val_f1:.4f}")
        print("\nRaport(walidacyjny):")
        print(classification_report(df_val["label"], y_pred))
        ###################

        #### Ewaluacja właściwa####
        print("\nEwaluacja na zbiorze testowym:")
        y_pred = np.argmax(trainer.predict(test_ds).predictions, axis=-1)
        print(f"F1: {f1_score(df_test['label'], y_pred):.4f}")
        print("\nRaport:")
        print(classification_report(df_test["label"], y_pred))
        ###################
        progress.update(1)

    print(f"\nCalkowity czas: {time.perf_counter() - started_at:.1f} s")


if __name__ == "__main__":
    main()