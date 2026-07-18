import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from harmonizer import FeatureHarmonizer

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")


class DataLoader:
    def __init__(self, data_dir="data", random_state=42):
        self.harmonizer = FeatureHarmonizer()
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        self.scaler_fitted = False
        self.random_state = random_state

    # ====================== LABEL MAPPING ======================
    def _map_label(self, row):
        src = str(row.get("source", "")).upper()
        for col in ["label", "Label", "Attack", "class", "Class", "attack_cat"]:
            if col not in row or pd.isna(row[col]):
                continue
            val = str(row[col]).strip().lower()
            if val in {"0", "0.0", "benign", "normal", "false", "0_normal", "benigntraffic", "", " "}:
                return 0
            if val in {"1", "1.0", "attack", "malicious", "true"}:
                return 1
            if col == "attack_cat" and val not in {"", "normal"}:
                return 1
            attack_patterns = [
                "dos", "ddos", "portscan", "bruteforce", "web attack", "infiltration",
                "bot", "patator", "heartbleed", "exploit", "fuzz", "recon", "hulk",
                "goldeneye", "slowloris", "slowhttptest", "sql", "xss"
            ]
            if any(p in val for p in attack_patterns):
                return 1
        return 1 if "IOT" in src else 0

    # ====================== FIXED FILE LOADING ======================
    def _load_file(self, filepath):
        if not os.path.exists(filepath):
            print(f"⚠️ Missing: {os.path.basename(filepath)}")
            return pd.DataFrame()
        
        print(f"Loading {os.path.basename(filepath)}")
        df = pd.read_csv(filepath, low_memory=True)
        
        # CRITICAL FIX: Remove leading/trailing spaces from column names
        df.columns = df.columns.str.strip()
        
        return df

    def _load_cic_ids2017(self, files):
        path = f"{self.data_dir}/CIC-IDS2017"
        dfs = []
        for fname in files:
            df = self._load_file(os.path.join(path, fname))
            if not df.empty:
                df["source"] = "CIC-IDS2017"
                dfs.append(df)
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def _load_unswnb15(self, split):
        path = f"{self.data_dir}/UNSW-NB15"
        fname = f"UNSW_NB15_{split}-set.csv"
        df = self._load_file(os.path.join(path, fname))
        if not df.empty:
            df["source"] = "UNSW-NB15"
        return df

    def _load_cic_iot2023(self, start_part, num_parts):
        path = f"{self.data_dir}/CIC-IoT2023"
        if not os.path.exists(path):
            print(f"⚠️ CIC-IoT2023 folder not found")
            return pd.DataFrame()

        part_files = sorted([f for f in os.listdir(path) if f.startswith("part-")])
        selected = part_files[start_part:start_part + num_parts]
        if not selected:
            return pd.DataFrame()
        
        print(f"Loading IoT parts {start_part}–{start_part + len(selected) - 1}")
        dfs = []
        for f in selected:
            df = self._load_file(os.path.join(path, f))
            if not df.empty:
                df["source"] = "CIC-IoT2023"
                dfs.append(df)
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    # ====================== TRAIN / TEST ======================
    def load_train(self):
        print("=== Loading TRAINING split ===")
        dfs = []
        
        cic_files = [
            "Benign-Monday-no-metadata.csv",
            "Tuesday-WorkingHours.csv",
            "Wednesday-workingHours.csv",
            "Thursday-WorkingHours-Morning-WebAttacks.csv",
            "Thursday-WorkingHours-Afternoon-Infilteration.csv"
        ]
        dfs.append(self._load_cic_ids2017(cic_files))
        
        dfs.append(self._load_unswnb15("training"))
        dfs.append(self._load_cic_iot2023(0, 8))

        df = pd.concat([d for d in dfs if not d.empty], ignore_index=True)
        return self._post_load_process(df, name="TRAIN")

    def load_test(self):
        print("=== Loading TEST split ===")
        dfs = []
        
        cic_test_files = [
            "Friday-WorkingHours-Morning.csv",
            "Friday-WorkingHours-Afternoon-PortScan.csv",
            "Friday-WorkingHours-Afternoon-DDos.csv",
        ]
        dfs.append(self._load_cic_ids2017(cic_test_files))
        dfs.append(self._load_unswnb15("testing"))
        dfs.append(self._load_cic_iot2023(160, 8))

        df = pd.concat([d for d in dfs if not d.empty], ignore_index=True)
        return self._post_load_process(df, name="TEST")

    def _post_load_process(self, df, name=""):
        if df.empty:
            return df
        print(f"\n{name} - Raw rows: {len(df):,}")
        
        df["label_mapped"] = df.apply(self._map_label, axis=1)
        
        print("Label distribution by source:")
        print(df.groupby("source")["label_mapped"].value_counts())
        print("\nAttack ratio by source:")
        print(df.groupby("source")["label_mapped"].mean().round(3))
        print("-" * 60)
        return df

    # ====================== PREPROCESS ======================
    def preprocess(self, df, target_attack_ratio=0.55, target_total_samples=200_000):
        print("\nHarmonising features...")
        df = df.reset_index(drop=True)
        X = self.harmonizer.transform(df)
        X = X.reset_index(drop=True)
        
        X.replace([np.inf, -np.inf], 0, inplace=True)
        X.fillna(0, inplace=True)

        benign = df[df["label_mapped"] == 0]
        attack = df[df["label_mapped"] == 1]

        n_benign = int(target_total_samples * (1 - target_attack_ratio))
        n_attack = target_total_samples - n_benign

        benign_s = benign.sample(n=min(n_benign, len(benign)), random_state=self.random_state)
        attack_s = attack.sample(n=min(n_attack, len(attack)), random_state=self.random_state)

        df_bal = pd.concat([benign_s, attack_s]).sample(frac=1, random_state=self.random_state)

        X_bal = X.loc[df_bal.index].copy().reset_index(drop=True)
        y_bal = df_bal["label_mapped"].to_numpy()

        if not self.scaler_fitted:
            X_scaled = self.scaler.fit_transform(X_bal)
            self.scaler_fitted = True
        else:
            X_scaled = self.scaler.transform(X_bal)

        print(f"Final Dataset: {len(X_scaled):,} samples | Attack: {y_bal.mean():.1%}")
        return X_scaled, y_bal, list(X.columns), df_bal.reset_index(drop=True)


# ====================== USAGE ======================
if __name__ == "__main__":
    loader = DataLoader()

    df_train = loader.load_train()
    X_train, y_train, features, _ = loader.preprocess(
        df_train, target_attack_ratio=0.55, target_total_samples=200_000
    )

    df_test = loader.load_test()
    X_test, y_test, _, _ = loader.preprocess(
        df_test, target_attack_ratio=0.05, target_total_samples=40000
    )

    print("\n🎉 Success! Both train and test loaded without errors.")