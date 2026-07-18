import pandas as pd
import numpy as np


class FeatureHarmonizer:
    """
    Harmonizes heterogeneous network-flow datasets into a common feature space.

    V1:
        CIC-IDS2017
        UNSW-NB15
        CIC-IoT2023

    All outputs have identical columns.
    """

    CANONICAL_COLUMNS = [
        # ======================================================
        # Basic Flow
        # ======================================================
        "duration",
        "protocol",
        # ======================================================
        # Packet Counts
        # ======================================================
        "fwd_packets",
        "bwd_packets",
        "total_packets",
        # ======================================================
        # Byte Counts
        # ======================================================
        "fwd_bytes",
        "bwd_bytes",
        "total_bytes",
        # ======================================================
        # Rates
        # ======================================================
        "packet_rate",
        "byte_rate",
        # ======================================================
        # Packet Statistics
        # ======================================================
        "mean_packet_size",
        "std_packet_size",
        "min_packet_size",
        "max_packet_size",
        # ======================================================
        # Inter-arrival Time
        # ======================================================
        "iat_mean",
        "iat_std",
        # ======================================================
        # TCP Flags
        # ======================================================
        "syn_count",
        "ack_count",
        "fin_count",
        "rst_count",
    ]

    def _empty(self, df):
        """
        Create an empty dataframe with the canonical feature columns.
        """
        return pd.DataFrame(
            0.0,
            index=df.index,
            columns=self.CANONICAL_COLUMNS,
        )

    def transform(self, df):
        """
        Harmonize a dataframe containing one or more datasets.
        """
        outputs = []

        for source, group in df.groupby("source"):
            source = str(source).upper()

            if source == "CIC-IDS2017":
                out = self._cic(group)
            elif source == "UNSW-NB15":
                out = self._unsw(group)
            elif source == "CIC-IOT2023":
                out = self._iot(group)
            else:
                print(f"Skipping unsupported source: {source}")
                continue

            outputs.append(out)

        if not outputs:
            raise ValueError("No supported datasets found.")

        X = pd.concat(outputs)
        X = X.reindex(df.index)  # restore original row order
        return X

    def _cic(self, df):
        """Fixed column mapping for CIC-IDS2017"""
        out = self._empty(df)

        # ---------------- Basic Flow ----------------
        out["duration"] = df["Flow Duration"]
        # CIC-IDS2017 doesn't contain protocol in this version
        out["protocol"] = 0

        # ---------------- Packets ----------------
        out["fwd_packets"] = df["Total Fwd Packets"]
        out["bwd_packets"] = df["Total Backward Packets"]
        out["total_packets"] = (
            df["Total Fwd Packets"] + df["Total Backward Packets"]
        )

        # ---------------- Bytes ----------------
        out["fwd_bytes"] = df["Total Length of Fwd Packets"]
        out["bwd_bytes"] = df["Total Length of Bwd Packets"]
        out["total_bytes"] = (
            df["Total Length of Fwd Packets"]
            + df["Total Length of Bwd Packets"]
        )

        # ---------------- Rates ----------------
        out["packet_rate"] = df["Flow Packets/s"]
        out["byte_rate"] = df["Flow Bytes/s"]

        # ---------------- Packet Statistics ----------------
        out["mean_packet_size"] = df["Packet Length Mean"]
        out["std_packet_size"] = df["Packet Length Std"]
        out["min_packet_size"] = df["Min Packet Length"]
        out["max_packet_size"] = df["Max Packet Length"]

        # ---------------- IAT ----------------
        out["iat_mean"] = df["Flow IAT Mean"]
        out["iat_std"] = df["Flow IAT Std"]

        # ---------------- TCP Flags ----------------
        out["syn_count"] = df["SYN Flag Count"]
        out["ack_count"] = df["ACK Flag Count"]
        out["fin_count"] = df["FIN Flag Count"]
        out["rst_count"] = df["RST Flag Count"]

        return out

    def _unsw(self, df):
        out = self._empty(df)

        # ---------------- Basic Flow ----------------
        out["duration"] = df["dur"]
        out["protocol"] = pd.factorize(df["proto"])[0]

        # ---------------- Packets ----------------
        out["fwd_packets"] = df["spkts"]
        out["bwd_packets"] = df["dpkts"]
        out["total_packets"] = df["spkts"] + df["dpkts"]

        # ---------------- Bytes ----------------
        out["fwd_bytes"] = df["sbytes"]
        out["bwd_bytes"] = df["dbytes"]
        out["total_bytes"] = df["sbytes"] + df["dbytes"]

        # ---------------- Rates ----------------
        out["packet_rate"] = df["rate"]
        out["byte_rate"] = df["sload"] + df["dload"]

        # ---------------- Packet Statistics ----------------
        out["mean_packet_size"] = (df["smean"] + df["dmean"]) / 2

        # Not available in UNSW
        out["std_packet_size"] = 0
        out["min_packet_size"] = 0
        out["max_packet_size"] = 0

        # ---------------- IAT ----------------
        out["iat_mean"] = (df["sinpkt"] + df["dinpkt"]) / 2
        out["iat_std"] = (df["sjit"] + df["djit"]) / 2

        # ---------------- TCP Flags ----------------
        out["syn_count"] = (df["synack"] > 0).astype(int)
        out["ack_count"] = (df["ackdat"] > 0).astype(int)
        out["fin_count"] = 0
        out["rst_count"] = 0

        return out

    def _iot(self, df):
        out = self._empty(df)

        # ---------------- Basic Flow ----------------
        out["duration"] = df["flow_duration"]
        out["protocol"] = df["Protocol Type"]

        # ---------------- Packets ----------------
        out["total_packets"] = df["Number"]
        out["fwd_packets"] = df["Number"] / 2
        out["bwd_packets"] = df["Number"] / 2

        # ---------------- Bytes ----------------
        out["total_bytes"] = df["Tot size"]
        out["fwd_bytes"] = df["Tot size"] / 2
        out["bwd_bytes"] = df["Tot size"] / 2

        # ---------------- Rates ----------------
        out["packet_rate"] = df["Rate"]
        out["byte_rate"] = df["Rate"] * df["AVG"]

        # ---------------- Packet Statistics ----------------
        out["mean_packet_size"] = df["AVG"]
        out["std_packet_size"] = df["Std"]
        out["min_packet_size"] = df["Min"]
        out["max_packet_size"] = df["Max"]

        # ---------------- IAT ----------------
        out["iat_mean"] = df["IAT"]
        out["iat_std"] = 0

        # ---------------- TCP Flags ----------------
        out["syn_count"] = df["syn_count"]
        out["ack_count"] = df["ack_count"]
        out["fin_count"] = df["fin_count"]
        out["rst_count"] = df["rst_count"]

        return out