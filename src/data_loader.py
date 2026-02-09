import pandas as pd
import requests
from pathlib import Path
import zipfile
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://s3.amazonaws.com/capitalbikeshare-data"
YEARS = [2024, 2025]

RAW_DIR = Path("data/raw")
ZIP_DIR = RAW_DIR / "zips"
CSV_DIR = RAW_DIR / "csv"
ZIP_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest: Path) -> bool:
    """Télécharge un fichier. Renvoie True si download OK (ou déjà présent), False sinon."""
    if dest.exists():
        logger.info(f"SKIP: {dest.name} existe déjà")
        return True

    logger.info(f"DOWNLOAD: {url}")
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except requests.HTTPError:
        logger.warning(f"NOT FOUND: {dest.name}")
        return False


def unzip_file(zip_path: Path, extract_to: Path) -> None:
    """Décompresse une archive ZIP dans extract_to."""
    logger.info(f"UNZIP: {zip_path.name}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)


def download_and_extract_all():
    """Télécharge et extrait les données (2024-2025)."""
    for year in YEARS:
        for month in range(1, 13):
            zip_name = f"{year}{month:02d}-capitalbikeshare-tripdata.zip"
            url = f"{BASE_URL}/{zip_name}"
            zip_path = ZIP_DIR / zip_name

            ok = download_file(url, zip_path)
            if not ok:
                continue  # mois non disponible => skip

            try:
                unzip_file(zip_path, CSV_DIR)
            except zipfile.BadZipFile:
                logger.error(f"BAD ZIP: {zip_name}")


def aggregate_data(csv_dir: Path, output_path: Path):
    """Agrège tous les CSV en demande horaire (nombre de trajets par heure)."""
    all_files = sorted(csv_dir.glob("*.csv"))
    if not all_files:
        logger.warning("Aucun CSV trouvé pour l'agrégation.")
        return

    frames = []
    for f in all_files:
        logger.info(f"PROCESSING: {f.name}")
        df = pd.read_csv(f, low_memory=False)

        # Robustesse : différentes versions de colonnes possibles
        time_col = None
        if "started_at" in df.columns:
            time_col = "started_at"
        elif "start_time" in df.columns:
            time_col = "start_time"

        if time_col is None:
            logger.warning(f"SKIP (no time col): {f.name}")
            continue

        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col])
        frames.append(df[[time_col]].rename(columns={time_col: "datetime"}))

    if not frames:
        logger.warning("Aucune donnée valide après lecture des CSV.")
        return

    df_full = pd.concat(frames, ignore_index=True)
    df_full = df_full.sort_values("datetime")

    df_full = df_full.set_index("datetime")
    df_hourly = df_full.resample("h").size().rename("demand").to_frame().reset_index()

    # Optionnel : garantit une série horaire continue (utile pour ML)
    df_hourly = df_hourly.sort_values("datetime")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_hourly.to_csv(output_path, index=False)
    logger.info(f"SUCCESS: Données agrégées -> {output_path} | n={len(df_hourly)}")


def load_and_prepare_data(filepath: str | Path) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    return df


def create_temporal_split(df: pd.DataFrame, train_end="2025-06-30", val_end="2025-10-31"):
    """Split strict : train <= train_end, val (train_end, val_end], test > val_end."""
    train_end = pd.to_datetime(train_end)
    val_end = pd.to_datetime(val_end)

    train = df[df["datetime"] <= train_end].copy()
    val = df[(df["datetime"] > train_end) & (df["datetime"] <= val_end)].copy()
    test = df[df["datetime"] > val_end].copy()
    return train, val, test


if __name__ == "__main__":
    download_and_extract_all()
    aggregate_data(CSV_DIR, Path("data/processed/bikeshare_aggregated.csv"))
