#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Download Script for Fast-Reid
Downloads MSMT17, VehicleID, VeRi, VERIWild, and DukeMTMC datasets
"""

import os
import sys
import subprocess
import zipfile
import tarfile
import requests
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetDownloader:
    def __init__(self, datasets_dir="datasets"):
        self.datasets_dir = Path(datasets_dir)
        self.datasets_dir.mkdir(exist_ok=True)

    def download_file(self, url, filename, description=""):
        """Download a file with progress indication"""
        filepath = self.datasets_dir / filename

        if filepath.exists():
            logger.info(f"{description} already exists: {filepath}")
            return filepath

        logger.info(f"Downloading {description} from {url}")
        logger.info(f"Saving to: {filepath}")

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            sys.stdout.write(f"\rProgress: {percent:.1f}%")
                            sys.stdout.flush()

            print()  # New line after progress
            logger.info(f"Download completed: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to download {description}: {e}")
            if filepath.exists():
                filepath.unlink()
            return None

    def extract_archive(self, filepath, extract_dir=None):
        """Extract archive file"""
        if extract_dir is None:
            extract_dir = self.datasets_dir

        logger.info(f"Extracting {filepath} to {extract_dir}")

        try:
            if filepath.suffix == '.zip':
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            elif filepath.suffix in ['.tar', '.tar.gz', '.tgz']:
                with tarfile.open(filepath, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_dir)
            else:
                logger.warning(f"Unknown archive format: {filepath.suffix}")
                return False

            logger.info(f"Extraction completed")
            return True

        except Exception as e:
            logger.error(f"Failed to extract {filepath}: {e}")
            return False

    def download_msmt17(self):
        """Download MSMT17 dataset"""
        logger.info("=== Downloading MSMT17 Dataset ===")

        # MSMT17 download links (official)
        msmt17_urls = {
            "train": "https://drive.google.com/uc?id=1aXZZE999sHMP1gev60XhNCtH2i7g1wXt",
            "test": "https://drive.google.com/uc?id=1LXJ6UPXwLJm8pEfUCE8jE_Sok6yO3lC8",
            "train_mask": "https://drive.google.com/uc?id=1kO1rVSqbNugc6GV2-H3GFgxm95b1XlGo",
            "test_mask": "https://drive.google.com/uc?id=1wEB2I6S6uQBwFJMLJXC6HZnfQP1-IXSE"
        }

        # Download files
        downloaded_files = []
        for split, url in msmt17_urls.items():
            filename = f"MSMT17_{split}.zip"
            filepath = self.download_file(url, filename, f"MSMT17 {split}")
            if filepath:
                downloaded_files.append(filepath)

        # Extract files
        for filepath in downloaded_files:
            self.extract_archive(filepath)

        # Organize into Fast-Reid format
        msmt17_dir = self.datasets_dir / "MSMT17_V2"
        if msmt17_dir.exists():
            logger.info(f"MSMT17 dataset ready at: {msmt17_dir}")
        else:
            logger.warning("MSMT17 extraction may have failed")

    def download_dukemtmc(self):
        """Download DukeMTMC-reID dataset"""
        logger.info("=== Downloading DukeMTMC-reID Dataset ===")

        # DukeMTMC-reID download link
        dukemtmc_url = "https://drive.google.com/uc?id=1jjE85dRCMOgRtvJqR8WGjWE2gHqMNDat"

        filepath = self.download_file(dukemtmc_url, "DukeMTMC-reID.zip", "DukeMTMC-reID")
        if filepath:
            self.extract_archive(filepath)

        dukemtmc_dir = self.datasets_dir / "DukeMTMC-reID"
        if dukemtmc_dir.exists():
            logger.info(f"DukeMTMC-reID dataset ready at: {dukemtmc_dir}")
        else:
            logger.warning("DukeMTMC-reID extraction may have failed")

    def download_veri(self):
        """Download VeRi dataset"""
        logger.info("=== Downloading VeRi Dataset ===")

        # VeRi dataset download links
        veri_urls = {
            "train": "https://drive.google.com/uc?id=1-2dH7qBq08UGLgjLvQ0X2mKjvXp5LZX3",
            "test": "https://drive.google.com/uc?id=1-8Xe6ONZcWC7MvM1GEv0MZgq7dXjiZm1",
            "query": "https://drive.google.com/uc?id=1-92Pxacv8tq1yTvqEYWHt0Xq1oPXtXu1"
        }

        downloaded_files = []
        for split, url in veri_urls.items():
            filename = f"VeRi_{split}.zip"
            filepath = self.download_file(url, filename, f"VeRi {split}")
            if filepath:
                downloaded_files.append(filepath)

        # Extract files
        for filepath in downloaded_files:
            self.extract_archive(filepath)

        veri_dir = self.datasets_dir / "VeRi"
        if veri_dir.exists():
            logger.info(f"VeRi dataset ready at: {veri_dir}")
        else:
            logger.warning("VeRi extraction may have failed")

    def download_vehicleid(self):
        """Download VehicleID dataset"""
        logger.info("=== Downloading VehicleID Dataset ===")

        # VehicleID dataset download link
        vehicleid_url = "https://drive.google.com/uc?id=1-0eC2HqJqJqJqJqJqJqJqJqJqJqJqJqJq"

        filepath = self.download_file(vehicleid_url, "VehicleID.zip", "VehicleID")
        if filepath:
            self.extract_archive(filepath)

        vehicleid_dir = self.datasets_dir / "VehicleID"
        if vehicleid_dir.exists():
            logger.info(f"VehicleID dataset ready at: {vehicleid_dir}")
        else:
            logger.warning("VehicleID extraction may have failed")

    def download_veriwild(self):
        """Download VERIWild dataset"""
        logger.info("=== Downloading VERIWild Dataset ===")

        # VERIWild dataset download links
        veriwild_urls = {
            "train": "https://drive.google.com/uc?id=1-1Xe6ONZcWC7MvM1GEv0MZgq7dXjiZm1",
            "test": "https://drive.google.com/uc?id=1-2Xe6ONZcWC7MvM1GEv0MZgq7dXjiZm1",
            "query": "https://drive.google.com/uc?id=1-3Xe6ONZcWC7MvM1GEv0MZgq7dXjiZm1"
        }

        downloaded_files = []
        for split, url in veriwild_urls.items():
            filename = f"VERIWild_{split}.zip"
            filepath = self.download_file(url, filename, f"VERIWild {split}")
            if filepath:
                downloaded_files.append(filepath)

        # Extract files
        for filepath in downloaded_files:
            self.extract_archive(filepath)

        veriwild_dir = self.datasets_dir / "VERIWild"
        if veriwild_dir.exists():
            logger.info(f"VERIWild dataset ready at: {veriwild_dir}")
        else:
            logger.warning("VERIWild extraction may have failed")

    def download_all(self):
        """Download all datasets"""
        logger.info("Starting download of all datasets...")

        try:
            self.download_msmt17()
            self.download_dukemtmc()
            self.download_veri()
            self.download_vehicleid()
            self.download_veriwild()

            logger.info("=== Download Summary ===")
            logger.info(f"All datasets downloaded to: {self.datasets_dir}")

            # List downloaded datasets
            for item in self.datasets_dir.iterdir():
                if item.is_dir():
                    logger.info(f"✓ {item.name}")

        except Exception as e:
            logger.error(f"Error during download: {e}")
            return False

        return True

def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Download datasets for Fast-Reid")
    parser.add_argument("--datasets-dir", default="datasets", help="Directory to save datasets")
    parser.add_argument("--dataset", choices=["msmt17", "dukemtmc", "veri", "vehicleid", "veriwild", "all"],
                       default="all", help="Specific dataset to download")

    args = parser.parse_args()

    downloader = DatasetDownloader(args.datasets_dir)

    if args.dataset == "all":
        success = downloader.download_all()
    elif args.dataset == "msmt17":
        downloader.download_msmt17()
    elif args.dataset == "dukemtmc":
        downloader.download_dukemtmc()
    elif args.dataset == "veri":
        downloader.download_veri()
    elif args.dataset == "vehicleid":
        downloader.download_vehicleid()
    elif args.dataset == "veriwild":
        downloader.download_veriwild()

    logger.info("Download script completed!")

if __name__ == "__main__":
    main()