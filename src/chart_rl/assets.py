"""Download pinned official assets and safely unpack the ChartQA archive."""

import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile

from .data import sha256_file


def extract_dataset(archive, destination):
    archive, destination = Path(archive), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Extraction destination already exists; use the recorded completed dataset or a new directory")
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        # Validate every path before writing any archive member.
        for member in members:
            relative = PurePosixPath(member.filename)
            path = (destination / member.filename).resolve()
            if (relative.is_absolute() or ".." in relative.parts or "\\" in member.filename or ":" in member.filename
                    or not path.is_relative_to(destination)
                    or stat.S_ISLNK(member.external_attr >> 16)):
                raise ValueError("Unsafe archive member: " + member.filename)
        destination.mkdir(parents=True)
        for member in members:
            target = destination / member.filename
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
    candidates = [path.parent.parent for path in destination.rglob("train_human.json") if path.parent.name == "train"]
    if len(candidates) != 1:
        raise ValueError("Could not uniquely locate the official ChartQA directory after extraction")
    root = candidates[0]
    for split in ("train", "val", "test"):
        for source in ("human", "augmented"):
            if not (root / split / f"{split}_{source}.json").is_file() or not (root / split / "png").is_dir():
                raise ValueError("Extracted dataset is missing an official split or images")
    return root


def download_assets(config_path, output_root, model_only=False, data_only=False):
    from huggingface_hub import hf_hub_download, snapshot_download

    if model_only and data_only:
        raise ValueError("Choose at most one of model-only/data-only")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "assets.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {"schema_version": 1}
    for key in ("model_revision", "dataset_revision"):
        if key in metadata and metadata[key] != config[key]:
            raise ValueError("Asset revision changed; use a new asset directory")
    if not data_only:
        model_path = snapshot_download(
            repo_id=config["model_id"], revision=config["model_revision"],
            local_dir=output / "models" / config["model_id"].split("/")[-1], max_workers=4,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "README.md", "LICENSE*"],
        )
        metadata.update(model_id=config["model_id"], model_revision=config["model_revision"], model_path=str(Path(model_path).resolve()))
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not model_only:
        archive = hf_hub_download(repo_id=config["dataset_id"], repo_type="dataset", revision=config["dataset_revision"],
                                  filename=config["dataset_archive"], local_dir=output / "downloads")
        archive_hash = sha256_file(Path(archive))
        if "dataset_root" in metadata:
            if metadata.get("dataset_archive_sha256") != archive_hash or not Path(metadata["dataset_root"]).is_dir():
                raise ValueError("Previously downloaded dataset no longer matches its asset record")
        else:
            root = extract_dataset(archive, output / "chartqa")
            metadata.update(dataset_root=str(root), dataset_archive_sha256=archive_hash,
                            dataset_id=config["dataset_id"], dataset_revision=config["dataset_revision"])
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata
