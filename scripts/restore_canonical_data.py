"""Restore only newline-damaged CSVs from the verified minimal-data archive."""
import argparse
import hashlib
from pathlib import Path
import zipfile

from ptit_sentiment.common import file_hash, read_json, run_cli, write_json


def restore(archive, output):
    """Verify all four originals before writing; preserve current bytes in a ZIP."""
    manifest_path = Path("data/splits/manifest.json")
    manifest_hash = file_hash(manifest_path)
    manifest = read_json(manifest_path)
    expected = {"data/labeled/labeled_comments.csv": manifest["source_sha256"]}
    expected.update({f"data/splits/{name}.csv": info["sha256"]
                     for name, info in manifest["splits"].items()})
    if set(expected) != {"data/labeled/labeled_comments.csv", "data/splits/train.csv",
                         "data/splits/validation.csv", "data/splits/test.csv"}:
        raise ValueError("Manifest must describe exactly the four canonical CSVs.")
    output = Path(output)
    if output.exists():
        raise ValueError("Choose a new output directory; preserve the previous backup.")
    originals, current, records = {}, {}, {}
    with zipfile.ZipFile(archive) as source:
        for name, digest in expected.items():
            original, present = source.read(name), Path(name).read_bytes()
            if hashlib.sha256(original).hexdigest() != digest:
                raise ValueError(f"Archive does not match manifest: {name}")
            if present.replace(b"\r\n", b"\n") != original.replace(b"\r\n", b"\n"):
                raise ValueError(f"Changes beyond line endings found; refusing to overwrite: {name}")
            originals[name], current[name] = original, present
            records[name] = {"before_sha256": hashlib.sha256(present).hexdigest(),
                             "expected_sha256": digest, "newline_only_difference": True}
    output.mkdir(parents=True)
    with zipfile.ZipFile(output / "before_restore.zip", "w", zipfile.ZIP_DEFLATED) as backup:
        for name, content in current.items():
            backup.writestr(name, content)
    for name, content in originals.items():
        Path(name).write_bytes(content)
        records[name]["after_sha256"] = file_hash(name)
        if records[name]["after_sha256"] != expected[name]:
            raise ValueError(f"Restored bytes failed verification: {name}")
    if file_hash(manifest_path) != manifest_hash:
        raise ValueError("Manifest changed during restoration.")
    write_json(output / "restoration.json", {
        "archive": str(archive), "archive_sha256": file_hash(archive),
        "manifest_sha256": manifest_hash, "manifest_unchanged": True,
        "policy": "Exact verified bytes; no relabeling, resplitting or other archive extraction",
        "files": records,
    })
    print("Verified and restored exactly four canonical CSVs; manifest unchanged.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="outputs/archives/data_minimal_20261007.zip")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    restore(args.archive, args.output)


if __name__ == "__main__":
    run_cli(main)
