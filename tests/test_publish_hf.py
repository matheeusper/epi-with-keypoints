import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.publish_hf import stage_release, validate_release


class PublishHuggingFaceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.checkpoint = self.root / "helmet_only_576_best.pth"
        self.checkpoint.write_bytes(b"checkpoint de teste")
        self.card = self.root / "README.md"
        self.card.write_text("# Model card\n", encoding="utf-8")
        self.config = self.root / "training_config.yaml"
        self.config.write_text("model: {}\n", encoding="utf-8")
        self.metadata = self.root / "model_metadata.json"
        self.metadata.write_text(
            json.dumps(
                {
                    "checkpoint": {
                        "filename": self.checkpoint.name,
                        "size_bytes": self.checkpoint.stat().st_size,
                        "sha256": hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(),
                    }
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_validate_release_accepts_matching_manifest(self):
        metadata = validate_release(
            self.checkpoint, self.metadata, self.card, self.config
        )
        self.assertEqual(metadata["checkpoint"]["filename"], self.checkpoint.name)

    def test_validate_release_rejects_tampered_checkpoint(self):
        self.checkpoint.write_bytes(b"conteudo alterado")
        with self.assertRaisesRegex(ValueError, "tamanho"):
            validate_release(self.checkpoint, self.metadata, self.card, self.config)

    def test_stage_contains_only_the_public_model_manifest(self):
        destination = self.root / "release"
        destination.mkdir()
        with patch("scripts.publish_hf.MODEL_CARD", self.card), patch(
            "scripts.publish_hf.MODEL_METADATA", self.metadata
        ), patch("scripts.publish_hf.TRAINING_CONFIG", self.config):
            stage_release(self.checkpoint, destination)

        self.assertEqual(
            sorted(path.name for path in destination.iterdir()),
            [
                "README.md",
                "helmet_only_576_best.pth",
                "model_metadata.json",
                "training_config.yaml",
            ],
        )


if __name__ == "__main__":
    unittest.main()
