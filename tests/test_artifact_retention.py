import unittest
from pathlib import Path
from unittest.mock import patch

from adcg.artifacts import keep_only, remove_pipeline_stage


class ArtifactRetentionTests(unittest.TestCase):
    def test_keep_only_removes_generated_siblings(self):
        root = Path("C:/pipeline-artifacts").resolve()
        final_image = root / "final.png"
        diagnostic = root / "diagnostic.png"
        nested = root / "diagnostics"

        def is_dir(path):
            return path in {root, nested}

        with patch.object(
            Path, "is_dir", autospec=True, side_effect=is_dir
        ), patch.object(
            Path,
            "iterdir",
            autospec=True,
            return_value=iter((final_image, diagnostic, nested)),
        ), patch.object(Path, "unlink", autospec=True) as unlink, patch(
            "adcg.artifacts.shutil.rmtree"
        ) as rmtree:
            keep_only(root, (final_image,))

        unlink.assert_called_once_with(diagnostic)
        rmtree.assert_called_once_with(nested)

    def test_remove_pipeline_stage_is_scoped_to_named_child(self):
        root = Path("C:/pipeline-artifacts").resolve()
        stage = root / "03_generated"
        with patch.object(
            Path, "is_dir", autospec=True, return_value=True
        ), patch("adcg.artifacts.shutil.rmtree") as rmtree:
            remove_pipeline_stage(root, "03_generated")

        rmtree.assert_called_once_with(stage)

    def test_remove_pipeline_stage_rejects_parent_traversal(self):
        with self.assertRaises(ValueError):
            remove_pipeline_stage("C:/pipeline-artifacts", "../outside")


if __name__ == "__main__":
    unittest.main()