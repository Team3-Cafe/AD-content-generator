import os
from pathlib import Path
import unittest
from unittest import mock

from adcg.prompt_layout.renderer import (
    PROJECT_KOREAN_FONT,
    _resolve_font_path,
)


class ProjectFontResolutionTests(unittest.TestCase):
    def test_project_korean_font_is_the_first_default_candidate(self):
        with (
            mock.patch.dict(os.environ, {"ADCG_FONT_PATH": ""}),
            mock.patch.object(Path, "is_file", return_value=True),
        ):
            resolved = _resolve_font_path(None, bold=True, text="한글 제목")

        self.assertEqual(resolved, PROJECT_KOREAN_FONT)


if __name__ == "__main__":
    unittest.main()
