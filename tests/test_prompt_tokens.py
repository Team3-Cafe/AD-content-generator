import contextlib
import io
import unittest

from adcg.prompt_tokens import fit_clip_prompt


class FakeTokenizer:
    model_max_length = 77

    def __call__(self, text, *, add_special_tokens, truncation):
        tokens = str(text).split()
        ids = list(range(len(tokens)))
        if add_special_tokens:
            ids = [-1, *ids, -2]
        return {"input_ids": ids}

    def decode(self, token_ids, *, skip_special_tokens):
        return " ".join(f"token{index}" for index in token_ids)


class PromptTokenTests(unittest.TestCase):
    def test_keeps_prompt_within_limit_and_prints_count(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            prompt = fit_clip_prompt(
                FakeTokenizer(),
                "short background prompt",
                label="generation positive",
            )

        self.assertEqual(prompt, "short background prompt")
        self.assertIn("5/77", output.getvalue())
        self.assertIn("kept", output.getvalue())

    def test_preserves_required_prefix_when_truncating(self):
        required = " ".join(f"required{index}" for index in range(10))
        prompt = " ".join(f"context{index}" for index in range(100))

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            fitted = fit_clip_prompt(
                FakeTokenizer(),
                prompt,
                label="identity negative",
                required_prefix=required,
            )

        token_count = len(fitted.split()) + 2
        self.assertLessEqual(token_count, 77)
        self.assertTrue(fitted.startswith("token0 token1"))
        self.assertIn("truncated", output.getvalue())


if __name__ == "__main__":
    unittest.main()
