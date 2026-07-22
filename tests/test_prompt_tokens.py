import contextlib
import io
import unittest

from adcg.prompt_tokens import fit_clip_prompt, fit_clip_prompt_parts


class FakeTokenizer:
    model_max_length = 77

    def __init__(self):
        self.vocabulary = {}
        self.reverse_vocabulary = {}

    def __call__(self, text, *, add_special_tokens, truncation):
        ids = []
        for token in str(text).split():
            if token not in self.vocabulary:
                token_id = len(self.vocabulary)
                self.vocabulary[token] = token_id
                self.reverse_vocabulary[token_id] = token
            ids.append(self.vocabulary[token])
        if add_special_tokens:
            ids = [-1, *ids, -2]
        return {"input_ids": ids}

    def decode(self, token_ids, *, skip_special_tokens):
        return " ".join(
            self.reverse_vocabulary[index]
            for index in token_ids
            if index in self.reverse_vocabulary
        )


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
        self.assertTrue(fitted.startswith("required0 required1"))
        self.assertIn("truncated", output.getvalue())

    def test_removes_duplicate_required_prompt_parts(self):
        fitted = fit_clip_prompt(
            FakeTokenizer(),
            "halo, scene-specific artifact",
            label="identity negative",
            required_prefix="halo, jagged edge",
        )

        self.assertEqual(fitted.count("halo"), 1)
        self.assertIn("scene-specific artifact", fitted)

    def test_composes_whole_clauses_under_positive_target(self):
        tokenizer = FakeTokenizer()
        oversized = " ".join(f"detail{index}" for index in range(80))
        prompt, metadata = fit_clip_prompt_parts(
            tokenizer,
            [
                "ordinary service location",
                "physically grounded surface",
                oversized,
                "available daylight",
            ],
            label="generation everyday positive",
            required_prefix="everyday documentary setting",
        )

        self.assertLessEqual(metadata["token_count"], 72)
        self.assertIn("ordinary service location", prompt)
        self.assertIn("available daylight", prompt)
        self.assertNotIn("detail0", prompt)
        self.assertEqual(metadata["dropped_clauses"], [oversized])


if __name__ == "__main__":
    unittest.main()
