import ast
import unittest
from pathlib import Path

APP_PATH = Path(__file__).with_name("app.py")
APP_TREE = ast.parse(APP_PATH.read_text(encoding="utf-8"))
TARGET_NAMES = {"category_prompt_instructions", "ensure_nonempty_limitations"}
TARGET_NODES = [node for node in APP_TREE.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in TARGET_NAMES]
NAMESPACE = {}
exec(compile(ast.Module(body=TARGET_NODES, type_ignores=[]), str(APP_PATH), "exec"), NAMESPACE)
category_prompt_instructions = NAMESPACE["category_prompt_instructions"]
ensure_nonempty_limitations = NAMESPACE["ensure_nonempty_limitations"]


class CategoryPromptInstructionTests(unittest.TestCase):
    def test_nonverbal_abstention_guidance_is_exclusive_to_nonverbal(self):
        nonverbal = category_prompt_instructions("Nonverbal Communication")
        self.assertIn("visual evidence is genuinely insufficient", nonverbal)
        for category in ("Vocal Delivery", "Fluency", "Word Choice"):
            instructions = category_prompt_instructions(category)
            self.assertNotIn("visual evidence is genuinely insufficient", instructions)
            self.assertNotIn("hand-detection", instructions)
            self.assertNotIn("pose tracking", instructions)

    def test_each_category_receives_its_own_boundary(self):
        self.assertIn("filler words", category_prompt_instructions("Vocal Delivery"))
        self.assertIn("intonation quality", category_prompt_instructions("Fluency"))
        self.assertIn("technical terminology", category_prompt_instructions("Word Choice"))
        self.assertIn("visual tracking", category_prompt_instructions("Nonverbal Communication"))

    def test_unknown_category_is_rejected(self):
        with self.assertRaises(ValueError):
            category_prompt_instructions("Overall Presentation")


class LimitationsNormalizationTests(unittest.TestCase):
    def test_missing_or_null_limitations_get_nonempty_default(self):
        for value in (None, "", "   ", [], {}):
            result = ensure_nonempty_limitations({"limitations": value})
            self.assertIsInstance(result["limitations"], str)
            self.assertTrue(result["limitations"].strip())

    def test_existing_limitations_are_preserved(self):
        assessment = {"limitations": "The camera view is partially obstructed."}
        self.assertIs(ensure_nonempty_limitations(assessment), assessment)
        self.assertEqual(assessment["limitations"], "The camera view is partially obstructed.")


if __name__ == "__main__":
    unittest.main()
