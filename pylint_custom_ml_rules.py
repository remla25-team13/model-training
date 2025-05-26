from pylint.checkers import BaseChecker
from pylint.lint import PyLinter
import astroid

# https://hynn01.github.io/ml-smells/posts/codesmells/14-randomness-uncontrolled/

class RandomnessChecker(BaseChecker):
    name = "randomness-checker"
    priority = -1
    msgs = {
        "W9002": (
            "Randomness used without setting a seed",
            "uncontrolled-randomness",
            "You should set a random seed when using randomness for reproducibility.",
        )
    }

    def __init__(self, linter=None):
        super().__init__(linter)
        self.random_used = False
        self.seed_set = False

    def visit_import(self, node):
        for name, _ in node.names:
            if name in ("random", "numpy", "numpy.random", "torch"):
                self.random_used = True

    def visit_importfrom(self, node):
        if node.modname in ("random", "numpy.random", "torch"):
            self.random_used = True

    def visit_call(self, node):
        if isinstance(node.func, astroid.Attribute):
            if node.func.attrname == "seed":
                self.seed_set = True

    def close(self):
        if self.random_used and not self.seed_set:
            self.add_message("uncontrolled-randomness", line=1)

def register(linter: PyLinter):
    linter.register_checker(RandomnessChecker(linter))