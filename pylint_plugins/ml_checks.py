from pylint.checkers import BaseChecker


class MLCodeSmellChecker(BaseChecker):
    name = "ml-code-smells"
    priority = -1
    msgs = {
        "W9001": (
            "predict() called on training data (X_train) — did you mean X_test?",
            "predict-on-training-data",
            "Avoid evaluating model on training data. Use X_test instead of X_train.",
        ),
        "W9002": (
            "Use df.to_numpy() instead of df.values for conversion",
            "values-attribute-misused",
            "Using .values may return inconsistent types; prefer df.to_numpy().",
        ),
    }

    def visit_call(self, node):
        func_name = node.func.as_string()

        # --- Check for `predict(X_train)` ---
        if (func_name.endswith(".predict") or func_name == "predict") and node.args:
            arg = node.args[0]
            if hasattr(arg, "name") and arg.name == "X_train":
                self.add_message("predict-on-training-data", node=node)

    def visit_attribute(self, node):
        # --- Check for `df.values` usage ---
        if node.attrname == "values":
            self.add_message("values-attribute-misused", node=node)


def register(linter):
    print("Running custom checks...")
    linter.register_checker(MLCodeSmellChecker(linter))
    print("Custom checks registered.")
