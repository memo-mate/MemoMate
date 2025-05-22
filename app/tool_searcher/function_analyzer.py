import pydantic


class FunctionAnalyzer:
    @staticmethod
    def analyze_function(function_) -> dict:
        """
        Analyzes a python function and returns a description compatible with the OpenAI API
        Assumptions:
        * docstring includes a function description and parameter descriptions separated by 2 linebreaks
        * docstring includes parameter descriptions indicated by :param x:
        """
        name = function_.__name__

        # analyze type hints
        parameters = pydantic.TypeAdapter(function_).json_schema()
        # remove references to self for methods
        parameters["properties"].pop("self", None)
        if "required" in parameters:
            parameters["required"] = [p for p in parameters["required"] if p != "self"]

        # analyze doc string
        if function_.__doc__:
            descriptions = [e.strip() for e in function_.__doc__.split(":param ")]
            function_description, parameter_descriptions = descriptions[0], descriptions[1:]
        else:
            function_description = ""
            parameter_descriptions = {}
        parameter_descriptions = {
            k.strip(): v.strip()
            for (k, v) in [e.split(":return:")[0].strip().split(": ", 1) for e in parameter_descriptions if e]
        }
        for parameter, parameter_description in parameter_descriptions.items():
            parameters["properties"][parameter]["description"] = parameter_description

        return {
            "type": "function",
            "function": {
                "name": name,
                "description": function_description,
                "parameters": parameters,
            },
            "strict": True,
        }

    def analyze_class(self, class_: object) -> list:
        """
        Analyzes a python class and returns a description of all its non-private functions
            compatible with the OpenAI API
        """
        functions = [
            self.analyze_function(getattr(class_, func))
            for func in dir(class_)
            if callable(getattr(class_, func)) and not func.startswith("_")
        ]
        return functions
