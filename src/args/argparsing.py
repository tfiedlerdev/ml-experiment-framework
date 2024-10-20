import argparse
from typing import Any, Literal, Type, cast
from pydantic import BaseModel
from src.args.yaml_config import YamlConfig
from src.experiments.base_experiment import BaseExperiment, BaseExperimentArgs
from src.args.experiment_registry import experiments


def str_to_bool(value):
    if value.lower() in ["true", "t"]:
        return True
    elif value.lower() in ["false", "f"]:
        return False
    elif value.lower() in ["none", "n"]:
        return None
    else:
        raise argparse.ArgumentTypeError("Invalid boolean value: {}".format(value))


def str_to_list(value):
    import json

    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise argparse.ArgumentTypeError("Invalid list value: {}".format(value))
    return parsed


def _parser_from_model(parser: argparse.ArgumentParser, model: Type[BaseModel]):
    "Add Pydantic model to an ArgumentParser"
    fields = model.model_fields

    for name, field in fields.items():
        if name == "hidden_sizes":
            pass

        def get_type_args():
            is_literal = getattr(field.annotation, "__origin__", None) is Literal
            is_bool = getattr(field.annotation, "__name__", None) == "bool"
            annotation_origin = getattr(field.annotation, "__origin__", None)
            is_list = (
                annotation_origin is not None
                and getattr(annotation_origin, "__name__", None) == "list"
            )

            if is_literal:
                return {"type": str, "choices": cast(Any, field.annotation).__args__}
            if is_bool:
                return {"type": str_to_bool}
            if is_list:
                return {"type": str_to_list}
            return {"type": field.annotation}

        parser.add_argument(
            f"--{name}",
            dest=name,
            default=field.default,
            help=field.description,
            **get_type_args(),  # type: ignore
        )
    return parser


def _create_arg_parser():
    base_parser = argparse.ArgumentParser()
    base_parser = _parser_from_model(base_parser, BaseExperimentArgs)
    base_args, _ = base_parser.parse_known_args()

    experiment_model = experiments[base_args.experiment_type].get_args_model()
    parser = argparse.ArgumentParser(
        description="Machine Learning Experiment Configuration"
    )
    parser = _parser_from_model(parser, experiment_model)
    return parser


def get_experiment_from_args() -> BaseExperiment:
    arg_parser = _create_arg_parser()
    args = arg_parser.parse_args()
    yaml_config = YamlConfig()

    experiment = experiments[args.experiment_type](vars(args), yaml_config.config)
    return experiment
