import logging

from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np
import torch
import re
from PIL import ImageDraw, ImageFont, Image


class BoxQuantizer(object):
    def __init__(self, mode, bins):
        self.mode = mode
        self.bins = bins

    def quantize(self, boxes: torch.Tensor, size):
        bins_w, bins_h = self.bins  # Quantization bins.
        size_w, size_h = size  # Original image size.
        size_per_bin_w = size_w / bins_w
        size_per_bin_h = size_h / bins_h
        xmin, ymin, xmax, ymax = boxes.split(1, dim=-1)  # Shape: 4 * [N, 1].

        if self.mode == "floor":
            quantized_xmin = (xmin / size_per_bin_w).floor().clamp(0, bins_w - 1)
            quantized_ymin = (ymin / size_per_bin_h).floor().clamp(0, bins_h - 1)
            quantized_xmax = (xmax / size_per_bin_w).floor().clamp(0, bins_w - 1)
            quantized_ymax = (ymax / size_per_bin_h).floor().clamp(0, bins_h - 1)

        elif self.mode == "round":
            raise NotImplementedError()

        else:
            raise ValueError("Incorrect quantization type.")

        quantized_boxes = torch.cat(
            (quantized_xmin, quantized_ymin, quantized_xmax, quantized_ymax), dim=-1
        ).int()

        return quantized_boxes

    def dequantize(self, boxes: torch.Tensor, size):
        bins_w, bins_h = self.bins  # Quantization bins.
        size_w, size_h = size  # Original image size.
        size_per_bin_w = size_w / bins_w
        size_per_bin_h = size_h / bins_h
        xmin, ymin, xmax, ymax = boxes.split(1, dim=-1)  # Shape: 4 * [N, 1].

        if self.mode == "floor":
            # Add 0.5 to use the center position of the bin as the coordinate.
            dequantized_xmin = (xmin + 0.5) * size_per_bin_w
            dequantized_ymin = (ymin + 0.5) * size_per_bin_h
            dequantized_xmax = (xmax + 0.5) * size_per_bin_w
            dequantized_ymax = (ymax + 0.5) * size_per_bin_h

        elif self.mode == "round":
            raise NotImplementedError()

        else:
            raise ValueError("Incorrect quantization type.")

        dequantized_boxes = torch.cat(
            (dequantized_xmin, dequantized_ymin, dequantized_xmax, dequantized_ymax),
            dim=-1,
        )

        return dequantized_boxes


class CoordinatesQuantizer(object):
    """
    Quantize coornidates (Nx2)
    """

    def __init__(self, mode, bins):
        self.mode = mode
        self.bins = bins

    def quantize(self, coordinates: torch.Tensor, size):
        bins_w, bins_h = self.bins  # Quantization bins.
        size_w, size_h = size  # Original image size.
        size_per_bin_w = size_w / bins_w
        size_per_bin_h = size_h / bins_h
        assert coordinates.shape[-1] == 2, "coordinates should be shape (N, 2)"
        x, y = coordinates.split(1, dim=-1)  # Shape: 4 * [N, 1].

        if self.mode == "floor":
            quantized_x = (x / size_per_bin_w).floor().clamp(0, bins_w - 1)
            quantized_y = (y / size_per_bin_h).floor().clamp(0, bins_h - 1)

        elif self.mode == "round":
            raise NotImplementedError()

        else:
            raise ValueError("Incorrect quantization type.")

        quantized_coordinates = torch.cat((quantized_x, quantized_y), dim=-1).int()

        return quantized_coordinates

    def dequantize(self, coordinates: torch.Tensor, size):
        bins_w, bins_h = self.bins  # Quantization bins.
        size_w, size_h = size  # Original image size.
        size_per_bin_w = size_w / bins_w
        size_per_bin_h = size_h / bins_h
        assert coordinates.shape[-1] == 2, "coordinates should be shape (N, 2)"
        x, y = coordinates.split(1, dim=-1)  # Shape: 4 * [N, 1].

        if self.mode == "floor":
            # Add 0.5 to use the center position of the bin as the coordinate.
            dequantized_x = (x + 0.5) * size_per_bin_w
            dequantized_y = (y + 0.5) * size_per_bin_h

        elif self.mode == "round":
            raise NotImplementedError()

        else:
            raise ValueError("Incorrect quantization type.")

        dequantized_coordinates = torch.cat((dequantized_x, dequantized_y), dim=-1)

        return dequantized_coordinates


class Florence2PostProcesser(object):
    """
    Florence-2 post process for converting text prediction to various tasks results.

    Args:
        config: A dict of configs.
        tokenizer: A tokenizer for decoding text to spans.
        sample config:
            UNIFIED_POST_PROCESS:
                # commom configs
                NUM_BBOX_HEIGHT_BINS: 1000
                NUM_BBOX_WIDTH_BINS: 1000
                COORDINATES_HEIGHT_BINS: 1000
                COORDINATES_WIDTH_BINS: 1000
                # task specific configs, override the common configs
                PRASE_TASKS:
                    - TASK_NAME: 'video_dense_caption'
                      PATTERN: 'r<time_(\d+)><time_(\d+)>([a-zA-Z0-9 ]+)'
                      SCORE_MODE: 'avg_cat_name_scores'
                      NUM_BINS: 100
                    - TASK_NAME: 'od'
                      PATTERN: 'r<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>([a-zA-Z0-9 ]+)'
                      SCORE_MODE: 'avg_cat_name_scores'

    Returns:
        parsed_dict (dict): A dict of parsed results.
    """

    def __init__(self, tokenizer=None):
        parse_tasks = []
        parse_task_configs = {}
        config = self._create_default_config()
        for task in config["PARSE_TASKS"]:
            parse_tasks.append(task["TASK_NAME"])
            parse_task_configs[task["TASK_NAME"]] = task

        self.config = config
        self.parse_tasks = parse_tasks
        self.parse_tasks_configs = parse_task_configs

        self.tokenizer = tokenizer
        if self.tokenizer is not None:
            self.all_special_tokens = set(self.tokenizer.all_special_tokens)

        self.init_quantizers()
        self.black_list_of_phrase_grounding = (
            self._create_black_list_of_phrase_grounding()
        )

    def _create_black_list_of_phrase_grounding(self):
        black_list = {}

        if (
            "phrase_grounding" in self.parse_tasks
            and self.parse_tasks_configs["phrase_grounding"]["FILTER_BY_BLACK_LIST"]
        ):
            black_list = set(["it", "I", "me", "mine", "you", "your", "yours", "he", "him", "his", "she", "her", "hers", "they", "them", "their", "theirs", "one", "oneself", "we", "us", "our", "ours", "you", "your", "yours", "they", "them", "their", "theirs", "mine", "yours", "his", "hers", "its", "ours", "yours", "theirs", "myself", "yourself", "himself", "herself", "itself", "ourselves", "yourselves", "themselves", "this", "that", "these", "those", "who", "whom", "whose", "which", "what", "who", "whom", "whose", "which", "that", "all", "another", "any", "anybody", "anyone", "anything", "each", "everybody", "everyone", "everything", "few", "many", "nobody", "none", "one", "several", "some", "somebody", "someone", "something", "each other", "one another", "myself", "yourself", "himself", "herself", "itself", "ourselves", "yourselves", "themselves", "the image", "image", "images", "the", "a", "an", "a group", "other objects", "lots", "a set"]) 
        return black_list

    def _create_default_config(self):
        config = {
            "NUM_BBOX_HEIGHT_BINS": 1000,
            "NUM_BBOX_WIDTH_BINS": 1000,
            "BOX_QUANTIZATION_MODE": "floor",
            "COORDINATES_HEIGHT_BINS": 1000,
            "COORDINATES_WIDTH_BINS": 1000,
            "COORDINATES_QUANTIZATION_MODE": "floor",
            "PARSE_TASKS": [
                {
                    "TASK_NAME": "od",
                    "PATTERN": r"([a-zA-Z0-9 ]+)<loc_(\\d+)><loc_(\\d+)><loc_(\\d+)><loc_(\\d+)>",
                },
                {
                    "TASK_NAME": "ocr",
                    "PATTERN": r"(.+?)<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>",
                    "AREA_THRESHOLD": 0.01,
                },
                {"TASK_NAME": "phrase_grounding", "FILTER_BY_BLACK_LIST": True},
                {
                    "TASK_NAME": "pure_text",
                },
                {
                    "TASK_NAME": "description_with_bboxes",
                },
                {
                    "TASK_NAME": "description_with_polygons",
                },
                {
                    "TASK_NAME": "polygons",
                },
                {
                    "TASK_NAME": "bboxes",
                },
                {
                    "TASK_NAME": "description_with_bboxes_or_polygons",
                },
            ],
        }

        return config

    def init_quantizers(self):
        # we have box_quantizer (od, grounding) and coordinates_quantizer (ocr, referring_segmentation)
        num_bbox_height_bins = self.config.get("NUM_BBOX_HEIGHT_BINS", 1000)
        num_bbox_width_bins = self.config.get("NUM_BBOX_WIDTH_BINS", 1000)
        box_quantization_mode = self.config.get("BOX_QUANTIZATION_MODE", "floor")
        self.box_quantizer = BoxQuantizer(
            box_quantization_mode,
            (num_bbox_width_bins, num_bbox_height_bins),
        )

        num_bbox_height_bins = (
            self.config["COORDINATES_HEIGHT_BINS"]
            if "COORDINATES_HEIGHT_BINS" in self.config
            else self.config.get("NUM_BBOX_HEIGHT_BINS", 1000)
        )
        num_bbox_width_bins = (
            self.config["COORDINATES_WIDTH_BINS"]
            if "COORDINATES_WIDTH_BINS" in self.config
            else self.config.get("NUM_BBOX_WIDTH_BINS", 1000)
        )
        box_quantization_mode = (
            self.config.get("COORDINATES_QUANTIZATION_MODE")
            if "COORDINATES_QUANTIZATION_MODE" in self.config
            else self.config.get("BOX_QUANTIZATION_MODE", "floor")
        )
        self.coordinates_quantizer = CoordinatesQuantizer(
            box_quantization_mode,
            (num_bbox_width_bins, num_bbox_height_bins),
        )

    def decode_with_spans(self, tokenizer, token_ids):
        filtered_tokens = tokenizer.convert_ids_to_tokens(
            token_ids, skip_special_tokens=False
        )
        assert len(filtered_tokens) == len(token_ids)

        # To avoid mixing byte-level and unicode for byte-level BPT
        # we need to build string separately for added tokens and byte-level tokens
        # cf. https://github.com/huggingface/transformers/issues/1133
        sub_texts = []
        for token in filtered_tokens:
            if token in self.all_special_tokens:
                sub_texts.append(token)
            else:
                raise ValueError(f"type {type(tokenizer)} not supported")

        text = ""
        spans = []
        for sub_text in sub_texts:
            span = (len(text), len(text) + len(sub_text))  # [start index, end index).
            text += sub_text
            spans.append(span)

        # Text format:
        # 1. T5Tokenizer/T5TokenizerFast:
        #      "<loc_1><loc_2><loc_3><loc_4> transplanting dog<loc_1><loc_2><loc_3><loc_4> cat</s>"
        #    Equivalent to t5_tokenizer.decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False, spaces_between_special_tokens=False)
        # 2. BartTokenizer (need to double check):
        #      "<s><loc_1><loc_2><loc_3><loc_4>transplanting dog<loc_1><loc_2><loc_3><loc_4>cat</s>"
        #    Equivalent to bart_tokenizer.decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False, spaces_between_special_tokens=False)
        return text, spans

    def parse_od_from_text_and_spans(
        self, text, pattern, image_size, phrase_centric=False
    ):
        parsed = list(re.finditer(pattern, text))

        instances = []
        for i in range(len(parsed)):
            # Prepare instance.
            instance = {}

            if phrase_centric:
                bbox_bins = [int(parsed[i].group(j)) for j in range(2, 6)]
            else:
                bbox_bins = [int(parsed[i].group(j)) for j in range(1, 5)]
            instance["bbox"] = self.box_quantizer.dequantize(
                boxes=torch.tensor(bbox_bins), size=image_size
            ).tolist()

            if phrase_centric:
                instance["cat_name"] = parsed[i].group(1).lower().strip()
            else:
                instance["cat_name"] = parsed[i].group(5).lower().strip()
            instances.append(instance)

        return instances

    def parse_ocr_from_text_and_spans(
        self,
        text,
        pattern,
        image_size,
        area_threshold=-1.0,
    ):
        bboxes = []
        labels = []
        text = text.replace("<s>", "")
        # ocr with regions
        parsed = re.findall(pattern, text)
        instances = []
        image_width, image_height = image_size

        for ocr_line in parsed:
            ocr_content = ocr_line[0]
            quad_box = ocr_line[1:]
            quad_box = [int(i) for i in quad_box]
            quad_box = (
                self.coordinates_quantizer.dequantize(
                    torch.tensor(np.array(quad_box).reshape(-1, 2)), size=image_size
                )
                .reshape(-1)
                .tolist()
            )

            if area_threshold > 0:
                x_coords = [i for i in quad_box[0::2]]
                y_coords = [i for i in quad_box[1::2]]

                # apply the Shoelace formula
                area = 0.5 * abs(
                    sum(
                        x_coords[i] * y_coords[i + 1] - x_coords[i + 1] * y_coords[i]
                        for i in range(4 - 1)
                    )
                )

                if area < (image_width * image_height) * area_threshold:
                    continue

            bboxes.append(quad_box)
            labels.append(ocr_content)
            instances.append(
                {
                    "quad_box": quad_box,
                    "text": ocr_content,
                }
            )
        return instances

    def parse_phrase_grounding_from_text_and_spans(self, text, pattern, image_size):
        # ignore <s> </s> and <pad>
        cur_span = 0
        if text.startswith("<s>"):
            cur_span += 3

        text = text.replace("<s>", "")
        text = text.replace("</s>", "")
        text = text.replace("<pad>", "")

        pattern = r"([^<]+(?:<loc_\d+>){4,})"
        phrases = re.findall(pattern, text)

        # pattern should be text pattern and od pattern
        pattern = r"^\s*(.*?)(?=<od>|</od>|<box>|</box>|<bbox>|</bbox>|<loc_)"
        box_pattern = r"<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>"

        instances = []
        for pharse_text in phrases:
            phrase_text_strip = pharse_text.replace("<ground>", "", 1)
            phrase_text_strip = pharse_text.replace("<obj>", "", 1)

            if phrase_text_strip == "":
                cur_span += len(pharse_text)
                continue

            # Prepare instance.
            instance = {}

            # parse phrase, get string
            phrase = re.search(pattern, phrase_text_strip)
            if phrase is None:
                cur_span += len(pharse_text)
                continue

            # parse bboxes by box_pattern
            bboxes_parsed = list(re.finditer(box_pattern, pharse_text))
            if len(bboxes_parsed) == 0:
                cur_span += len(pharse_text)
                continue

            phrase = phrase.group()
            # remove leading and trailing spaces
            phrase = phrase.strip()

            if phrase in self.black_list_of_phrase_grounding:
                cur_span += len(pharse_text)
                continue

            # a list of list
            bbox_bins = [
                [int(_bboxes_parsed.group(j)) for j in range(1, 5)]
                for _bboxes_parsed in bboxes_parsed
            ]
            instance["bbox"] = self.box_quantizer.dequantize(
                boxes=torch.tensor(bbox_bins), size=image_size
            ).tolist()

            # exclude non-ascii characters
            phrase = phrase.encode("ascii", errors="ignore").decode("ascii")
            instance["cat_name"] = phrase

            instances.append(instance)

        return instances

    def parse_description_with_bboxes_from_text_and_spans(
        self, text, pattern, image_size, allow_empty_phrase=False
    ):
        # temporary parse solution, split by '.'
        # ignore <s> </s> and <pad>

        text = text.replace("<s>", "")
        text = text.replace("</s>", "")
        text = text.replace("<pad>", "")

        if allow_empty_phrase:
            pattern = rf"(?:(?:<loc_\d+>){{4,}})"
        else:
            pattern = r"([^<]+(?:<loc_\d+>){4,})"
        phrases = re.findall(pattern, text)

        # pattern should be text pattern and od pattern
        pattern = r"^\s*(.*?)(?=<od>|</od>|<box>|</box>|<bbox>|</bbox>|<loc_)"
        box_pattern = r"<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>"

        instances = []
        for pharse_text in phrases:
            phrase_text_strip = pharse_text.replace("<ground>", "", 1)
            phrase_text_strip = pharse_text.replace("<obj>", "", 1)

            if phrase_text_strip == "" and not allow_empty_phrase:
                continue

            # parse phrase, get string
            phrase = re.search(pattern, phrase_text_strip)
            if phrase is None:
                continue

            phrase = phrase.group()
            # remove leading and trailing spaces
            phrase = phrase.strip()

            # parse bboxes by box_pattern
            bboxes_parsed = list(re.finditer(box_pattern, pharse_text))
            if len(bboxes_parsed) == 0:
                continue

            # a list of list
            bbox_bins = [
                [int(_bboxes_parsed.group(j)) for j in range(1, 5)]
                for _bboxes_parsed in bboxes_parsed
            ]

            bboxes = self.box_quantizer.dequantize(
                boxes=torch.tensor(bbox_bins), size=image_size
            ).tolist()

            phrase = phrase.encode("ascii", errors="ignore").decode("ascii")
            for _bboxes in bboxes:
                # Prepare instance.
                instance = {}
                instance["bbox"] = _bboxes
                # exclude non-ascii characters
                instance["cat_name"] = phrase
                instances.append(instance)

        return instances

    def parse_description_with_polygons_from_text_and_spans(
        self,
        text,
        pattern,
        image_size,
        allow_empty_phrase=False,
        polygon_sep_token="<sep>",
        polygon_start_token="<poly>",
        polygon_end_token="</poly>",
        with_box_at_start=False,
    ):

        # ref_seg format: '<expression><x1><y1><x2><y2><><><sep><><><><>'
        # ignore <s> </s> and <pad>

        text = text.replace("<s>", "")
        text = text.replace("</s>", "")
        text = text.replace("<pad>", "")

        if allow_empty_phrase:
            pattern = rf"(?:(?:<loc_\d+>|{re.escape(polygon_sep_token)}|{re.escape(polygon_start_token)}|{re.escape(polygon_end_token)}){{4,}})"
        else:
            # [^<]+: This part matches one or more characters that are not the < symbol.
            # The ^ inside the square brackets [] is a negation, meaning it matches anything except <.
            #
            pattern = rf"([^<]+(?:<loc_\d+>|{re.escape(polygon_sep_token)}|{re.escape(polygon_start_token)}|{re.escape(polygon_end_token)}){{4,}})"
        phrases = re.findall(pattern, text)

        phrase_string_pattern = (
            r"^\s*(.*?)(?=<od>|</od>|<box>|</box>|<bbox>|</bbox>|<loc_|<poly>)"
        )
        box_pattern = rf"((?:<loc_\d+>)+)(?:{re.escape(polygon_sep_token)}|$)"

        # one polygons instance is separated by polygon_start_token and polygon_end_token
        polygons_instance_pattern = (
            rf"{re.escape(polygon_start_token)}(.*?){re.escape(polygon_end_token)}"
        )

        instances = []
        for phrase_text in phrases:

            # exclude loc_\d+>
            # need to get span if want to include category score
            phrase_text_strip = re.sub(r"^loc_\d+>", "", phrase_text, count=1)

            # phrase = phrase.replace('<poly>', '')
            # phrase = phrase.replace('poly>', '')

            if phrase_text_strip == "" and not allow_empty_phrase:
                continue

            # parse phrase, get string
            phrase = re.search(phrase_string_pattern, phrase_text_strip)
            if phrase is None:
                continue
            phrase = phrase.group()
            # remove leading and trailing spaces
            phrase = phrase.strip()

            # parse bboxes by box_pattern

            # split by polygon_start_token and polygon_end_token first using polygons_instance_pattern
            if polygon_start_token in phrase_text and polygon_end_token in phrase_text:
                polygons_instances_parsed = list(
                    re.finditer(polygons_instance_pattern, phrase_text)
                )
            else:
                polygons_instances_parsed = [phrase_text]

            for _polygons_instances_parsed in polygons_instances_parsed:
                # Prepare instance.
                instance = {}

                # polygons_parsed= list(re.finditer(box_pattern, phrase_text))
                if isinstance(_polygons_instances_parsed, str):
                    polygons_parsed = list(
                        re.finditer(box_pattern, _polygons_instances_parsed)
                    )
                else:
                    polygons_parsed = list(
                        re.finditer(box_pattern, _polygons_instances_parsed.group(1))
                    )
                if len(polygons_parsed) == 0:
                    continue

                # a list of list (polygon)
                bbox = []
                polygons = []
                for _polygon_parsed in polygons_parsed:
                    # group 1: whole <loc_\d+>...</loc_\d+>
                    _polygon = _polygon_parsed.group(1)
                    # parse into list of int
                    _polygon = [
                        int(_loc_parsed.group(1))
                        for _loc_parsed in re.finditer(r"<loc_(\d+)>", _polygon)
                    ]
                    if with_box_at_start and len(bbox) == 0:
                        if len(_polygon) > 4:
                            # no valid bbox prediction
                            bbox = _polygon[:4]
                            _polygon = _polygon[4:]
                        else:
                            bbox = [0, 0, 0, 0]
                    # abandon last element if is not paired
                    if len(_polygon) % 2 == 1:
                        _polygon = _polygon[:-1]

                    # reshape into (n, 2)
                    _polygon = (
                        self.coordinates_quantizer.dequantize(
                            torch.tensor(np.array(_polygon).reshape(-1, 2)),
                            size=image_size,
                        )
                        .reshape(-1)
                        .tolist()
                    )
                    # reshape back
                    polygons.append(_polygon)

                instance["cat_name"] = phrase
                instance["polygons"] = polygons
                if len(bbox) != 0:
                    instance["bbox"] = self.box_quantizer.dequantize(
                        boxes=torch.tensor([bbox]), size=image_size
                    ).tolist()[0]

                instances.append(instance)

        return instances

    def __call__(
        self,
        text=None,
        image_size=None,
        parse_tasks=None,
    ):
        """
        Args:
            text: model outputs
            image_size: (width, height)
            parse_tasks: a list of tasks to parse, if None, parse all tasks.

        """
        if parse_tasks is not None:
            if isinstance(parse_tasks, str):
                parse_tasks = [parse_tasks]
            for _parse_task in parse_tasks:
                assert (
                    _parse_task in self.parse_tasks
                ), f"parse task {_parse_task} not supported"

        # sequence or text should be provided
        assert text is not None, "text should be provided"

        parsed_dict = {"text": text}

        for task in self.parse_tasks:
            if parse_tasks is not None and task not in parse_tasks:
                continue

            pattern = self.parse_tasks_configs[task].get("PATTERN", None)

            if task == "ocr":
                instances = self.parse_ocr_from_text_and_spans(
                    text,
                    pattern=pattern,
                    image_size=image_size,
                    area_threshold=self.parse_tasks_configs[task].get(
                        "AREA_THRESHOLD", 0.01
                    ),
                )
                parsed_dict["ocr"] = instances
            elif task == "phrase_grounding":
                instances = self.parse_phrase_grounding_from_text_and_spans(
                    text,
                    pattern=pattern,
                    image_size=image_size,
                )
                parsed_dict["phrase_grounding"] = instances
            elif task == "pure_text":
                parsed_dict["pure_text"] = text
            elif task == "description_with_bboxes":
                instances = self.parse_description_with_bboxes_from_text_and_spans(
                    text,
                    pattern=pattern,
                    image_size=image_size,
                )
                parsed_dict["description_with_bboxes"] = instances
            elif task == "description_with_polygons":
                instances = self.parse_description_with_polygons_from_text_and_spans(
                    text,
                    pattern=pattern,
                    image_size=image_size,
                )
                parsed_dict["description_with_polygons"] = instances
            elif task == "polygons":
                instances = self.parse_description_with_polygons_from_text_and_spans(
                    text,
                    pattern=pattern,
                    image_size=image_size,
                    allow_empty_phrase=True,
                )
                parsed_dict["polygons"] = instances
            elif task == "bboxes":
                instances = self.parse_description_with_bboxes_from_text_and_spans(
                    text,
                    pattern=pattern,
                    image_size=image_size,
                    allow_empty_phrase=True,
                )
                parsed_dict["bboxes"] = instances
            elif task == "description_with_bboxes_or_polygons":
                if "<poly>" in text:
                    # only support either polygons or bboxes, not both at the same time
                    instances = (
                        self.parse_description_with_polygons_from_text_and_spans(
                            text,
                            pattern=pattern,
                            image_size=image_size,
                        )
                    )
                else:
                    instances = self.parse_description_with_bboxes_from_text_and_spans(
                        text,
                        pattern=pattern,
                        image_size=image_size,
                    )
                parsed_dict["description_with_bboxes_or_polygons"] = instances
            else:
                raise ValueError("task {} is not supported".format(task))

        return parsed_dict


tasks_answer_post_processing_type = {
    "<OCR>": "pure_text",
    "<OCR_WITH_REGION>": "ocr",
    "<CAPTION>": "pure_text",
    "<DETAILED_CAPTION>": "pure_text",
    "<MORE_DETAILED_CAPTION>": "pure_text",
    "<OD>": "description_with_bboxes",
    "<DENSE_REGION_CAPTION>": "description_with_bboxes",
    "<CAPTION_TO_PHRASE_GROUNDING>": "phrase_grounding",
    "<REFERRING_EXPRESSION_SEGMENTATION>": "polygons",
    "<REGION_TO_SEGMENTATION>": "polygons",
    "<OPEN_VOCABULARY_DETECTION>": "description_with_bboxes_or_polygons",
    "<REGION_TO_CATEGORY>": "pure_text",
    "<REGION_TO_DESCRIPTION>": "pure_text",
    "<REGION_TO_OCR>": "pure_text",
    "<REGION_PROPOSAL>": "bboxes",
}


def init_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokens_to_add = {
        "additional_special_tokens": tokenizer.additional_special_tokens
        + ["<od>", "</od>", "<ocr>", "</ocr>"]
        + [f"<loc_{x}>" for x in range(1000)]
        + [ "<cap>", "</cap>", "<ncap>", "</ncap>", "<dcap>", "</dcap>", "<grounding>", "</grounding>", "<seg>", "</seg>", "<sep>", "<region_cap>", "</region_cap>", "<region_to_desciption>", "</region_to_desciption>", "<proposal>", "</proposal>", "<poly>", "</poly>", "<and>"]
    }
    tokenizer.add_special_tokens(tokens_to_add)
    return tokenizer


def generate_text(task_prompt, tensor, onnx_file, tokenizer, text_input=None, dev="cpu"):

    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    input_ids_np = tokenizer(prompt, return_tensors="pt")["input_ids"]
    input_ids_np= input_ids_np.to(dev).cpu().numpy()
    pixel_values_np = tensor.cpu().numpy()

    # model export
    ort_session = ort.InferenceSession(onnx_file)

    # generated_tokens = [start_token_id]
    generated_tokens = [0]
    max_length = 60

    for step in range(max_length):
        decoder_input_ids = torch.tensor([generated_tokens], dtype=torch.long)
        decoder_input_ids_np = decoder_input_ids.cpu().numpy()

        outputs = ort_session.run(
            None,
            {
                "input_ids": input_ids_np,
                "pixel_values": pixel_values_np,
                "decoder_input_ids": decoder_input_ids_np,
            },
        )
        logits = outputs[0]
        next_token_logits = logits[:, -1, :]
        next_token_id = np.argmax(next_token_logits, axis=-1)[0]

        generated_tokens.append(int(next_token_id))

        if next_token_id == 1:
            # actually should be == model.config.eos_token_id, but we don't have model
            # just in case ask Imran
            break

    # Декодирование сгенерированных токенов
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
    return generated_text


def post_process_generation(text, task, image_size, tokenizer):
    """
    Post-process the output of the model to each of the task outputs.

    Args:
        tokenizer:
        text (`str`): The text to post-process.
        task (`str`): The task to post-process the text for.
        image_size (`Tuple[int, int]`): The size of the image. height x width.
    """

    post_processor = Florence2PostProcesser(tokenizer=tokenizer)
    task_answer_post_processing_type = tasks_answer_post_processing_type.get(
        task, "pure_text"
    )
    task_answer = post_processor(
        text=text,
        image_size=image_size,
        parse_tasks=task_answer_post_processing_type,
    )[task_answer_post_processing_type]

    if task_answer_post_processing_type == "pure_text":
        final_answer = task_answer
        # remove the special tokens
        final_answer = final_answer.replace("<s>", "").replace("</s>", "\n")
    elif task_answer_post_processing_type in [
        "od",
        "description_with_bboxes",
        "bboxes",
    ]:
        od_instances = task_answer
        bboxes_od = [_od_instance["bbox"] for _od_instance in od_instances]
        labels_od = [str(_od_instance["cat_name"]) for _od_instance in od_instances]
        final_answer = {"bboxes": bboxes_od, "labels": labels_od}
    elif task_answer_post_processing_type in ["ocr"]:
        bboxes = [_od_instance["quad_box"] for _od_instance in task_answer]
        labels = [str(_od_instance["text"]) for _od_instance in task_answer]
        final_answer = {"quad_boxes": bboxes, "labels": labels}
    elif task_answer_post_processing_type in ["phrase_grounding"]:
        bboxes = []
        labels = []
        for _grounded_phrase in task_answer:
            for _bbox in _grounded_phrase["bbox"]:
                bboxes.append(_bbox)
                labels.append(_grounded_phrase["cat_name"])
        final_answer = {"bboxes": bboxes, "labels": labels}
    elif task_answer_post_processing_type in ["description_with_polygons", "polygons"]:
        labels = []
        polygons = []
        for result in task_answer:
            label = result["cat_name"]
            _polygons = result["polygons"]
            labels.append(label)
            polygons.append(_polygons)
        final_answer = {"polygons": polygons, "labels": labels}
    elif task_answer_post_processing_type in ["description_with_bboxes_or_polygons"]:
        bboxes = []
        bboxes_labels = []
        polygons = []
        polygons_labels = []
        for result in task_answer:
            label = result["cat_name"]
            if "polygons" in result:
                _polygons = result["polygons"]
                polygons.append(_polygons)
                polygons_labels.append(label)
            else:
                _bbox = result["bbox"]
                bboxes.append(_bbox)
                bboxes_labels.append(label)
        final_answer = {
            "bboxes": bboxes,
            "bboxes_labels": bboxes_labels,
            "polygons": polygons,
            "polygons_labels": polygons_labels,
        }
    else:
        raise ValueError(
            "Unknown task answer post processing type: {}".format(
                task_answer_post_processing_type
            )
        )

    final_answer = {task: final_answer}
    return final_answer


def draw_bbox_on_image(image, results):
    bboxes = results["<CAPTION_TO_PHRASE_GROUNDING>"]["bboxes"]
    labels = results["<CAPTION_TO_PHRASE_GROUNDING>"]["labels"]

    draw = ImageDraw.Draw(image)

    bbox_color = (255, 0, 0)  # red
    bbox_width = 3

    for bbox, label in zip(bboxes, labels):
        x_min, y_min, x_max, y_max = bbox
        draw.rectangle(
            [x_min, y_min, x_max, y_max], outline=bbox_color, width=bbox_width
        )

        # marks
        try:
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            original_size = 12
            font_size = original_size * 2
            font = ImageFont.truetype(font_path, font_size)
            text_position = (x_min, y_min - font_size - 1)  # Положение текста над bbox
            draw.text(text_position, label, fill=bbox_color, font=font)
        except Exception as e:
            logging.error(f"Не удалось загрузить шрифт: {e}")
    return image


