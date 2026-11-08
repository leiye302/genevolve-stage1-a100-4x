"""Run the pinned trainer's real multimodal label builder on a published row."""
from pathlib import Path

from .common import rows
from .data import sft_registry


def check_sample(base, source, media_dir):
    from llamafactory.data.converter import SharegptDatasetConverter
    from llamafactory.data.parser import DatasetAttr
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.hparams import DataArguments, ModelArguments
    from llamafactory.model.loader import load_tokenizer

    data_args = DataArguments(template="qwen3_vl_nothink", cutoff_len=32768,
                              train_on_prompt=False, mask_history=False, media_dir=str(media_dir))
    attr = DatasetAttr("file", Path(source).name)
    attr.join(sft_registry()["genevolve_train"])
    source_row = next(rows(source))
    converted = SharegptDatasetConverter(attr, data_args)(source_row)
    model_args = ModelArguments(model_name_or_path=str(base), trust_remote_code=False,
                                image_max_pixels=589824, image_min_pixels=1024)
    objects = load_tokenizer(model_args)
    tokenizer, processor = objects["tokenizer"], objects["processor"]
    if processor is None:
        raise ValueError("The real Qwen3-VL processor did not load")
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    encoded = SupervisedDatasetProcessor(template, tokenizer, processor, data_args).preprocess_dataset(
        {key: [value] for key, value in converted.items()}
    )
    (ids,), (labels,) = encoded["input_ids"], encoded["labels"]
    tokens = [x for x in labels if x != -100]
    text = tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    assistant = [m["content"] for m in source_row["messages"] if m["role"] == "assistant"]
    if any(t.strip() not in text for t in assistant):
        raise ValueError("The real trainer did not preserve each complete assistant turn in the fixture")
    if any(m["content"] in text for m in source_row["messages"] if m["role"] != "assistant"):
        raise ValueError("The real trainer supervised a non-assistant message")
    visual = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    visual_count = ids.count(visual)
    if source_row["images"] and (visual_count <= 0 or visual in tokens):
        raise ValueError("The image tokens are absent or incorrectly supervised")
    return {"input_tokens": len(ids), "assistant_target_tokens": len(tokens),
            "assistant_turns": len(assistant), "masked_visual_tokens": visual_count,
            "image_assets": len(source_row["images"])}
