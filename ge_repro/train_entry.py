"""The paper's assistant-token mean, using the stock HF causal loss.

Qwen3-VL in the pinned Transformers version ignores num_items_in_batch in
its internal forward loss. Use the Trainer loss hook so the denominator
includes every valid assistant token across the 4x8 optimizer batch.
"""


def token_mean_loss(outputs, labels, num_items_in_batch=None):
    from transformers.loss.loss_utils import ForCausalLMLoss
    return ForCausalLMLoss(
        outputs.logits, labels, outputs.logits.shape[-1],
        num_items_in_batch=num_items_in_batch,
    )


def main():
    from llamafactory.train.sft import workflow
    from llamafactory.train.tuner import run_exp

    class TokenMeanTrainer(workflow.CustomSeq2SeqTrainer):
        def __init__(self, *args, **kwargs):
            if kwargs.get("compute_loss_func") is not None:
                raise ValueError("Unexpected second loss override")
            kwargs["compute_loss_func"] = token_mean_loss
            super().__init__(*args, **kwargs)

    workflow.CustomSeq2SeqTrainer = TokenMeanTrainer
    run_exp()


if __name__ == "__main__":
    main()
