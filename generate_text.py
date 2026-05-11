import torch


def generate_text_simple(model, current_context, max_new_tokens, context_size):
    # current_context is (batch, n_tokens) array of indices in the current context

    for _ in range(max_new_tokens):
        # crop the current context if it exceeds the supported context
        # size example, if LLM supports only 5 tokens, and the context
        # size is 10 then only the last 5 tokens are used as context
        cropped_context = current_context[:, -context_size:]

        # compute prediction
        # this tells PyTorch to disable the gradient calculation engine,
        # thereby saving memory (stops taking derivatives, so no training).
        # otherwise PyTorch remembers every operation (mul, add, etc.) so
        # that it can calc gradients during backprop much quicker.
        with torch.no_grad():
            logits, _ = model(cropped_context)

        # focus only on the last token
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # for the last index, pick the token with the highest
        # probability
        probas = torch.softmax(logits, dim=-1)  # batch, vocab_size
        next_idx = torch.argmax(probas, dim=-1, keepdim=True)  # batch, 1

        current_context = torch.cat(
            (current_context, next_idx), dim=1
        )  # batch, n_tokens+1

    return current_context


def generate_text_with_kv_cache(model, current_context, max_new_tokens, context_size):
    # current_context is (batch, n_tokens) array of indices in the current context

    kv_cache = None

    for _ in range(max_new_tokens):
        # crop the current context if it exceeds the supported context
        # size example, if LLM supports only 5 tokens, and the context
        # size is 10 then only the last 5 tokens are used as context
        cropped_context = current_context[:, -context_size:]

        # In the first step, we pass the whole prompt.
        # In all following steps, we only pass the LAST token [:, -1:]
        cropped_context = (
            cropped_context if kv_cache is None else cropped_context[:, -1:]
        )

        # compute prediction
        # this tells PyTorch to disable the gradient calculation engine,
        # thereby saving memory (stops taking derivatives, so no training).
        # otherwise PyTorch remembers every operation (mul, add, etc.) so
        # that it can calc gradients during backprop much quicker.
        with torch.no_grad():
            logits, kv_cache = model(cropped_context, kv_cache)

        # focus only on the last token
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # for the last index, pick the token with the highest
        # probability
        probas = torch.softmax(logits, dim=-1)  # batch, vocab_size
        next_idx = torch.argmax(probas, dim=-1, keepdim=True)  # batch, 1

        current_context = torch.cat(
            (current_context, next_idx), dim=1
        )  # batch, n_tokens+1

    return current_context
