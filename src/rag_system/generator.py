from transformers import AutoModelForCausalLM, AutoTokenizer


class Generator:
    """Generate answers with Qwen."""

    def __init__(self) -> None:
        model_name = "Qwen/Qwen3-0.6B"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )

    def generate(self, prompt: str) -> str:
        """
        Generate an answer from a prompt.
        START
        1. Load Qwen tokenizer.
        2. Load Qwen model.

        3. RECEIVE a prompt from the user.

        4. PUT the prompt inside a message structure.

        5. CONVERT the message into Qwen chat format.

        6. CONVERT the formatted text into token IDs.
        MOVE the input to the model's device.

        7. GIVE the input to Qwen.
        GENERATE new token IDs.

        8. REMOVE the original input tokens.
        KEEP only the generated tokens.

        9. TRY to find the last </think> token.
        IF found:
            START after that token.
        IF not found:
            START from the beginning.

        10. CONVERT the selected token IDs into normal text.
            REMOVE extra newline characters.

        11. RETURN the final answer.

        END
        """

        messages = [
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )


        model_inputs = self.tokenizer(
            [text],
            return_tensors="pt",
        ).to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=32768,
        )
        output_ids = generated_ids[0][
            len(model_inputs.input_ids[0]):
        ].tolist()

        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        content = self.tokenizer.decode(
            output_ids[index:],
            skip_special_tokens=True,
        ).strip("\n")

        return content

generator = Generator()

answer = generator.generate("what is python")
