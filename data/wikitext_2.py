import re
from datasets import load_dataset


class DataSet:

    @staticmethod
    def wiki_text_2(
        max_length=64,
        stride=48,
    ):
        print("Loading WikiText-2...")

        dataset = load_dataset(
            "Self-GRIT/wikitext-2-raw-v1-preprocessed",
            split="train"
        )

        corpus = []

        for sample in dataset:

            text = sample["text"].strip()

            if not text:
                continue

            # lowercase
            text = text.lower()

            # collapse whitespace
            text = re.sub(r"\s+", " ", text)

            # split into sentences
            sentences = re.split(r"[.!?]+", text)

            for sentence in sentences:

                sentence = sentence.strip()

                if not sentence:
                    continue

                words = sentence.split()

                # ignore tiny sentences
                if len(words) < 2:
                    continue

                # short sentence
                if len(words) <= max_length - 2:
                    corpus.append(" ".join(words))
                    continue

                # long sentence -> overlapping windows
                start = 0

                while start < len(words):

                    chunk = words[start:start + max_length - 2]

                    if len(chunk) < 2:
                        break

                    corpus.append(" ".join(chunk))

                    if start + max_length - 2 >= len(words):
                        break

                    start += stride

        return corpus