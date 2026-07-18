import random


def generate_synthetic_corpus(num_sentences=100, seed=42):
    """
    Generates a synthetic corpus with high structural similarity to the original
    7 sentences (Subject-Verb-Object/Adverb, short length, limited vocab).
    """
    random.seed(seed)

    # Vocabulary derived from your original dataset + slight expansions
    subjects = ['dog', 'cat', 'bird', 'fish', 'bear', 'wolf', 'fox', 'deer', 'rabbit', 'lion',
                'tiger', 'eagle', 'shark', 'whale', 'tree', 'flower', 'river', 'mountain', 'cloud', 'storm',
                'wind', 'rain', 'sun', 'moon', 'star', 'king', 'queen', 'knight', 'wizard', 'dragon',
                'ghost', 'robot', 'child', 'boy', 'girl', 'man', 'woman', 'teacher', 'doctor', 'artist',
                'writer', 'singer', 'dancer', 'farmer', 'builder', 'leader', 'hero', 'villain', 'monster', 'alien']

    verbs = ['run', 'walk', 'jump', 'swim', 'fly', 'sing', 'dance', 'write', 'read', 'eat',
             'sleep', 'fight', 'hide', 'seek', 'build', 'break', 'hold', 'drop', 'push', 'pull',
             'think', 'feel', 'know', 'see', 'hear', 'speak', 'listen', 'give', 'take', 'find',
             'lose', 'make', 'destroy', 'heal', 'grow', 'shrink', 'rise', 'fall', 'shine', 'watch']

    objects = ['ball', 'stick', 'stone', 'book', 'sword', 'shield', 'crown', 'food', 'water', 'ice',
               'gold', 'silver', 'wood', 'metal', 'cloth', 'glass', 'paper', 'rope', 'chain', 'car',
               'boat', 'train', 'plane', 'house', 'castle', 'bridge', 'road', 'path', 'key', 'lock',
               'door', 'window', 'wall', 'gate', 'garden', 'field', 'forest', 'desert', 'ocean', 'lake',
               'valley', 'island', 'city', 'town', 'village', 'room', 'hall', 'tower', 'temple', 'ship']

    adjectives = ['big', 'small', 'fast', 'slow', 'hot', 'cold', 'bright', 'dark', 'loud', 'quiet',
                  'heavy', 'light', 'old', 'new', 'good', 'bad', 'happy', 'sad', 'brave', 'scared',
                  'strong', 'weak', 'tall', 'short', 'wide', 'narrow', 'thick', 'thin', 'rough', 'smooth']

    adverbs = ['quickly', 'slowly', 'loudly', 'quietly', 'brightly', 'softly', 'happily', 'sadly', 'bravely',
               'cowardly',
               'carefully', 'recklessly', 'easily', 'hardly', 'often', 'rarely', 'always', 'never', 'gently',
               'fiercely']

    # Templates mimicking the original sentence structures
    templates = [
        "{subj} {verb}.",
        "{subj} {verb} {adv}.",
        "the {subj} can {verb}.",
        "{subj} {verb} the {obj}.",
        "the {subj} likes the {obj}.",
        "the {subj} sees the {obj}.",
        "the {adj} {subj} {verb}.",
        "the {adj} {subj} {verb} {adv}.",
        "the {subj} lives near the {obj}.",
        "the {subj} walks with the {obj}.",
    ]
    corpus = []
    for _ in range(num_sentences):
        template = random.choice(templates)
        sentence = template.format(
            subj=random.choice(subjects),
            verb=random.choice(verbs),
            adv=random.choice(adverbs),
            obj=random.choice(objects),
            adj=random.choice(adjectives)
        )
        # Ensure lowercase to match the tokenizer's expectations
        corpus.append(sentence.lower())

    return corpus
