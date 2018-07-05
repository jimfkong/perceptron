from Weight import Feat


class Feature:
    """
    Context surrounding a word.
    """
    def __init__(self, word, pos_tag, chunk, gold_standard, additional_feats, word_stem):
        """
        Create a new feature.
        :param word: Word the feature is about.
        :param pos_tag: PoS tag of the word.
        :param chunk: Syntactic chunk of the word.
        :param gold_standard: Actual NER tag.
        :param additional_feats: Counter of additional features.
        :param word_stem: Stemmed version of the word.
        """
        self.prev_word = None
        self.next_word = None

        self.values = [None] * len(Feat.__members__)

        self.values[Feat.BOW] = word
        self.values[Feat.POS] = pos_tag
        self.values[Feat.CHUNK] = chunk
        if word_stem is not None:
            self.values[Feat.STEM] = word_stem

        self.additional_feats = additional_feats
        self.gold_standard = gold_standard

    def set_next(self, next_word):
        """
        Set the next word. If possible, also sets the previous word's next-next word.
        :param next_word: Feature representing the next word.
        """
        self.next_word = next_word
        self.values[Feat.BOW_NEXT] = next_word.values[Feat.BOW]
        self.values[Feat.NEXT_POS] = next_word.values[Feat.POS]

        if self.prev_word is not None:
            self.prev_word.values[Feat.BOW_NEXT_NEXT] = next_word.values[Feat.BOW]
            self.prev_word.values[Feat.NEXT_NEXT_POS] = next_word.values[Feat.POS]

    def set_prev(self, prev_word):
        """
        Set the previous word. If possible, also sets the next word's previous-previous word.
        :param prev_word: Feature representing the previous word.
        """
        self.prev_word = prev_word
        self.values[Feat.BOW_PREV] = prev_word.values[Feat.BOW]
        self.values[Feat.PREV_POS] = prev_word.values[Feat.POS]

        if prev_word.prev_word is not None:
            self.values[Feat.BOW_PREV_PREV] = prev_word.values[Feat.BOW_PREV]
            self.values[Feat.PREV_PREV_POS] = prev_word.values[Feat.PREV_POS]
