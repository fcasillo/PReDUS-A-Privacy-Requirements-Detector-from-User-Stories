import collections
from string import punctuation

class Liwc():
    """
    Class for the Linguistic Inquiry and Word Count (LIWC) dictionairy.
    The dictionary files are proprietary and can be obtained by liwc.net
    """

    def __init__(self, filepath):
        """
        :param filepath: path to the LIWC .dic file.
        """
        self.categories, self.lexicon = self._load_dict_file(filepath)
        self._trie = self._build_char_trie(self.lexicon)

    def getCategoryDescription(self,category):
        if category=="NegativePrivacy":
            return "Antecedents and consequences of negative privacy experiences"
        elif category=="Restriction":
            return "Restrictive and regulatory behaviors for maintaining privacy"
        elif category== "NormsRequisites":
            return "Norms, beliefs and expectations in relation to achieving privacy"
        elif category=="OutcomeState":
            return "Behavioral states and the outcomes that are served through privacy"
        elif category=="OpenVisible":
            return "Open and public access to people"
        elif category=="PrivateSecret":
            return "The 'content' of privacy, i.e., what is considered private"
        elif category=="Intimacy":
            return "Small group privacy marked by group inclusion and intimacy"
        elif category=="SafetyProtect":
            return "Feeling safe and protecting or guarding oneself"

    def search(self, word):
        """
        Search a word in the liwc dictionairy.
        :param word:
        :return: a list of the liwc categories the word belongs.
                 an empty list if the word is not found in the dictionary.
        """
        return self._search_trie(self._trie, word)

    def parse(self, tokens):
        """
        Parses a document and extracts raw counts of words that fall into the
        various LIWC categories.
        :param tokens: a list of tokens, a tokeniSed document
        :return: a counter with the linguistic categories found in the doc,
                and the raw count of words that fall in each category.
        """
        cat_counter = collections.Counter()
        keywords = []
        words_category = []

        for token in tokens:
            # Find in which categories this token falls, if any
            cats = self.search(token)
            for cat in cats:
                if cat != 'PrivacyTotal':
                    words_category.append([token,cat,self.getCategoryDescription(cat)])
                    cat_counter[cat] += 1
            if cats != []:
                for character in token:  
                    if character in punctuation:  
                        token = token.replace(character, "")
                keywords.append(token)
        return cat_counter, keywords, words_category

    def _load_dict_file(self, filepath):
        liwc_file = open(filepath)

        # Key, category dict
        categories = {}

        # Word, cat_name dict
        lexicon = {}

        # '%' signals a change in the .dic file.
        # (0-1) Cats, ids
        # (>1) Words, cat_ids
        percent_sign_count = 0

        for line in liwc_file:
            stp = line.strip()

            if stp:
                parts = stp.split('\t')

                if parts[0] == '%':
                    percent_sign_count += 1
                else:
                    # If the percent sign counter equals 1, parse the LIWC
                    # categories
                    if percent_sign_count == 1:
                        categories[parts[0]] = parts[1]
                    # Else, parse lexicon
                    else:
                        lexicon[parts[0]] = [categories[cat_id]
                                             for cat_id in parts[1:]]

        return categories, lexicon

    @staticmethod
    def _build_char_trie(lexicon):
        """
        Builds a char trie, to cater for wildcard ('*') matches.
        """
        trie = {}
        for pattern, cat_names in lexicon.items():
            cursor = trie
            for char in pattern:
                if char == '*':
                    cursor['*'] = cat_names
                    break

                if char not in cursor:
                    cursor[char] = {}

                cursor = cursor[char]

            # $ signifies end of token
            cursor['$'] = cat_names

        return trie

    @staticmethod
    def _search_trie(trie, token, i=0):
        """
        Search the given char trie for paths that match the token.
        """
        if '*' in trie:
            return trie['*']
        elif '$' in trie and i == len(token):
            return trie['$']
        elif i < len(token):
            char = token[i]
            if char in trie:
                return Liwc._search_trie(trie[char], token, i + 1)
        return []
