# -*- coding: utf-8 -*-


from gensim.summarization import bm25


class BM25Util(object):
    def __init__(self, corpus):
        self.bm25 = bm25.BM25(corpus)
        self.average_idf = sum(
            map(lambda k: float(self.bm25.idf[k]), self.bm25.idf.keys())
        ) / len(self.bm25.idf.keys())

    def similarity(self, query, size=10):
        # scores = self.bm25.get_scores(query, self.average_idf)
        scores = self.bm25.get_scores(query)
        scores_sort = sorted(
            list(enumerate(scores)), key=lambda item: item[1], reverse=True
        )
        return scores_sort[:size]

