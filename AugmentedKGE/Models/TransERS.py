from Models.TransE import TransE


class TransERS(TransE):
    """
    Xiaofei Zhou, Qiannan Zhu, Ping Liu, Li Guo: Learning Knowledge Embeddings by Combining Limit-based Scoring Loss.
        CIKM 2017: 1009-1018.
    """
    def get_default_loss(self):
        return 'margin_limit'
