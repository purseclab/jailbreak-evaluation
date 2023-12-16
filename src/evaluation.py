import abc


class Evaluation(abc.ABC):
    @abc.abstractmethod
    def __call__(self, safeguard: str, intent: str, response: str) -> bool:
        pass

class ChaoEvaluation(Evaluation):
    # [1] P. Chao, A. Robey, E. Dobriban, H. Hassani, G. J. Pappas, and E. Wong, “Jailbreaking Black Box Large Language Models in Twenty Queries.” arXiv, Oct. 13, 2023. doi: 10.48550/arXiv.2310.08419.

    def __call__(self, safeguard: str, intent: str, response: str) -> bool:
        raise NotImplementedError

class HuangEvaluation(Evaluation):
    # [1] Y. Huang, S. Gupta, M. Xia, K. Li, and D. Chen, “Catastrophic Jailbreak of Open-source LLMs via Exploiting Generation.” arXiv, Oct. 10, 2023. doi: 10.48550/arXiv.2310.06987.

    def __call__(self, safeguard: str, intent: str, response: str) -> bool:
        raise NotImplementedError

class ZouEvaluation(Evaluation):
    # [1] A. Zou, Z. Wang, J. Z. Kolter, and M. Fredrikson, “Universal and Transferable Adversarial Attacks on Aligned Language Models.” arXiv, Jul. 27, 2023. doi: 10.48550/arXiv.2307.15043.

    def __call__(self, safeguard: str, intent: str, response: str) -> bool:
        raise NotImplementedError