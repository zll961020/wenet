# Copyright    2023  Xiaomi Corp.        (authors: Wei Kang)
#              2023  Binbin Zhang (binbzha@qq.com)
#              2023  Kaixun Huang
#              2023  Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
# See ../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from wenet.text.tokenize_utils import tokenize_by_bpe_model
from wenet.text.whisper_tokenizer import WhisperTokenizer
from typing import Dict, List, Tuple
from collections import deque
from typing import Dict, List, Tuple, Optional            # + Optional
try:                                                     # 新增：检测 graphviz
    import graphviz
    _GRAPHVIZ_AVAILABLE = True
except ImportError:                                      # pragma: no cover
    _GRAPHVIZ_AVAILABLE = False

def tokenize(context_list_path, symbol_table, bpe_model=None, tokenizer=None):
    """ Read biasing list from the biasing list address, tokenize and convert it
        into token id
    """
    if bpe_model is not None:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(bpe_model)
    else:
        sp = None

    with open(context_list_path, "r") as fin:
        context_txts = fin.readlines()

    context_list = []
    for context_txt in context_txts:
        context_txt = context_txt.strip()

        labels = []
        tokens = []
        if bpe_model is not None:
            tokens = tokenize_by_bpe_model(sp, context_txt)
        else:
            for ch in context_txt:
                if ch == ' ':
                    ch = "▁"
                tokens.append(ch)
        if tokenizer is not None and isinstance(tokenizer, WhisperTokenizer):
            _, ids = tokenizer.tokenize(context_txt)
            labels.extend(ids)          # id 本身就是 int，可直接用
        else:
            for ch in tokens:
                if ch in symbol_table:
                    labels.append(symbol_table[ch])
                elif '<unk>' in symbol_table:
                    labels.append(symbol_table['<unk>'])
        context_list.append(labels)
    return context_list


class ContextState:
    """The state in ContextGraph"""

    def __init__(
        self,
        id: int,
        token: int,
        token_score: float,
        node_score: float,
        output_score: float,
        is_end: bool,
    ):
        """Create a ContextState.

        Args:
          id:
            The node id, only for visualization now. A node is in [0, graph.num_nodes).
            The id of the root node is always 0.
          token:
            The token id.
          token_score:
            The bonus for each token during decoding, which will hopefully
            boost the token up to survive beam search.
          node_score:
            The accumulated bonus from root of graph to current node, it will be
            used to calculate the score for fail arc.
          output_score:
            The total scores of matched phrases, sum of the node_score of all
            the output node for current node.
          is_end:
            True if current token is the end of a context.
        """
        self.id = id
        self.token = token
        self.token_score = token_score
        self.node_score = node_score
        self.output_score = output_score
        self.is_end = is_end
        self.next = {}
        self.fail = None
        self.output = None


class ContextGraph:
    """The ContextGraph is modified from Aho-Corasick which is mainly
    a Trie with a fail arc for each node.
    See https://en.wikipedia.org/wiki/Aho%E2%80%93Corasick_algorithm for more details
    of Aho-Corasick algorithm.

    A ContextGraph contains some words / phrases that we expect to boost their
    scores during decoding. If the substring of a decoded sequence matches the word / phrase  # noqa
    in the ContextGraph, we will give the decoded sequence a bonus to make it survive
    beam search.
    """

    def __init__(self,
                 context_list_path: str,
                 symbol_table: Dict[str, int],
                 bpe_model: str = None,
                 context_score: float = 6.0,
                 tokenizer=None):
        """Initialize a ContextGraph with the given ``context_score``.

        A root node will be created (**NOTE:** the token of root is hardcoded to -1).

        Args:
          context_score:
            The bonus score for each token(note: NOT for each word/phrase, it means longer  # noqa
            word/phrase will have larger bonus score, they have to be matched though).
        """
        self.tokenizer = tokenizer
        self.context_score = context_score
        self.context_list = tokenize(context_list_path, symbol_table,
                                     bpe_model, tokenizer)
        self.inv_sym_table = {v: k for k, v in symbol_table.items()}
        self.num_nodes = 0
        self.root = ContextState(
            id=self.num_nodes,
            token=-1,
            token_score=0,
            node_score=0,
            output_score=0,
            is_end=False,
        )
        self.root.fail = self.root
        self.build_graph(self.context_list)

    def build_graph(self, token_ids: List[List[int]]):
        """Build the ContextGraph from a list of token list.
        It first build a trie from the given token lists, then fill the fail arc
        for each trie node.

        See https://en.wikipedia.org/wiki/Trie for how to build a trie.

        Args:
          token_ids:
            The given token lists to build the ContextGraph, it is a list of token list,
            each token list contains the token ids for a word/phrase. The token id
            could be an id of a char (modeling with single Chinese char) or an id
            of a BPE (modeling with BPEs).
        """
        for tokens in token_ids:
            node = self.root
            for i, token in enumerate(tokens):
                if token not in node.next:
                    self.num_nodes += 1
                    is_end = i == len(tokens) - 1
                    node_score = node.node_score + self.context_score
                    node.next[token] = ContextState(
                        id=self.num_nodes,
                        token=token,
                        token_score=self.context_score,
                        node_score=node_score,
                        output_score=node_score if is_end else 0,
                        is_end=is_end,
                    )
                node = node.next[token]
        self._fill_fail_output()  # AC

    def _fill_fail_output(self):
        """This function fills the fail arc for each trie node, it can be computed
        in linear time by performing a breadth-first search starting from the root.
        See https://en.wikipedia.org/wiki/Aho%E2%80%93Corasick_algorithm for the
        details of the algorithm.
        """
        queue = deque()
        for token, node in self.root.next.items():
            node.fail = self.root
            queue.append(node)
        while queue:
            current_node = queue.popleft()
            for token, node in current_node.next.items():
                fail = current_node.fail
                if token in fail.next:
                    fail = fail.next[token]
                else:
                    fail = fail.fail
                    while token not in fail.next:
                        fail = fail.fail
                        if fail.token == -1:  # root
                            break
                    if token in fail.next:
                        fail = fail.next[token]
                node.fail = fail
                # fill the output arc
                output = node.fail
                while not output.is_end:
                    output = output.fail
                    if output.token == -1:  # root
                        output = None
                        break
                node.output = output
                node.output_score += 0 if output is None else output.output_score
                queue.append(node)

    def forward_one_step(self, state: ContextState,
                         token: int) -> Tuple[float, ContextState]:
        """Search the graph with given state and token.

        Args:
          state:
            The given token containing trie node to start.
          token:
            The given token.

        Returns:
          Return a tuple of score and next state.
        """
        node = None
        score = 0
        # token matched
        if token in state.next:
            node = state.next[token]
            score = node.token_score
        else:
            # token not matched
            # We will trace along the fail arc until it matches the token or reaching
            # root of the graph.
            node = state.fail
            while token not in node.next:
                node = node.fail
                if node.token == -1:  # root
                    break

            if token in node.next:
                node = node.next[token]

            # The score of the fail path
            score = node.node_score - state.node_score
        assert node is not None
        return (score + node.output_score, node)

    def finalize(self, state: ContextState) -> Tuple[float, ContextState]:
        """When reaching the end of the decoded sequence, we need to finalize
        the matching, the purpose is to subtract the added bonus score for the
        state that is not the end of a word/phrase.

        Args:
          state:
            The given state(trie node).

        Returns:
          Return a tuple of score and next state. If state is the end of a word/phrase
          the score is zero, otherwise the score is the score of a implicit fail arc
          to root. The next state is always root.
        """
        # The score of the fail arc
        score = -state.node_score
        return (score, self.root)


# ------------------- 新增：可视化 -------------------
    def draw(
        self,
        title: Optional[str] = None,
        font: str = "Noto Sans CJK SC"
    ):
        """
        使用 graphviz 将 ContextGraph(Aho-Corasick 自动机) 绘制成有向图。

        节点标签:  id/(node_score, output_score)
        黑色实线 : 正常 next 转移
        红色虚线 : fail  失败边
        绿色点线 : output 输出边

        返回 graphviz.Digraph 对象，可直接 dot.render(...) 导出。
        """
        if not _GRAPHVIZ_AVAILABLE:
            raise ImportError("graphviz is not installed, please `pip install graphviz`")

        graph_attr = {
            "rankdir": "LR",
            "size": "8.5,11",
            "center": "1",
            "ranksep": "0.3",
            "nodesep": "0.25",
            "charset": "utf-8",
        }
        if title is not None:
            graph_attr["label"] = title

        default_node_attr = {
            "shape": "circle",
            "fontsize": "12",
            "fontname": font,
        }
        final_node_attr   = {
            "shape": "doublecircle",
            "fontsize": "12",
            "fontname": font,
        }
        default_edge_attr = {
            "fontsize": "12",
            "fontname": font,
        }

        dot = graphviz.Digraph(
            name="ContextGraph",
            graph_attr=graph_attr,
            node_attr=default_node_attr,
            edge_attr=default_edge_attr,
        )

        visited, queue = set(), deque([self.root])
        while queue:
            node = queue.popleft()

            # 1. 打印节点
            if node.id not in visited:
                lbl = f"{node.id}/({node.node_score}, {node.output_score})"
                if node.is_end:
                    dot.node(str(node.id), label=lbl, **final_node_attr)
                else:
                    dot.node(str(node.id), label=lbl)
                visited.add(node.id)

            # 2. next 边（黑色；标签=token/score）
            for tok, nxt in node.next.items():
                if isinstance(self.tokenizer, WhisperTokenizer):
                    # Whisper: 用 detokenize 把单个 id 反解为字符 组合的id才有结果 还是限制token id
                    tok_lbl = str(tok)
                else:
                    tok_lbl = self.inv_sym_table.get(tok, str(tok))
                label   = f"{tok_lbl}/{nxt.token_score}"   # ★ 新增：拼上分数
                dot.edge(str(node.id), str(nxt.id), label=label)   # 默认黑色实线
                if nxt.id not in visited:
                    queue.append(nxt)

            # 3. fail 边（红色实线）
            if node.fail is not None and node.fail.id != node.id:
                dot.edge(
                    str(node.id),
                    str(node.fail.id),
                    label="ε",
                    color="red",                    # 红色
                    style="solid",                  # 实线
                    arrowhead="normal",             # 有箭头
                )

            # 4. output 边（绿色点线）
            if node.output is not None:
                print(f'output node is not None')
                dot.edge(str(node.id), str(node.output.id), label="out", color="green", style="dotted", penwidth="2")

        return dot
