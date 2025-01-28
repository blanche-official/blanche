from typing import final

from blanche.browser.context import Context
from blanche.browser.node_type import BlancheNode
from blanche.browser.snapshot import BrowserSnapshot
from blanche.pipe.preprocessing.a11y.tree import ProcessedA11yTree


@final
class ActionA11yPipe:

    @staticmethod
    def forward(snapshot: BrowserSnapshot) -> Context:
        processed_tree = ProcessedA11yTree.from_a11y_tree(snapshot.a11y_tree)
        return Context(
            snapshot=snapshot,
            node=BlancheNode.from_a11y_node(
                node=processed_tree.processed_tree,
                path=snapshot.metadata.url,
            ),
        )
