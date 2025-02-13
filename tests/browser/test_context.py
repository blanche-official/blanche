import pytest

from blanche.actions.base import Action
from blanche.browser.context import Context
from blanche.browser.node_type import A11yNode, A11yTree, NodeRole, BlancheNode
from blanche.browser.snapshot import BrowserSnapshot, SnapshotMetadata


@pytest.fixture
def nested_graph() -> BlancheNode:
    return BlancheNode(
        id=None,
        role=NodeRole.GROUP,
        text="root",
        children=[
            BlancheNode(id="A1", role=NodeRole.BUTTON, text="A1"),
            BlancheNode(id="A2", role=NodeRole.BUTTON, text="A2"),
            BlancheNode(id="A3", role=NodeRole.BUTTON, text="A3"),
            BlancheNode(id=None, role=NodeRole.TEXT, text="text"),
            BlancheNode(
                id=None,
                role=NodeRole.GROUP,
                text="B1",
                children=[
                    BlancheNode(id="B1", role=NodeRole.BUTTON, text="B1"),
                    BlancheNode(id="B2", role=NodeRole.BUTTON, text="B2"),
                    BlancheNode(id=None, role=NodeRole.TEXT, text="text"),
                ],
            ),
            BlancheNode(id="A4", role=NodeRole.BUTTON, text="A4"),
            BlancheNode(
                id=None,
                role=NodeRole.GROUP,
                text="B2",
                children=[
                    BlancheNode(id="B3", role=NodeRole.BUTTON, text="B3"),
                    BlancheNode(id="B4", role=NodeRole.BUTTON, text="B4"),
                    BlancheNode(id=None, role=NodeRole.TEXT, text="text"),
                    BlancheNode(
                        id=None,
                        role=NodeRole.GROUP,
                        text="C",
                        children=[
                            BlancheNode(id="C1", role=NodeRole.BUTTON, text="C1"),
                            BlancheNode(id="C2", role=NodeRole.BUTTON, text="C2"),
                            BlancheNode(id=None, role=NodeRole.TEXT, text="text"),
                        ],
                    ),
                ],
            ),
        ],
    )


@pytest.fixture
def browser_snapshot() -> BrowserSnapshot:
    empty_a11y_tree = A11yNode(
        role="root",
        name="root",
        children=[],
    )
    return BrowserSnapshot(
        metadata=SnapshotMetadata(
            url="https://example.com",
            title="example",
        ),
        html_content="my html content",
        a11y_tree=A11yTree(empty_a11y_tree, empty_a11y_tree),
        screenshot=None,
    )


def test_subgraph_without_existing_actions(nested_graph: BlancheNode, browser_snapshot: BrowserSnapshot):
    context = Context(snapshot=browser_snapshot, node=nested_graph)
    assert len(context.interaction_nodes()) == 10
    # test with A1
    subgraph = context.subgraph_without([Action(id="A1", description="A1", category="A1")])
    assert subgraph is not None
    assert subgraph.node.find("A1") is None
    assert len(subgraph.interaction_nodes()) == 9
    # test with A1, A2, A3
    subgraph = context.subgraph_without(
        [
            Action(id="A1", description="A1", category="A1"),
            Action(id="A2", description="A2", category="A2"),
            Action(id="A3", description="A3", category="A3"),
        ]
    )
    assert subgraph is not None
    assert subgraph.node.find("A1") is None
    assert subgraph.node.find("A2") is None
    assert subgraph.node.find("A3") is None
    assert len(subgraph.interaction_nodes()) == 7
    # test with B1, B2, C2
    subgraph = context.subgraph_without(
        [
            Action(id="A1", description="A1", category="A1"),
            Action(id="A2", description="A2", category="A2"),
            Action(id="A3", description="A3", category="A3"),
            Action(id="B1", description="B1", category="B1"),
            Action(id="B2", description="B2", category="B2"),
            Action(id="C2", description="C2", category="C2"),
        ]
    )
    assert subgraph is not None
    assert subgraph.node.find("B1") is None
    assert subgraph.node.find("B2") is None
    assert subgraph.node.find("C2") is None
    assert len(subgraph.interaction_nodes()) == 4
    # exclude all
    subgraph = context.subgraph_without(
        [
            Action(id="A1", description="A1", category="A1"),
            Action(id="A2", description="A2", category="A2"),
            Action(id="A3", description="A3", category="A3"),
            Action(id="A4", description="A4", category="A4"),
            Action(id="B1", description="B1", category="B1"),
            Action(id="B2", description="B2", category="B2"),
            Action(id="B3", description="B3", category="B3"),
            Action(id="B4", description="B4", category="B4"),
            Action(id="C1", description="C1", category="C1"),
            Action(id="C2", description="C2", category="C2"),
        ]
    )
    assert subgraph is None
