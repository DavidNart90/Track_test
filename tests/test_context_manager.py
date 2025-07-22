from src.trackrealties.agents.context import ContextManager


def test_context_persistence():
    cm = ContextManager()
    ctx1 = cm.get_or_create_context("s1", user_id="u1", user_role="investor")
    ctx1.user_preferences["budget"] = "$100k"
    cm.update_context("s1", ctx1)

    ctx2 = cm.get_or_create_context("s2", user_id="u1", user_role="investor")
    assert ctx2.user_preferences.get("budget") == "$100k"
