from mecha_agent_cli.sandbox.command_policy import CommandPolicy


def test_command_policy_allows_pytest() -> None:
    assert CommandPolicy().check(["pytest", "-q"]).allowed


def test_command_policy_rejects_curl() -> None:
    assert not CommandPolicy().check(["curl", "https://example.com"]).allowed
