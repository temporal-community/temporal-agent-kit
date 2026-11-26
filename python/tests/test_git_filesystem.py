# ABOUTME: Tests for git_repository_filesystem: a read-only ActivityFileSystem over a git
# repository. It reads a ref's tree without a checkout, picks up new commits once its clone is
# old, never moves a full commit id, and refuses a config that names anything but a repository
# the worker listed.

from __future__ import annotations

import asyncio
import subprocess
from collections import OrderedDict
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from temporalio.exceptions import ApplicationError
from temporalio.testing import ActivityEnvironment

from temporal_agent_kit import agent
from temporal_agent_kit.code_mode import git_fs
from temporal_agent_kit.code_mode.activity_fs import FsCall, FsRead, OKFGraphCall

NOTE = "---\ntype: Fact\ntitle: A\n---\nSee [b](b.md).\n"


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return done.stdout.strip()


def _commit(repo: Path, files: dict[str, bytes]) -> str:
    for name, data in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_bytes(data)
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "change")
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def origin(tmp_path: Path) -> Path:
    repo = tmp_path / "origin"
    repo.mkdir()
    _git(repo, "init", "--quiet", "--initial-branch=main")
    _commit(
        repo,
        {
            "README.md": b"top",
            "wiki/a.md": NOTE.encode(),
            "wiki/b.md": b"---\ntype: Fact\n---\n",
            "wiki/img/logo.bin": b"\x00\xff",
        },
    )
    return repo


def _filesystem(
    origin: Path, tmp_path: Path, refresh_after: timedelta = timedelta(0)
) -> dict[str, Any]:
    fs = agent.git_repository_filesystem(
        "test-git",
        repositories={"kb": str(origin)},
        cache_dir=tmp_path / "cache",
        refresh_after=refresh_after,
    )
    return {a.__temporal_activity_definition.name: a for a in fs.activities}


def test_it_is_read_only(origin: Path, tmp_path: Path):
    assert sorted(_filesystem(origin, tmp_path)) == [
        "vfs.test-git.list",
        "vfs.test-git.okf_graph",
        "vfs.test-git.read",
        "vfs.test-git.stat",
        "vfs.test-git.view",
    ]


async def test_it_reads_the_tree_of_a_ref(origin: Path, tmp_path: Path):
    fs = _filesystem(origin, tmp_path)
    env = ActivityEnvironment()
    config = {"repository": "kb", "ref": "main"}

    assert await env.run(fs["vfs.test-git.list"], FsCall(config=config, path=".")) == [
        "README.md",
        "wiki",
    ]
    assert await env.run(fs["vfs.test-git.list"], FsCall(config=config, path="wiki")) == [
        "a.md",
        "b.md",
        "img",
    ]
    text = await env.run(fs["vfs.test-git.read"], FsRead(config=config, path="wiki/a.md"))
    assert (text.content, text.encoding) == (NOTE, "utf-8")
    part = await env.run(
        fs["vfs.test-git.read"], FsRead(config=config, path="wiki/a.md", offset=4, length=4)
    )
    assert part.content == "type"
    binary = await env.run(fs["vfs.test-git.read"], FsRead(config=config, path="wiki/img/logo.bin"))
    assert binary.encoding == "base64"

    file = (await env.run(fs["vfs.test-git.stat"], FsCall(config=config, path="wiki/a.md"))).found
    assert (file.is_dir, file.size) == (False, len(NOTE))
    assert file.mtime == float(_git(origin, "show", "-s", "--format=%ct", "HEAD"))
    folder = (await env.run(fs["vfs.test-git.stat"], FsCall(config=config, path="wiki"))).found
    assert folder.is_dir
    missing = await env.run(fs["vfs.test-git.stat"], FsCall(config=config, path="nope.md"))
    assert missing.found is None
    # The clone is bare: nothing is checked out on the worker.
    assert not list((tmp_path / "cache").rglob("README.md"))


async def test_a_subdirectory_is_the_root_of_the_mount(origin: Path, tmp_path: Path):
    fs = _filesystem(origin, tmp_path)
    env = ActivityEnvironment()
    config = {"repository": "kb", "ref": "main", "subdirectory": "wiki"}

    assert await env.run(fs["vfs.test-git.list"], FsCall(config=config, path=".")) == [
        "a.md",
        "b.md",
        "img",
    ]
    outside = await env.run(fs["vfs.test-git.stat"], FsCall(config=config, path="README.md"))
    assert outside.found is None
    graph = await env.run(fs["vfs.test-git.okf_graph"], OKFGraphCall(config=config, path="."))
    assert [(c.id, c.title) for c in graph.concepts] == [("a", "A"), ("b", "b")]
    assert [(l.source, l.target) for l in graph.links] == [("a", "b")]


async def test_a_branch_follows_new_commits_and_a_commit_id_does_not(origin: Path, tmp_path: Path):
    fs = _filesystem(origin, tmp_path)
    env = ActivityEnvironment()
    first = _git(origin, "rev-parse", "HEAD")
    read = fs["vfs.test-git.read"]
    head = {"repository": "kb"}
    assert (await env.run(read, FsRead(config=head, path="README.md"))).content == "top"

    _commit(origin, {"README.md": b"changed"})

    branch = {"repository": "kb", "ref": "main"}
    assert (await env.run(read, FsRead(config=branch, path="README.md"))).content == "changed"
    pinned = {"repository": "kb", "ref": first}
    assert (await env.run(read, FsRead(config=pinned, path="README.md"))).content == "top"


async def test_a_recent_clone_is_not_fetched_again(origin: Path, tmp_path: Path):
    fs = _filesystem(origin, tmp_path, refresh_after=timedelta(hours=1))
    env = ActivityEnvironment()
    config = {"repository": "kb", "ref": "main"}
    read = fs["vfs.test-git.read"]
    assert (await env.run(read, FsRead(config=config, path="README.md"))).content == "top"

    second = _commit(origin, {"README.md": b"changed"})

    assert (await env.run(read, FsRead(config=config, path="README.md"))).content == "top"
    # A commit id that the clone does not have yet is fetched whatever the clone's age.
    pinned = {"repository": "kb", "ref": second}
    assert (await env.run(read, FsRead(config=pinned, path="README.md"))).content == "changed"


async def test_one_key_for_two_urls_gives_two_clones(tmp_path: Path):
    reads = {}
    for name in ("a", "b"):
        repo = tmp_path / name
        repo.mkdir()
        _git(repo, "init", "--quiet", "--initial-branch=main")
        _commit(repo, {"README.md": f"repo {name}".encode()})
        fs = agent.git_repository_filesystem(
            f"test-git-{name}", repositories={"kb": str(repo)}, cache_dir=tmp_path / "cache"
        )
        activities = {a.__temporal_activity_definition.name: a for a in fs.activities}
        reads[name] = activities[f"vfs.test-git-{name}.read"]
    env = ActivityEnvironment()
    call = FsRead(config={"repository": "kb", "ref": "main"}, path="README.md")

    for name in ("a", "b", "a"):
        assert (await env.run(reads[name], call)).content == f"repo {name}"
    assert len(list((tmp_path / "cache").glob("kb-*.git"))) == 2


@pytest.fixture
def git_runs(monkeypatch: pytest.MonkeyPatch) -> list[tuple[tuple[Any, ...], Any]]:
    runs: list[tuple[tuple[Any, ...], Any]] = []
    spawn = asyncio.create_subprocess_exec

    async def spy(*args: Any, **kwargs: Any) -> Any:
        process = await spawn(*args, **kwargs)
        runs.append((args, process))
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spy)
    return runs


async def test_a_commits_tree_is_listed_once_for_many_calls(
    origin: Path, tmp_path: Path, git_runs: list[tuple[tuple[Any, ...], Any]]
):
    fs = _filesystem(origin, tmp_path, refresh_after=timedelta(hours=1))
    env = ActivityEnvironment()
    config = {"repository": "kb", "ref": "main"}
    for path in ("README.md", "wiki", "wiki/a.md"):
        await env.run(fs["vfs.test-git.stat"], FsCall(config=config, path=path))
    assert await env.run(fs["vfs.test-git.list"], FsCall(config=config, path="wiki")) == [
        "a.md",
        "b.md",
        "img",
    ]

    assert [args[1] for args, _ in git_runs].count("ls-tree") == 1


async def test_old_trees_are_dropped_once_too_many_paths_are_kept(
    origin: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(git_fs, "_trees", OrderedDict())
    monkeypatch.setattr(git_fs, "_PATHS_KEPT", 1)
    fs = _filesystem(origin, tmp_path)
    env = ActivityEnvironment()
    first = _git(origin, "rev-parse", "HEAD")
    second = _commit(origin, {"README.md": b"changed"})
    for ref in (first, second):
        call = FsCall(config={"repository": "kb", "ref": ref}, path=".")
        await env.run(fs["vfs.test-git.list"], call)

    assert [commit for _, commit, _ in git_fs._trees] == [second]


@pytest.mark.skipif(not git_fs._KILL_GROUP, reason="stopping git's helpers needs process groups")
async def test_a_cancelled_git_call_stops_git_and_its_children(
    tmp_path: Path, git_runs: list[tuple[tuple[Any, ...], Any]]
):
    started, stopped = tmp_path / "started", tmp_path / "stopped"
    script = f'trap \'touch "{stopped}"; exit 1\' TERM; touch "{started}"; sleep 10 & wait'
    call = asyncio.create_task(git_fs._git("-c", f"alias.slow=!{script}", "slow", cwd=tmp_path))
    async with asyncio.timeout(5):
        while not started.exists():
            await asyncio.sleep(0.01)
    call.cancel()
    # The sleep holds git's pipes, so the call only ends quickly if the sleep was stopped too.
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(call, timeout=5)

    assert git_runs[0][1].returncode is not None
    # Only SIGTERM runs the trap, and git needs SIGTERM to remove its lock files.
    assert stopped.exists()


@pytest.mark.parametrize(
    ("config", "path", "error_type"),
    [
        ({"repository": "other"}, "README.md", "ValueError"),
        ({"repository": "kb", "ref": "--upload-pack=x"}, "README.md", "ValueError"),
        ({"repository": "kb", "ref": "no-such-branch"}, "README.md", "ValueError"),
        ({"repository": "kb", "subdirectory": "../x"}, "README.md", "ValueError"),
        ({"repository": "kb", "subdirectory": "README.md"}, ".", "ValueError"),
        ({"repository": "kb"}, "nope.md", "FileNotFoundError"),
        ({"repository": "kb"}, "wiki", "IsADirectoryError"),
    ],
)
async def test_a_bad_config_or_path_fails_without_a_retry(
    origin: Path, tmp_path: Path, config: dict[str, str], path: str, error_type: str
):
    fs = _filesystem(origin, tmp_path)
    with pytest.raises(ApplicationError) as raised:
        await ActivityEnvironment().run(fs["vfs.test-git.read"], FsRead(config=config, path=path))
    assert raised.value.type == error_type
    assert raised.value.non_retryable


def test_a_repository_key_must_be_a_plain_name():
    with pytest.raises(ValueError, match="repository key"):
        agent.git_repository_filesystem("test-git", repositories={"../kb": "url"})
