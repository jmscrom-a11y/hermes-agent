from dataclasses import dataclass
from pathlib import Path
import secrets
import shlex
import subprocess
import time

from app.config.settings import ALLOWED_FILE_DIRS, ALLOWED_SHELL_COMMANDS


class SafetyError(ValueError):
    pass


@dataclass
class PendingAction:
    token: str
    kind: str
    payload: dict
    preview: str
    expires_at: float = 0.0  # epoch seconds; 0 = 무기한 (기존 호환)


_pending_actions = {}


def _allowed_dirs():
    return [Path(path).expanduser().resolve() for path in ALLOWED_FILE_DIRS]


def resolve_allowed_path(file_path):
    target = Path(file_path).expanduser()
    if not target.is_absolute():
        target = Path.cwd() / target
    target = target.resolve()

    for allowed_dir in _allowed_dirs():
        # 허용 디렉토리 내부(하위 경로 포함)인지 체크
        # trailing separator로 접두어 매칭 우회 방지 (/tmp vs /tmpfoo)
        if target == allowed_dir or str(target).startswith(str(allowed_dir) + "/"):
            return target

    allowed = ", ".join(str(path) for path in _allowed_dirs())
    raise SafetyError(f"Path is outside allowed folders: {target}. Allowed: {allowed}")


def _new_pending_action(kind, payload, preview, ttl: int = 300):
    token = secrets.token_urlsafe(8)
    action = PendingAction(
        token=token,
        kind=kind,
        payload=payload,
        preview=preview,
        expires_at=time.time() + ttl,
    )
    _pending_actions[token] = action
    return action


def _cleanup_expired():
    now = time.time()
    expired = [t for t, a in _pending_actions.items() if a.expires_at and now >= a.expires_at]
    for token in expired:
        _pending_actions.pop(token, None)


def dry_run_write_file(file_path, content):
    target = resolve_allowed_path(file_path)
    exists = target.exists()
    preview = "\n".join(
        [
            "DRY-RUN file write",
            f"path: {target}",
            f"exists: {exists}",
            f"bytes: {len(content.encode('utf-8'))}",
        ]
    )
    return _new_pending_action(
        "write_file",
        {"path": str(target), "content": content},
        preview,
    )


def dry_run_delete_file(file_path):
    target = resolve_allowed_path(file_path)
    if target.exists() and not target.is_file():
        raise SafetyError(f"Only files can be deleted: {target}")

    preview = "\n".join(
        [
            "DRY-RUN file delete",
            f"path: {target}",
            f"exists: {target.exists()}",
        ]
    )
    return _new_pending_action(
        "delete_file",
        {"path": str(target)},
        preview,
    )


def _parse_shell_command(command):
    parts = shlex.split(command)
    if not parts:
        raise SafetyError("Shell command is empty.")
    if parts[0] not in ALLOWED_SHELL_COMMANDS:
        allowed = ", ".join(ALLOWED_SHELL_COMMANDS)
        raise SafetyError(f"Shell command is not whitelisted: {parts[0]}. Allowed: {allowed}")
    return parts


def dry_run_shell(command):
    parts = _parse_shell_command(command)
    preview = "\n".join(
        [
            "DRY-RUN shell command",
            f"command: {shlex.join(parts)}",
            f"allowed: {parts[0]}",
        ]
    )
    return _new_pending_action(
        "shell",
        {"command": command},
        preview,
    )


def confirm_action(token):
    _cleanup_expired()
    action = _pending_actions.pop(token, None)
    if action is None:
        raise SafetyError("Confirmation token is invalid or expired.")

    if action.kind == "write_file":
        path = resolve_allowed_path(action.payload["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(action.payload["content"], encoding="utf-8")
        return f"file written: {path}"

    if action.kind == "delete_file":
        path = resolve_allowed_path(action.payload["path"])
        if path.exists() and not path.is_file():
            raise SafetyError(f"Only files can be deleted: {path}")
        if path.exists():
            path.unlink()
            return f"file deleted: {path}"
        return f"file already missing: {path}"

    if action.kind == "shell":
        parts = _parse_shell_command(action.payload["command"])
        result = subprocess.run(
            parts,
            cwd=str(_allowed_dirs()[0]),
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = result.stdout.strip() or result.stderr.strip()
        if not output:
            output = "(no output)"
        return f"exit_code: {result.returncode}\n{output}"

    raise SafetyError(f"Unknown action kind: {action.kind}")


def cancel_action(token):
    _cleanup_expired()
    action = _pending_actions.pop(token, None)
    if action is None:
        raise SafetyError("Confirmation token is invalid or expired.")
    return f"cancelled: {action.kind}"


def clear_pending_actions():
    _pending_actions.clear()
